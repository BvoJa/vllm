# Plan v6: CPU Hot-Path Optimization for 3-Core Environments

## Diagnosis Summary

The prefix cache code (`cache_blocks`, `cache_full_blocks`) **already short-circuits
correctly during decode**. The early return at `single_type_kv_cache_manager.py:341`:

```python
if num_cached_blocks >= num_full_blocks:
    return
```

fires on 15/16 decode steps, costing ~1 dict get + 1 int division + 1 comparison
per request — about **4 μs per step for 8 concurrent requests**. Not the bottleneck.

The 2–5 ms per step comes from **two sources**:

| Source | Est. cost | Fixed by |
|---|---|---|
| Kernel launch overhead (~100 PyTorch ops × 10–30 μs) | 1–3 ms | CUDA graphs |
| Python scheduler loop (state checks, iteration, transitions) | 1–2 ms | profile-guided trimming |

CUDA graphs are enabled by default at optimization level O2
(`cudagraph_mode = FULL_AND_PIECEWISE` in `compilation.py`). If they are working
correctly, the 2–5 ms is **entirely Python scheduling overhead**, not kernel launches.

---

## Optimizations

### 1. Config: Verify CUDA Graphs Are Active

**File:** `vllm/config/vllm.py:207` — default preset sets `cudagraph_mode: NONE`,
but the `optimization_level` field (default `O2`) overrides it to
`FULL_AND_PIECEWISE` at `vllm.py:1243`.

**Action:** Confirm with runtime logs that CUDA graphs are being captured and
replayed. If decode steps show `"mode": "FULL"` or `"mode": "PIECEWISE"` in
the model runner output, graphs are working.

If graphs are NOT active (fallback to `NONE`), the override at `vllm.py:1217`
may have triggered due to lack of attention backend support. Check:

```python
# In vllm/config/vllm.py, around 1217:
if self.optimization_level > OptimizationLevel.O0:
    # ... may downgrade cudagraph_mode to NONE if attention backend
    # doesn't support it
```

For a 1.2B model on H200, the flash-attention backend should support
piecewise CUDA graphs. If not, set explicitly:

```bash
--compilation-config '{"cudagraph_mode": "piecewise"}'
```

### 2. Config: Disable Non-Essential I/O

In a 3-core environment, async I/O threads compete for CPU time.

```bash
--disable-log-requests --disable-log-stats
```

If KV cache events are not needed for monitoring, ensure they are disabled at
the block-pool level (`enable_kv_cache_events: False` in config). This skips
the `_build_block_stored_event` path in `cache_full_blocks`.

### 3. Code: Remove List Slice in `cache_full_blocks`

**File:** `vllm/v1/core/block_pool.py:271`

```python
new_block_hashes = block_hashes[num_cached_blocks:]
```

allocates a new Python list on every call. While this is on the 1-in-16
decode path, it is still ~8,000 allocations over the full 70×6-turn trace.
Replace with index-based access in the loop:

```python
# Before:
new_block_hashes = block_hashes[num_cached_blocks:]
for i, blk in enumerate(new_full_blocks):
    block_hash = new_block_hashes[i]
    ...

# After:
block_hashes_len = len(block_hashes)
for i, blk in enumerate(new_full_blocks):
    idx = num_cached_blocks + i
    if idx >= block_hashes_len:
        break  # safety, should not happen
    block_hash = block_hashes[idx]
    ...
```

This also avoids slicing into the `block_hashes` list in the event block
at line 310 where `block_hashes[num_cached_blocks - 1]` is accessed
(a single-element index, fine as-is).

### 4. Code: Skip `new_hashes` List Creation When Events Disabled

**File:** `vllm/v1/core/block_pool.py:272–274`

```python
new_hashes: list[ExternalBlockHash] | None = (
    [] if self.enable_kv_cache_events else None
)
```

This already correctly skips allocation when events are off. No change needed.

### 5. Profile: Instrument the Scheduler Hot Path

The 2–5 ms per step estimate needs a profile to pinpoint. Add timed spans at
the `record_function` markers in `scheduler.py` around:

- `schedule: allocate_slots` (line 539)
- `schedule: schedule_running` (decode iteration)
- `schedule: schedule_waiting` (prefill iteration)
- `schedule: finalize` (output processing)

Run the trace with a small number of conversations (e.g., 4 conversations,
2 turns) and measure the breakdown. If `allocate_slots` is `<100 μs` per
step but `schedule_running` is >> 1 ms, the bottleneck is in the scheduler
iteration itself (state checks, list operations) — not the KV cache.

**If `schedule_running` dominates:** Optimize the scheduler loop.
Potential targets:
- `_preempt` logic: called only when blocks are exhausted (rare)
- `_schedule` iteration: `for req in self.running:` + state dispatch
- `request.status` checks — cheap property access

---

## Expected Impact

| Change | Per-step savings | Relative impact |
|---|---|---|
| CUDA graph verification (config) | 1–3 ms if graphs were off | **High** |
| Slice avoidance (#3) | ~0.1 μs | Negligible |
| Logging off (#2) | ~0.1–0.5 ms | Low–Medium |

The largest gains come from ensuring CUDA graphs are actually running and
reducing non-KV work in the scheduler. The prefix cache code is not the
bottleneck for this hardware.

---

## Summary

The earlier SLRU work (v1–v4) produced a correct eviction policy. Plan v5
was wrong (output blocks ARE reused across turns). Plan v6 shifts focus to
the actual bottleneck for 3-core hardware: **CPU scheduling overhead**, not
cache policy.

**Recommended next step:** Run a profile on 4 conversations × 2 turns with
the `record_function` markers to confirm where the 2–5 ms per step is spent.
If scheduler iteration dominates, optimize the scheduler loop; if CUDA
graphs are unexpectedly off, fix the graph configuration.
