# Real SLRU for vLLM Prefix Cache — v2

## Key Insight from Review

The reviewer correctly identified that 2 separate internal lists are cleaner than a
split-pointer single list for `remove()` disambiguation. However, after deeper
analysis, **the simplest correct design needs zero internal queue changes** —
the natural eviction cycle provides free demotion.

---

## 1. What SLRU Does (Refresher)

```
Eviction order:  [head] Probationary (A1) ... Protected (Am) [tail]
                      ▲ pop here first              ▲ pop here last
```

| Event | Placement |
|---|---|
| First free (`num_hits == 0`) | → Probationary (head side) |
| Prefix hit (`touch()`) | `num_hits++`, leaves free list |
| Second free (`num_hits >= 1`) | → Protected (tail side) |
| Eviction | Pop from head — probationary first, protected only when probationary empty |
| Demotion | Block evicted from protected → hash cleared → `num_hits` reset → back to probationary on next free |

---

## 2. Why The Queue Needs No Change

The current `FreeKVCacheBlockQueue` operations naturally implement SLRU segments:

```python
# In BlockPool.free_blocks():
self.free_block_queue.prepend_n(probationary)   # → head (evict first)
self.free_block_queue.append_n(protected)        # → tail (evict last)
```

After N rounds, the queue looks like:

```
[P3, P2, P1, R1, R2, R3]
```

- **Head** is always probationary → evicted first ✓
- **Tail** is always protected → evicted last ✓
- **Demotion is free**: when `popleft()` eventually reaches `R1`, it pops it for a
  new request. `_maybe_evict_cached_block()` clears the hash AND resets
  `num_hits`. On next `free_blocks()`, the block goes to probationary.
- **No 2-list complexity** — same single linked list, same public API
- **`remove()` in `touch()`** — always O(1), no segment ambiguity

The key enabler: **`num_hits` is reset on hash eviction**, not on every free.
A block stays protected as long as it keeps getting hit. Once it stops getting
hit, the eviction cycle naturally reclaims it.

---

## 3. Changes

### 3.1 `KVCacheBlock` — new field

**File:** `vllm/v1/core/kv_cache_utils.py:118`

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    # NEW: incremented on each prefix-cache reuse (touch).
    # Reset on hash eviction. Controls SLRU segment assignment.
    num_hits: int = 0
    # ... rest unchanged ...
```

### 3.2 Reset on hash eviction

**File:** `block_pool.py:661` (`_maybe_evict_cached_block`) and
`kv_cache_utils.py:159` (`reset_hash`)

Current `reset_hash()`:
```python
def reset_hash(self):
    self._block_hash = None
    self._block_hash_num_tokens = None
```

Change to:
```python
def reset_hash(self):
    self._block_hash = None
    self._block_hash_num_tokens = None
    self.num_hits = 0    # NEW: block starts fresh when re-cached
```

### 3.3 `BlockPool.touch()` — record hits

**File:** `block_pool.py:684-699`

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
            block.num_hits += 1        # <--- one new line
        block.ref_cnt += 1
```

### 3.4 `BlockPool.free_blocks()` — classify by hits

**File:** `block_pool.py:701-758`

Replace the SLRU branch (lines 722–741):

```python
if self.prefix_cache_policy == "slru":
    probationary: list[KVCacheBlock] = []
    protected: list[KVCacheBlock] = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.num_hits >= 1:
                protected.append(block)
            else:
                probationary.append(block)
    self.free_block_queue.prepend_n(probationary)
    self.free_block_queue.append_n(protected)
```

Notable simplifications vs current code:
- No `unhashed` tier (unhashed blocks naturally have `num_hits == 0` → probationary)
- No `block_hash_num_tokens` check
- Config parameter `slru_protected_tokens` becomes unused

### 3.5 Config deprecation

**File:** `vllm/config/cache.py:112-129`

`prefix_cache_policy` stays as-is (`"lru"` or `"slru"`).

`slru_protected_tokens` is deprecated — it is no longer read by the SLRU code path.
Add a warning log if it is set to a non-default value under `"slru"` policy.

Optionally add `slru_protected_ratio: float = 1.0` as a safety valve
(maximum fraction of free list that can be protected). Default 1.0 = unlimited.

### 3.6 `FreeKVCacheBlockQueue` — unchanged

No structural changes. The single doubly-linked list with `prepend_n`/`append_n`
already produces the correct segment ordering. Only the docstring is updated:

```python
class FreeKVCacheBlockQueue:
    """...existing text...

    Under ``prefix_cache_policy == "slru"``, the caller places
    probationary blocks at the head via ``prepend_n`` and protected
    blocks at the tail via ``append_n``. The natural eviction order
    (head → tail) implements SLRU: evict probationary first.
    """
```

---

## 4. Summary of Code Changes

| File | Change | Lines |
|---|---|---|
| `kv_cache_utils.py:118` | Add `num_hits: int = 0` to `KVCacheBlock` | +1 |
| `kv_cache_utils.py:159` | Reset `num_hits` in `reset_hash()` | +1 |
| `block_pool.py:695` | `block.num_hits += 1` in `touch()` | +1 |
| `block_pool.py:722-741` | Replace SLRU branch with `num_hits` classification | ~20 |
| `block_pool.py` docstring | Update `free_blocks()` docstring | ~5 |
| `cache.py:112-129` | Deprecate `slru_protected_tokens`, warning log | ~5 |
| `kv_cache_utils.py:179` docstring | Update `FreeKVCacheBlockQueue` docstring | ~5 |

**Total:** ~40 lines changed, zero new classes, zero architectural changes.

---

## 5. Verification

| Scenario | Expected |
|---|---|
| New block allocated, freed → `free_blocks()` | `num_hits == 0` → probationary head |
| Block hit once (`touch()`), freed again | `num_hits == 1` → protected tail |
| Block hit multiple times | `num_hits > 1` → protected tail (same logic) |
| Protected block evicted via `popleft()` | `_maybe_evict_cached_block` → `reset_hash()` → `num_hits = 0` |
| Same block re-allocated, freed | `num_hits == 0` → probationary (demoted) |
| LRU mode unchanged (`policy="lru"`) | No `num_hits` involvement; existing behavior |
