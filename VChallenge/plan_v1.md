# Plan: Real SLRU for vLLM Prefix Cache

## 1. What is SLRU?

SLRU (Segmented LRU) splits the free list into two segments ordered by **actual reuse count**, not by a static heuristic:

```
[ head ] ↔ P1 ↔ P2 ↔ P3 ↔ ... ↔ R1 ↔ R2 ↔ R3 ↔ ... ↔ [ tail ]
           ^^^^^^^ probationary ^^^^^^   ^^^^^^^ protected ^^^^^^^
           evict here first               evict here last
```

**Lifecycle of a block:**

```
First alloc → First free → [Probationary] ──hit──→ Alloc again
                                       ↑                ↓
                                       │         [num_hits += 1]
                                       │                ↓
                                       │         Second free → [Protected]
                                       │
                  └── evicted → hash reset → next alloc starts over
```

| Event | Segment | Action |
|---|---|---|
| First time freed | → Probationary tail | `prepend_n` (front) |
| Hit (touch) while free | N/A (leaves free list) | `num_hits += 1` |
| Freed again (was hit) | → Protected tail | `append_n` (back) |
| Eviction needed | Pop from head | Probationary first; protected only if empty |
| Protected too large | Demote protected LRU | Move first protected block to probationary |

---

## 2. The Current Implementation Gap

Current `BlockPool.free_blocks()` (`block_pool.py:722-741`) classifies by **prefix depth**:

```python
block.block_hash_num_tokens <= self.slru_protected_tokens  # depth check
```

This is a **static heuristic**, not SLRU:
- Shallow prefixes → "protected" regardless of actual reuse
- Deep prefixes → "probationary" even if frequently reused
- No feedback from real cache hits/misses
- The `slru_protected_tokens` knob has no grounding in theory

The free list queue (`FreeKVCacheBlockQueue`) is structurally correct (O(1) doubly-linked list) but lacks:
- Hit tracking on blocks
- Segment classification by hit count
- Protected segment size enforcement and demotion

---

## 3. Changes Needed

### 3.1. `KVCacheBlock` — add hit counter

**File:** `vllm/v1/core/kv_cache_utils.py:118`

Add:

```python
@dataclass(slots=True)
class KVCacheBlock:
    # ... existing fields ...

    # How many times this block was re-accessed via prefix-cache hit
    # while it was in the free list. Controls SLRU segment assignment.
    # Reset to 0 on hash eviction.
    num_hits: int = 0
```

**Reset logic:** In `BlockPool._maybe_evict_cached_block()` (line 661), which calls `_remove_cached_block_hashes()` → `block.reset_hash()`, also set `block.num_hits = 0`. This ensures a block re-entering the cache with new content starts fresh.

### 3.2. `BlockPool.touch()` — record hits

**File:** `block_pool.py:684-699`

Current:
```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

Change to:
```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
            block.num_hits += 1      # <--- record the prefix-cache hit
        block.ref_cnt += 1
```

This is the **only** place `num_hits` is incremented. First-time allocations (`get_new_blocks()`) do not touch this counter.

### 3.3. `BlockPool.free_blocks()` — classify by hits

**File:** `block_pool.py:701-758`

Replace the depth-based condition:

```python
if block.num_hits >= 1:
    protected.append(block)
else:
    probationary.append(block)
```

Remove `slru_protected_tokens` from the SLRU path entirely (it becomes a no-op / deprecated config key).

The ordering in the free list remains:
1. `prepend_n(unhashed + probationary)` → head (evict first)
2. `append_n(protected)` → tail (evict last)

### 3.4. `FreeKVCacheBlockQueue` — protected size limit

**File:** `vllm/v1/core/kv_cache_utils.py:179`

Without a cap, protected grows until every block is protected, at which point SLRU degrades to LRU. We need a lazy demotion mechanism.

#### 3.4.1 New fields

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks, protected_ratio=0.8):
        # ... existing init ...
        self._protected_ratio = protected_ratio
        self._num_protected = 0
        self._protected_head: KVCacheBlock | None = None
```

`_protected_head` points to the first protected block (or `None`). Invariant:

```
[fake_head] ↔ P1 ↔ ... ↔ Pn ↔ [_protected_head] ↔ R1 ↔ ... ↔ Rm ↔ [fake_tail]
```

#### 3.4.2 New append operations

```python
def append_probationary_n(self, blocks: list[KVCacheBlock]):
    """Add blocks to the tail of probationary (before protected)."""
    if not blocks:
        return
    if self._protected_head is not None:
        # Insert right before _protected_head
        prev = self._protected_head.prev_free_block
        for block in blocks:
            block.prev_free_block = prev
            prev.next_free_block = block
            prev = block
        block.next_free_block = self._protected_head
        self._protected_head.prev_free_block = block
    else:
        # No protected blocks, just prepend to front like before
        self.prepend_n(blocks)
    self.num_free_blocks += len(blocks)

def append_protected_n(self, blocks: list[KVCacheBlock]):
    """Add blocks to the tail of protected (very end of queue)."""
    self.append_n(blocks)
    self._num_protected += len(blocks)
    if self._protected_head is None:
        self._protected_head = blocks[0]
```

#### 3.4.3 Modified `remove()`

```python
def remove(self, block: KVCacheBlock):
    if block is self._protected_head:
        # Advance protected head
        self._protected_head = (
            block.next_free_block
            if block.next_free_block is not self.fake_free_list_tail
            else None
        )
    # ... existing remove logic (link prev/next) ...
    self.num_free_blocks -= 1
    # Decrement protected count
    # We don't know for sure if the block was protected without checking
    # its position relative to _protected_head. Simplest: check if it
    # was after _protected_head (or was _protected_head itself).
    # Since remove is only called from touch() for prefix hits, and
    # the block is leaving the free list entirely, we just need to
    # keep the invariant correct. We can determine this by checking
    # if the block was before or after _protected_head.
```

**Simplification:** Since `remove()` is only called from `touch()`, and the block is being allocated (leaving the free list), we can determine which segment it was in by comparing it against `_protected_head`. But this requires walking — which we don't want.

**Better approach:** Don't track `_num_protected` precisely. Instead, on `popleft()`, if the head block has `num_hits >= 1` and it's been a while since we last demoted, just demote it. This is a heuristic but works well in practice.

**Best approach: use `_num_protected` with `remove` updating it.**

In `remove()`, we can check if the block was in the protected segment by scanning from `_protected_head` forward. But we don't want O(n) remove.

**Practical solution:** Add a `_in_protected_segment` bool flag to `KVCacheBlock`:

Actually, the simplest correct solution: since we know `popleft()` always pops from the head, and the head is always probationary (we maintain the invariant), the only way protected blocks are popped is through demotion or when probationary is empty.

Let me redesign more carefully. The demotion should happen in `free_blocks()` after classification, not in `popleft()`. This is simpler:

In `free_blocks()`, after computing probationary and protected:

```python
# Cap protected size
total_free = self.get_num_free_blocks() + len(probationary) + len(protected)
max_protected = int(total_free * self._protected_ratio)
excess = len(protected) + self._current_protected_count - max_protected
if excess > 0:
    # Demote excess protected blocks from the current protected pool
    # by moving them to probationary
    for _ in range(min(excess, self._current_protected_count)):
        # pop oldest protected, add to probationary
        ...
```

But this requires `FreeKVCacheBlockQueue` to expose a way to pop from protected's head and prepend to probationary's tail.

This is getting complex. For the plan, I'll describe the simplest viable approach:

**Simplest viable approach: lazy demotion in `popleft()`**

```python
def popleft(self):
    first = self.fake_free_list_head.next_free_block
    if first is self.fake_free_list_tail:
        raise ValueError("No free blocks available")

    # If the head is a protected block, demote it first.
    if first.num_hits >= 1 and first is self._protected_head:
        # Demote this block to probationary and try again
        self._demote_head_to_probationary()
        return self.popleft()

    # Normal pop from head
    ...
```

```python
def _demote_head_to_probationary(self):
    block = self._protected_head
    # Remove from current position
    block.prev_free_block.next_free_block = block.next_free_block
    block.next_free_block.prev_free_block = block.prev_free_block

    # Advance protected head
    self._protected_head = (
        block.next_free_block
        if block.next_free_block is not self.fake_free_list_tail
        else None
    )

    # Re-insert at the tail (this is now probationary)
    # Actually we want it at the end of probationary, which is just before
    # the (new) _protected_head.
    if self._protected_head is not None:
        prev = self._protected_head.prev_free_block
        prev.next_free_block = block
        block.prev_free_block = prev
        block.next_free_block = self._protected_head
        self._protected_head.prev_free_block = block
    else:
        # No protected blocks left, append at very end
        tail = self.fake_free_list_tail.prev_free_block
        tail.next_free_block = block
        block.prev_free_block = tail
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
```

This is sound: when `popleft()` finds a protected block at the head, it must mean probationary is empty (by the invariant), so it demotes it to the end of probationary (right before the new protected head) and recurses.

### 3.5. Deprecate `slru_protected_tokens`

The `slru_protected_tokens` config field becomes meaningless for real SLRU. Options:
1. **Remove it entirely** — clean break.
2. **Deprecate with warning** — keep the field but ignore it under `prefix_cache_policy="slru"` with a warning log.
3. **Rename to `slru_protected_ratio`** — repurpose the field as the max fraction of the free list that can be protected (default 0.8).

Option 3 is best: the user's intent ("protect valuable blocks") is preserved, just with a ratio instead of a token count.

```python
slru_protected_ratio: float | None = Field(default=0.8, ge=0, le=1)
```

The config validation should warn if both `slru_protected_tokens` and `slru_protected_ratio` are set, preferring the new one.

---

## 4. Migration Path

| Step | Files | Description |
|---|---|---|
| 1 | `kv_cache_utils.py` | Add `num_hits: int = 0` to `KVCacheBlock` |
| 2 | `block_pool.py` | Add `num_hits += 1` in `touch()` |
| 3 | `block_pool.py` | Change `free_blocks()` SLRU branch to use `num_hits` |
| 4 | `kv_cache_utils.py` | Add `protected_ratio`, `_protected_head`, segment-aware methods to `FreeKVCacheBlockQueue` |
| 5 | `block_pool.py` | Add `num_hits = 0` in `_maybe_evict_cached_block()` path |
| 6 | `cache.py` | Deprecate `slru_protected_tokens`, add `slru_protected_ratio` |
| 7 | All coordinators | Thread `slru_protected_ratio` instead of `slru_protected_tokens` |
| 8 | Tests | Add unit tests for SLRU classification and demotion |

---

## 5. Verification

| Test | Method |
|---|---|
| Hit tracking | Alloc block A, free it, touch it, free it → verify `num_hits == 1` and block goes to protected |
| Eviction order | Free multiple blocks with different hit counts → verify probationary pops before protected |
| Demotion | Fill protected to >80%, verify further allocations demote from protected |
| LRU fallback | `prefix_cache_policy="lru"` unchanged — verify blocks freed in FIFO order |
| Integration | Run prefix-caching benchmarks; compare hit rates before/after |

---

## 6. Summary

| Aspect | Current (fake SLRU) | Real SLRU |
|---|---|---|
| Classification basis | Prefix token depth (`block_hash_num_tokens`) | Actual reuse count (`num_hits`) |
| Adapts to traffic? | No (static threshold) | Yes (learns from hits) |
| Protected size limit | None (grows unbounded) | Ratio-based lazy demotion |
| Config meaning | `slru_protected_tokens` | `slru_protected_ratio` |
| Hit tracking | Not tracked | `touch()` increments `num_hits` |
