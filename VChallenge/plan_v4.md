# SLRU Final Plan — True Segmented LRU

## The Policy (one sentence)

Two LRU segments: **probationary** (`num_hits == 0`) evicted first, **protected** (`num_hits >= 1`) evicted last. Each segment is true LRU: **oldest-freed at the head**, **newest at the tail**.

## Queue invariant

```
fake_head ↔ P_oldest ↔ P_newer ↔ P_newest ↔ R_oldest ↔ R_newer ↔ R_newest ↔ fake_tail
            ^^^^^^^^^^^^^ probationary ^^^^^^^^^^^^^   ^^^^^^^^^^^ protected ^^^^^^^^^^^
            ▲ popleft here first                       ▲ popleft here only when empty ↖
```

## Block lifecycle

```
             ┌─ first free (num_hits=0) ──→ Probationary tail
             │
Allocated ───┤                                      Cache hit (touch)
             │    num_hits += 1 ←─────────────────────┘
             │                              │
             │                              └─ second free (num_hits>=1) ──→ Protected tail
             │
             └─ evicted → reset_hash() → num_hits=0 → next free → Probationary
```

## Three changes needed

### 1. `FreeKVCacheBlockQueue` — split marker

Add `_use_slru` flag and `_protected_start` pointer.

**New `__init__` param:**

```python
def __init__(self, blocks, use_slru=False):
    self._use_slru = use_slru
    self._protected_start = None
    # ... existing init ...
```

**`append_n`** — update `_protected_start` on first protected append:

```python
def append_n(self, blocks):
    if not blocks:
        return
    if self._use_slru and self._protected_start is None:
        self._protected_start = blocks[0]
    # ... existing append_n logic ...
```

**New `append_probationary_n`** — insert right before `_protected_start`:

```python
def append_probationary_n(self, blocks):
    if not blocks:
        return
    if self._protected_start is not None:
        insert_at = self._protected_start.prev_free_block
        for b in blocks:
            b.prev_free_block = insert_at
            insert_at.next_free_block = b
            insert_at = b
        b.next_free_block = self._protected_start
        self._protected_start.prev_free_block = b
    else:
        # No protected yet, just go to tail
        tail_prev = self.fake_free_list_tail.prev_free_block
        for b in blocks:
            b.prev_free_block = tail_prev
            tail_prev.next_free_block = b
            tail_prev = b
        b.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = b
    self.num_free_blocks += len(blocks)
```

**`popleft`** — advance `_protected_start` if it was popped:

```python
def popleft(self):
    first = self.fake_free_list_head.next_free_block
    if self._use_slru and first is self._protected_start:
        self._protected_start = first.next_free_block
        if self._protected_start is self.fake_free_list_tail:
            self._protected_start = None
    # ... existing popleft logic (link around, clear pointers, decrement count) ...
```

**`popleft_n`** — same tracking during the bulk pop:

```python
def popleft_n(self, n):
    # ... existing code ...
    # Add: flag when _protected_start is popped
    # After the loop: advance _protected_start if needed
```

**`remove`** — advance `_protected_start` if removing the boundary block:

```python
def remove(self, block):
    if self._use_slru and block is self._protected_start:
        self._protected_start = block.next_free_block
        if self._protected_start is self.fake_free_list_tail:
            self._protected_start = None
    # ... existing remove logic ...
```

### 2. `BlockPool.free_blocks()` — use new method

```python
if self.prefix_cache_policy == "slru":
    probationary, protected = [], []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.num_hits >= 1:
                protected.append(block)
            else:
                probationary.append(block)
    self.free_block_queue.append_probationary_n(probationary)  # LRU ordering
    self.free_block_queue.append_n(protected)                  # LRU ordering
```

### 3. `BlockPool.__init__()` — pass SLRU flag to queue

```python
self.free_block_queue = FreeKVCacheBlockQueue(
    self.blocks, use_slru=(self.prefix_cache_policy == "slru")
)
```

## Why this is correct

| Scenario | Ordering | Why |
|---|---|---|
| Round 1: free P1, R1 | `[P1, R1]` | P1 appended to probationary tail, R1 to protected |
| Round 2: free P2, R2 | `[P1, P2, R1, R2]` | P2 inserted before R1, R2 after R1 |
| Evict one | P1 (oldest probationary) | Head |
| Evict two more | P2, R1 | P2 next, then oldest protected |
| Touch R1 (hit) | `remove(R1)`, `_protected_start` → R2 | R1 leaves free list |
| Free R1 again | Goes to protected tail | R1 behind R2 |

## Changes summary

| File | Change |
|---|---|
| `kv_cache_utils.py` | `FreeKVCacheBlockQueue`: add `_use_slru`, `_protected_start`, `append_probationary_n()`, update `popleft/popleft_n/remove/append_n` |
| `block_pool.py` | Pass `use_slru` to queue; change `prepend_n(probationary)` → `append_probationary_n(probationary)` |
| `kv_cache_utils.py` | `KVCacheBlock.num_hits` (already done) |
| `block_pool.py` | `touch()` num_hits increment (already done) |
| `cache.py` | Deprecation (already done) |
