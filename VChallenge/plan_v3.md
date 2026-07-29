# SLRU v3: True LRU Within Segments

## Current problem

```
free_blocks([P1]) → prepend_n → [P1]
free_blocks([P2]) → prepend_n → [P2, P1]
Eviction: P2 first (newest) — LIFO, not LRU
```

## Fix: insert probationary right before the protected boundary

### In `FreeKVCacheBlockQueue`, maintain a `_protected_start` pointer

```
fake_head ↔ P_old ↔ P_new ↔ _protected_start ↔ R_old ↔ R_new ↔ fake_tail
           ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           probationary                 protected
```

### New method `append_probationary_n(blocks)`

Inserts blocks at the tail of the probationary segment (right before `_protected_start`).
Blocks within the batch keep their input order → first batch element is closest to
_head_ → evicted first.

```python
def append_probationary_n(self, blocks):
    """Append blocks to the end of the probationary segment (before protected)."""
    if not blocks:
        return
    # Insertion point is right before _protected_start
    if self._protected_start is not None:
        insert_point = self._protected_start.prev_free_block
    else:
        insert_point = self.fake_free_list_tail.prev_free_block

    for block in blocks:
        block.prev_free_block = insert_point
        insert_point.next_free_block = block
        insert_point = block

    insert_point.next_free_block = (
        self._protected_start if self._protected_start is not None
        else self.fake_free_list_tail
    )
    if self._protected_start is not None:
        self._protected_start.prev_free_block = insert_point
    else:
        self.fake_free_list_tail.prev_free_block = insert_point

    self.num_free_blocks += len(blocks)
```

### Existing `append_n(protected)` stays the same

Appends to the very tail (after all existing protected blocks). True LRU.

### Updated `popleft()` — pops from head (always probationary first)

```python
def popleft(self):
    first = self.fake_free_list_head.next_free_block
    if first is self._protected_start:
        # Probationary is empty; before evicting from protected,
        # advance the marker so the first protected block becomes
        # the new (temporary) head.
        # (If we wanted demotion, we'd move it to probationary here.)
        pass
    # ... standard popleft logic ...
```

### Updated `remove(block)` — maintain `_protected_start`

When `touch()` removes a block from the free list:

```python
def remove(self, block):
    if block is self._protected_start:
        self._protected_start = block.next_free_block
        if self._protected_start is self.fake_free_list_tail:
            self._protected_start = None
    # ... standard remove logic (link prev → next) ...
    self.num_free_blocks -= 1
```

### Updated `free_blocks()` SLRU branch

```python
self.free_block_queue.append_probationary_n(probationary)
self.free_block_queue.append_n(protected)
```

Now both segments have true LRU ordering:

```
Round 1: append_probationary_n([P1]) → [P1]
         append_n([R1])             → [P1, R1]

Round 2: append_probationary_n([P2]) → [P1, P2, R1]
         append_n([R2])             → [P1, P2, R1, R2]

Eviction: P1 → P2 → R1 → R2
          ^ oldest first in each segment ✓
          ^ all probationary before protected ✓
```

## Changes

| File | Change |
|---|---|
| `kv_cache_utils.py` | Add `_protected_start: KVCacheBlock \| None`, `append_probationary_n()`, update `remove()`, update `popleft()` |
| `block_pool.py` | Change `prepend_n(probationary)` → `append_probationary_n(probationary)` |
