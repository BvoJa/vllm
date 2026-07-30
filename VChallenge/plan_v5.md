# Plan v5: Skip Caching Output-Only Blocks

## Problem

For multi-turn conversation workloads, **output tokens are never shared**.
Yet `cache_full_blocks()` inserts every full block into the prefix cache
hash table — including blocks entirely in the generated output.

For your trace:
- 300 output tokens/turn = ~19 blocks/turn
- 6 turns × 70 conversations = 420 generations
- ~7,980 useless cache insertions, all immediately evicted

## Impact

| Metric | Before | After |
|---|---|---|
| Cache entries per turn | ~28 (user+output) | ~9 (user only) |
| Total cache churn | 420 × 19 = 7,980 useless inserts | zero |
| Hash table size | 3× larger | compact |
| Cache hit rate | same | same |
| Block memory | same (blocks still allocated for compute) | same |

This is pure CPU/overhead savings — no GPU memory change because output
blocks are still allocated (they must hold KV data for decoding). We just
stop registering their hashes in `cached_block_hash_to_block`.

## Implementation

### 1. Override `reachable_block_mask` in `FullAttentionManager`

**File:** `vllm/v1/core/single_type_kv_cache_manager.py:565`

Current: `FullAttentionManager` inherits the default `reachable_block_mask`
from `SingleTypeKVCacheManager` which returns `None` (cache all).

Add:

```python
class FullAttentionManager(SingleTypeKVCacheManager):
    # ... existing ...

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        retention_interval: int | None = None,
        num_prompt_tokens: int | None = None,
    ) -> list[bool] | None:
        """Skip caching blocks that are entirely in the generated output.

        Output tokens are never a prefix-cache hit for future requests,
        so skip their blocks to reduce hash-table pressure.
        """
        if num_prompt_tokens is None:
            return None
        block_size = kv_cache_spec.block_size
        prompt_blocks = cdiv(num_prompt_tokens, block_size)
        mask = []
        for i in range(start_block, end_block):
            mask.append(i < prompt_blocks)
        return mask if not all(mask) else None
```

A block at index `i` covers token range `[i * block_size, (i+1) * block_size)`.
Since `num_prompt_tokens` may not be block-aligned, the condition `i < prompt_blocks`
uses `cdiv` — a block that straddles the prompt/output boundary (contains at
least one prompt token) is still cached. Only blocks that are **entirely** in
the output are skipped.

The `if not all(mask) else None` preserves the existing contract: `None` means
"cache everything" and avoids allocating a trivial all-True list.

### 2. (Optional) Same override for other attention manager classes

The same logic applies to any spec type that can have output blocks:

- `TQFullAttentionManager` (if exists as a separate class)
- `MLAAttentionManager` (DeepSeek MLA — also has prompt/output distinction)
- `HiddenStateCacheManager`

Each would get the same `reachable_block_mask` override. For the first pass,
only `FullAttentionManager` is sufficient for the common case.

### 3. What does NOT change

- `BlockPool.cache_full_blocks()` — already supports `block_mask`; no change
- `BlockPool.free_blocks()` — SLRU unaffected
- `KVCacheBlock.num_hits` — unaffected
- Block allocation — output blocks still allocated normally
- `num_cached_block` tracking — still updated to `num_full_blocks`

## Edge cases

| Case | Behavior |
|---|---|
| Single-turn prompt (no output yet) | All blocks within prompt → mask all True → `None` → cache all |
| Block straddles prompt/output boundary | `i < prompt_blocks` → True → cached (partially useful) |
| `num_prompt_tokens` unavailable | `None` guard → return `None` → cache all (safe fallback) |
| Speculative decoding (draft tokens) | Draft output is also generated text → still correctly skipped |
| Pure decode (no prompt) | `num_prompt_tokens = 0` → `prompt_blocks = 0` → mask all False → cache nothing |

## Files changed

| File | Change | Lines |
|---|---|---|
| `single_type_kv_cache_manager.py` | Add `reachable_block_mask` to `FullAttentionManager` | ~15 |
| (optional) same file | Same override for MLA, HiddenState managers | ~15 each |

## Verification

| Test | Method |
|---|---|
| Output blocks not cached | Alloc blocks, fill prompt+output, call `cache()` → verify hash table size equals prompt block count |
| Straddle block cached | `num_prompt_tokens = 17`, `block_size = 16` → block 1 (tokens 16-31) is cached |
| Single-turn unaffected | `num_prompt_tokens = full` → mask all True → `None` → caches everything |
| `num_prompt_tokens = None` | `None` guard → `None` → backward-compatible |
