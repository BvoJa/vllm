### Objective
Rewrite the cache replacement policy based on the proposals in `guide.md`. This involves implementing an SLRU policy and a "skip filler hashing" optimization.

### Implementation Plan

#### 1. Update Configuration
- **File:** `vllm/config/cache.py`
  - Add `prefix_cache_policy: PrefixCachePolicy` to `CacheConfig`.
  - Add `slru_protected_tokens: int` to `CacheConfig`.
  - Add `enable_skip_filler_hashing: bool` to `CacheConfig`.
  - Add `filler_depth_tokens: int` to `CacheConfig`.
- **File:** `vllm/engine/arg_utils.py`
  - Add command-line arguments `--prefix-cache-policy`, `--slru-protected-tokens`, `--enable-skip-filler-hashing`, and `--filler-depth-tokens` to `EngineArgs`.
  - Ensure these arguments are used to initialize the new `CacheConfig` attributes.

#### 2. Propagate Configuration
- **File:** `vllm/vllm/v1/core/sched/scheduler.py`
  - In `Scheduler.__init__`, pass the new `CacheConfig` attributes to the `KVCacheManager` constructor.
- **File:** `vllm/vllm/v1/core/kv_cache_manager.py`
  - In `KVCacheManager.__init__`, accept the new arguments.
  - Pass `prefix_cache_policy`, `slru_protected_tokens`, `enable_skip_filler_hashing`, and `filler_depth_tokens` to the `get_kv_cache_coordinator` call.
- **File:** `vllm/vllm/v1/core/kv_cache_coordinator.py`
  - In `KVCacheCoordinator.__init__`, accept the new arguments and pass them to the `BlockPool` constructor within each `SingleTypeKVCacheManager`.
- **File:** `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
    - In `SingleTypeKVCacheManager.__init__`, accept the new arguments and pass them to the `BlockPool` constructor.

#### 3. Implement SLRU Policy
- **File:** `vllm/vllm/v1/core/block_pool.py`
  - **Modify `BlockPool.__init__`:**
    - Accept and store `prefix_cache_policy` and `slru_protected_tokens`.
  - **Modify `free_block()` method:**
    - If `self.prefix_cache_policy == 'slru'`, a block is only added to the free queue if `block.block_hash_num_tokens > self.slru_protected_tokens`.
  - **Modify `touch()` method:**
    - If `self.prefix_cache_policy == 'slru'`, skip touching blocks where `block.block_hash_num_tokens <= self.slru_protected_tokens` because they are not in the free queue.

#### 4. Implement Skip Filler Hashing
- **File:** `vllm/vllm/v1/core/block_pool.py`
  - **Modify `BlockPool.__init__`:**
    - Accept and store `enable_skip_filler_hashing` and `filler_depth_tokens`.
  - **Modify `cache_full_blocks()` method:**
    - If `self.enable_skip_filler_hashing` is true, skip caching blocks where `num_hash_tokens > self.filler_depth_tokens`.
  - **Modify