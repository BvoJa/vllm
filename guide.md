I want to propose these 4 new arguments

#  --prefix-cache-policy  
--prefix-cache-policy=lru: The current cache replacement policy of vllm

--prefix-cache-policy=slru: The new cache replacement policy I propose
- The free queue now only accept cache blocks with ref_cnt = 0 and block_hash_num_tokens > a bound

# --slru-protected-tokens  
--slru-protected-tokens=1000: The bound for the prefix-cache-policy=slru


# --enable-skip-filler-hashing and --filler-depth-tokens 
--enable-skip-filler-hashing=true: A block that has block_hash_num_tokens > a bound is not cached
--filler-depth-tokens=3000: The bound for skipping hashing scheme


# Noting
- In the intial code, all num-gpu-blocks initialized are all moved to the free queue
vllm/vllm/v1/core/block_pool.py
** 
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
**

- For each KVCacheBlock, this is the token position + 1 that this block ends at 
vllm/vllm/v1/core/kv_cache_utils.py
**
        # Number of prefix tokens covered by _block_hash. For full blocks this is
        # the full block boundary; partial aliases can end inside a cache block.
        _block_hash_num_tokens: int | None = None
        def block_hash_num_tokens(self) -> int | None:
            return self._block_hash_num_tokens
**

- Use block_hash_num_tokens() to be the condition for --slru-protected-tokens and --filler-depth-tokens 

- Check vllm/vllm/v1/core/block_pool.py for implementation of the current free queue: def touch(), def free_block(), def get_new_blocks()

- My purpose is to create some cache blocks that are called static block, this will be used thorughout all conversation, so there is no need to move them to the free queue

- free_block() method: now free queue only accept cache blocks that block_hash_num_tokens > slru-protected-tokens

- touch() method: now blocks that block_hash_num_tokens <= slru-protected-tokens are not in the free queue anymore, use this condition to skip these blocks



- 




