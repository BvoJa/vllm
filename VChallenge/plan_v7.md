# Plan v7: Config-Only Optimization for 3-Core CPU Bottleneck

## Context

Only docker-compose is submitted. Stock `vllm/vllm-openai:v0.22.1` — no source
modifications possible. All v1–v5 code changes are irrelevant for scoring.

---

## Bottleneck Analysis

With 3 CPU cores, the scheduler loop overhead per step scales as **O(max_num_seqs)**.
Each concurrent sequence adds dict lookups, state checks, and function calls on
every step. The GPU (H200 MIG, 1.2B model) does one decode token in ~0.25–0.5 ms,
but the CPU takes 2–5 ms.

**Target:** Reduce per-step Python work by lowering concurrency and eliminating
wasted metadata.

---

## Proposed Changes (vs. docker-compose-6.yml)

| Flag | Current | Proposed | Reason |
|---|---|---|---|
| `--max-num-seqs` | 32 | **8** | O(32) → O(8) scheduler work. GPU only needs 4–8 sequences to saturate decode. |
| `--max-model-len` | 4800 | **2500** | Trace max is 2450. Halves block table metadata allocation. |
| `--max-num-batched-tokens` | 4800 | **2500** | Match `max-model-len`. |
| `--long-prefill-token-threshold` | 2500 | **512** | 2500 > trace max prompt (2150) → chunked prefill never fires. 512 enables it. |
| `--disable-log-requests` | (missing) | **add** | Free CPU. Also suppresses per-request JSON serialization. |
| `--block-size` | 32 | **32** (keep) | Halves block metadata vs. default 16. Correct for CPU-constrained. |
| `--async-scheduling` | set | keep | Overlaps CPU scheduling with GPU execution. |
| `--enable-chunked-prefill` | set | keep | Now actually works (threshold fixed to 512). |
| `--kv-cache-dtype` | fp8 | keep | Maximizes KV cache capacity. |
| `--quantization` | fp8 | keep | Reduces model weight memory to ~1.2 GB. |
| `--gpu-memory-utilization` | 0.95 | keep | Maximizes block count. |

### Why `--max-num-seqs=8`?

On H200 MIG, a decode step per sequence is memory-bandwidth-bound (~0.25 ms).
With 8 sequences batched:

- 8 × 0.25 ms ≈ 2 ms GPU time (some overlap from batching)
- CPU scheduling for 8 sequences ≈ 1–2 ms
- Async scheduling overlaps CPU with GPU → effective step time ≈ **2 ms**

vs. current 32 sequences:

- 32 × 0.25 ms ≈ 8 ms GPU time (memory BW is shared, so it's more like 0.5–1 ms)
- CPU scheduling for 32 sequences ≈ 4–5 ms
- Step time ≈ **4–5 ms** (CPU bottleneck)

Reducing from 32 to 8 cuts CPU overhead ~4×, directly improving TPOT.

### Why `--long-prefill-token-threshold=512`?

Current value of 2500 exceeds all prompt lengths in the trace (max 2150).
Chunked prefill is a no-op. With 512:

- Turn 1 prompt (1150 tokens): 3 chunks (512 + 512 + 126)
- Turn 6 prompt (2150 tokens): 5 chunks (512 × 4 + 102)
- Each chunk is interleaved with decode steps, reducing TTFT impact on
  other active conversations.

### Memory check

With `--max-model-len=2500`, `--block-size=32`, `--kv-cache-dtype=fp8`:
- Max blocks per sequence: 2500 ÷ 32 = 79
- With 8 concurrent: 8 × 79 = 632 blocks actively used
- Total blocks available (~15.4 GB / 1.5 MB per block ≈ 10,000+)
- Utilization: 6%. No eviction pressure.

---

## Expected ERS Impact

| Component | Current estimate | With changes |
|---|---|---|
| TPOT | 4–5 ms (CPU bottleneck) | **2 ms** (GPU-bound) |
| TTFT | 150–300 ms (queueing + prefill) | **100–200 ms** (less queueing) |
| s_tpot (γ=2, floor=1ms) | 0.12–0.25 | **0.50** |
| s_ttft (γ=2, floor=50ms) | 0.30–0.60 | **0.50–0.70** |
| **ERS** | 0.21–0.43 | **0.50–0.60** |

---

## Final docker-compose-7.yml

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      - --enable-prefix-caching
      - --kv-cache-dtype=fp8
      - --quantization=fp8
      - --max-model-len=2500
      - --max-num-batched-tokens=2500
      - --max-num-seqs=8
      - --block-size=32
      - --async-scheduling
      - --disable-log-stats
      - --disable-log-requests
      - --gpu-memory-utilization=0.95
      - --enable-chunked-prefill
      - --long-prefill-token-threshold=512
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
