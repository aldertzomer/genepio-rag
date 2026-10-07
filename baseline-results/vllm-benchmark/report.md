# vLLM benchmark — 2026-10-07

RTX 4070 Ti SUPER (16 GiB); vLLM 0.13.0; model mistralai/Ministral-3-8B-Instruct-2512.
Context 8192; temperature 0; extraction 128 tokens; grounding 96 tokens.
Prefix caching disabled; first three benchmark.tsv records cyclically repeated for batches.
One extraction warmup and engine initialization excluded from measured times.
GPU memory/utilization includes other processes and reserved KV cache.

**Every measured response hit the output limit and failed JSON parsing.**
Grounding was measured independently with baseline extracted concepts, using the unchanged grounding prompt and validator.
These are throughput results, not successful end-to-end annotations.

| Mode | Pass | Prompt tokens | Generated tokens | Wall s | Gen tok/s | Total tok/s | Records/s | GPU mean % | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential (1) | extraction | 221 | 128 | 2.125 | 60.2 | 164.2 | 0.471 | 93.3 | 13.10 |
| sequential (1) | grounding | 2108 | 96 | 1.874 | 51.2 | 1176.1 | 0.534 | 94.2 | 13.22 |
| sequential (2) | extraction | 221 | 128 | 2.114 | 60.6 | 165.1 | 0.473 | 93.6 | 13.22 |
| sequential (2) | grounding | 2145 | 96 | 2.351 | 40.8 | 953.3 | 0.425 | 80.1 | 13.33 |
| sequential (3) | extraction | 241 | 128 | 2.114 | 60.6 | 174.6 | 0.473 | 93.6 | 13.33 |
| sequential (3) | grounding | 4353 | 96 | 2.220 | 43.2 | 2004.2 | 0.450 | 94.9 | 13.57 |
| batch_8 (8) | extraction | 1808 | 1024 | 2.429 | 421.7 | 1166.1 | 3.294 | 93.5 | 13.57 |
| batch_8 (8) | grounding | 21465 | 768 | 4.664 | 164.7 | 4766.5 | 1.715 | 97.2 | 14.44 |
| batch_16 (16) | extraction | 3636 | 2048 | 2.711 | 755.5 | 2096.8 | 5.902 | 92.9 | 14.44 |
| batch_16 (16) | grounding | 45138 | 1536 | 11.136 | 137.9 | 4191.1 | 1.437 | 96.1 | 14.44 |
| batch_32 (32) | extraction | 7272 | 4096 | 3.408 | 1202.0 | 3336.0 | 9.390 | 92.9 | 14.44 |
| batch_32 (32) | grounding | 90313 | 3072 | 20.777 | 147.9 | 4494.6 | 1.540 | 96.6 | 14.44 |

Engine initialization: 172.7 seconds.

AST comparison confirmed both prompts and all top-level functions except CLI token-limit configuration are unchanged.
The inference class is the only pipeline implementation replaced.
