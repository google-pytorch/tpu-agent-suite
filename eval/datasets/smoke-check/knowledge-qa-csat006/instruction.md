I am serving a model with the vLLM TPU backend on TPU v6e.

I set the KV cache memory budget to 24 GB.

With the vLLM engine argument `--block-size 256`, the process uses about 24 GB for the cache. Most of my requests are short, around 80 to 180 tokens.

I changed `--block-size` from 256 to 64 because I expected the smaller blocks to waste less memory.

After the change, the KV cache still uses about 24 GB of HBM. The number of cache blocks is about 4 times larger, and I can fit more short requests before the cache fills up. But decode latency is worse.

If the smaller blocks are saving memory, why does my HBM usage stay the same? Where is the memory saving actually showing up?

Also, does TPU v6e have a hardware block size I should match, or should I choose the KV cache block size another way? Answer clearly, citing any references you can for grounding. Drop a JSON at `/logs/artifacts/answer.json` with keys `answer` and `reference`, for scoring.
