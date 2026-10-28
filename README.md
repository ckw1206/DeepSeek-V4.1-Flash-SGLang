# DeepSeek-V4.1-Flash on SGLang

One-command SGLang deployment of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), serving an OpenAI-compatible API.

Recipe: [SGLang cookbook, DeepSeek-V4.1](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1) (H200 cell, the verified one). GPUs with more memory than H200 (B200, B300, ...) are auto-detected and get a relaxed config.

## Architecture

```mermaid
---
config:
  layout: elk
---
flowchart LR
    A(OpenAI-compatible client)

    subgraph Server ["GPU Server"]
        S(start.sh)
        subgraph C ["Docker container: lmsysorg/sglang:dev-dsv41"]
            API(SGLang HTTP server :8000<br/>reasoning + tool-call parsers)
            SCH(Scheduler<br/>batching, KV cache, CUDA graphs)
            subgraph G ["GPUs: TP = EP = GPU count"]
                direction LR
                G0(GPU 0<br/>attention shard + experts)
                G1(GPU 1<br/>attention shard + experts)
                GN(GPU N<br/>attention shard + experts)
            end
            SPEC(DSpark draft<br/>optional, DSPARK=1)
        end
        W[(HF cache<br/>~510GB weights)]
    end

    A -- "HTTP /v1/chat/completions" --> API
    S -- "docker run" --> C
    API --> SCH
    SCH --> G
    SPEC -.-> SCH
    W -- "mounted volume" --> C
    G0 <-->|"NVLink"| G1
    G1 <-->|"NVLink"| GN
```

`start.sh` sizes the config from the GPU memory it finds (see [Parallelism](#parallelism)) and launches one SGLang container. Weights (fp8 dense + fp4 experts) are downloaded once into the mounted Hugging Face cache. Attention is tensor-parallel and the MoE experts are expert-parallel across the same GPUs. Each request goes through the scheduler, which batches it with others, and the reply comes back through the API server. The server splits thinking into `reasoning_content` and parses tool calls.

## Server Deployment

Requires Docker + NVIDIA Container Toolkit on the server.

```bash
./install_docker.sh   # one-time; skips anything already installed
cp .env.sample .env   # optional; edit overrides
./start.sh            # streams the boot log, returns once /v1/models answers
./stop.sh
```

First boot downloads ~510GB of weights into `~/.cache/huggingface` (override with `HF_CACHE`).

### Parallelism

TP = EP = number of GPUs in `GPU_ID` (2, 4 or 8). `start.sh` reads per-GPU memory from `nvidia-smi` (override with `GPU_MEM_GB`, in GiB) and picks a profile:

| Per-GPU memory | Profile |
|---|---|
| < 160 GiB (H200) | Cookbook pins `--attention-backend dsv4`, `--moe-runner-backend flashinfer_mxfp4`, `--cuda-graph-max-bs-decode 64`; `MEM_FRACTION` 0.8 |
| >= 160 GiB (B200, B300, ...) | Pins dropped so SGLang resolves backends and the decode graph list itself; `MEM_FRACTION` 0.9. Untested on these GPUs: check the startup log for the resolved backends |

| Total memory (GPUs x per-GPU) | Status |
|---|---|
| >= 600 GiB (8x H200, 4x B200/B300) | Full window, no caps |
| 500-600 GiB (4x H200, 2x B300) | Tight vs ~510GB weights; defaults to `MEM_FRACTION` 0.92 plus `--max-running-requests 16 --chunked-prefill-size 1024 --context-length ${CONTEXT_LENGTH}` (default 65536; override via `CONTEXT_LENGTH` or `EXTRA_ARGS`) |
| < 500 GiB (2x H200) | Rejected: can't hold the weights |

### Running TP vs DP attention

> **Hint: built for NVL8 nodes** (HGX H200 / B200 / B300 NVL8: 8 GPUs on one NVSwitch NVLink domain, all-to-all full bandwidth). The TP all-reduce and the EP dispatch/combine below all ride NVLink, which is why TP8/EP8 and DP-attention splits are cheap here. On PCIe-only boxes or across nodes the cross-GPU traffic is far more expensive, so expect different results and re-benchmark before copying these settings.

Default is pure tensor parallel: attention TP and expert EP both span every GPU (TP8/EP8 on 8 GPUs). Best latency at low concurrency.

```bash
./start.sh                                   # TP8 / EP8
```

`DP_SIZE=N` switches attention to data parallel (`--dp-size N --enable-dp-attention`). Each group of `GPUs / N` GPUs serves its own requests with its own KV cache, while the experts stay sharded across all GPUs (EP = GPU count). `DP_SIZE` must divide the GPU count.

```bash
DP_SIZE=4 EXTRA_ARGS="--max-running-requests 512" ./start.sh   # 8 GPUs: attention DP4 x TP2, experts EP8
DP_SIZE=8 EXTRA_ARGS="--max-running-requests 512" ./start.sh   # 8 GPUs: attention DP8 x TP1, experts EP8
```

| Mode | Attention | Experts | Pick it when |
|---|---|---|---|
| `DP_SIZE=1` (default) | TP8 | EP8 | Few concurrent users, lowest per-request latency |
| `DP_SIZE=4` | DP4 x TP2 | EP8 | High concurrency: KV is not replicated across all GPUs, smaller attention all-reduce, bigger decode batches |
| `DP_SIZE=8` | DP8 x TP1 | EP8 | Highest concurrency; dense/attention weights are replicated 8x, so it needs the memory headroom of 160+ GiB GPUs |

Notes:
- `--max-running-requests` is the total across all DP groups (512 is about 128 per group of 4). Raising it does nothing if real concurrency never reaches it.
- DP attention adds MoE dispatch/combine traffic and can leave some groups idle under uneven load, so it can lose to TP8 at low concurrency. Benchmark both (`python3 -m sglang.bench_serving` at concurrency 16, 64, 128, 256, 512) and compare output tok/s with p99 inter-token latency.
- **Known issue (8x B300, `dev-dsv41`, 2026-10-28):** `DP_SIZE=4` boots and captures CUDA graphs, then the first chat request dies with `repeat_interleave ... output_size (194) must be the same as the sum of the elements in the repeats tensor (193)` (device-side assert, scheduler exit -3, whole container killed; `Xid 43` in `dmesg` is the assert, not a hardware fault). `DP_SIZE=1` (TP8/EP8) on the same host works. Until a newer SGLang build fixes it, treat DP attention as experimental here and re-test after updating the image.

### Settings (`.env` or shell env)

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_ID` | `0,1,2,3,4,5,6,7` | GPUs passed to the container (2, 4 or 8; see Parallelism) |
| `GPU_MEM_GB` | auto (`nvidia-smi`) | Per-GPU memory in GiB; picks the H200 vs larger-GPU profile |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | API bind address |
| `IMAGE` | `lmsysorg/sglang:dev-dsv41` | Preview build; V4.1 is not in an SGLang release yet |
| `MEM_FRACTION` | `0.8` (< 160 GiB GPUs) / `0.9` (>= 160 GiB) / `0.92` (tight) | `--mem-fraction-static` |
| `THINKING` | `0` | `1` = think by default (`SGLANG_DEFAULT_THINKING`) |
| `CONTEXT_LENGTH` | `65536` | Tight configs only: `--context-length` cap (otherwise the model window) |
| `DP_SIZE` | `1` | `>1` = DP attention (`--dp-size N --enable-dp-attention`); e.g. `DP_SIZE=4` on 8 GPUs is DP4 x TP2 attention, experts still EP8. Better at high concurrency, worse at low; pair with `EXTRA_ARGS="--max-running-requests 512"`. **Known issue:** crashed on the first request on `dev-dsv41` (see below); keep `1` for now |
| `DSPARK` | `0` | `1` = DSpark speculative decoding (bundled draft, block size 5) |
| `HF_CACHE` | `~/.cache/huggingface` | Host model cache |
| `HF_TOKEN` | unset | Optional (public model) |
| `EXTRA_ARGS` | unset | Extra SGLang flags, appended last |

Thinking is off by default. Send `reasoning_effort` (`low`, `high`, `xhigh`, `max`) in the request to turn it on, or set `THINKING=1`. `--reasoning-parser auto` puts the thinking in `reasoning_content`. Tool calling uses `--tool-call-parser auto` (resolves to `deepseekv41`). Send `tools` in the request; no extra flag is needed.

On H200 the cookbook pins `--attention-backend dsv4`, `--moe-runner-backend flashinfer_mxfp4` and `--cuda-graph-max-bs-decode 64`, because the default decode CUDA-graph batch list does not fit in 140 GiB. For the high-throughput variant, add `EXTRA_ARGS="--max-running-requests 256"`.

DSpark (`DSPARK=1`): the cookbook's low-latency cells on other GPUs turn it on, but the verified H200 cells run without it. It is untested here, so it is off by default.

## Test

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": "Who won the world series in 2020?"}]}'
```

## Files

- `install_docker.sh`: installs Docker + NVIDIA Container Toolkit (on the GPU host)
- `start.sh` / `stop.sh`: run and stop the SGLang server (on the GPU host)
- `.env.sample`: optional overrides for `start.sh`
