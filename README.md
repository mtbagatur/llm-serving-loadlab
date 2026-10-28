# llm-serving-loadlab

**Is your LLM slow because of load, or because of prompt length?** A small, reproducible
load-test + analysis kit that answers that question with data, and turns the answer into a
one-page decision memo.

## The problem

"The chatbot is slow" usually triggers the reflex of scaling whatever sits in front of the
model: the API gateway, the auth service, the ingress. If the time is actually spent *inside*
the model server, that changes nothing.

LLM latency has two very different phases:

- **Prefill**: the whole prompt is processed in one compute-heavy pass. Its cost grows with
  prompt length (long system prompts, RAG context, chat history).
- **Decode**: output tokens are generated one at a time.

Time to first token (TTFT) = queueing + prefill. If latency grows with *prompt length* and most
of it is TTFT, the bottleneck is prefill on the serving device, and the fix is more serving
capacity (another model replica on idle hardware) or less prefill work, not more gateway pods.

This repo reproduces that diagnostic method with public models on a laptop.

## Method

1. **Controlled load test** (`loadlab/loadgen.py`): async, streaming, OpenAI-compatible.
   Runs a full factorial matrix *prompt length x concurrency* (closed loop: exactly `c`
   requests in flight). Per request: TTFT, total latency, decode tokens/s, prompt and
   completion tokens (from the server's `usage`). Every prompt starts with a unique nonce so
   prefix caching cannot hide prefill cost. Output: CSV + JSON with run metadata.
2. **Analysis** (`loadlab/analyze.py`): per scenario, fit
   `log(latency) = a + e_p*log(prompt_tokens) + e_c*log(concurrency)`.
   The coefficients are elasticities ("2x prompt -> 2^e_p x latency"). The design is balanced,
   so the variance each factor explains alone is directly comparable. It also reports the TTFT
   share of latency and the **throughput gain** from 1 to max concurrency (~1.0 = the device is
   saturated, extra requests just queue). Charts (PNG) + `decision_memo.md`.
3. **Intervention**: run the same matrix against two replicas on separate (idle) compute and
   compare.

## Results (real run on this machine)

All numbers below come from `results/real/` and were produced by the code in this repo on
2026-10-28. Nothing is extrapolated.

**Setup.** No GPU available, so this is a CPU stand-in for "one accelerator":

| | |
|---|---|
| Hardware | Laptop, AMD Ryzen 7 7735HS (8 cores / 16 threads), 13 GB RAM, no GPU |
| Engine | Ollama 0.6.8 (llama.cpp), `OLLAMA_NUM_PARALLEL=4`, context 8192 |
| Model | `qwen2.5:0.5b` (Q4_K_M, public) |
| one-replica | one Ollama container pinned to CPUs 0-7 (CPUs 8-15 idle) |
| two-replicas | + a second container pinned to CPUs 8-15, client-side round robin |
| Matrix | prompt ~128/512/1024/2048 tokens x concurrency 1/2/4/8, 8 requests per cell, `max_tokens=64` |
| Requests | 128 per scenario, **0 errors** in both |

**Finding (one-replica)** — from `results/real/analysis/decision_memo.md`:

| driver | elasticity of latency | variance explained alone (R², log) |
|---|---|---|
| prompt length | 1.00 (2x prompt -> 2.0x latency) | 0.65 |
| concurrency | 0.84 (2x in-flight -> 1.8x latency) | 0.32 |

- With **one request in flight** (no queue), going from ~147 to ~1878 prompt tokens moves
  median latency from 1.99 s to 20.61 s. The cost exists without any load.
- At the longest prompt, **87%** of latency is TTFT (queue + prefill). TTFT elasticity to prompt
  length is 1.40 (super-linear: attention cost on CPU).
- Throughput changes by **x1.00** from 1 to 8 in-flight requests: the device is saturated;
  extra concurrency only adds waiting time. That is why concurrency *looks* like a driver too:
  requests queue behind other requests' prefill.
- Verdict produced by the analyzer: **`prefill`**.

![latency vs prompt](results/real/analysis/one-replica_latency_vs_prompt.png)
![where the time goes](results/real/analysis/one-replica_ttft_vs_decode.png)

**Intervention (two-replicas on the idle cores):**

- Throughput at concurrency 8: **x1.49** (median over prompt lengths).
- Median p50 latency speedup where more than one request is in flight: **x1.28**; at
  concurrency 1 there is no gain (x1.00 for 512-2048 tokens), as expected.
- Not x2: both replicas share one laptop chip (memory bandwidth, L3 cache, power/thermal
  budget). On separate GPUs/nodes the gain should be closer to linear; that is **not measured here**.
- Noise: the 128-token cells were *slower* in the two-replica run (e.g. 128x1: 1.99 s -> 3.75 s,
  with replica A itself at 4.05 s vs 1.99 s earlier). These are 8-request cells on a laptop with
  other containers running; treat single cells as noisy and look at the trend.

![scenario comparison](results/real/analysis/scenario_comparison.png)

Full per-cell table: `results/real/analysis/cell_summary.csv`. Raw requests:
`results/real/one-replica.csv`, `results/real/two-replicas.csv`.

### Mock demo (simulated, not a model)

`results/mock/` contains output from `loadlab/mock_server.py`, a **simulated** server that
sleeps according to a parametric prefill/decode model. It exists for tests and for running the
pipeline without a model. Its numbers are not results and are labelled MOCK in the memo.

## What I'd tell the customer

> We measured instead of guessing. Response time grows with how much text each request sends
> to the model, and it does so even when only one user is active. Most of the waiting happens
> before the first word appears, while the model reads the prompt. The model server is already
> fully busy: sending it more requests at once does not get more work done, it only makes
> everyone wait longer.
>
> So scaling the services in front of the model (gateway, authentication) will not help, they
> are not where the time goes. What helps, in order:
> 1. Put a second model replica on hardware that is currently idle, behind the same service
>    (in our test: ~1.5x more requests per second under load).
> 2. Send less text per request: cap retrieved context, shorten system prompts, enable prefix
>    caching for the shared part.
> 3. Watch time-to-first-token by prompt size, and scale on queue length, not CPU.

## How to run

```bash
make venv          # creates .venv and installs requirements.txt
make test          # pytest (uses the mock server, no model needed)
make mock-demo     # full pipeline against the simulated server -> results/mock/

# Real run (Docker, ~400 MB model pull, CPU-only is fine; took ~40 min for both scenarios on the laptop above)
make ollama-up     # two Ollama containers pinned to CPUs 0-7 and 8-15, pulls qwen2.5:0.5b
make run-one       # one-replica scenario  -> results/real/one-replica.{csv,json}
make run-two       # two-replicas scenario -> results/real/two-replicas.{csv,json}
make analyze       # charts + decision memo -> results/real/analysis/
make ollama-down
```

Any OpenAI-compatible server works (vLLM, SGLang, llama.cpp server, TGI with the OpenAI API):

```bash
.venv/bin/python -m loadlab.loadgen --base-url http://HOST:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-lengths 128,512,1024,2048,4096 --concurrency 1,2,4,8,16 --requests-per-cell 16 \
  --scenario vllm-1gpu --out results/vllm/one
.venv/bin/python -m loadlab.analyze results/vllm/one.csv --out-dir results/vllm/analysis
```

Note: `prompts.py` calibrates prompt length for the Qwen2.5 tokenizer; for other models the
target length is approximate, but the analysis always uses the server-reported `prompt_tokens`.

## Kubernetes (`k8s/`) — planned

Kustomize base (Namespace, Deployment, Service) and overlays for one and two replicas of an
Ollama + `qwen2.5:0.5b` server. Replicas are spread across nodes so the second one lands on
idle compute. `k8s/hpa-note.md` explains why there is no CPU-based HPA and sketches a
queue-depth-based KEDA scaler.

Status: **planned.** The manifests render with `kubectl kustomize` and pass a server-side
dry run against a local kind cluster; they have **not** been benchmarked on kind yet.

```bash
make kind-up       # kubectl apply -k k8s/overlays/one-replica
kubectl -n loadlab port-forward svc/llm-server 8080:80 &
.venv/bin/python -m loadlab.loadgen --base-url http://127.0.0.1:8080/v1 --model qwen2.5:0.5b \
  --scenario kind-one --out results/kind/one
make kind-two      # scale to two replicas, repeat with --scenario kind-two
make kind-down
```

## Planned / not done

- GPU run with vLLM (needs an NVIDIA GPU; this machine has none).
- Benchmark on kind / a real cluster (manifests ready, not yet run).
- Open-loop (fixed requests/s) mode for capacity planning against a TTFT SLO.
- Larger cells (more requests per cell) to tighten the noisy short-prompt measurements.

## Layout

```
loadlab/prompts.py      unique, length-controlled prompts (defeats prefix cache)
loadlab/loadgen.py      async streaming load generator -> CSV/JSON
loadlab/analyze.py      elasticities, throughput, charts, decision memo
loadlab/mock_server.py  SIMULATED OpenAI-compatible server (tests / offline demo)
k8s/                    kustomize base + one/two replica overlays, HPA note
tests/                  pytest
results/real/           real run on this machine (raw + analysis)
results/mock/           mock demo output
docs/WALKTHROUGH.md     design notes (Turkish)
```

## License

MIT
