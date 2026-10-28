# llm-serving-loadlab — see README.md
PY      ?= .venv/bin/python
MODEL   ?= qwen2.5:0.5b
LENGTHS ?= 128,512,1024,2048
CONC    ?= 1,2,4,8
N       ?= 8
OUT     ?= results/real

.PHONY: venv test mock-demo ollama-up ollama-down run-one run-two analyze kind-up kind-two kind-down clean

venv:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q

# Offline demo: simulated server, clearly labelled MOCK in every output.
mock-demo:
	$(PY) -m loadlab.mock_server --port 8099 --prefill-tok-s 2000 --decode-tok-s 40 & echo $$! > .mock.pid; sleep 1
	$(PY) -m loadlab.loadgen --base-url http://127.0.0.1:8099/v1 --model mock --scenario mock \
		--prompt-lengths $(LENGTHS) --concurrency $(CONC) --requests-per-cell 4 --max-tokens 32 --out results/mock/mock; \
		kill `cat .mock.pid`; rm -f .mock.pid
	$(PY) -m loadlab.analyze results/mock/mock.csv --out-dir results/mock/analysis --source-note "MOCK server (simulated timings, no model)"

# Real run on a CPU-only machine: two Ollama replicas pinned to disjoint cores.
# Replica A = cores 0-7, replica B = cores 8-15 (the "idle accelerator").
ollama-up:
	docker volume create loadlab-ollama >/dev/null
	docker run -d --name loadlab-ollama-a --cpuset-cpus 0-7 -p 127.0.0.1:11501:11434 \
		-e OLLAMA_NUM_PARALLEL=4 -e OLLAMA_CONTEXT_LENGTH=8192 -e OLLAMA_KEEP_ALIVE=-1 \
		-v loadlab-ollama:/root/.ollama ollama/ollama:0.6.8
	docker run -d --name loadlab-ollama-b --cpuset-cpus 8-15 -p 127.0.0.1:11502:11434 \
		-e OLLAMA_NUM_PARALLEL=4 -e OLLAMA_CONTEXT_LENGTH=8192 -e OLLAMA_KEEP_ALIVE=-1 \
		-v loadlab-ollama:/root/.ollama ollama/ollama:0.6.8
	sleep 3; docker exec loadlab-ollama-a ollama pull $(MODEL)

ollama-down:
	-docker rm -f loadlab-ollama-a loadlab-ollama-b

run-one:
	$(PY) -m loadlab.loadgen --base-url http://127.0.0.1:11501/v1 --model $(MODEL) --scenario one-replica \
		--prompt-lengths $(LENGTHS) --concurrency $(CONC) --requests-per-cell $(N) --out $(OUT)/one-replica

run-two:
	$(PY) -m loadlab.loadgen --base-url http://127.0.0.1:11501/v1 --base-url http://127.0.0.1:11502/v1 \
		--model $(MODEL) --scenario two-replicas \
		--prompt-lengths $(LENGTHS) --concurrency $(CONC) --requests-per-cell $(N) --out $(OUT)/two-replicas

analyze:
	$(PY) -m loadlab.analyze $(OUT)/one-replica.csv $(OUT)/two-replicas.csv --out-dir $(OUT)/analysis \
		--baseline one-replica --source-note "real run, Ollama 0.6.8 + $(MODEL) on CPU (see README)"

# Kubernetes variant (PLANNED: manifests are validated with a server-side dry run, not yet benchmarked).
kind-up:
	kubectl apply -k k8s/overlays/one-replica
	kubectl -n loadlab rollout status deploy/llm-server --timeout=10m
kind-two:
	kubectl apply -k k8s/overlays/two-replicas
	kubectl -n loadlab rollout status deploy/llm-server --timeout=10m
kind-down:
	kubectl delete namespace loadlab --ignore-not-found

clean:
	rm -rf results/mock .pytest_cache
