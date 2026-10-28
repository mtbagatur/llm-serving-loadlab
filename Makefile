# llm-serving-loadlab — see README.md
PY      ?= .venv/bin/python
MODEL   ?= qwen2.5:0.5b
LENGTHS ?= 128,512,1024,2048
CONC    ?= 1,2,4,8
N       ?= 8
OUT     ?= results/real

.PHONY: venv test mock-demo mock-open ollama-up ollama-down run-one run-two analyze run-open analyze-open \
	kind-create kind-load kind-up kind-two kind-forward kind-run-one kind-run-two kind-run-open kind-analyze kind-down kind-delete clean

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

# Open loop: Poisson arrivals at fixed average rates (req/s), one prompt length.
RATES   ?= 0.05,0.1,0.15,0.25
OPEN_LEN ?= 512
OPEN_N  ?= 16
mock-open:
	$(PY) -m loadlab.mock_server --port 8099 --prefill-tok-s 2000 --decode-tok-s 40 & echo $$! > .mock.pid; sleep 1
	$(PY) -m loadlab.loadgen --mode open --base-url http://127.0.0.1:8099/v1 --model mock --scenario mock-open \
		--prompt-lengths $(OPEN_LEN) --rates 1,2,4,8 --requests-per-cell 24 --max-tokens 32 --out results/mock/mock-open; \
		kill `cat .mock.pid`; rm -f .mock.pid
	$(PY) -m loadlab.openloop results/mock/mock-open.csv --out-dir results/mock/analysis-openloop --slo-ttft 1 \
		--source-note "MOCK server (simulated timings, no model)"
run-open:
	$(PY) -m loadlab.loadgen --mode open --base-url http://127.0.0.1:11501/v1 --model $(MODEL) --scenario open-one-replica \
		--prompt-lengths $(OPEN_LEN) --rates $(RATES) --requests-per-cell $(OPEN_N) --out $(OUT)/open-one-replica
analyze-open:
	$(PY) -m loadlab.openloop $(OUT)/open-*.csv --out-dir $(OUT)/analysis-openloop --slo-ttft 10 \
		--source-note "real run, Ollama 0.6.8 + $(MODEL) on CPU (see README)"

# ---------------------------------------------------------------- Kubernetes (kind)
# Separate cluster "loadlab" with its own kubeconfig, so no other cluster/context is touched.
# Model files are mounted read-only from MODEL_DIR (a directory containing models/manifests
# and models/blobs for qwen2.5:0.5b); the Ollama image is loaded from the local Docker cache.
KIND_NAME  ?= loadlab
KUBECONFIG_KIND ?= $(CURDIR)/.kind-kubeconfig
MODEL_DIR  ?= $(HOME)/.ollama
KOUT       ?= results/real-k8s
K          = KUBECONFIG=$(KUBECONFIG_KIND) kubectl
kind-create:
	sed "s#__MODEL_DIR__#$(MODEL_DIR)#" k8s/kind/kind-config.yaml.tmpl > .kind-config.yaml
	kind create cluster --config .kind-config.yaml --kubeconfig $(KUBECONFIG_KIND)
kind-load:
	# 'kind load docker-image' fails with Docker's containerd image store on multi-arch images; use an archive.
	docker save --platform linux/amd64 ollama/ollama:0.6.8 -o .ollama-image.tar
	kind load image-archive .ollama-image.tar --name $(KIND_NAME); rm -f .ollama-image.tar
kind-up:
	$(K) apply -k k8s/overlays/kind-one
	$(K) -n loadlab rollout status deploy/llm-server --timeout=10m
kind-two:
	$(K) apply -k k8s/overlays/kind-two
	$(K) -n loadlab rollout status deploy/llm-server --timeout=10m
# One port-forward per pod (18081, 18082, ...). Used because kube-proxy could not start on the
# test machine (see README); with a healthy kube-proxy use http://127.0.0.1:18080/v1 (NodePort).
kind-forward:
	-pkill -f "^kubectl -n loadlab port-forward" ; i=1; for p in `$(K) -n loadlab get pod -l app=llm-server -o name`; do \
		nohup $(K) -n loadlab port-forward $$p 1808$$i:11434 >/dev/null 2>&1 & i=$$((i+1)); done; sleep 2
kind-run-one:
	$(PY) -m loadlab.loadgen --base-url http://127.0.0.1:18081/v1 --model $(MODEL) --scenario kind-one \
		--prompt-lengths $(LENGTHS) --concurrency $(CONC) --requests-per-cell $(N) --out $(KOUT)/kind-one
kind-run-two:
	$(PY) -m loadlab.loadgen --base-url http://127.0.0.1:18081/v1 --base-url http://127.0.0.1:18082/v1 \
		--model $(MODEL) --scenario kind-two \
		--prompt-lengths $(LENGTHS) --concurrency $(CONC) --requests-per-cell $(N) --out $(KOUT)/kind-two
kind-run-open:
	$(PY) -m loadlab.loadgen --mode open --base-url http://127.0.0.1:18081/v1 --model $(MODEL) --scenario open-kind-one \
		--prompt-lengths $(OPEN_LEN) --rates $(RATES) --requests-per-cell $(OPEN_N) --out $(KOUT)/open-kind-one
kind-analyze:
	$(PY) -m loadlab.analyze $(KOUT)/kind-one.csv $(KOUT)/kind-two.csv --out-dir $(KOUT)/analysis \
		--baseline kind-one --source-note "real run on kind, Ollama 0.6.8 + $(MODEL), 6 exclusive CPUs per pod (see README)"
	$(PY) -m loadlab.openloop $(KOUT)/open-*.csv --out-dir $(KOUT)/analysis-openloop --slo-ttft 10 \
		--source-note "real run on kind, Ollama 0.6.8 + $(MODEL), 6 exclusive CPUs per pod (see README)"
kind-down:
	-pkill -f "^kubectl -n loadlab port-forward"
	$(K) delete namespace loadlab --ignore-not-found
kind-delete:
	kind delete cluster --name $(KIND_NAME) --kubeconfig $(KUBECONFIG_KIND)

clean:
	rm -rf results/mock .pytest_cache .kind-config.yaml .kind-kubeconfig
