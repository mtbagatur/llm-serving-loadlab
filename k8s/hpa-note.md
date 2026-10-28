# Autoscaling note (HPA)

No HPA manifest is applied on purpose. A CPU- or request-rate-based HPA is the
wrong signal for an LLM server:

- During prefill the accelerator is saturated while request rate can be flat;
  a long-prompt workload can be slow at low RPS.
- GPU pods take minutes to become ready (image + model weights), so reactive
  scaling on a lagging metric overshoots or arrives late.

What I would use instead:

1. Scale on **in-flight requests / queue depth per replica** exported by the
   engine (vLLM exposes `vllm:num_requests_waiting` and `vllm:num_requests_running`
   on `/metrics`) through Prometheus Adapter or KEDA.
2. Keep a floor of replicas sized from a load test like the one in this repo
   (the prompt-length distribution decides the per-replica capacity).
3. Alert on TTFT p95 by prompt-length bucket.

Sketch (planned, not tested here — requires Prometheus + KEDA):

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {name: llm-server, namespace: loadlab}
spec:
  scaleTargetRef: {name: llm-server}
  minReplicaCount: 1
  maxReplicaCount: 4
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: sum(vllm:num_requests_waiting{namespace="loadlab"})
        threshold: "4"
```
