"""A *simulated* OpenAI-compatible LLM server for tests and offline demos.

THIS IS A MOCK. It does not run a model. It sleeps according to a simple
parametric model of a single accelerator:

    prefill:  one prefill at a time (the accelerator is compute-bound during
              prefill), taking  prompt_tokens / prefill_tok_s
                                + quadratic_coef * prompt_tokens**2  seconds
    decode:   all active sequences decode together (continuous batching); each
              token takes  (1 / decode_tok_s) * (1 + batch_penalty * (active - 1))
    overhead: fixed per-request overhead (gateway, auth, network)

Because prefill is serialized, concurrent long prompts queue behind each other.
That lets the analysis code be tested against a server whose ground truth we know.

Run:  python -m loadlab.mock_server --port 8099
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

from aiohttp import web

from .prompts import estimate_tokens


@dataclass
class MockParams:
    prefill_tok_s: float = 4000.0
    quadratic_coef: float = 0.0
    decode_tok_s: float = 50.0
    batch_penalty: float = 0.05
    overhead_s: float = 0.01
    time_scale: float = 1.0  # multiply all sleeps (tests use a small value)


class MockEngine:
    def __init__(self, params: MockParams):
        self.p = params
        self.prefill_lock = asyncio.Lock()
        self.active_decode = 0

    def prefill_seconds(self, prompt_tokens: int) -> float:
        return prompt_tokens / self.p.prefill_tok_s + self.p.quadratic_coef * prompt_tokens ** 2

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds * self.p.time_scale)

    async def generate(self, prompt_tokens: int, max_tokens: int):
        await self.sleep(self.p.overhead_s)
        async with self.prefill_lock:
            await self.sleep(self.prefill_seconds(prompt_tokens))
        self.active_decode += 1
        try:
            for i in range(max_tokens):
                if i > 0:
                    per_tok = (1.0 / self.p.decode_tok_s) * (1 + self.p.batch_penalty * (self.active_decode - 1))
                    await self.sleep(per_tok)
                yield f"tok{i} "
        finally:
            self.active_decode -= 1


def make_app(params: MockParams | None = None) -> web.Application:
    engine = MockEngine(params or MockParams())
    app = web.Application()

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        text = " ".join(m.get("content", "") for m in body.get("messages", []))
        prompt_tokens = estimate_tokens(text)
        max_tokens = int(body.get("max_tokens") or 16)
        created = int(time.time())
        if not body.get("stream"):
            out = [t async for t in engine.generate(prompt_tokens, max_tokens)]
            return web.json_response({
                "id": "mock", "object": "chat.completion", "created": created, "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(out)}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out), "total_tokens": prompt_tokens + len(out)},
            })
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        n = 0
        async for tok in engine.generate(prompt_tokens, max_tokens):
            n += 1
            chunk = {"id": "mock", "object": "chat.completion.chunk", "created": created, "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"id": "mock", "object": "chat.completion.chunk", "created": created, "model": body.get("model"),
                     "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                              "total_tokens": prompt_tokens + n}}
            await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "mock-model", "object": "model"}]})

    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--prefill-tok-s", type=float, default=MockParams.prefill_tok_s)
    p.add_argument("--quadratic-coef", type=float, default=MockParams.quadratic_coef)
    p.add_argument("--decode-tok-s", type=float, default=MockParams.decode_tok_s)
    p.add_argument("--batch-penalty", type=float, default=MockParams.batch_penalty)
    p.add_argument("--overhead-s", type=float, default=MockParams.overhead_s)
    a = p.parse_args()
    params = MockParams(a.prefill_tok_s, a.quadratic_coef, a.decode_tok_s, a.batch_penalty, a.overhead_s)
    web.run_app(make_app(params), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
