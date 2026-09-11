"""FastAPI inference server.

  GET  /health              - liveness + which model is loaded
  POST /call                - run one request through base OR fine-tuned
  POST /compare             - base vs fine-tuned side by side (the demo endpoint)
  POST /benchmark           - p50 / p95 latency of the fine-tuned model

One model is loaded; the LoRA adapter is toggled on/off with PeftModel.disable_adapter(), so
base and fine-tuned share the same weights in memory. Defaults to the transformers backend
(not Unsloth) for portability - override with use_unsloth=True on a GPU box.
"""

from __future__ import annotations

import time
from typing import Any

from .metrics import parse_calls
from .prompts import build_messages


def create_app(base: str, adapter: str, use_unsloth: bool = False):
    from fastapi import FastAPI
    from pydantic import BaseModel

    from .models import generate, load_model

    model, tok = load_model(base, adapter=adapter, use_unsloth=use_unsloth)

    def _infer(messages: list[dict], adapter_on: bool) -> dict:
        can_toggle = hasattr(model, "disable_adapter")
        t0 = time.time()
        if not adapter_on and can_toggle:
            with model.disable_adapter():
                out = generate(model, tok, [messages], max_new_tokens=512, batch_size=1)[0]
        else:
            out = generate(model, tok, [messages], max_new_tokens=512, batch_size=1)[0]
        return {
            "raw": out,
            "calls": parse_calls(out),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    app = FastAPI(title="ftpipe function-calling API")

    class CallIn(BaseModel):
        query: str
        tools: list[dict]
        model: str = "finetuned"  # finetuned | base

    class CompareIn(BaseModel):
        query: str
        tools: list[dict]

    class BenchIn(BaseModel):
        n: int = 20

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "base": base, "adapter": adapter}

    @app.post("/call")
    def call(body: CallIn) -> dict:
        return _infer(build_messages(body.query, body.tools), adapter_on=body.model != "base")

    @app.post("/compare")
    def compare(body: CompareIn) -> dict:
        msgs = build_messages(body.query, body.tools)
        b = _infer(msgs, adapter_on=False)
        f = _infer(msgs, adapter_on=True)
        return {"base": b, "finetuned": f, "agree": b["calls"] == f["calls"]}

    @app.post("/benchmark")
    def benchmark(body: BenchIn) -> dict:
        sample = build_messages(
            "What is the weather in Paris and what time is it there?",
            [
                {"name": "get_weather", "parameters": {"city": {"type": "str"}}},
                {"name": "get_time", "parameters": {"city": {"type": "str"}}},
            ],
        )
        lat = []
        for _ in range(max(2, body.n)):
            t0 = time.time()
            generate(model, tok, [sample], max_new_tokens=128, batch_size=1)
            lat.append((time.time() - t0) * 1000)
        lat.sort()
        return {
            "n": len(lat),
            "p50_ms": round(lat[len(lat) // 2], 1),
            "p95_ms": round(lat[max(0, int(len(lat) * 0.95) - 1)], 1),
            "mean_ms": round(sum(lat) / len(lat), 1),
        }

    return app


def serve(
    base: str,
    adapter: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    use_unsloth: bool = False,
) -> None:
    import uvicorn

    uvicorn.run(create_app(base, adapter, use_unsloth=use_unsloth), host=host, port=port)
