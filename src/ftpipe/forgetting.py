"""Catastrophic-forgetting check.

Fine-tuning can degrade general ability. We measure base vs fine-tuned on two standard
multiple-choice benchmarks (ARC-Easy, HellaSwag) by scoring each answer choice with the
model's log-likelihood - the same idea lm-eval-harness uses for `acc` / `acc_norm`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import EvalConfig, load_config


def _score_choice(model, tok, context: str, continuation: str) -> tuple[float, int]:
    import torch

    ctx_ids = tok(context, return_tensors="pt").input_ids
    full_ids = tok(context + continuation, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        logits = model(full_ids).logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = full_ids[0, 1:]
    tok_logp = logp[torch.arange(targets.shape[0]), targets]
    n_cont = full_ids.shape[1] - ctx_ids.shape[1]
    if n_cont <= 0:
        return -1e9, 1
    return tok_logp[-n_cont:].sum().item(), n_cont


def _accuracy(model, tok, items) -> dict:
    correct = correct_norm = 0
    for context, choices, answer in items:
        scored = [_score_choice(model, tok, context, c) for c in choices]
        by_sum = max(range(len(choices)), key=lambda i: scored[i][0])
        by_norm = max(range(len(choices)), key=lambda i: scored[i][0] / scored[i][1])
        correct += by_sum == answer
        correct_norm += by_norm == answer
    n = max(1, len(items))
    return {"acc": round(correct / n, 4), "acc_norm": round(correct_norm / n, 4), "n": n}


def _load_arc(n: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(n))
    items = []
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        items.append(
            (f"Question: {r['question']}\nAnswer:", [f" {t}" for t in texts], labels.index(r["answerKey"]))
        )
    return items


def _load_hellaswag(n: int):
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation").select(range(n))
    return [
        (r["ctx"], [f" {e}" for e in r["endings"]], int(r["label"]))
        for r in ds
        if r["label"] != ""
    ]


def run(cfg: EvalConfig, base: str, adapter: str | None, use_unsloth: bool = True) -> dict:
    from .models import load_model

    arc = _load_arc(cfg.forgetting.arc_n)
    hs = _load_hellaswag(cfg.forgetting.hellaswag_n)

    report: dict = {}
    for tag, adp in [("base", None), ("finetuned", adapter)]:
        model, tok = load_model(base, adapter=adp, use_unsloth=use_unsloth)
        report[tag] = {
            "arc_easy": _accuracy(model, tok, arc),
            "hellaswag": _accuracy(model, tok, hs),
        }
        del model

    def _delta(bench: str) -> dict:
        b = report["base"][bench]["acc"]
        f = report["finetuned"][bench]["acc"]
        return {
            "base_acc": b,
            "finetuned_acc": f,
            "delta": round(f - b, 4),
            "retention_pct": round(100 * f / b, 1) if b else None,
        }

    report["summary"] = {"arc_easy": _delta("arc_easy"), "hellaswag": _delta("hellaswag")}
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "forgetting.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return report


def main(config_path: str, base: str, adapter: str | None) -> dict:
    return run(load_config(config_path, EvalConfig), base=base, adapter=adapter)
