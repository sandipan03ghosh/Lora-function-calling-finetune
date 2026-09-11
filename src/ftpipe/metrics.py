"""Pure-Python scoring for function-calling predictions.

No torch / transformers imports here on purpose: this module is unit-tested in CI and used to
score model outputs after generation.

A prediction is a string produced by a model. `parse_calls` recovers a list of
``{"name": str, "arguments": dict}`` from it (or ``None`` if nothing call-shaped is found).
An unparseable or non-canonical prediction is scored as wrong - that rewards the fine-tuned
model's formatting reliability.
"""

from __future__ import annotations

import json
import re
from collections import Counter

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*", "", t).strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def parse_calls(text: str | None) -> list[dict] | None:
    """Recover a list of tool calls from raw model output."""
    if not text or not text.strip():
        return None
    t = _strip_fences(text)

    raw = None
    try:
        raw = json.loads(t)
    except Exception:
        m = _ARRAY_RE.search(t)
        if m:
            try:
                raw = json.loads(m.group(0))
            except Exception:
                raw = None
        if raw is None:
            m = _OBJECT_RE.search(t)
            if m:
                try:
                    raw = json.loads(m.group(0))
                except Exception:
                    raw = None
    if raw is None:
        return None

    if isinstance(raw, dict):
        if isinstance(raw.get("tool_calls"), list):
            raw = raw["tool_calls"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        return None

    calls: list[dict] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or c.get("function") or c.get("tool")
        if isinstance(name, dict):  # openai-style {"function": {"name": ...}}
            name = name.get("name")
        args = c.get("arguments")
        if args is None:
            args = c.get("parameters")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if not name:
            continue
        calls.append({"name": str(name), "arguments": args})
    return calls


def _val_eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    return str(a).strip().lower() == str(b).strip().lower()


def _args_match(gold: dict, pred: dict) -> tuple[int, int]:
    matched = sum(1 for k, gv in gold.items() if k in pred and _val_eq(gv, pred[k]))
    return matched, len(gold)


def score_example(pred_calls, gold_calls, tool_names) -> dict:
    """Per-example raw counts. Aggregated later by `_aggregate`."""
    valid = pred_calls is not None
    pred_calls = pred_calls or []
    tool_names = set(tool_names or [])

    gold_names = Counter(c["name"] for c in gold_calls)
    pred_names = Counter(c["name"] for c in pred_calls)
    tp = sum((gold_names & pred_names).values())
    fp = sum((pred_names - gold_names).values())
    fn = sum((gold_names - pred_names).values())

    hallucinated = any(c["name"] not in tool_names for c in pred_calls) if tool_names else False

    used = [False] * len(pred_calls)
    exact = 0
    arg_matched = 0
    arg_total = 0
    for g in gold_calls:
        pick = -1
        for i, p in enumerate(pred_calls):
            if used[i] or p["name"] != g["name"]:
                continue
            m, tot = _args_match(g["arguments"], p["arguments"])
            if m == tot and len(p["arguments"]) == tot:
                pick = i
                break
            if pick == -1:
                pick = i
        if pick >= 0:
            used[pick] = True
            p = pred_calls[pick]
            m, tot = _args_match(g["arguments"], p["arguments"])
            arg_matched += m
            arg_total += tot
            if m == tot and len(p["arguments"]) == tot:
                exact += 1
        else:
            arg_total += len(g["arguments"])

    return {
        "valid_json": valid,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_calls": exact,
        "gold_calls": len(gold_calls),
        "arg_matched": arg_matched,
        "arg_total": arg_total,
        "hallucinated": hallucinated,
        "extra_calls": max(0, len(pred_calls) - len(gold_calls)),
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    gold = sum(r["gold_calls"] for r in rows)
    argt = sum(r["arg_total"] for r in rows)
    return {
        "n": len(rows),
        "function_name_f1": round(f1, 4),
        "function_name_precision": round(prec, 4),
        "function_name_recall": round(rec, 4),
        "exact_call_match": round(sum(r["exact_calls"] for r in rows) / gold, 4) if gold else 0.0,
        "arg_value_accuracy": round(sum(r["arg_matched"] for r in rows) / argt, 4) if argt else 0.0,
        "json_validity": round(sum(1 for r in rows if r["valid_json"]) / len(rows), 4),
        "hallucinated_function_rate": round(
            sum(1 for r in rows if r["hallucinated"]) / len(rows), 4
        ),
        "over_calling_rate": round(sum(1 for r in rows if r["extra_calls"] > 0) / len(rows), 4),
    }


def score_records(records: list[dict]) -> dict:
    """records: [{id, query, tools, gold_calls, split_kind, prediction}] -> overall + per-split metrics."""
    scored = []
    for r in records:
        pred = parse_calls(r.get("prediction"))
        tool_names = {t.get("name") for t in r.get("tools", [])}
        s = score_example(pred, r["gold_calls"], tool_names)
        s["split_kind"] = r.get("split_kind", "all")
        scored.append(s)
    kinds = sorted({s["split_kind"] for s in scored})
    return {
        "overall": _aggregate(scored),
        "by_split": {k: _aggregate([s for s in scored if s["split_kind"] == k]) for k in kinds},
    }


def _example_score(record: dict) -> float:
    pred = parse_calls(record.get("prediction"))
    tool_names = {t.get("name") for t in record.get("tools", [])}
    s = score_example(pred, record["gold_calls"], tool_names)
    return s["exact_calls"] / s["gold_calls"] if s["gold_calls"] else 0.0


def head_to_head(base_records: list[dict], ft_records: list[dict], max_examples: int = 15) -> dict:
    """Align base vs fine-tuned predictions by id; bucket into helped / hurt."""
    b = {r["id"]: r for r in base_records}
    f = {r["id"]: r for r in ft_records}
    helped, hurt = [], []
    for _id in b.keys() & f.keys():
        bs, fs = _example_score(b[_id]), _example_score(f[_id])
        item = {
            "id": _id,
            "split_kind": b[_id].get("split_kind", "all"),
            "query": b[_id]["query"],
            "gold": json.dumps(b[_id]["gold_calls"], ensure_ascii=False)[:500],
            "base": (b[_id].get("prediction") or "")[:500],
            "finetuned": (f[_id].get("prediction") or "")[:500],
        }
        if fs > bs:
            helped.append(item)
        elif fs < bs:
            hurt.append(item)
    return {
        "n_helped": len(helped),
        "n_hurt": len(hurt),
        "helped": helped[:max_examples],
        "hurt": hurt[:max_examples],
    }
