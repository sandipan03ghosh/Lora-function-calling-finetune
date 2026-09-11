"""Build the function-calling dataset from Salesforce/xlam-function-calling-60k.

Pipeline:
  1. parse the JSON columns (`tools`, `answers`)
  2. drop every row whose gold answer calls a tool or an argument NOT in its own schema
     (a real data-quality filter - keeps the training signal clean)
  3. dedup by normalised query
  4. hold out a fraction of unique tool names so the OOD test set contains APIs the model
     never saw during training
  5. 80/10/10 split of the in-distribution rows

`build_splits` is pure (no network) and is unit-tested. `prepare` wraps it with the download.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .config import DataConfig


def _loads(x):
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


def _norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def valid_row(tools, calls) -> bool:
    """True iff every gold call names a listed tool and uses only its documented parameters."""
    if not isinstance(tools, list) or not isinstance(calls, list) or not calls:
        return False
    tool_map = {t.get("name"): t for t in tools if isinstance(t, dict) and t.get("name")}
    for c in calls:
        if not isinstance(c, dict) or c.get("name") not in tool_map:
            return False
        params = tool_map[c["name"]].get("parameters") or {}
        if not isinstance(params, dict):
            return False
        for arg in c.get("arguments") or {}:
            if arg not in params:
                return False
    return True


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_splits(rows: list[dict], cfg: DataConfig) -> dict:
    """rows: [{id, query, tools, gold_calls}] already parsed + schema-validated."""
    rng = random.Random(cfg.seed)

    seen: set[str] = set()
    deduped: list[dict] = []
    for r in rows:
        key = _norm_query(r["query"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    all_tools = sorted({c["name"] for r in deduped for c in r["gold_calls"]})
    rng.shuffle(all_tools)
    n_hold = max(1, int(len(all_tools) * cfg.ood_tool_fraction))
    heldout = set(all_tools[:n_hold])

    id_rows, ood_rows = [], []
    for r in deduped:
        if any(c["name"] in heldout for c in r["gold_calls"]):
            ood_rows.append({**r, "split_kind": "ood"})
        else:
            id_rows.append({**r, "split_kind": "id"})

    rng.shuffle(id_rows)
    if cfg.max_examples and len(id_rows) > cfg.max_examples:
        id_rows = id_rows[: cfg.max_examples]

    n = len(id_rows)
    n_train = int(n * cfg.split_ratios[0])
    n_val = int(n * cfg.split_ratios[1])
    train = id_rows[:n_train]
    val = id_rows[n_train : n_train + n_val]
    test_id = id_rows[n_train + n_val :]

    rng.shuffle(ood_rows)
    test_ood = ood_rows[: cfg.ood_test_cap]

    # integrity checks
    train_tools = {c["name"] for r in train for c in r["gold_calls"]}
    assert not (train_tools & heldout), "OOD leak: a held-out tool appears in train"
    queries = [_norm_query(r["query"]) for r in (train + val + test_id + test_ood)]
    assert len(queries) == len(set(queries)), "duplicate query across splits"

    splits = {"train": train, "val": val, "test_id": test_id, "test_ood": test_ood}
    stats = {
        "n_raw_rows": len(rows),
        "n_after_dedup": len(deduped),
        "n_unique_tools": len(all_tools),
        "n_heldout_tools": len(heldout),
        "heldout_tools_sample": sorted(heldout)[:20],
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "avg_tools_per_example": round(
            sum(len(r["tools"]) for r in deduped) / max(1, len(deduped)), 2
        ),
        "avg_calls_per_example": round(
            sum(len(r["gold_calls"]) for r in deduped) / max(1, len(deduped)), 2
        ),
        "multi_call_fraction": round(
            sum(1 for r in deduped if len(r["gold_calls"]) > 1) / max(1, len(deduped)), 3
        ),
    }
    return {"splits": splits, "stats": stats}


def prepare(cfg: DataConfig) -> dict:
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_id, split="train")
    rows: list[dict] = []
    kept_invalid = 0
    for i, r in enumerate(ds):
        tools = _loads(r.get("tools"))
        answers = r.get("answers") if r.get("answers") is not None else r.get("answer")
        calls = _loads(answers)
        query = r.get("query") or r.get("question")
        if not query or tools is None or calls is None:
            continue
        if len(query.strip()) < cfg.min_query_chars:
            continue
        if not valid_row(tools, calls):
            kept_invalid += 1
            continue
        rows.append(
            {
                "id": r.get("id", i),
                "query": query.strip(),
                "tools": tools,
                "gold_calls": [
                    {"name": c["name"], "arguments": c.get("arguments", {})} for c in calls
                ],
            }
        )

    result = build_splits(rows, cfg)
    result["stats"]["n_dropped_schema_invalid"] = kept_invalid

    out = Path(cfg.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split_rows in result["splits"].items():
        _write_jsonl(out / f"{name}.jsonl", split_rows)
    (out / "dataset_stats.json").write_text(json.dumps(result["stats"], indent=2), encoding="utf-8")

    print("Dataset ready:\n" + json.dumps(result["stats"], indent=2))
    return result["stats"]
