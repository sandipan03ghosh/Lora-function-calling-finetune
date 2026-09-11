"""Validate data/benchmark/handcrafted.jsonl.

The .jsonl file is the source of truth (hand-written, committed). This script checks every row:
  - it parses
  - every gold call names a tool that appears in that row's `tools` list
  - every gold-call argument is a documented parameter of that tool
  - `split_kind` is "handcrafted"

Run:  python scripts/build_handcrafted_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ftpipe.data import valid_row  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / "data" / "benchmark" / "handcrafted.jsonl"


def main() -> None:
    rows = [json.loads(x) for x in BENCH.read_text(encoding="utf-8").splitlines() if x.strip()]
    problems = []
    for r in rows:
        if r.get("split_kind") != "handcrafted":
            problems.append((r.get("id"), "split_kind != 'handcrafted'"))
        calls = r["gold_calls"]
        if calls and not valid_row(r["tools"], calls):
            problems.append((r.get("id"), "call references a tool/arg not in schema"))
    if problems:
        for pid, msg in problems:
            print(f"  BAD  {pid}: {msg}")
        raise SystemExit(f"{len(problems)} invalid row(s)")
    empties = sum(1 for r in rows if not r["gold_calls"])
    print(f"OK - {len(rows)} rows, {empties} no-tool-needed cases, all schema-valid")


if __name__ == "__main__":
    main()
