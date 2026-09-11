"""LLM-as-judge: blind A/B comparison of base vs fine-tuned tool calls.

Optional. Enabled by setting `judge.provider` in configs/eval.yaml to `anthropic` or `openai`
and providing the matching API key (as a Colab secret / env var). The judge sees the request,
the tool list, the reference answer, and the two responses in a random order.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from .config import EvalConfig, load_config

_PROMPT = """You are grading two assistants that must call the correct tool(s) for a user request.

User request:
{query}

Available tools:
{tools}

Reference (correct) answer:
{gold}

Response A:
{a}

Response B:
{b}

Score each response 1-5 (5 = matches the reference intent with the right tool and arguments;
1 = wrong tool or invalid output). Reply with JSON only:
{{"score_a": <int>, "score_b": <int>, "winner": "A" | "B" | "tie", "reason": "<short>"}}
"""


def _call_anthropic(model: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def _call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    r = client.chat.completions.create(
        model=model, max_tokens=300, messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(cfg: EvalConfig) -> dict:
    jc = cfg.judge
    if jc.provider == "none":
        print("[judge] provider = none; skipping (set eval.yaml judge.provider to enable).")
        return {"skipped": True}

    base = _read(Path(cfg.output_dir) / "base" / "predictions.jsonl")
    ft = {r["id"]: r for r in _read(Path(cfg.output_dir) / "finetuned" / "predictions.jsonl")}
    pairs = [(b, ft[b["id"]]) for b in base if b["id"] in ft][: jc.n_examples]

    rng = random.Random(cfg.seed)
    caller = _call_anthropic if jc.provider == "anthropic" else _call_openai
    rows, wins_ft, wins_base, ties, sum_base, sum_ft = [], 0, 0, 0, 0, 0

    for b, f in pairs:
        swap = rng.random() < 0.5
        a_txt, b_txt = (
            (f["prediction"], b["prediction"]) if swap else (b["prediction"], f["prediction"])
        )
        prompt = _PROMPT.format(
            query=b["query"],
            tools=json.dumps(b["tools"], ensure_ascii=False),
            gold=json.dumps(b["gold_calls"], ensure_ascii=False),
            a=a_txt,
            b=b_txt,
        )
        try:
            raw = caller(jc.model, prompt)
            verdict = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        except Exception as exc:  # pragma: no cover
            print(f"[judge] skipped one example ({exc})")
            continue

        ft_score = verdict["score_a"] if swap else verdict["score_b"]
        base_score = verdict["score_b"] if swap else verdict["score_a"]
        sum_base += base_score
        sum_ft += ft_score
        real_winner = {
            "A": "finetuned" if swap else "base",
            "B": "base" if swap else "finetuned",
        }.get(verdict.get("winner"), "tie")
        wins_ft += real_winner == "finetuned"
        wins_base += real_winner == "base"
        ties += real_winner == "tie"
        rows.append(
            {
                "id": b["id"],
                "base_score": base_score,
                "finetuned_score": ft_score,
                "winner": real_winner,
                "reason": verdict.get("reason", ""),
            }
        )

    n = max(1, len(rows))
    summary = {
        "n": len(rows),
        "provider": jc.provider,
        "model": jc.model,
        "mean_base_score": round(sum_base / n, 3),
        "mean_finetuned_score": round(sum_ft / n, 3),
        "finetuned_win_rate": round(wins_ft / n, 3),
        "base_win_rate": round(wins_base / n, 3),
        "tie_rate": round(ties / n, 3),
    }
    out = Path(cfg.output_dir)
    (out / "judge.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(config_path: str) -> dict:
    return run(load_config(config_path, EvalConfig))
