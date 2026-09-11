"""Evaluate a model (base or fine-tuned) on the held-out + handcrafted benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from .config import EvalConfig, load_config
from .metrics import head_to_head, score_records


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(
    cfg: EvalConfig,
    base: str,
    adapter: str | None = None,
    tag: str = "base",
    use_unsloth: bool = True,
) -> dict:
    from .models import generate_for, load_model

    records: list[dict] = []
    for path in cfg.benchmarks:
        default_kind = Path(path).stem.replace("test_", "")
        for row in _read_jsonl(path):
            row.setdefault("split_kind", default_kind)
            records.append(row)

    model, tok = load_model(base, adapter=adapter, use_unsloth=use_unsloth)
    preds = generate_for(
        model, tok, records, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size
    )
    for row, pred in zip(records, preds):
        row["prediction"] = pred

    summary = score_records(records)

    out_dir = Path(cfg.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import mlflow

        mlflow.set_experiment("function-calling-eval")
        with mlflow.start_run(run_name=f"eval-{tag}"):
            mlflow.log_metrics(
                {k: v for k, v in summary["overall"].items() if isinstance(v, (int, float))}
            )
            for split, m in summary["by_split"].items():
                mlflow.log_metrics(
                    {f"{split}_{k}": v for k, v in m.items() if isinstance(v, (int, float))}
                )
    except Exception as exc:  # pragma: no cover
        print(f"[evaluate] mlflow logging skipped ({exc})")

    _maybe_head_to_head(cfg)
    print(json.dumps(summary, indent=2))
    return summary


def _maybe_head_to_head(cfg: EvalConfig) -> None:
    base_p = Path(cfg.output_dir) / "base" / "predictions.jsonl"
    ft_p = Path(cfg.output_dir) / "finetuned" / "predictions.jsonl"
    if not (base_p.exists() and ft_p.exists()):
        return

    h2h = head_to_head(_read_jsonl(base_p), _read_jsonl(ft_p))
    (Path(cfg.output_dir) / "head_to_head.json").write_text(
        json.dumps(h2h, indent=2), encoding="utf-8"
    )

    lines = [
        "# Base vs fine-tuned - head to head",
        "",
        f"- fine-tuning **helped** on {h2h['n_helped']} examples",
        f"- fine-tuning **hurt** on {h2h['n_hurt']} examples (regressions)",
        "",
    ]
    for title, key in [
        ("Examples where fine-tuning helped", "helped"),
        ("Regressions - fine-tuning made it worse", "hurt"),
    ]:
        lines.append(f"## {title}")
        for ex in h2h[key][:5]:
            lines += [
                f"**Query ({ex['split_kind']}):** {ex['query']}",
                "",
                f"- gold: `{ex['gold']}`",
                f"- base: `{ex['base']}`",
                f"- finetuned: `{ex['finetuned']}`",
                "",
            ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/head_to_head.md").write_text("\n".join(lines), encoding="utf-8")


def main(config_path: str, base: str, adapter: str | None, tag: str) -> dict:
    return run(load_config(config_path, EvalConfig), base=base, adapter=adapter, tag=tag)
