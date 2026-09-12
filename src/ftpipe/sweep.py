"""Small hyperparameter sweep -> comparison table -> winning config.

Runs a fast profile (capped steps + a subset of training data) so 6 runs finish in ~30-40 min
on a Colab T4. The final model is trained separately at full length with the winner.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import train as train_mod
from .config import TrainConfig, load_config


def _write_reports(results: list[dict], complete: bool) -> None:
    ranked = sorted(results, key=lambda r: r.get("val_function_name_f1", 0.0), reverse=True)

    Path("reports").mkdir(exist_ok=True)
    lines = [
        "# Hyperparameter sweep results",
        "",
        "_Fast profile: capped steps + subset of training data. The final model is trained "
        "separately at full length with the winning config._",
        "",
        "_status: IN PROGRESS - partial results, sweep not yet finished_" if not complete else "",
        "",
        "| run | rank | lr | epochs | val function-name F1 | val exact-call match | runtime (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ranked:
        lines.append(
            f"| {r['run']} | {r['lora_r']} | {r['lr']} | {r['epochs']} | "
            f"{r.get('val_function_name_f1', '-')} | {r.get('val_exact_call_match', '-')} | "
            f"{r.get('runtime_s', '-')} |"
        )
    if complete and ranked:
        best = ranked[0]
        lines += [
            "",
            f"**Winning config:** rank {best['lora_r']}, lr {best['lr']}, epochs {best['epochs']} "
            f"(val function-name F1 {best.get('val_function_name_f1')}).",
            "",
            "Set these in `configs/train.yaml`, then run `ftpipe train` for the full-length model.",
        ]
    Path("reports/sweep_results.md").write_text("\n".join(lines), encoding="utf-8")
    Path("outputs/sweep").mkdir(parents=True, exist_ok=True)
    Path("outputs/sweep/comparison.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")


def run(cfg: TrainConfig) -> list[dict]:
    combos = cfg.sweep or [
        {"lora_r": 8, "lr": 2e-4, "epochs": 1},
        {"lora_r": 16, "lr": 2e-4, "epochs": 1},
        {"lora_r": 32, "lr": 2e-4, "epochs": 1},
    ]

    results: list[dict] = []
    for i, combo in enumerate(combos):
        run_cfg = cfg.model_copy(deep=True)
        for key, value in combo.items():
            setattr(run_cfg, key, value)
        run_cfg.lora_alpha = 2 * run_cfg.lora_r
        run_cfg.max_steps = None  # let each combo's own `epochs` govern run length
        run_cfg.train_subset = cfg.sweep_max_examples
        run_cfg.val_eval_n = min(cfg.val_eval_n, 80)
        run_cfg.early_stopping_patience = 10**6  # let short capped runs finish
        run_cfg.run_name = f"sweep-{i:02d}-r{run_cfg.lora_r}-lr{run_cfg.lr}-e{run_cfg.epochs}"
        run_cfg.output_dir = f"outputs/sweep/{run_cfg.run_name}"

        print(f"\n=== sweep {i + 1}/{len(combos)}: {combo} ===")
        summary = train_mod.run(run_cfg)
        results.append(
            {
                "run": run_cfg.run_name,
                "lora_r": run_cfg.lora_r,
                "lr": run_cfg.lr,
                "epochs": run_cfg.epochs,
                **summary.get("task_metrics", {}),
                "runtime_s": summary.get("train_runtime_s"),
            }
        )
        _write_reports(results, complete=False)  # survive a disconnect/interrupt mid-sweep

    _write_reports(results, complete=True)
    with open("reports/sweep_results.md", encoding="utf-8") as f:
        print(f.read())
    return sorted(results, key=lambda r: r.get("val_function_name_f1", 0.0), reverse=True)


def main(config_path: str) -> list[dict]:
    return run(load_config(config_path, TrainConfig))
