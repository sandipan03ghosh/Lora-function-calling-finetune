"""QLoRA fine-tuning with PEFT + TRL, tracked in MLflow.

Known-good with unsloth (2024-2025) + trl>=0.12. If a newer TRL renames SFTConfig fields or
the `tokenizer=` argument of SFTTrainer, adjust the two spots marked "TRL API spot".
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from .config import TrainConfig, load_config
from .metrics import parse_calls, score_example
from .prompts import build_messages, format_target


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _quick_task_eval(model, tok, rows: list[dict]) -> dict:
    """A small greedy eval so we log a task metric next to eval_loss."""
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    prompts = [
        tok.apply_chat_template(
            build_messages(r["query"], r["tools"]), tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    tp = fp = fn = exact = gold_total = 0
    for start in range(0, len(prompts), 8):
        chunk = prompts[start : start + 8]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(
            model.device
        )
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id
            )
        plen = enc["input_ids"].shape[1]
        for j, out_row in enumerate(gen):
            text = tok.decode(out_row[plen:], skip_special_tokens=True)
            r = rows[start + j]
            s = score_example(
                parse_calls(text), r["gold_calls"], {t["name"] for t in r["tools"]}
            )
            tp += s["tp"]
            fp += s["fp"]
            fn += s["fn"]
            exact += s["exact_calls"]
            gold_total += s["gold_calls"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    FastLanguageModel.for_training(model)
    return {
        "val_function_name_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "val_exact_call_match": round(exact / gold_total, 4) if gold_total else 0.0,
    }


def run(cfg: TrainConfig) -> dict:
    _set_seed(cfg.seed)

    import mlflow
    import torch
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    proc = Path(cfg.processed_dir)
    train_rows = _read_jsonl(proc / "train.jsonl")
    val_rows = _read_jsonl(proc / "val.jsonl")
    if cfg.train_subset:
        train_rows = train_rows[: cfg.train_subset]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg.model_4bit,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    def to_text(row: dict) -> dict:
        msgs = build_messages(row["query"], row["tools"]) + [
            {"role": "assistant", "content": format_target(row["gold_calls"])}
        ]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    train_ds = Dataset.from_list([to_text(r) for r in train_rows])
    val_ds = Dataset.from_list([to_text(r) for r in val_rows])

    args = SFTConfig(
        dataset_text_field="text",  # TRL API spot #1
        max_length=cfg.max_seq_length,  # TRL renamed max_seq_length -> max_length
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps if cfg.max_steps else -1,
        learning_rate=cfg.lr,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        seed=cfg.seed,
        output_dir=f"{cfg.output_dir}/checkpoints/{cfg.run_name}",
        report_to="mlflow",
        run_name=cfg.run_name,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,  # TRL API spot #2 (newer TRL: processing_class=tok)
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part=cfg.instruction_part,
            response_part=cfg.response_part,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[train] completion-only masking unavailable ({exc}); training on full text.")

    mlflow.set_experiment(cfg.experiment_name)
    t0 = time.time()
    trainer.train()
    duration = time.time() - t0

    best_dir = Path(cfg.output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tok.save_pretrained(str(best_dir))

    task_metrics = _quick_task_eval(model, tok, val_rows[: cfg.val_eval_n]) if cfg.val_eval_n else {}
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    summary = {
        "run_name": cfg.run_name,
        "model": cfg.model,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "n_train": len(train_rows),
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "task_metrics": task_metrics,
        "train_runtime_s": round(duration, 1),
        "peak_gpu_mem_mb": round(peak_mem, 1),
        "adapter_path": str(best_dir),
    }
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # append custom metrics/artifacts to the run the TRL MLflow callback just created
    run = mlflow.last_active_run()
    if run is not None:
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_params(
                {
                    "cfg_lora_r": cfg.lora_r,
                    "cfg_lora_alpha": cfg.lora_alpha,
                    "cfg_lora_dropout": cfg.lora_dropout,
                    "cfg_target_modules": ",".join(cfg.target_modules),
                    "cfg_lr": cfg.lr,
                    "cfg_epochs": cfg.epochs,
                    "cfg_n_train": len(train_rows),
                    "cfg_seed": cfg.seed,
                }
            )
            mlflow.log_metrics(
                {
                    **task_metrics,
                    "train_runtime_s": round(duration, 1),
                    "peak_gpu_mem_mb": round(peak_mem, 1),
                    "best_eval_loss": float(trainer.state.best_metric or 0.0),
                }
            )
            mlflow.log_artifact(str(Path(cfg.output_dir) / "train_summary.json"))

    print(json.dumps(summary, indent=2))
    return summary


def main(config_path: str, run_name: str | None = None) -> dict:
    cfg = load_config(config_path, TrainConfig)
    if run_name:
        cfg.run_name = run_name
    return run(cfg)
