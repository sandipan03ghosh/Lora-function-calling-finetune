"""Typed config objects loaded from the YAML files in configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_id: str = "Salesforce/xlam-function-calling-60k"
    max_examples: int = 1500
    ood_tool_fraction: float = 0.15
    ood_test_cap: int = 200
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    min_query_chars: int = 10
    seed: int = 42
    processed_dir: str = "data/processed"


class JudgeConfig(BaseModel):
    provider: Literal["none", "anthropic", "openai", "gemini"] = "none"
    model: str = "claude-3-5-haiku-20241022"
    n_examples: int = 40


class ForgettingConfig(BaseModel):
    arc_n: int = 100
    hellaswag_n: int = 100


class EvalConfig(BaseModel):
    benchmarks: list[str]
    max_new_tokens: int = 512
    batch_size: int = 8
    seed: int = 42
    output_dir: str = "outputs/eval"
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    forgetting: ForgettingConfig = Field(default_factory=ForgettingConfig)


class TrainConfig(BaseModel):
    experiment_name: str = "function-calling-lora"

    model: str = "unsloth/Llama-3.2-3B-Instruct"
    model_4bit: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    use_unsloth: bool = True
    max_seq_length: int = 2048

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_steps: int | None = None
    train_subset: int | None = None

    eval_steps: int = 25
    save_steps: int = 25
    early_stopping_patience: int = 3
    val_eval_n: int = 100

    seed: int = 42
    processed_dir: str = "data/processed"
    output_dir: str = "outputs/final"
    run_name: str = "final"

    instruction_part: str = "<|start_header_id|>user<|end_header_id|>\n\n"
    response_part: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    sweep_max_examples: int = 800
    sweep: list[dict] = Field(default_factory=list)


def load_config(path: str | Path, model: type[BaseModel]):
    """Parse a YAML file into the given pydantic model."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return model(**data)
