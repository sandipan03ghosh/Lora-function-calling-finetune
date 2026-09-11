# Convenience wrappers around `python -m ftpipe.cli`. Every step is reproducible from a config file.
.PHONY: prepare sweep train eval-base eval-ft head-to-head judge forgetting serve test lint all

CONFIG_DATA  ?= configs/data.yaml
CONFIG_TRAIN ?= configs/train.yaml
CONFIG_EVAL  ?= configs/eval.yaml
BASE         ?= unsloth/Llama-3.2-3B-Instruct
ADAPTER      ?= outputs/final/best

prepare:
	python -m ftpipe.cli prepare --config $(CONFIG_DATA)

sweep:
	python -m ftpipe.cli sweep --config $(CONFIG_TRAIN)

train:
	python -m ftpipe.cli train --config $(CONFIG_TRAIN)

eval-base:
	python -m ftpipe.cli evaluate --config $(CONFIG_EVAL) --base $(BASE) --tag base

eval-ft:
	python -m ftpipe.cli evaluate --config $(CONFIG_EVAL) --base $(BASE) --adapter $(ADAPTER) --tag finetuned

judge:
	python -m ftpipe.cli judge --config $(CONFIG_EVAL)

forgetting:
	python -m ftpipe.cli forgetting --config $(CONFIG_EVAL) --base $(BASE) --adapter $(ADAPTER)

serve:
	python -m ftpipe.cli serve --base $(BASE) --adapter $(ADAPTER)

test:
	pytest

lint:
	ruff check .

# full GPU pipeline (run on Colab/Kaggle)
all: prepare sweep train eval-base eval-ft forgetting
