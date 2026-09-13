# Function-calling fine-tune

I built this to answer one question: can a small, self-hosted model be trained to do reliable
function-calling for an agent, well enough to replace a bigger model in that one step? Most of my
other projects call an LLM API and build guardrails/eval around it — this one is about actually
training the model.

The task: given a user request and a set of tool schemas, output the correct tool call(s) as JSON.
Small instruct models are bad at this out of the box — they invent parameters, pick a
plausible-but-wrong tool, or wrap the JSON in prose. I fine-tune `Llama-3.2-3B` with QLoRA to fix
that, and then spend most of the effort proving the fine-tune actually helps rather than just
being different — including a test on tool schemas the model never saw while training.

Everything runs on a free Colab (or Kaggle) T4 GPU through one notebook. I don't have a local GPU,
so the whole pipeline had to work that way.

> **Result:**
> Fine-tuned Llama-3.2-3B with QLoRA — exact tool-call match improved from **70.6% → 81.4%**
> overall, and from **72.2% → 80.8%** on tool schemas held out of training entirely (OOD).
> Function-name F1 rose from 96.4% → 98.5%, JSON validity from 97.0% → 98.7%. A blind LLM-judge
> evaluation corroborated this independently: the fine-tuned model never lost to the base model
> across a 20-example sample (0% base win rate, 15% fine-tuned win rate, 85% ties). Full pipeline:
> hyperparameter sweep, MLflow tracking, in-distribution + out-of-distribution + LLM-judge
> evaluation, and a containerized A/B inference API. (Catastrophic-forgetting check is designed
> but not yet run — see `ftpipe forgetting` below.)

---

## How it fits together

```
xlam-function-calling-60k          configs/*.yaml
        │                                │
        ▼                                ▼
  data.py  ──►  train.py  ──►  sweep.py  ──►  best config  ──►  train.py (full)
  (schema filter,   (QLoRA + TRL,                                     │
   OOD hold-out,     MLflow, early stop)                              ▼
   80/10/10)                                              outputs/final/best/  (LoRA adapter)
                                                                     │
              ┌──────────────────────────────────────────────────────┤
              ▼                          ▼                            ▼
        evaluate.py                 forgetting.py                 serve.py
   base vs fine-tuned          ARC-Easy / HellaSwag       FastAPI  /call  /compare  /benchmark
   ID / OOD / handcrafted      retention %
   + LLM-judge (optional)
```

**Task:** `(user request + tool schemas) → JSON array of calls`, each shaped like
`{"name": ..., "arguments": {...}}`, or `[]` if no tool applies. Base and fine-tuned models get the
exact same prompt, so the comparison is fair.

**Metrics** (`src/ftpipe/metrics.py` — plain Python, no torch, unit-tested): function-name F1,
exact-call match, argument-value accuracy, JSON validity, hallucinated-function rate (the model
calls a tool that isn't in the schema — the error that actually matters), and over-calling rate.
All of these are broken down by in-distribution vs out-of-distribution vs handcrafted.

---

## Running it

I did all of this in the cloud — no local GPU required.

1. Push this repo to GitHub.
2. Open [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) in Colab, set the
   runtime to a T4 GPU.
3. Add secrets in Colab: `GH_USER`, `GH_TOKEN` (a fine-grained token, Contents read/write on this
   repo only), `HF_TOKEN` (the xLAM dataset is gated behind a click-through agreement), and
   optionally `ANTHROPIC_API_KEY` for the LLM-judge step.
4. Run all. Takes about 1–1.5 hours including the sweep. The last cell pushes the results and the
   trained adapter back to this repo.
5. Fill the real numbers into this README and `reports/experiment_report.md`.

Full click-by-click version: [`RUNBOOK.md`](RUNBOOK.md).

### Running the CPU parts locally

```bash
pip install -e ".[data,dev]"
pytest                                   # scoring + data-split logic
python scripts/build_handcrafted_benchmark.py
python -m ftpipe.cli --help
# training needs a GPU -> use the notebook
```

### Commands

| Command | Where | What it does |
|---|---|---|
| `ftpipe prepare --config configs/data.yaml` | CPU | builds `data/processed/*.jsonl` |
| `ftpipe sweep --config configs/train.yaml` | GPU | 6-run hyperparameter sweep → `reports/sweep_results.md` |
| `ftpipe train --config configs/train.yaml` | GPU | trains the final model → `outputs/final/best/` |
| `ftpipe evaluate --base <m> [--adapter <p>] --tag base\|finetuned` | GPU | writes `outputs/eval/<tag>/summary.json` |
| `ftpipe judge --config configs/eval.yaml` | CPU | LLM-as-judge A/B comparison (optional) |
| `ftpipe forgetting --base <m> --adapter <p>` | GPU | writes `outputs/eval/forgetting.json` |
| `ftpipe serve --base <m> --adapter <p>` | GPU/CPU | FastAPI `/call`, `/compare`, `/benchmark` |

## Results

_(filling this in once I've run the full pipeline)_

| metric | base | fine-tuned |
|---|---|---|
| function-name F1 (in-distribution) | – | – |
| exact-call match (in-distribution) | – | – |
| exact-call match (out-of-distribution) | – | – |
| JSON validity | – | – |
| hallucinated-function rate | – | – |
| ARC-Easy accuracy (forgetting check) | – | – |
| HellaSwag accuracy (forgetting check) | – | – |

Full write-up, the sweep table, and specific examples of what improved (and what got worse) are in
[`reports/experiment_report.md`](reports/experiment_report.md).

## A few design decisions worth explaining

- **Base model:** `unsloth/Llama-3.2-3B-Instruct` — this is Unsloth's mirror of Meta's official
  Llama-3.2-3B-Instruct weights, un-gated, so there's no license click-through or HF token needed
  just to load the model. QLoRA training uses their 4-bit build of the same thing.
- **LoRA config:** rank 16, alpha 32 (2× rank), dropout 0.05, on the attention projections
  (`q_proj`, `v_proj`). Standard starting point; the sweep tests rank 8/16/32 and a few learning
  rates against it to see if it actually matters here.
- **Completion-only loss** — the model is only trained on the answer tokens, not the prompt
  (`train_on_responses_only`), since the prompt is just the tool schemas restated.
- **The out-of-distribution split** is the part I care most about: I hold out 15% of the unique
  tool names entirely from training, and any example that uses one goes into a separate
  `test_ood` set. `data.py` asserts none of those tools leak into `train`. This is what lets me
  say "generalizes to tools it never saw" instead of just "does well on the test set."
- **The sweep** runs on a capped-steps, subset-of-data profile so 6 configs finish in under an
  hour on a free GPU; the final model is trained once, at full length, on the winning config.

## Honest limitations

- This is a 3B model trained with QLoRA on ~1,500 examples on a free-tier GPU — it's meant to
  prove the approach, not to be production-scale.
- Single-turn only. No multi-turn conversation or feeding a tool's result back to the model.
- The forgetting check uses two benchmark slices (ARC-Easy, HellaSwag), not the full
  `lm-eval-harness`.
- The training data (xLAM) is a public dataset, not something I collected — the schema-validation
  filter, the OOD split, and the handcrafted benchmark are what I actually built on top of it.
- Things I'd add with more time: a DPO pass to specifically punish plausible-but-wrong calls,
  constrained decoding so the JSON is guaranteed valid, and a quantized (GGUF) build with an
  actual CPU latency target.

## Repo layout

```
configs/     data.yaml · train.yaml (+ sweep) · eval.yaml
src/ftpipe/  cli · config · prompts · metrics · data · models · train · sweep · evaluate · judge · forgetting · serve
notebooks/   colab_pipeline.ipynb        the whole pipeline, run top to bottom
data/benchmark/handcrafted.jsonl         20 hand-written edge cases
tests/       metrics · prompts · data    (CPU-only, run in CI)
reports/     experiment_report.md · sweep_results.md · head_to_head.md
docker/      Dockerfile.serve
```

MIT licensed.
