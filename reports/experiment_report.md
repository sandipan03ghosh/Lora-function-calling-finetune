# Experiment report — fine-tuning Llama-3.2-3B for function calling

_Fill every `_fill_` marker after the Colab run. Sources: `outputs/eval/*/summary.json`,
`outputs/eval/head_to_head.json`, `outputs/eval/forgetting.json`, `reports/sweep_results.md`,
`outputs/eval/judge_summary.json`._

## 1. Problem and why fine-tuning

Agents need a model that, given a user request and a set of tool schemas, emits the **right tool
call(s) with correct arguments in valid JSON**. Small instruct models do this unreliably: they
invent parameters, pick a plausible-but-wrong tool, or wrap the JSON in prose.

**Why not few-shot prompting?** The tool schemas already fill the prompt; adding several worked
examples per call inflates every request's token cost and latency, and still doesn't fix format
drift.

**Why not RAG?** There is nothing to retrieve — the tools are given in-context. This is a
behaviour/format problem, which is what fine-tuning is for.

**Goal:** match a much larger model's tool-use reliability with a 3B model that can be self-hosted.

## 2. Dataset

- **Source:** `Salesforce/xlam-function-calling-60k` (Apache-2.0).
- **Schema-validation filter:** every row whose gold answer calls a tool or an argument absent
  from its own schema is dropped — `_fill_` rows removed (`n_dropped_schema_invalid` in
  `dataset_stats.json`).
- **Dedup:** by normalised query.
- **Out-of-distribution hold-out:** 15% of unique tool names (`_fill_` of `_fill_` tools) are
  removed from training entirely; any example that calls one is routed to `test_ood`. `data.py`
  asserts no held-out tool appears in `train`.
- **Split:** 80/10/10 of the in-distribution rows.

| split | size |
|---|---|
| train | _fill_ |
| val | _fill_ |
| test_id | _fill_ |
| test_ood | _fill_ |
| handcrafted | 20 |

Avg tools per example: _fill_ · avg calls per example: _fill_ · multi-call fraction: _fill_.

## 3. Training setup

- **Base:** `unsloth/Llama-3.2-3B-Instruct`, QLoRA 4-bit.
- **LoRA:** rank _fill_, alpha _fill_, dropout 0.05, target modules `q_proj,v_proj`.
- **Optimiser:** AdamW 8-bit, cosine schedule, lr _fill_, warmup 0.05, weight decay 0.01.
- **Loss:** completion-only (answer tokens only).
- **Early stopping:** on `eval_loss`, patience 3; best checkpoint = step _fill_.
- **Hardware:** Colab T4 16 GB · wall-clock _fill_ min · peak GPU mem _fill_ MB.
- **Cost:** free tier (≈ $0 · _fill_ compute-hours).

## 4. Hyperparameter sweep

6 runs on a fast profile (capped steps + `_fill_` training examples). Full table in
`reports/sweep_results.md`.

| run | rank | lr | epochs | val function-name F1 | val exact-call match |
|---|---|---|---|---|---|
| _fill_ | | | | | |

**Winning config:** rank _fill_, lr _fill_, epochs _fill_ — chosen on validation function-name F1.
Observations: `_fill_ (e.g. rank had little effect above 16; lr 5e-4 was unstable)`.

## 5. Results — base vs fine-tuned

| metric | base | fine-tuned | Δ |
|---|---|---|---|
| function-name F1 — in-distribution | _fill_ | _fill_ | |
| function-name F1 — **out-of-distribution** | _fill_ | _fill_ | |
| exact-call match — in-distribution | _fill_ | _fill_ | |
| exact-call match — **out-of-distribution** | _fill_ | _fill_ | |
| exact-call match — handcrafted | _fill_ | _fill_ | |
| argument-value accuracy | _fill_ | _fill_ | |
| JSON validity | _fill_ | _fill_ | |
| hallucinated-function rate | _fill_ | _fill_ | |
| over-calling rate | _fill_ | _fill_ | |

**LLM-as-judge** (`_fill_` provider, n=`_fill_`): mean score base `_fill_` vs fine-tuned `_fill_`;
fine-tuned win rate `_fill_`.

**Head-to-head** (`outputs/eval/head_to_head.json`): fine-tuning helped on `_fill_` examples,
hurt on `_fill_`.

### Examples where fine-tuning helped
`_fill_ — paste 2-3 from reports/head_to_head.md, at least one from test_ood`

### Regressions — where fine-tuning made it worse
`_fill_ — paste 2-3, and say why (e.g. over-fit to always emitting an array; lost a rare
zero-call case)`

## 6. Catastrophic forgetting

| benchmark | base acc | fine-tuned acc | Δ | retention |
|---|---|---|---|---|
| ARC-Easy | _fill_ | _fill_ | _fill_ | _fill_ % |
| HellaSwag | _fill_ | _fill_ | _fill_ | _fill_ % |

Interpretation: `_fill_ — small LoRA rank on attention-only projections keeps general ability
largely intact; note any drop and whether fewer epochs would trade task gain for retention.`

## 7. Honest limitations — when NOT to use this

- **Compliance-critical routing:** if a wrong tool call is unacceptable, a deterministic
  router or schema-constrained decoding is safer than any fine-tune.
- **Frequently-changing tool catalogue:** the model generalizes to unseen tools (see OOD), but a
  large catalogue shift still favours re-prompting a bigger model.
- **Multi-turn / tool-result reasoning:** out of scope here — single-turn only.
- **Tiny budgets of examples (<200):** few-shot prompting a larger model likely wins.

## 8. Deployment

- **Adapter size:** _fill_ MB (vs ~6 GB for the 4-bit base) — swappable without reloading the base.
- **Serving:** FastAPI (`src/ftpipe/serve.py`), `/compare` returns base and fine-tuned side by
  side; `docker/Dockerfile.serve` builds a CPU image.
- **Latency** (`/benchmark`, `_fill_` hardware): p50 `_fill_` ms · p95 `_fill_` ms.

---

**CV line:**
> Fine-tuned Llama-3.2-3B (LoRA/QLoRA, PEFT + TRL) for agent tool-use: +`_fill_` tool-name F1,
> hallucinated-call rate → `_fill_`%, generalizes to unseen APIs; reproducible pipeline with
> hyperparameter sweep, MLflow tracking, ID/OOD + LLM-judge evaluation, forgetting analysis, and a
> Dockerized inference API.
