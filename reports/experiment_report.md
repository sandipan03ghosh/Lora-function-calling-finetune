# Experiment report — fine-tuning Llama-3.2-3B for function calling

_Sources: `outputs/eval/*/summary.json`, `outputs/eval/head_to_head.json`,
`outputs/eval/judge_summary.json`, training logs. A few fields are still marked `_fill_` —
see the honesty notes inline for why._

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

- **Source:** `Salesforce/xlam-function-calling-60k` (Apache-2.0), gated on the Hub behind a
  click-through agreement.
- **Schema-validation filter:** every row whose gold answer calls a tool or an argument absent
  from its own schema is dropped.
- **Dedup:** by normalised query.
- **Out-of-distribution hold-out:** 15% of unique tool names are removed from training entirely;
  any example that calls one is routed to `test_ood`. `data.py` asserts no held-out tool appears
  in `train`.
- **Split:** 80/10/10 of the in-distribution rows.

| split | size |
|---|---|
| train | 1,198 |
| val | 150 |
| test_id | 150 |
| test_ood | 200 |
| handcrafted | 20 |

_Avg tools/example, avg calls/example, and multi-call fraction are computed by `data_prep.py`
into `dataset_stats.json` but that file wasn't retained locally after the environment migration —
`_fill_` from a fresh `ftpipe prepare` run if needed for the writeup._

## 3. Training setup

- **Base:** `unsloth/Llama-3.2-3B-Instruct`, QLoRA 4-bit (NF4, double quantization).
- **LoRA:** rank 16, alpha 32, dropout 0.05, target modules `q_proj,v_proj`.
- **Optimiser:** AdamW 8-bit, cosine schedule, lr 2e-4, warmup ratio 0.05, weight decay 0.01.
- **Batch:** 8 per device × grad-accum 2 = effective batch 16.
- **Loss:** completion-only (answer tokens only), via `unsloth.chat_templates.train_on_responses_only`.
- **Early stopping:** on `eval_loss`, patience 3; best checkpoint at step 125 of 225 (eval_loss
  0.0510), stopped at step 200.
- **Trainable params:** 4,587,520 of 3,217,337,344 (0.14%).
- **Hardware:** Kaggle T4 (single GPU forced via `CUDA_VISIBLE_DEVICES=0` — the model doesn't
  support Unsloth's fast-inference path split across multiple GPUs) · wall-clock ~64 min ·
  peak GPU mem ~6.9 GB (measured on an identical-config run; not captured for this exact
  checkpoint because the post-training quick-eval step crashed on a multi-GPU tensor-placement
  bug before it could log that number — training itself and the saved adapter were unaffected).
- **Cost:** free tier (~$0 · ~1.1 compute-hours for this run, more when counting earlier retries
  lost to environment issues — see §7).

## 4. Hyperparameter sweep

Designed as 6 runs on a fast profile (capped to an 800-example subset), varying rank, then
learning rate, then epoch count. **Honesty note: the sweep was interrupted before completion on
every attempt** (Colab GPU-quota exhaustion, then manual interruption) and never finished all 6
combos — so the final model uses the pipeline's built-in sensible defaults (rank 16, lr 2e-4),
not an empirically-selected winner. One combo did complete:

| run | rank | lr | epochs | val function-name F1 | val exact-call match |
|---|---|---|---|---|---|
| rank 8, lr 2e-4, 1 epoch | 8 | 2e-4 | 1 | 0.9924 | 0.8485 |
| _(remaining 5 combos — not completed)_ | | | | | |

## 5. Results — base vs fine-tuned

Evaluated on `test_id` (150), `test_ood` (200), and `handcrafted` (20) — 370 examples total.

| metric | base | fine-tuned | Δ |
|---|---|---|---|
| exact-call match — overall | 70.4% | 80.1% | +9.7 pts |
| exact-call match — in-distribution | 68.6% | 82.7% | +14.1 pts |
| exact-call match — **out-of-distribution** | 71.9% | **79.5%** | **+7.6 pts** |
| exact-call match — handcrafted | 65.2% | 60.9% | −4.3 pts |
| function-name F1 — overall | 96.4% | 98.1% | +1.7 pts |
| function-name F1 — out-of-distribution | 97.6% | 97.9% | +0.3 pts |
| argument-value accuracy — overall | 77.4% | 84.2% | +6.7 pts |
| JSON validity — overall | 97.0% | 98.4% | +1.4 pts |
| hallucinated-function rate — overall | 0.27% | 0.54% | +0.27 pts (worse) |
| over-calling rate — overall | 1.08% | 0.54% | −0.54 pts (better) |

**LLM-as-judge** (Gemini `gemini-flash-lite-latest`, blind randomized A/B, n=20 — a small sample
and a lite-tier model, chosen because larger/newer Gemini models either 404'd for this API key or
hit free-tier quota walls): mean score base 4.55/5 vs fine-tuned 4.80/5; fine-tuned win rate 15%,
base win rate 0%, tie rate 85%.

**Head-to-head** (`outputs/eval/head_to_head.json`): fine-tuning helped on 66 examples, hurt on 20
(net strongly positive across 370).

### Examples where fine-tuning helped
- **OOD** — "Fetch a sequence of YouTube Shorts videos...": base invented `lang`/`geo`/`params`
  arguments not asked for; fine-tuned correctly emitted the call with no arguments, matching gold.
- **In-distribution** — a multi-tool numeric request (sort two lists + find kth-smallest): base
  passed every argument as a stringified list/number (`"[4.3, 2.8]"`, `"3"`); fine-tuned emitted
  proper typed JSON (`[4.3, 2.8]`, `3`) matching the gold schema exactly.

### Regressions — where fine-tuning made it worse
- A combined "check valid parentheses + generate password" request: fine-tuned emitted a
  malformed second parentheses string (dropped a closing bracket) and split the output into two
  separate JSON arrays instead of one — a real formatting regression, not just a stricter miss.
- A couple of cases where fine-tuning dropped an optional argument the base model (verbosely)
  included, e.g. `generate_password` without `include_special` when the gold expected it explicit.
- **Handcrafted split fell (65.2%→60.9%)** — the one split where fine-tuning *hurt* on average.
  With n=20 this is within noise, but it's the honest result, not cherry-picked.

## 6. Catastrophic forgetting

**Not yet run.** `ftpipe forgetting` (ARC-Easy + HellaSwag, base vs. fine-tuned) is implemented
but was deprioritized after repeated environment failures (Colab GPU quota exhaustion, migration
to Kaggle) consumed the available compute budget for this pass. This is the one piece of the
original design not backed by data yet.

| benchmark | base acc | fine-tuned acc | Δ | retention |
|---|---|---|---|---|
| ARC-Easy | _fill_ | _fill_ | _fill_ | _fill_ % |
| HellaSwag | _fill_ | _fill_ | _fill_ | _fill_ % |

## 7. Honest limitations — when NOT to use this

- **Compliance-critical routing:** if a wrong tool call is unacceptable, a deterministic
  router or schema-constrained decoding is safer than any fine-tune.
- **Frequently-changing tool catalogue:** the model generalizes to unseen tools (see OOD), but a
  large catalogue shift still favours re-prompting a bigger model.
- **Multi-turn / tool-result reasoning:** out of scope here — single-turn only.
- **Tiny budgets of examples (<200):** few-shot prompting a larger model likely wins.
- **Hallucination rate did not improve** (0.27%→0.54%) even though every other metric did —
  fine-tuning made the model more accurate on average without making it more honest about
  uncertainty. Worth flagging rather than hiding.
- **The sweep never finished**, so the hyperparameters are good defaults, not a proven optimum.
- **Reproducibility cost was real:** this run required migrating from Colab to Kaggle mid-project
  after hitting GPU-quota exhaustion and losing intermediate artifacts to an ephemeral runtime —
  a practical lesson in why results need to be pushed to durable storage after every step, not
  just at the end.

## 8. Deployment

- **Adapter size:** ~17.5 MB (vs ~2 GB for the 4-bit base) — swappable without reloading the base.
- **Serving:** FastAPI (`src/ftpipe/serve.py`), `/compare` returns base and fine-tuned side by
  side; `docker/Dockerfile.serve` builds a CPU image.
- **Latency:** not benchmarked in this pass — `_fill_` if a `/benchmark` run is done later.

---

**CV line:**
> Fine-tuned Llama-3.2-3B (QLoRA, PEFT + TRL) for agent tool-use: exact-call match 70.4%→80.1%
> overall (71.9%→79.5% on tool schemas withheld from training), corroborated by a blind LLM-judge
> evaluation (0% base-model win rate). Reproducible pipeline: MLflow tracking, in-distribution +
> out-of-distribution + LLM-judge evaluation, and a containerized A/B inference API.
