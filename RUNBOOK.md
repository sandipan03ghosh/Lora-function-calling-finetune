# Runbook — from empty repo to portfolio piece

No local GPU needed. Steps 1–2 are one-time; step 3 is the run; steps 4–6 are the write-up.

## 1. Put the code on GitHub

You don't need git installed locally — use the web UI.

1. github.com → **New repository** → name it `fine-tuning-pipeline` → **Create**.
2. On the empty repo page → **uploading an existing file** → drag in the whole project folder →
   **Commit changes**.

(Or, if you have git: `git init && git add . && git commit -m "init" && git remote add origin <url> && git push -u origin main`.)

## 2. Create the secrets

**GitHub token** (lets the notebook push results back):
- github.com → Settings → Developer settings → **Fine-grained tokens** → Generate new token
- Repository access: only `fine-tuning-pipeline` · Permissions: **Contents → Read and write**
- Copy the token.

**In Colab** (key icon, left sidebar) add:
| Name | Value |
|---|---|
| `GH_USER` | your GitHub username |
| `GH_TOKEN` | the token above |
| `ANTHROPIC_API_KEY` | *(optional)* enables the LLM-judge step |

## 3. Run the pipeline

1. Open `notebooks/colab_pipeline.ipynb` in Colab (`File → Open notebook → GitHub → your repo`).
2. `Runtime → Change runtime type → T4 GPU`.
3. Edit the first code cell: set `REPO_NAME = 'fine-tuning-pipeline'`.
4. `Runtime → Run all`. Expect ~1–1.5 h:
   - dataset build — ~3 min
   - sweep (6 fast runs) — ~30–40 min
   - final training — ~20–30 min
   - eval base + fine-tuned + forgetting — ~15 min
5. **After the sweep**, look at the printed `reports/sweep_results.md`, edit `configs/train.yaml`
   (`lora_r`, `lr`, `epochs`) to the winning row, and re-run the "Train the final model" cell
   onward. (Or accept the defaults for a first pass.)
6. The last cell pushes `reports/`, `outputs/eval/`, and `outputs/final/best/` (the adapter) back
   to GitHub, and offers `mlruns.zip` for download.

If the Colab session dies: the adapter and eval JSON are already on GitHub if you got past the
push cell; otherwise re-run from "Train the final model".

## 4. Fill in the numbers

Edit directly on github.com (pencil icon):

- **`README.md`** — the headline `X`/`Y` placeholders and the results table, from
  `outputs/eval/base/summary.json` and `outputs/eval/finetuned/summary.json`
  (`overall` and `by_split`).
- **`reports/experiment_report.md`** — every `_fill_` marker, from `summary.json`,
  `outputs/eval/head_to_head.json`, `outputs/eval/forgetting.json`, `reports/sweep_results.md`,
  and (if run) `outputs/eval/judge_summary.json`.

## 5. Record the demo (< 4 min)

1. A tool request where the **base model picks the wrong tool or emits invalid JSON**, then the
   **fine-tuned model gets it right** — via the notebook's `/compare` cell or the tunnelled API.
2. The **MLflow UI** — training loss curves + the sweep runs.
3. The **head-to-head** table and one out-of-distribution example it got right.
4. Say the headline sentence with your real numbers.

## 6. Polish

- Add the demo video link + the results table to the top of `README.md`.
- Commit a screenshot of the MLflow sweep view to `reports/`.
- On your CV: the one-line version from the bottom of `reports/experiment_report.md`.
