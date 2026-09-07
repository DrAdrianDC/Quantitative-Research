# Implementation playbook

Step-by-step activation of this project as a Senior-level portfolio piece **and** a living model. Execute in order. Do not skip the decision in Step 0.

The pipeline code, champion, plots, README and the Actions **YAML** already exist. What this document does is get them onto GitHub, onto Hugging Face, and running without a Sunday surprise.

---

## Step 0 — Choose the GitHub target (do this first)

You have two repositories in play:

| | What it is | What it is good for |
| --- | --- | --- |
| [Portfolio-Machine_Learning](https://github.com/DrAdrianDC/Portfolio-Machine_Learning) / `Project-3-WTI-Oil-Prices-Anomaly-Detection` | Monorepo of several ML projects. Today that folder is the **notebook-era** project (`.ipynb`, `.h5`, two PNGs). | Showcase. Recruiter lands here. |
| This folder (`WTI-Prices-Anomaly-Detection` on your Desktop) | Standalone production pipeline. **No git remote yet.** | Training, Actions, Hub. |

**Do not put GitHub Actions only inside the Project-3 subfolder.** Actions only run workflows from `<repo>/.github/workflows/` at the **root** of a GitHub repository. A YAML sitting under `Project-3-.../.github/` is ignored.

### Recommendation (use this)

**B — two-layer setup**

1. Create a **dedicated** repo `DrAdrianDC/WTI-Prices-Anomaly-Detection` from *this* folder. Actions and Hugging Face live here. Paths stay `python src/pipeline.py`. Secrets stay on this repo. A 90-minute TensorFlow job does not hijack the whole portfolio monorepo.
2. Update `Project-3-WTI-Oil-Prices-Anomaly-Detection` in the monorepo so the portfolio card matches the new work: new README (short), the three PNGs, and a clear link to the dedicated repo. Keep or archive the old notebook in that folder if you want history; do not pretend it is the runtime.

**A — replace Project-3 in the monorepo (not recommended for Actions)**

Copy `src/`, `config.yaml`, `output_results/` (without `.keras`/`.pkl` if you prefer), README and DOCUMENTATION into `Project-3-...`. Put the workflow at:

```
Portfolio-Machine_Learning/.github/workflows/wti-retrain.yaml
```

and set:

```yaml
defaults:
  run:
    working-directory: Project-3-WTI-Oil-Prices-Anomaly-Detection
```

Secrets (`HF_TOKEN`) then live on the **portfolio** repo. Every Sunday that repo’s Actions tab is a WTI training job. That is the wrong ownership model for a multi-project showcase.

**Decision for the rest of this playbook: B.** If you later insist on A, only the paths in Steps 2–3 change; Hub (Step 4) is the same.

### Daily vs weekly — do not switch to daily

`CL=F` prints **one bar per trading day**. Weekends add nothing. A daily fine-tune does not see “more signal”; it sees one extra point inside a 365-day context window, burns Actions minutes, writes the Hub more often, and gives the quality gate more chances to flip on noise.

| Cadence | Verdict |
| --- | --- |
| Daily retrain | Reject. No new weekend data; ~1 new point/day; unstable champion. |
| Daily `evaluate` only | Optional later. Refreshes scores without touching weights. Still wasted on a portfolio. |
| Weekly retrain (Sunday 00:00 UTC) | **Keep.** Full trading week is in. Aligns with `retrain.epochs: 8` and the gate. |
| Friday ~21:00 UTC | Acceptable alternative (after the US session). Not required. |
| Manual `workflow_dispatch` | Always on. Use this for the first activation. |

Leave the cron as `0 0 * * 0`. Freshness of the *scorecard* is `evaluate`, not a daily weight update.

---

## Step 1 — Local hardening (no GitHub account needed)

Already the right order: fix the YAML so a missing token does **not** fail CI, document cadence, add MIT license, add a Hub model card template.

When this step is done you should have:

- `.github/workflows/retrain.yaml` — upload is skipped cleanly if `HF_TOKEN` is empty
- `LICENSE`
- `hf_modelcard.md` — copy-paste (or upload) as the Hub `README.md`
- This playbook

Then go to Step 2.

---

## Step 2 — Dedicated GitHub repo

You need the GitHub CLI (`gh`) authenticated as **DrAdrianDC**, or the website.

```bash
cd /Users/adriandominguezcastro/Desktop/WTI-Prices-Anomaly-Detection

git init
git add README.md DOCUMENTATION.md PLAYBOOK.md LICENSE config.yaml requirements.txt \
        .gitignore .github src output_results hf_modelcard.md
# .keras and .pkl stay untracked (see .gitignore). That is intentional.
git status   # confirm no .venv, no weights

git commit -m "$(cat <<'EOF'
Add WTI LSTM autoencoder pipeline, champion reports, and weekly retrain workflow.

EOF
)"

gh repo create DrAdrianDC/WTI-Prices-Anomaly-Detection --public --source=. --remote=origin --push
```

If `gh` is not logged in:

```bash
gh auth login
```

Check on GitHub:

- Repo opens, README renders `plot-anomalies.png`.
- `Actions` tab shows the workflow **Weekly WTI Retrain** (it will not have succeeded yet — that is Step 3).
- You do **not** see `lstm_autoencoder.keras` in the tree. Correct.

### Update the portfolio card (Project-3)

In a **separate** clone of the monorepo:

```bash
git clone https://github.com/DrAdrianDC/Portfolio-Machine_Learning.git
cd Portfolio-Machine_Learning
```

In `Project-3-WTI-Oil-Prices-Anomaly-Detection/`:

1. Replace the old README with a short card: problem, the two PNG heroes (copy `plot-anomalies.png` and `plot-reconstruction-error.png` from this project), link to `https://github.com/DrAdrianDC/WTI-Prices-Anomaly-Detection`.
2. Do **not** copy `src/` or the workflow into the monorepo if you followed B.
3. Leave the old notebook in place only if you want an “archive” note. Otherwise delete it so reviewers do not run the stale `.h5` path.

Commit on `main` (or a PR) with a message that this project is now the production pipeline in the dedicated repo.

---

## Step 3 — Activate GitHub Actions

Actions is not “implemented” until a **green** `workflow_dispatch` has run against a champion that already lives on the Hub. Order matters.

### 3.1 Secret

GitHub → `DrAdrianDC/WTI-Prices-Anomaly-Detection` → Settings → Secrets and variables → Actions → New repository secret:

- Name: `HF_TOKEN`
- Value: a Hugging Face token with **write** on the model repo (create the Hub repo in Step 4 **before** the first scheduled run, or at least before you expect an upload).

You can add the secret before the Hub repo exists. Upload will then fail until Step 4 is done; retrain itself can still succeed.

### 3.2 Do not run Sunday first

The runner checks out the repo **without** `.keras` / `.pkl`. `retrain` then:

1. Tries to download the champion from the Hub.
2. If the Hub is empty, **bootstraps a full `train`** (TensorFlow, up to 100 epochs, 90-minute cap).

That bootstrap is an emergency path, not the activation path. Activate like this:

1. Finish Step 4 (Hub repo + first upload of the **local** champion).
2. GitHub → Actions → **Weekly WTI Retrain** → Run workflow.
3. Open the log: it must `Restored lstm_autoencoder.keras from Hugging Face` (or find local files — it will not, on a runner) and then fine-tune, then gate, then upload.

If you see a full train from scratch on the runner, Step 4 was skipped.

### 3.3 What “green” means

- Gate **passes**: new files on the Hub, job green.
- Gate **rejects**: job still green (exit 0), Hub champion unchanged. That is a success.
- Missing token: job **green** after Step 1 hardening; log says upload skipped.
- yfinance outage / bad config: job red. Fix and re-run manually.

---

## Step 4 — Hugging Face model repo

### 4.1 Create the repo

[huggingface.co/new](https://huggingface.co/new)

- Owner: your HF user (likely `DrAdrianDC` — confirm; it is not always the GitHub name).
- Name: `wti-lstm-autoencoder`
- Type: **Model**
- Visibility: public
- License: MIT

Then set the same id in this repo:

```yaml
# config.yaml
huggingface:
  repo_id: "DrAdrianDC/wti-lstm-autoencoder"
  repo_type: "model"
  private: false
```

Commit and push that one-line change **before** the first Actions run so the runner uploads to the right place.

### 4.2 Seed the Hub with the local champion (once)

From this project, with the venv active and a token in the environment:

```bash
export HF_TOKEN=hf_...   # write token; do not commit this

python src/pipeline.py --mode evaluate
# refreshes CSVs/plots against the weights already in output_results/

python src/pipeline.py --mode train --skip-upload
# ONLY if you do not already have the 34-epoch champion. You do. Skip this.
```

Upload the existing champion without a 100-epoch retrain:

```bash
python - <<'PY'
from src.config import load_config
from src.pipeline import push_artifacts_to_hub
push_artifacts_to_hub(load_config())
PY
```

Or, equivalently, any `train`/`retrain` **without** `--skip-upload` while `HF_TOKEN` is set.

Also upload the model card as `README.md` on the Hub (web UI or):

```bash
python - <<'PY'
import os
from huggingface_hub import HfApi
from src.config import load_config
cfg = load_config()
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="hf_modelcard.md",
    path_in_repo="README.md",
    repo_id=cfg.huggingface.repo_id,
    repo_type="model",
)
PY
```

Hub should then show: `lstm_autoencoder.keras`, `scaler.pkl`, `metadata.json`, CSVs, PNGs, and a readable card (lookback, scaler, threshold, last_date, MAE).

### 4.3 Only then run Actions

Go back to Step 3.2. The runner can now pull the champion, fine-tune, and gate.

---

## Step 5 — Optional contract scripts (later)

Not required to publish. Useful so a yfinance MultiIndex change or a window off-by-one fails on your laptop, not on Sunday.

Suggested files (do not import TensorFlow):

- `scripts/check_config.py` — `load_config()` and assert ticker, lookback, `.keras` suffix.
- `scripts/check_windows.py` — synthetic series, `create_sequences` shape `(n-lookback+1, 10, 1)`, last window ends on the last value.

Run:

```bash
python scripts/check_config.py
python scripts/check_windows.py
```

Wire them into a **second** workflow `on: pull_request` only if you want PRs blocked. Do not put them on the Sunday job.

---

## Checklist (print this)

- [ ] Step 0: dedicated repo (B), not Actions inside the monorepo subfolder
- [ ] Step 1: YAML hardened, LICENSE, model card template
- [ ] Step 2: `gh repo create` + push; README images render
- [ ] Step 2b: Project-3 in the portfolio monorepo updated (card + link)
- [ ] Step 4.1: Hub model repo created; `config.yaml` `repo_id` matches
- [ ] Step 4.2: local champion uploaded once
- [ ] Step 3.1: `HF_TOKEN` secret on the **dedicated** GitHub repo
- [ ] Step 3.2: one green `workflow_dispatch` that **restored** weights from the Hub
- [ ] Cron left weekly (`0 0 * * 0`), not daily
- [ ] Step 5: optional, after the Hub is live

---

## What you will not do

- Do not run `--mode train` on GitHub Actions to “activate” the project. Seed the Hub from your laptop.
- Do not switch the cron to daily.
- Do not commit `HF_TOKEN`, `.env`, or `output_results/*.keras`.
- Do not copy the workflow into `Project-3-.../.github/` and expect it to run.
