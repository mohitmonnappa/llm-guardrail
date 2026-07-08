# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CPU-friendly safety classifier for LLM prompts — detects jailbreaks, prompt injections, toxic content, and PII extraction attempts. Ships as a Python library (`src/classifier.py`) plus a Streamlit dashboard (`ui/app.py`).

## Commands

```bash
# Setup
pip install -r requirements.txt

# Run the dashboard
streamlit run ui/app.py          # http://localhost:8501

# Fine-tune DistilBERT on data/test_suite.csv (writes models/fine_tuned_model/)
python tools/fine_tune.py

# Run full evaluation across thresholds 0.1-0.9 (writes output/*.csv, output/*.png)
python tools/evaluate.py

# Generate per-prompt verdict report (writes output/test_results.csv, .json)
python generate_results.py

# Quick manual smoke test
python -c "from src.classifier import PromptClassifier; c = PromptClassifier(); print(c.classify('Tell me how to hack'))"
```

There is no test suite (pytest is in requirements.txt but no tests exist yet) and no lint config beyond `black`/`flake8` being listed as dependencies — no invocation scripts are defined for either.

## Architecture — read this before touching `classify()`

**The classifier's decision path is rule-based only; the loaded DistilBERT model is not used for inference.**

`PromptClassifier.__init__` (`src/classifier.py`) loads a tokenizer and `AutoModelForSequenceClassification` (base or fine-tuned from `models/fine_tuned_model/`) and moves it to device — but `classify()` never calls `self.model(...)`. The actual logic is:

1. `_rule_based_classify()` — lowercases the input and checks four hardcoded keyword lists (toxic, PII, jailbreak, prompt_injection). First match wins, returns a hardcoded confidence (0.87–0.95) for that category.
2. If no keyword matches, `classify()` falls through to a hardcoded "safe" response (confidence 0.92) — this is *not* a model prediction, just a fixed dict literal (see the comment "prevents false positives... default to SAFE (safer approach than relying on untrained model)").

So `threshold` (config.yaml, `Config.get_threshold()`, `PromptClassifier.set_threshold()`) is plumbed through the whole stack (evaluator sweeps 9 threshold values, UI has a strictness slider) but **is never actually read inside `classify()` or `_rule_based_classify()`** — changing it has no effect on verdicts today. Any threshold-driven behavior change would need to be wired into `_rule_based_classify()`.

`tools/fine_tune.py` does perform real training (standard PyTorch loop: `AdamW`, linear warmup schedule, saves to `models/fine_tuned_model/` with `training_history.json`) — the model artifact is real and gets loaded at runtime — it's just not consulted for predictions. If you're asked to make the classifier "actually use the model," the fix point is `PromptClassifier.classify()`: add a branch that runs `self.model(**tokenized_input)`, softmaxes logits into the `scores` dict, and only falls back to rule-based/safe-default when appropriate — not "add a new model," since one is already loaded and sitting idle.

### Data flow

```
data/test_suite.csv (45 hand-labeled prompts: prompt, category, label)
  → tools/fine_tune.py (80/20 split, trains DistilBERT, saves models/fine_tuned_model/)
  → tools/evaluate.py / generate_results.py (loads classifier, runs classify() over test suite,
    computes sklearn metrics via src/evaluator.py, writes output/)
  → ui/app.py (Streamlit dashboard: live classification, metrics, threshold-analysis tab, test-suite explorer)
```

`src/config.py` (`Config` class) is a thin YAML wrapper with dot-notation `get`/`set` (e.g. `config.get("threshold.default")`) and hardcoded defaults mirroring `config.yaml`; `src/logger.py` sets up a `"guardrails"` logger writing to `output/guardrails.log`. Neither is currently imported by `src/classifier.py` (classifier takes its params directly via constructor args, not through `Config`).

### Known documentation vs. code mismatches

`README.md` and `docs/DEVELOPMENT.md` present accuracy/latency numbers (84% accuracy, 180ms P95, per-category detection rates) as if measured from the DistilBERT model. Since inference is rule-based only, any numbers in those docs or in `output/` reflect keyword-matching performance on the 45-prompt suite, not learned-model performance. Be skeptical of these figures if asked to update or verify them — regenerate via `tools/evaluate.py` / `generate_results.py` rather than trusting the docs.

### Categories

Fixed 5-way taxonomy used everywhere (`CATEGORIES` constant duplicated in `src/classifier.py`, `src/evaluator.py`, and `tools/fine_tune.py`'s `PromptDataset`): `safe`, `jailbreak`, `pii_extraction`, `toxic_content`, `prompt_injection`. Binary safe/unsafe metrics collapse the four attack categories into `unsafe`.

### Paths that matter

- `data/test_suite.csv` — the only labeled dataset; both training and evaluation draw from this same 45-row file (no separate held-out set beyond the fine-tuner's random 80/20 split, reseeded each run via `np.random.seed(42)`).
- `models/fine_tuned_model/` — gitignored (see `.gitignore`); regenerate with `tools/fine_tune.py` if missing. `ui/app.py` and `tools/evaluate.py` both silently fall back to the untrained base model if this directory doesn't exist.
- `output/` — all generated reports/plots/logs land here; gitignored contents (`*.log`) but the directory itself is tracked.
