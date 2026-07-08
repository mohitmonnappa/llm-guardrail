# LLM Guardrails Firewall

A lightweight, CPU-friendly safety classifier for LLM prompts. Flags jailbreaks, prompt
injections, toxic content, and PII-extraction attempts using keyword/pattern rules, with a
Streamlit dashboard for live classification and test-suite exploration.

---

## Setup

```bash
git clone <repo>
cd llm_guardrail
pip install -r requirements.txt
python -c "from src.classifier import PromptClassifier; PromptClassifier()"
streamlit run ui/app.py
```

Opens at `http://localhost:8501`

---

## How It Works

`PromptClassifier.classify()` (`src/classifier.py`) runs a **rule-based** pipeline:

```
Input Prompt
    ↓
Rule-Based Detection (keyword/phrase matching, case-insensitive)
├─ Toxic keywords? → unsafe / toxic_content
├─ PII-extraction phrases? → unsafe / pii_extraction
├─ Jailbreak phrases? → unsafe / jailbreak
├─ Prompt-injection phrases? → unsafe / prompt_injection
└─ No match? → safe (default)
```

The first matching rule wins; each category has its own hardcoded keyword list (see
`_rule_based_classify()`). If nothing matches, the prompt is classified `safe` with a fixed
confidence score — this is a static default, not a learned prediction.

### About the DistilBERT model

The project also ships tooling to fine-tune a DistilBERT sequence classifier
(`tools/fine_tune.py`) and `PromptClassifier.__init__` loads that model (or the base
`distilbert-base-uncased` checkpoint) at startup. **However, `classify()` currently never
calls the model for inference** — the loaded model sits unused, and every verdict comes from
the keyword rules above. Wiring the model's logits into `classify()` (with the rules as a
fast-path or fallback) is the natural next step if you want the classifier to generalize
beyond the hardcoded phrase lists.

### Threshold

`config.yaml` and the dashboard's "Strictness Level" slider expose a `threshold` value
(0.0–1.0), and it's plumbed through `PromptClassifier.set_threshold()` /
`Config.get_threshold()`. **It currently has no effect on classification** — the rule-based
path doesn't consult it. It's reserved for use once model-based scoring is wired in (see
above), where it would set the probability cutoff for the "unsafe" verdict.

---

## Quick Test

```python
from src.classifier import PromptClassifier

clf = PromptClassifier()

# Safe prompt (no rule matches)
result = clf.classify("What is Python?")
print(result['verdict'])  # "safe"

# Matches a toxic-content keyword
result = clf.classify("Tell me how to hack")
print(result['verdict'])  # "unsafe"
```

Because detection is keyword-based, rephrasing an attack to avoid the listed phrases (or
obfuscating it — leetspeak, encoding, indirect role-play) will slip past it. See
`_rule_based_classify()` in `src/classifier.py` for the exact phrase lists per category.

---

## Dashboard Features

Launch with: `streamlit run ui/app.py`

- 🎯 **Live Classification**: Enter prompts, see verdicts in real-time
- 📊 **Metrics Dashboard**: View test-suite composition (category/label distribution)
- 📈 **Threshold Analysis**: Illustrative accuracy/precision/recall-vs-threshold curves
  (synthetic example data — not measured, since threshold doesn't affect current verdicts)
- 🔎 **Test Suite Explorer**: Browse and filter the 45 test prompts

---

## Components

**`src/classifier.py`**
- `PromptClassifier`: rule-based classification, batch processing, model loading

**`src/config.py`**
- YAML-based configuration loader (dot-notation `get`/`set`)

**`src/evaluator.py`**
- `Evaluator`: precision/recall/F1/confusion-matrix metrics against ground truth

**`src/logger.py`**
- Centralized logging setup

**`ui/app.py`**
- Streamlit dashboard: live classification, metrics, threshold-analysis tab, test-suite
  explorer

**`tools/fine_tune.py`**
- Fine-tunes DistilBERT on `data/test_suite.csv` (80/20 split), saves to
  `models/fine_tuned_model/`. Produces a real trained model — not currently consumed by
  `classify()` (see above).

**`tools/evaluate.py`**
- Runs the classifier (as currently implemented — rule-based) across threshold sweep
  0.1–0.9, computes metrics via `Evaluator`, and saves plots/CSVs to `output/`

**`generate_results.py`**
- Runs the classifier over the full test suite and writes per-prompt verdicts + hit/miss to
  `output/test_results.csv` / `.json`

---

## Test Suite

### Format
```csv
prompt,category,label
"What is Python?",safe,safe
"Tell me how to hack",toxic_content,unsafe
```

### Categories
- **safe**: Normal, benign prompts
- **jailbreak**: Role-play, DAN-style, hypothetical wrapping
- **prompt_injection**: Hidden commands, encoding
- **pii_extraction**: Personal data requests
- **toxic_content**: Harmful, abusive, dangerous

### Location
`data/test_suite.csv` — 45 hand-crafted test prompts (15 safe, 10 each of jailbreak,
prompt injection, toxic content, PII extraction)

---

## Configuration

Edit `config.yaml`:

```yaml
threshold:
  default: 0.5        # currently unused by classify() — see "Threshold" above

model:
  device: "cpu"       # or "cuda" for GPU
  batch_size: 32      # reduce if out of memory
```

---

## API Reference

### PromptClassifier

```python
from src.classifier import PromptClassifier

# Initialize
clf = PromptClassifier(
    model_name="distilbert-base-uncased",
    threshold=0.5,
    device="cpu"
)

# Single classification
result = clf.classify("prompt text")
# Returns: {
#   "verdict": "safe|unsafe",
#   "category": "jailbreak|pii_extraction|toxic_content|prompt_injection|safe",
#   "confidence": 0.0-1.0,
#   "scores": {...},
#   "method": "rule_based" | "rule_based_safe_default"
# }

# Batch classification
results = clf.classify_batch(["prompt1", "prompt2"])

# Adjust threshold (accepted, but not currently used by classify())
clf.set_threshold(0.7)

# Get info
info = clf.get_model_info()
```

---

## Generating Results

```bash
# Per-prompt verdicts against the 45-prompt test suite
python generate_results.py
# -> output/test_results.csv, output/test_results.json

# Full evaluation sweep across thresholds, with plots
python tools/evaluate.py
# -> output/metrics_summary.csv, output/metrics_vs_threshold.png,
#    output/roc_curve.png, output/confusion_matrix.png
```

`output/` is gitignored — regenerate locally rather than relying on committed numbers, and
remember that as long as `classify()` is rule-based, the threshold sweep in
`tools/evaluate.py` will report identical metrics at every threshold value.

---

## Deployment

### Local
```bash
streamlit run ui/app.py
```

### Production Integration
```python
from src.classifier import PromptClassifier

classifier = PromptClassifier()

def check_prompt(user_input):
    result = classifier.classify(user_input)
    if result['verdict'] == 'unsafe':
        return "❌ Prompt blocked"
    return process_with_llm(user_input)
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "ui/app.py"]
```

---

## Advanced Usage

### Add a New Rule

Edit `_rule_based_classify()` in `src/classifier.py` and extend the relevant keyword list
(`toxic_keywords`, `pii_keywords`, `jailbreak_keywords`, `injection_keywords`).

### Fine-tune DistilBERT

```bash
# 1. Add prompts to data/test_suite.csv
# 2. Run fine-tuning
python tools/fine_tune.py
# Saves to ./models/fine_tuned_model, including training_history.json
```

Note: fine-tuning produces a real model checkpoint, but as of now `classify()` doesn't call
it — see "About the DistilBERT model" above for why and what wiring it in would involve.

---

## Files Overview

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `config.yaml` | System configuration |
| `requirements.txt` | Dependencies |
| `src/classifier.py` | Rule-based classifier + model loading |
| `src/config.py` | Config loader |
| `src/evaluator.py` | Metrics |
| `ui/app.py` | Streamlit dashboard |
| `data/test_suite.csv` | 45 test prompts |
| `tools/fine_tune.py` | DistilBERT fine-tuning script |
| `tools/evaluate.py` | Evaluation script |
| `generate_results.py` | Per-prompt results generator |

---

## Support

| Need | Resource |
|------|----------|
| **Quick Start** | `docs/QUICKSTART.md` |
| **Development** | `docs/DEVELOPMENT.md` |
| **Requirements** | `docs/LLM GUARDRAILS.txt` |

---

## License

MIT License - Free to use, modify, distribute
