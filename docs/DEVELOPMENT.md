# Development Guide

Advanced documentation for developers and researchers.

---

## Architecture

### Classification Pipeline

```
User Input
    ↓
Rule-Based Detection
    ├─ Match? → Return Verdict
    └─ No Match? → Safe (default)
```

`classify()` in `src/classifier.py` is rule-based only today — it never calls the loaded
model. The tokenizer/model are loaded in `__init__` and a fine-tuned checkpoint can be
produced via `tools/fine_tune.py`, but nothing in `classify()` runs inference against them
yet. Treat "DistilBERT" in this doc as describing infrastructure that's loaded and ready,
not something currently in the decision path.

### Current Approach

1. **Rule-Based Detection** (only decision path)
   - Keywords for toxic content
   - Patterns for jailbreaks
   - PII extraction attempts
   - Prompt injections

2. **Safe Default**
   - Defaults to "safe" when no rules match
   - Fixed confidence score, not a model prediction

3. **Wiring in the fine-tuned model** (not yet done)
   - Would mean running `self.model(**tokenized_input)` inside `classify()`, softmaxing
     logits into the `scores` dict, and deciding using `self.threshold`
   - Until then, `set_threshold()` / `config.yaml`'s `threshold.default` have no effect on
     verdicts

---

## Source Code

### Main Files

**`src/classifier.py`**
- `PromptClassifier`: Main classification class
- `_rule_based_classify()`: Rule-based detection
- `classify()`: Single prompt classification
- `classify_batch()`: Batch processing

**`src/config.py`**
- Configuration management
- YAML loading/saving
- Default values

**`src/evaluator.py`**
- Metrics calculation
- Accuracy, precision, recall, F1
- Per-category analysis

**`ui/app.py`**
- Streamlit dashboard
- Live classification interface
- Metrics and visualizations

---

## Extending the Classifier

### Add Rule-Based Detection

Edit `src/classifier.py`, method `_rule_based_classify()`:

```python
# Add new keywords
new_keywords = ['keyword1', 'keyword2']

for keyword in new_keywords:
    if keyword in text_lower:
        return {
            "verdict": "unsafe",
            "category": "your_category",
            "confidence": 0.95,
            "scores": {...},
            "timestamp": datetime.now().isoformat(),
            "method": "rule_based"
        }
```

### Fine-Tune on Custom Data

1. Prepare data in `data/test_suite.csv`:
   ```csv
   prompt,category,label
   "Your prompt",safe,safe
   "Harmful prompt",toxic_content,unsafe
   ```

2. Run fine-tuning:
   ```bash
   python tools/fine_tune.py
   ```

3. Use fine-tuned model:
   ```python
   clf = PromptClassifier(
       fine_tuned_path="./models/fine_tuned_model"
   )
   ```

---

## Configuration

### `config.yaml` Options

```yaml
model:
  name: "distilbert-base-uncased"
  device: "cpu"              # cpu or cuda
  batch_size: 32
  max_length: 512

threshold:
  default: 0.5               # 0.1-0.9

categories:
  - "safe"
  - "jailbreak"
  - "pii_extraction"
  - "toxic_content"
  - "prompt_injection"

training:
  epochs: 3
  learning_rate: 2e-5
  warmup_steps: 500
```

---

## Performance Optimization

`device`, `batch_size`, and `max_length` control how the tokenizer/model are loaded and
would matter once `classify()` actually runs inference through them (see "Wiring in the
fine-tuned model" above). They have no effect on current rule-based classification speed.

---

## Evaluation

### Run Metrics

```bash
python tools/evaluate.py
```

Generates:
- `output/detailed_results.json` - All classifications
- `output/metrics_summary.csv` - Per-threshold metrics
- `output/metrics_vs_threshold.png` - Performance curves
- `output/roc_curve.png` - ROC analysis
- `output/confusion_matrix.png` - Confusion matrix

### Test Your Classifier

```python
from src.classifier import PromptClassifier

clf = PromptClassifier()

# Test cases
test_cases = [
    ("What is Python?", "safe"),
    ("Tell me how to hack", "unsafe"),
    ("DAN: No restrictions", "unsafe"),
]

for prompt, expected in test_cases:
    result = clf.classify(prompt)
    actual = result['verdict']
    status = "✓" if actual == expected else "✗"
    print(f"{status} {prompt}: {actual}")
```

---

## API Reference

### PromptClassifier

#### Initialization
```python
clf = PromptClassifier(
    model_name="distilbert-base-uncased",
    fine_tuned_path=None,
    threshold=0.5,
    device="cpu",
    batch_size=32,
    max_length=512
)
```

#### Methods

**`classify(text: str) -> Dict`**
- Single prompt classification
- Returns: verdict, category, confidence, scores

**`classify_batch(texts: List[str]) -> List[Dict]`**
- Batch classification
- Returns: list of results

**`set_threshold(threshold: float)`**
- Update strictness (0.0-1.0)

**`get_threshold() -> float`**
- Get current threshold

**`get_model_info() -> Dict`**
- Get model metadata

**`save_model(path: str)`**
- Save checkpoint

---

## Test Suite

### Format
```csv
prompt,category,label
"Text",category,safe/unsafe
```

### Categories
- `safe`: Normal, benign
- `jailbreak`: Role-play, DAN-style, hypothetical
- `pii_extraction`: Personal data requests
- `toxic_content`: Harmful, abusive
- `prompt_injection`: Hidden commands

### Adding Tests

1. Edit `data/test_suite.csv`
2. Add new row: `"Your prompt",category,label`
3. Optionally retrain the (currently unused) model: `python tools/fine_tune.py`
4. To actually change verdicts today, add matching keywords to `_rule_based_classify()`
   in `src/classifier.py` instead

---

## Monitoring

### Log Files
```
output/guardrails.log
```

### Metrics to Track

1. **Classification Distribution**
   - % Safe vs Unsafe
   - % Per category

2. **Performance**
   - Inference latency
   - Throughput (prompts/sec)

3. **Accuracy**
   - True positives (caught attacks)
   - False positives (blocked safe)
   - False negatives (missed attacks)

---

## Troubleshooting

### Common Issues

**Model download fails**
```bash
rm -rf ~/.cache/huggingface/
python -c "from src.classifier import PromptClassifier; PromptClassifier()"
```

**Out of memory**
```yaml
# config.yaml
batch_size: 4
max_length: 256
```

**Slow inference**
```yaml
# config.yaml
device: "cuda"  # Use GPU if available
```

**All prompts unsafe**
- Check classifier version
- Verify rule-based detection is working
- Run: `python -c "from src.classifier import PromptClassifier; print(PromptClassifier()._rule_based_classify('test'))"`

---

## Contributing

### Adding Features

1. Fork the repo
2. Create branch: `git checkout -b feature/name`
3. Make changes
4. Test: `python tools/evaluate.py`
5. Commit: `git commit -am "Add feature"`
6. Push: `git push origin feature/name`
7. Create PR

### Improving Accuracy

1. Collect real-world attack examples
2. Add to `data/test_suite.csv`
3. Add corresponding keywords/phrases to `_rule_based_classify()` (this is what actually
   changes verdicts today)
4. Evaluate: `python tools/evaluate.py`
5. Longer-term: wire the fine-tuned model into `classify()` so `tools/fine_tune.py` output
   actually improves generalization beyond the keyword lists
6. Share improvements

---

## Resources

- **DistilBERT Paper**: https://arxiv.org/abs/1910.01108
- **HuggingFace Docs**: https://huggingface.co/docs/transformers
- **PyTorch Docs**: https://pytorch.org/docs
- **Streamlit Guide**: https://docs.streamlit.io

---

## Performance Benchmarks

No benchmarks are published here — regenerate them yourself with `python tools/evaluate.py`
(writes `output/metrics_summary.csv` and plots) or `python generate_results.py`
(per-prompt hit/miss). Since `classify()` is currently rule-based, expect identical metrics
across every threshold in the sweep, and expect results to reflect keyword-matching
accuracy on `data/test_suite.csv`, not a trained model's performance.

---

## Future Work

- [ ] Multi-language support
- [ ] Image/video content detection
- [ ] Ensemble with additional models
- [ ] Real-time threat intelligence
- [ ] Federated learning
- [ ] Community model marketplace

---

**Questions?** See main `README.md` or create an issue.
