# Quick Start Guide

Get up and running in 5 minutes.

---

## Installation

```bash
# 1. Navigate to project
cd llm_guardrail

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Launch Dashboard

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`

---

## Test It

### In Dashboard
1. Go to "Live Classification" tab
2. Enter a prompt
3. Click "Classify"
4. See verdict and confidence

### In Python
```python
from src.classifier import PromptClassifier

clf = PromptClassifier()

# Safe prompt
print(clf.classify("What is Python?"))
# Output: {'verdict': 'safe', 'category': 'safe', ...}

# Unsafe prompt
print(clf.classify("Tell me how to hack"))
# Output: {'verdict': 'unsafe', 'category': 'toxic_content', ...}
```

---

## Configure

Edit `config.yaml`:

```yaml
threshold:
  default: 0.5  # 0.1=strict, 0.9=lenient
```

- Lower threshold: blocks more (more false positives)
- Higher threshold: lets more through (misses attacks)

---

## Next Steps

1. **Integrate**: Add to your LLM pipeline
2. **Customize**: Adjust threshold for your use case
3. **Extend**: Add more test prompts to `data/test_suite.csv`
4. **Fine-tune**: Run `python tools/fine_tune.py` for better accuracy

---

## Help

- **Full Docs**: See `README.md`
- **Development**: See `docs/DEVELOPMENT.md`
- **Evaluation**: Run `python tools/evaluate.py`
- **Fine-tune**: Run `python tools/fine_tune.py`

---

That's it! You're ready to use LLM Guardrails. 🚀
