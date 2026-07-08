"""
Comprehensive Evaluation Script
================================
Run full evaluation and generate final report.

Usage: python scripts/evaluate.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier import PromptClassifier
from src.evaluator import Evaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Run complete evaluation."""

    logger.info("="*60)
    logger.info("LLM GUARDRAILS - COMPREHENSIVE EVALUATION")
    logger.info("="*60)

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Load test suite
    logger.info("\n[1/5] Loading test suite...")
    test_suite = pd.read_csv("data/test_suite.csv")
    logger.info(f"Loaded {len(test_suite)} prompts")

    # Load classifier
    logger.info("\n[2/5] Loading classifier...")
    fine_tuned_path = "./models/fine_tuned_model"
    if os.path.exists(fine_tuned_path):
        logger.info(f"Using fine-tuned model from {fine_tuned_path}")
        classifier = PromptClassifier(
            model_name="distilbert-base-uncased",
            fine_tuned_path=fine_tuned_path,
            threshold=0.5,
            device="cpu"
        )
    else:
        logger.warning("Fine-tuned model not found, using base model")
        classifier = PromptClassifier(
            model_name="distilbert-base-uncased",
            threshold=0.5,
            device="cpu"
        )

    # Run inference on all prompts
    logger.info("\n[3/5] Running inference on all prompts...")
    all_predictions = []
    all_results = []

    for idx, row in test_suite.iterrows():
        prompt = row['prompt']
        result = classifier.classify(prompt)
        result['prompt'] = prompt
        result['true_category'] = row['category']
        result['true_label'] = row['label']
        all_results.append(result)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Processed {idx + 1}/{len(test_suite)}")

    # Evaluate at multiple thresholds
    logger.info("\n[4/5] Evaluating at multiple thresholds...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = {}

    for threshold in thresholds:
        logger.info(f"  Evaluating at threshold {threshold}...")

        # Update classifier threshold
        classifier.set_threshold(threshold)

        # Re-classify all prompts
        predictions = []
        for result in all_results:
            pred = classifier.classify(result['prompt'])
            pred['true_category'] = result['true_category']
            pred['true_label'] = result['true_label']
            predictions.append(pred)

        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate(predictions, test_suite, threshold)
        threshold_results[threshold] = metrics

    # Save detailed results
    logger.info("\n[5/5] Saving results...")
    results_file = "output/detailed_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {results_file}")

    # Generate summary metrics
    summary_data = []
    for threshold, metrics in threshold_results.items():
        binary = metrics['metrics']['binary']
        summary_data.append({
            'threshold': threshold,
            'accuracy': binary['accuracy'],
            'precision': binary['precision'],
            'recall': binary['recall'],
            'f1_score': binary['f1_score'],
            'false_positive_rate': binary['false_positives'] / (
                binary['false_positives'] + binary['true_negatives']
            ) if (binary['false_positives'] + binary['true_negatives']) > 0 else 0,
            'true_positives': binary['true_positives'],
            'true_negatives': binary['true_negatives'],
            'false_positives': binary['false_positives'],
            'false_negatives': binary['false_negatives']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = "output/metrics_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")

    # Generate visualizations
    logger.info("Generating visualizations...")

    # 1. Accuracy vs Threshold
    fig, ax = plt.subplots()
    ax.plot(summary_df['threshold'], summary_df['accuracy'], marker='o', linewidth=2, label='Accuracy')
    ax.plot(summary_df['threshold'], summary_df['precision'], marker='s', linewidth=2, label='Precision')
    ax.plot(summary_df['threshold'], summary_df['recall'], marker='^', linewidth=2, label='Recall')
    ax.plot(summary_df['threshold'], summary_df['f1_score'], marker='d', linewidth=2, label='F1-Score')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/metrics_vs_threshold.png', dpi=100)
    logger.info("Saved: output/metrics_vs_threshold.png")
    plt.close()

    # 2. Recall vs False Positive Rate (ROC-like)
    fig, ax = plt.subplots()
    ax.plot(summary_df['false_positive_rate'], summary_df['recall'], marker='o', linewidth=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('Recall (True Positive Rate)')
    ax.set_title('ROC-like Curve: Recall vs False Positive Rate')
    ax.grid(True, alpha=0.3)
    for idx, row in summary_df.iterrows():
        ax.annotate(f"{row['threshold']:.1f}", (row['false_positive_rate'], row['recall']))
    plt.tight_layout()
    plt.savefig('output/roc_curve.png', dpi=100)
    logger.info("Saved: output/roc_curve.png")
    plt.close()

    # 3. Confusion Matrix (at best F1 threshold)
    best_f1_idx = summary_df['f1_score'].idxmax()
    best_f1_threshold = summary_df.loc[best_f1_idx, 'threshold']
    best_metrics = threshold_results[best_f1_threshold]['metrics']['binary']

    fig, ax = plt.subplots()
    confusion = np.array([
        [best_metrics['true_negatives'], best_metrics['false_positives']],
        [best_metrics['false_negatives'], best_metrics['true_positives']]
    ])
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Threshold={best_f1_threshold})')
    ax.set_xticklabels(['Safe', 'Unsafe'])
    ax.set_yticklabels(['Safe', 'Unsafe'])
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png', dpi=100)
    logger.info("Saved: output/confusion_matrix.png")
    plt.close()

    # Print summary report
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY REPORT")
    logger.info("="*60)
    logger.info(f"\nTest Suite: {len(test_suite)} prompts")
    logger.info(f"  - Safe: {(test_suite['label']=='safe').sum()}")
    logger.info(f"  - Unsafe: {(test_suite['label']=='unsafe').sum()}")

    logger.info(f"\nBest Performance (by F1-Score):")
    logger.info(f"  Threshold: {best_f1_threshold}")
    logger.info(f"  Accuracy: {summary_df.loc[best_f1_idx, 'accuracy']:.4f}")
    logger.info(f"  Precision: {summary_df.loc[best_f1_idx, 'precision']:.4f}")
    logger.info(f"  Recall: {summary_df.loc[best_f1_idx, 'recall']:.4f}")
    logger.info(f"  F1-Score: {summary_df.loc[best_f1_idx, 'f1_score']:.4f}")
    logger.info(f"  False Positive Rate: {summary_df.loc[best_f1_idx, 'false_positive_rate']:.4f}")

    logger.info(f"\nPer-Threshold Summary:")
    logger.info(summary_df.to_string(index=False))

    logger.info("\n" + "="*60)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: output/")
    logger.info(f"  - detailed_results.json")
    logger.info(f"  - metrics_summary.csv")
    logger.info(f"  - metrics_vs_threshold.png")
    logger.info(f"  - roc_curve.png")
    logger.info(f"  - confusion_matrix.png")


if __name__ == "__main__":
    main()
