"""
Evaluation Module
=================
Comprehensive evaluation metrics for LLM guardrails.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate classifier performance."""

    CATEGORIES = ["safe", "jailbreak", "pii_extraction", "toxic_content", "prompt_injection"]

    def __init__(self):
        self.results = []

    def evaluate(
        self,
        predictions: List[Dict],
        ground_truth: pd.DataFrame,
        threshold: float
    ) -> Dict:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of model predictions
            ground_truth: DataFrame with category and label columns
            threshold: Classification threshold

        Returns:
            Dictionary with metrics
        """
        # Extract predictions and labels
        pred_categories = [p['category'] for p in predictions]
        true_categories = ground_truth['category'].tolist()

        # Convert to safe/unsafe labels
        pred_labels = ['unsafe' if cat != 'safe' else 'safe' for cat in pred_categories]
        true_labels = ground_truth['label'].tolist()

        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }

        # Binary classification metrics (safe vs unsafe)
        metrics['metrics']['binary'] = self._calculate_binary_metrics(
            true_labels,
            pred_labels
        )

        # Multi-class metrics (all categories)
        metrics['metrics']['multiclass'] = self._calculate_multiclass_metrics(
            true_categories,
            pred_categories
        )

        # Per-category metrics
        metrics['metrics']['per_category'] = self._calculate_per_category_metrics(
            true_categories,
            pred_categories
        )

        self.results.append(metrics)
        return metrics

    def _calculate_binary_metrics(self, true_labels, pred_labels) -> Dict:
        """Calculate binary (safe/unsafe) metrics."""
        # Convert to numerical
        true_binary = [1 if label == 'unsafe' else 0 for label in true_labels]
        pred_binary = [1 if label == 'unsafe' else 0 for label in pred_labels]

        return {
            'accuracy': accuracy_score(true_binary, pred_binary),
            'precision': precision_score(true_binary, pred_binary, zero_division=0),
            'recall': recall_score(true_binary, pred_binary, zero_division=0),
            'f1_score': f1_score(true_binary, pred_binary, zero_division=0),
            'true_positives': sum((p == 1 and t == 1) for p, t in zip(pred_binary, true_binary)),
            'true_negatives': sum((p == 0 and t == 0) for p, t in zip(pred_binary, true_binary)),
            'false_positives': sum((p == 1 and t == 0) for p, t in zip(pred_binary, true_binary)),
            'false_negatives': sum((p == 0 and t == 1) for p, t in zip(pred_binary, true_binary)),
        }

    def _calculate_multiclass_metrics(self, true_categories, pred_categories) -> Dict:
        """Calculate multi-class metrics."""
        # Map categories to IDs
        cat_to_id = {cat: idx for idx, cat in enumerate(self.CATEGORIES)}
        true_ids = [cat_to_id.get(cat, 0) for cat in true_categories]
        pred_ids = [cat_to_id.get(cat, 0) for cat in pred_categories]

        return {
            'accuracy': accuracy_score(true_ids, pred_ids),
            'weighted_precision': precision_score(true_ids, pred_ids, average='weighted', zero_division=0),
            'weighted_recall': recall_score(true_ids, pred_ids, average='weighted', zero_division=0),
            'weighted_f1': f1_score(true_ids, pred_ids, average='weighted', zero_division=0),
        }

    def _calculate_per_category_metrics(self, true_categories, pred_categories) -> Dict:
        """Calculate metrics for each category."""
        metrics = {}

        for category in self.CATEGORIES:
            # Binary classification for this category
            true_binary = [1 if cat == category else 0 for cat in true_categories]
            pred_binary = [1 if cat == category else 0 for cat in pred_categories]

            metrics[category] = {
                'accuracy': accuracy_score(true_binary, pred_binary),
                'precision': precision_score(true_binary, pred_binary, zero_division=0),
                'recall': recall_score(true_binary, pred_binary, zero_division=0),
                'f1_score': f1_score(true_binary, pred_binary, zero_division=0),
                'count': sum(true_binary)
            }

        return metrics

    def get_summary(self) -> Dict:
        """Get summary across all thresholds."""
        if not self.results:
            return {}

        df = pd.DataFrame([
            {
                'threshold': r['threshold'],
                'accuracy': r['metrics']['binary']['accuracy'],
                'precision': r['metrics']['binary']['precision'],
                'recall': r['metrics']['binary']['recall'],
                'f1_score': r['metrics']['binary']['f1_score'],
                'false_positive_rate': r['metrics']['binary']['false_positives'] / (
                    r['metrics']['binary']['false_positives'] + r['metrics']['binary']['true_negatives']
                ) if (r['metrics']['binary']['false_positives'] + r['metrics']['binary']['true_negatives']) > 0 else 0
            }
            for r in self.results
        ])

        return {
            'summary_table': df.to_dict('records'),
            'best_threshold_by_f1': float(df.loc[df['f1_score'].idxmax(), 'threshold']),
            'best_threshold_by_accuracy': float(df.loc[df['accuracy'].idxmax(), 'threshold']),
        }

    def save_results(self, output_path: str) -> None:
        """Save results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    def save_summary(self, output_path: str) -> None:
        """Save summary to CSV."""
        summary = self.get_summary()
        if 'summary_table' in summary:
            df = pd.DataFrame(summary['summary_table'])
            df.to_csv(output_path, index=False)
            logger.info(f"Summary saved to {output_path}")
