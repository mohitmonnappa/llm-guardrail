#!/usr/bin/env python
"""
Generate test results with per-prompt verdicts and confidence scores.

Run with: python generate_results.py
"""

from src.classifier import PromptClassifier
import pandas as pd
import json

def main():
    # Load test suite
    test_suite = pd.read_csv("data/test_suite.csv")

    # Initialize classifier
    print("Loading classifier...")
    clf = PromptClassifier(threshold=0.5)

    # Classify all prompts
    print(f"Classifying {len(test_suite)} prompts...")
    results = []

    for idx, row in test_suite.iterrows():
        prompt = row['prompt']
        true_category = row['category']
        true_label = row['label']

        result = clf.classify(prompt)

        # Determine if hit or miss
        predicted_label = "unsafe" if result['verdict'] == "unsafe" else "safe"
        hit = (predicted_label == true_label)

        results.append({
            'prompt': prompt,
            'true_category': true_category,
            'true_label': true_label,
            'predicted_verdict': result['verdict'],
            'predicted_category': result['category'],
            'confidence': round(result['confidence'], 3),
            'hit': 'YES' if hit else 'NO'
        })

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(test_suite)} done")

    # Save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('output/test_results.csv', index=False)
    print("\n✅ Results saved!")

    # Save as JSON
    results_json = results_df.to_dict('records')
    with open('output/test_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print summary
    print(f"\nTest Suite Results (45 prompts)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Calculate metrics
    accuracy = (results_df['hit'] == 'YES').sum() / len(results_df)
    print("\n" + "=" * 80)
    print(f"Overall Accuracy: {accuracy*100:.1f}%")
    print(f"Correct: {(results_df['hit'] == 'YES').sum()}/45")
    print(f"\nPer-Category Accuracy:")
    for category in results_df['true_category'].unique():
        cat_data = results_df[results_df['true_category'] == category]
        cat_accuracy = (cat_data['hit'] == 'YES').sum() / len(cat_data)
        print(f"  {category}: {cat_accuracy*100:.1f}% ({(cat_data['hit'] == 'YES').sum()}/{len(cat_data)})")

    print(f"\nFiles saved:")
    print(f"  - output/test_results.csv")
    print(f"  - output/test_results.json")

if __name__ == "__main__":
    main()
