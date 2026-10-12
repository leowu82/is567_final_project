"""Fine-tune a transformer on real or synthetic data.

This script will run a single-epoch sanity check if no real dataset is available.
"""
import os
from fraud_detect.analyzer.slm_model import fine_tune
from fraud_detect.data_utils import load_combined_dataset, get_dataset_stats

def main():
    # Try to load real datasets; fall back to synthetic if unavailable
    try:
        print("Loading training datasets...")
        texts, labels = load_combined_dataset(include_sms=True, include_phishing=True)
        print(f"Dataset loaded: {len(texts)} samples")
        stats = get_dataset_stats(texts, labels)
        print(f"Label distribution: {stats['label_distribution']}")
    except Exception as e:
        print(f"⚠ Could not load real datasets: {e}")
        print("Using synthetic data for sanity check...")
        texts = ["You have won a prize, click here", "Meeting at 10am tomorrow"]
        labels = [1, 0]
    
    out = fine_tune(texts, labels, output_dir="models/slm")
    print(f"SLM model saved to: {out} (trained on {len(texts)} samples)")

if __name__ == '__main__':
    main()
