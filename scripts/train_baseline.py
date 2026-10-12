"""Train a TF-IDF + SVM baseline on real or synthetic data."""
from fraud_detect.gatekeeper.vectorizer import train_baseline, save_model
from fraud_detect.gatekeeper.preprocessing import normalize_text
from fraud_detect.data_utils import load_combined_dataset, get_dataset_stats

def main():
    # Try to load real datasets; fall back to synthetic if unavailable
    try:
        print("Loading training datasets...")
        texts, labels = load_combined_dataset(include_sms=True, include_phishing=True)
        print(f"Dataset loaded: {len(texts)} samples")
        stats = get_dataset_stats(texts, labels)
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Average text length: {stats['avg_text_length']:.1f} chars")
    except Exception as e:
        print(f"⚠ Could not load real datasets: {e}")
        print("Using synthetic data for smoke test...")
        texts = [
            "Please reset your password at http://example.com/login",
            "Hey buddy, long time no see, let's catch up",
            "Your account has been compromised, send 2FA code",
            "Free entry in 2 a weekly competition to win"
        ]
        labels = [1, 0, 1, 1]
    
    texts = [normalize_text(t) for t in texts]
    model = train_baseline(texts, labels)
    save_model(model, "models/baseline.joblib")
    print(f"Baseline trained and saved to models/baseline.joblib (on {len(texts)} samples)")

if __name__ == '__main__':
    main()
