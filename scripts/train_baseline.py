"""Train a small TF-IDF + SVM baseline on synthetic data for a smoke test."""
from fraud_detect.gatekeeper.vectorizer import train_baseline, save_model
from fraud_detect.gatekeeper.preprocessing import normalize_text

def main():
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
    print("Baseline trained and saved to models/baseline.joblib")

if __name__ == '__main__':
    main()
