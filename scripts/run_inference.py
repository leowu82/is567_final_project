import argparse
from fraud_detect.gatekeeper.preprocessing import scrub_pii, normalize_text
from fraud_detect.gatekeeper.rules import score_rules
from fraud_detect.gatekeeper.vectorizer import load_model, predict_baseline
from fraud_detect.output.explain import explain_decision

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--text', required=True)
    args = p.parse_args()
    t = scrub_pii(args.text)
    t = normalize_text(t)
    heur = score_rules(t)
    try:
        model = load_model('models/baseline.joblib')
        label = predict_baseline(model, [t])[0]
    except Exception:
        label = None
    explanation = explain_decision(t, heur, retrieved=None)
    out = {"label": label, "explanation": explanation}
    print(out)

if __name__ == '__main__':
    main()
