import argparse
import json
from fraud_detect.gatekeeper.preprocessing import scrub_pii, normalize_text
from fraud_detect.gatekeeper.rules import score_rules
from fraud_detect.gatekeeper.vectorizer import load_model as load_baseline, predict_baseline
from fraud_detect.analyzer.slm_model import load_model as load_slm, predict_slm
from fraud_detect.output.explain import explain_decision

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--text', required=True)
    p.add_argument('--model', choices=['slm', 'baseline'], default='slm', 
                   help='Which model to use (default: slm with fallback to baseline)')
    args = p.parse_args()
    
    t = scrub_pii(args.text)
    t = normalize_text(t)
    heur = score_rules(t)
    
    label = None
    model_used = None
    
    # Try SLM model first (unless explicitly choosing baseline)
    if args.model != 'baseline':
        slm_model = load_slm('models/slm')
        if slm_model is not None:
            try:
                predictions, probs = predict_slm(slm_model, [t])
                label = int(predictions[0])
                model_used = 'slm'
            except Exception as e:
                print(f"SLM inference failed: {e}", file=__import__('sys').stderr)
    
    # Fallback to baseline if SLM not available or failed
    if label is None:
        try:
            baseline_model = load_baseline('models/baseline.joblib')
            if baseline_model is not None:
                label = int(predict_baseline(baseline_model, [t])[0])
                model_used = 'baseline'
        except Exception as e:
            print(f"Baseline inference failed: {e}", file=__import__('sys').stderr)
    
    explanation = explain_decision(t, heur, retrieved=None)
    out = {
        "label": label, 
        "explanation": explanation,
        "model_used": model_used
    }
    print(json.dumps(out))

if __name__ == '__main__':
    main()
