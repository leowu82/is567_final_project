import re

URL_RE = re.compile(r"https?://[\w\.-/\?=&%]+", re.IGNORECASE)
CRYPTO_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")

PATTERNS = {
    "url": URL_RE,
    "crypto_address": CRYPTO_RE,
}

def score_rules(text: str) -> dict:
    """Return matched rule evidence and a simple heuristic score."""
    evidence = {}
    score = 0
    for name, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            evidence[name] = matches[:5]
            score += 1
    return {"score": score, "evidence": evidence}
