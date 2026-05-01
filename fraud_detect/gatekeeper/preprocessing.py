import re

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,3}\)|\d{2,4})[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")

def scrub_pii(text: str) -> str:
    """Redact simple PII: emails and phone-like patterns."""
    if not text:
        return text
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = PHONE_RE.sub("[PHONE]", t)
    return t


def normalize_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t
