from fraud_detect.gatekeeper.preprocessing import scrub_pii

def test_scrub_email():
    s = "Contact me at alice@example.com"
    out = scrub_pii(s)
    assert "[EMAIL]" in out
