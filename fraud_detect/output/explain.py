def explain_decision(text: str, heuristic: dict, retrieved: list = None) -> str:
    parts = []
    parts.append(f"Input length: {len(text)}")
    parts.append(f"Heuristic score: {heuristic.get('score',0)}")
    evidence = heuristic.get('evidence', {})
    for k, v in evidence.items():
        parts.append(f"Matched {k}: {v[:3]}")
    if retrieved:
        parts.append(f"Top retrieved KB items: {len(retrieved)}")
    return " | ".join(parts)
