try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

_MODEL = None

def _ensure_model(name: str = "all-MiniLM-L6-v2"):
    global _MODEL
    if _MODEL is None:
        if _ST_AVAILABLE:
            _MODEL = SentenceTransformer(name)
        else:
            _MODEL = None
    return _MODEL

def get_embeddings(texts, model_name: str = "all-MiniLM-L6-v2"):
    """Return dense embeddings for a list of texts. Falls back to simple hash vector if package missing."""
    model = _ensure_model(model_name)
    if model is not None:
        return model.encode(list(texts), show_progress_bar=False)
    # fallback: simple deterministic vector (not meaningful for production)
    import numpy as np
    emb = []
    for t in texts:
        arr = np.array([ord(c) for c in t[:256]], dtype=np.float32)
        # reduce or pad to length 128
        if arr.size >= 128:
            arr = arr[:128]
        else:
            arr = np.pad(arr, (0, 128-arr.size))
        emb.append(arr)
    return np.stack(emb)
