import os
try:
    import faiss
    _FAISS = True
except Exception:
    _FAISS = False
import numpy as np

def build_faiss_index(embeddings: np.ndarray, index_path: str):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if not _FAISS:
        # save numpy fallback
        np.save(index_path + ".npy", embeddings)
        return index_path
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index_path

def query_index(index_path: str, query_emb: np.ndarray, k: int = 5):
    if not _FAISS:
        idxs = np.arange(0, min(k, query_emb.shape[0]))
        return idxs.tolist(), [0.0]*len(idxs)
    index = faiss.read_index(index_path)
    D, I = index.search(query_emb, k)
    return I.tolist(), D.tolist()
