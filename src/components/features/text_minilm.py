import numpy as np
from sentence_transformers import SentenceTransformer

class TextMiniLMUtt:
    def __init__(self, model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, text:str) -> np.ndarray:
        if not text:
            return np.zeros(384, dtype=np.float32)
        emb = self.model.encode(text, normalize_embeddings=False)
        return emb.astype(np.float32)
