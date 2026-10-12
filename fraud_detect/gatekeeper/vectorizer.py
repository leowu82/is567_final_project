from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import List

def train_baseline(texts: List[str], labels: List[int]):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LinearSVC())
    ])
    pipe.fit(texts, labels)
    return pipe

def predict_baseline(pipe, texts: List[str]):
    return pipe.predict(texts)

def save_model(pipe, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(pipe, path)

def load_model(path: str):
    return joblib.load(path)
