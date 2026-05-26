import os
import re
import pickle
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.pkl")


def _normalize_text(text: str) -> str:
    # Minimal normalization to improve consistency
    text = text or ""
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Keep only first two columns (Category, Message)
    df = df.iloc[:, :2].copy()
    df.columns = ["Category", "Message"]

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Normalize labels
    # Dataset sometimes contains ham; we'll map to Safe/non-spam
    df["Category"] = df["Category"].replace({"ham": "not spam", "ham ": "not spam"})

    # Normalize message text
    df["Message"] = df["Message"].astype(str).map(_normalize_text)

    return df


def train_and_save(csv_path: str = None, force_retrain: bool = False):
    csv_path = csv_path or os.path.join(os.path.dirname(__file__), "spam.csv")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if (not force_retrain) and os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return

    df = _load_dataset(csv_path)

    msg = df["Message"]
    cat = df["Category"]

    # Stratify if possible
    stratify = cat if cat.nunique() > 1 else None

    msg_train, msg_test, cat_train, cat_test = train_test_split(
        msg,
        cat,
        test_size=0.2,
        random_state=1,
        stratify=stratify,
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(msg_train)
    X_test = vectorizer.transform(msg_test)

    model = MultinomialNB()
    model.fit(X_train, cat_train)

    # Quick metric (not used by app directly, but helpful)
    y_pred = model.predict(X_test)
    _acc = accuracy_score(cat_test, y_pred)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def get_model():
    """Load trained artifacts; if missing, train automatically."""
    train_and_save(force_retrain=False)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer

