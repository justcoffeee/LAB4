"""LAB 04 - preprocessing helpers (no leakage by construction).

Pipeline dung:
  raw texts -> train/val/test split (stratified)
  -> fit TF-IDF / Word2Vec chi tren TRAIN
  -> transform val/test
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

URL_RE = re.compile(r"http\S+|www\.\S+")
NUM_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")


def clean_text(s):
    """Lowercase + chuan hoa URL/number, giu chu cai (don gian, tai hien duoc)."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub(" <url> ", s)
    s = NUM_RE.sub(" <num> ", s)
    s = re.sub(r"[^a-z_<>\s]", " ", s)
    return WS_RE.sub(" ", s).strip()


def load_sms(path="sms.tsv"):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df["label_id"] = (df["label"] == "spam").astype(int)  # spam=1 (positive), ham=0
    df["clean"] = df["text"].astype(str).map(clean_text)
    return df


def stratified_split(df, seed=42):
    """70% train / 15% val / 15% test, stratified theo label."""
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label_id"])
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["label_id"])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def fit_tfidf(train_texts, ngram=(1, 1), max_features=None, stopwords=None):
    """Fit vectorizer CHI TREN TRAIN (tranh leakage)."""
    vec = TfidfVectorizer(ngram_range=ngram, max_features=max_features,
                          stop_words=stopwords, token_pattern=r"(?u)\b\w+\b")
    X_train = vec.fit_transform(train_texts)
    return vec, X_train


def document_average_vectors(texts, embedding, dim):
    """Average Word2Vec: mean vector cua cac tu co trong embedding; cau rong -> zeros."""
    X = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        ids = [embedding[w] for w in str(t).split() if w in embedding]
        if ids:
            X[i] = np.mean(ids, axis=0)
    return X


if __name__ == "__main__":
    df = load_sms("sms.tsv")
    print("shape:", df.shape)
    print(df["label"].value_counts().to_string())
    print("dup rows:", int(df.duplicated().sum()),
          "| missing text:", int(df["text"].isna().sum()))
    print("len describe:\n", df["text"].str.len().describe().to_string())
    tr, va, te = stratified_split(df)
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")
    print("spam rate:", {k: round(v['label_id'].mean(), 4)
                         for k, v in [("train", tr), ("val", va), ("test", te)]})
