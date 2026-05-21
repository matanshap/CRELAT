#!/usr/bin/env python3
"""
Genre classification from Folger XML dialogue only: <sp> speeches → TF-IDF → LOOCV.

Standalone: uses xml.etree only (no xmlparser / BERT / torch).
Play list matches the 37 canonical Folger corpus plays used elsewhere in CRELAT.

Run from repo root:
  python3 src/genre_from_xml_speech_text.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

# (xml_path relative to repo root, title, genre) — same 37 plays as genre_analysis.PLAYs
PLAYS: list[tuple[str, str, str]] = [
    ("Data/XML/folger_corpus/Err.xml", "The Comedy of Errors", "comedy"),
    ("Data/XML/folger_corpus/TGV.xml", "The Two Gentlemen of Verona", "comedy"),
    ("Data/XML/folger_corpus/Shr.xml", "The Taming of the Shrew", "comedy"),
    ("Data/XML/folger_corpus/LLL.xml", "Love's Labour's Lost", "comedy"),
    ("Data/XML/folger_corpus/MND.xml", "A Midsummer Night's Dream", "comedy"),
    ("Data/XML/folger_corpus/MV.xml", "The Merchant of Venice", "comedy"),
    ("Data/XML/folger_corpus/Wiv.xml", "The Merry Wives of Windsor", "comedy"),
    ("Data/XML/folger_corpus/Ado.xml", "Much Ado About Nothing", "comedy"),
    ("Data/XML/folger_corpus/AYL.xml", "As You Like It", "comedy"),
    ("Data/XML/folger_corpus/TN.xml", "Twelfth Night", "comedy"),
    ("Data/XML/folger_corpus/AWW.xml", "All's Well That Ends Well", "comedy"),
    ("Data/XML/folger_corpus/MM.xml", "Measure for Measure", "comedy"),
    ("Data/XML/folger_corpus/Tit.xml", "Titus Andronicus", "tragedy"),
    ("Data/XML/folger_corpus/Rom.xml", "Romeo and Juliet", "tragedy"),
    ("Data/XML/folger_corpus/JC.xml", "Julius Caesar", "tragedy"),
    ("Data/XML/folger_corpus/Ham.xml", "Hamlet", "tragedy"),
    ("Data/XML/folger_corpus/Oth.xml", "Othello", "tragedy"),
    ("Data/XML/folger_corpus/Lr.xml", "King Lear", "tragedy"),
    ("Data/XML/folger_corpus/Mac.xml", "Macbeth", "tragedy"),
    ("Data/XML/folger_corpus/Ant.xml", "Antony and Cleopatra", "tragedy"),
    ("Data/XML/folger_corpus/Cor.xml", "Coriolanus", "tragedy"),
    ("Data/XML/folger_corpus/Tim.xml", "Timon of Athens", "tragedy"),
    ("Data/XML/folger_corpus/Tro.xml", "Troilus and Cressida", "tragedy"),
    ("Data/XML/folger_corpus/1H6.xml", "Henry VI Part 1", "history"),
    ("Data/XML/folger_corpus/2H6.xml", "Henry VI Part 2", "history"),
    ("Data/XML/folger_corpus/3H6.xml", "Henry VI Part 3", "history"),
    ("Data/XML/folger_corpus/R3.xml", "Richard III", "history"),
    ("Data/XML/folger_corpus/R2.xml", "Richard II", "history"),
    ("Data/XML/folger_corpus/Jn.xml", "King John", "history"),
    ("Data/XML/folger_corpus/1H4.xml", "Henry IV Part 1", "history"),
    ("Data/XML/folger_corpus/2H4.xml", "Henry IV Part 2", "history"),
    ("Data/XML/folger_corpus/H5.xml", "Henry V", "history"),
    ("Data/XML/folger_corpus/H8.xml", "Henry VIII", "history"),
    ("Data/XML/folger_corpus/Per.xml", "Pericles", "comedy"),
    ("Data/XML/folger_corpus/Cym.xml", "Cymbeline", "comedy"),
    ("Data/XML/folger_corpus/WT.xml", "The Winter's Tale", "comedy"),
    ("Data/XML/folger_corpus/Tmp.xml", "The Tempest", "comedy"),
]

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def extract_speeches_concat(xml_path: str) -> tuple[str, int]:
    """
    Concatenate all spoken dialogue from tei:div2 // tei:sp / tei:ab (Folger encoding).
    Returns (single document string, number of non-empty speeches).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks: list[str] = []
    for div2 in root.findall(".//tei:div2", NS):
        for sp in div2.findall("tei:sp", NS):
            ab = sp.find("tei:ab", NS)
            if ab is None:
                continue
            text = "".join(ab.itertext())
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append(text)
    doc = " ".join(chunks)
    return doc, len(chunks)


def speech_stats(doc: str) -> np.ndarray:
    """Dense stylometric scalars from raw dialogue (lowercase)."""
    low = doc.lower()
    words = _TOKEN_RE.findall(low)
    n = max(len(words), 1)
    uniq = len(set(words))
    chars = max(len(low), 1)
    return np.array(
        [
            np.log1p(len(words)),
            uniq / n,
            low.count("?") / chars,
            low.count("!") / chars,
            (low.count(" thou ") + low.count(" thee ") + low.count(" thy ") + low.count(" thine ")) / n,
            (low.count(" king") + low.count(" crown") + low.count(" majesty")) / n,
            (low.count(" love") + low.count(" marry")) / n,
            (low.count(" death") + low.count(" blood") + low.count(" murder")) / n,
        ],
        dtype=np.float64,
    ).reshape(1, -1)


def build_dense_block(docs: list[str]) -> np.ndarray:
    return np.vstack([speech_stats(d) for d in docs])


def _docs_to_dense_matrix(X) -> np.ndarray:
    """FeatureUnion helper: X is object array or list of document strings."""
    if isinstance(X, np.ndarray):
        docs = X.ravel().tolist()
    else:
        docs = list(X)
    return build_dense_block(docs)


def main() -> int:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description="LOOCV genre from XML speech text + TF-IDF.")
    ap.add_argument(
        "--out-json",
        default=os.path.join(root_dir, "output", "genre_text_xml_loocv.json"),
    )
    ap.add_argument(
        "--out-plot",
        default=os.path.join(root_dir, "output", "genre_text_xml_loocv.svg"),
    )
    ap.add_argument("--max-features", type=int, default=6000)
    ap.add_argument("--c", type=float, default=2.0, help="LogisticRegression C")
    args = ap.parse_args()

    docs: list[str] = []
    titles: list[str] = []
    genres: list[str] = []
    n_sp: list[int] = []

    for rel, title, genre in PLAYS:
        path = os.path.join(root_dir, rel)
        if not os.path.isfile(path):
            print(f"Missing XML: {path}", file=sys.stderr)
            return 1
        doc, ns = extract_speeches_concat(path)
        if len(doc) < 200:
            print(f"Very little text in {title}, skipping.", file=sys.stderr)
            continue
        docs.append(doc)
        titles.append(title)
        genres.append(genre)
        n_sp.append(ns)

    if len(docs) < 5:
        print("Not enough plays loaded.", file=sys.stderr)
        return 1

    le = LabelEncoder()
    y = le.fit_transform(genres)
    labels = list(le.classes_)

    tfidf = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    stats_branch = Pipeline(
        [
            ("mk", FunctionTransformer(_docs_to_dense_matrix, validate=False)),
            ("sc", StandardScaler()),
        ]
    )
    features = FeatureUnion(
        [
            ("tfidf", tfidf),
            ("stats", stats_branch),
        ]
    )
    clf = Pipeline(
        [
            ("features", features),
            (
                "lr",
                LogisticRegression(
                    max_iter=8000,
                    solver="saga",
                    C=float(args.c),
                    random_state=0,
                ),
            ),
        ]
    )
    loo = LeaveOneOut()
    y_pred = cross_val_predict(clf, docs, y, cv=loo)
    pred_labels = le.inverse_transform(y_pred)

    acc = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    report = classification_report(genres, pred_labels, labels=labels, zero_division=0)
    cm = confusion_matrix(genres, pred_labels, labels=labels)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    payload = {
        "method": "Pipeline LOOCV: TF-IDF (1–2 grams) on <sp>/<ab> dialogue + 8 dialogue scalars (no vocab leakage)",
        "n_plays": len(docs),
        "max_tfidf_features": args.max_features,
        "logistic_C": args.c,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_play": [
            {
                "title": t,
                "n_speeches": ns,
                "true_genre": g,
                "predicted_genre": p,
                "correct": g == p,
            }
            for t, ns, g, p in zip(titles, n_sp, genres, pred_labels)
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="BuPu",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row fraction"},
        ax=ax,
    )
    ax.set_xlabel("Predicted (LOOCV, text only)")
    ax.set_ylabel("True genre (Folio-style)")
    ax.set_title(
        "Genre from XML speech text — TF-IDF + dialogue stats\n"
        f"n={len(docs)} plays · accuracy={acc:.2f} · macro-F1={macro_f1:.2f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(args.out_plot, bbox_inches="tight")
    plt.close()

    print(report)
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
