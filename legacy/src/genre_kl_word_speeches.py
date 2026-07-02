#!/usr/bin/env python3
"""
KL divergence between unigram word distributions (XML speeches) and genre references.

For each play, build a smoothed unigram P over a vocabulary fixed on training plays only
(leave-one-out). For each genre g, pool training plays of that genre → smoothed Q_g.
Prediction: argmin_g D_KL(P || Q_g) (smaller = closer match to genre's word bag).

Standalone: xml.etree only for extraction. Run from repo root:
  python3 src/genre_kl_word_speeches.py
  python3 src/genre_kl_word_speeches.py --predictor logreg
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import rel_entr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

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
GENRE_ORDER = ("comedy", "history", "tragedy")


def extract_speeches_concat(xml_path: str) -> tuple[str, int]:
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
    return " ".join(chunks), len(chunks)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def corpus_vocab(train_docs: list[str], max_vocab: int) -> list[str]:
    c: Counter[str] = Counter()
    for d in train_docs:
        c.update(tokenize(d))
    return [w for w, _ in c.most_common(max_vocab)]


def word_to_index(vocab: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(vocab)}


def count_vector(doc: str, w2i: dict[str, int], vocab_size: int) -> np.ndarray:
    """Counts over |vocab| + 1 buckets (last = UNK)."""
    out = np.zeros(vocab_size + 1, dtype=np.float64)
    unk = vocab_size
    for w in tokenize(doc):
        j = w2i.get(w, unk)
        out[j] += 1.0
    return out


def smoothed_distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    c = counts + float(alpha)
    s = c.sum()
    if s <= 0:
        n = len(counts)
        return np.full(n, 1.0 / n, dtype=np.float64)
    return c / s


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P || Q); P,Q on same finite alphabet, both sum to 1."""
    return float(np.sum(rel_entr(p, q)))


def loocv_kl_features_and_argmin(
    docs: list[str],
    genres: list[str],
    *,
    max_vocab: int,
    alpha: float,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Returns:
      X_kl: shape (n, 3) with columns [KL(P||Q_comedy), KL(P||Q_history), KL(P||Q_tragedy)]
      preds_argmin: list of predicted genre per row
      per_play: metadata including kl dict per play
    """
    n = len(docs)
    X_kl = np.zeros((n, len(GENRE_ORDER)), dtype=np.float64)
    preds: list[str] = []
    rows: list[dict] = []

    for k in range(n):
        train_idx = [i for i in range(n) if i != k]
        train_docs = [docs[i] for i in train_idx]
        vocab = corpus_vocab(train_docs, max_vocab)
        V = len(vocab)
        w2i = word_to_index(vocab)

        by_g: dict[str, list[int]] = defaultdict(list)
        for j, i in enumerate(train_idx):
            by_g[genres[i]].append(j)

        train_counts = [count_vector(train_docs[j], w2i, V) for j in range(len(train_idx))]
        c_test = count_vector(docs[k], w2i, V)
        p_play = smoothed_distribution(c_test, alpha)

        kl_row: dict[str, float] = {}
        q_genre: dict[str, np.ndarray] = {}

        for g in GENRE_ORDER:
            idxs = by_g[g]
            if not idxs:
                kl_row[g] = float("inf")
                q_genre[g] = np.ones(V + 1) / (V + 1)
                continue
            pooled = np.zeros(V + 1, dtype=np.float64)
            for j in idxs:
                pooled += train_counts[j]
            q_g = smoothed_distribution(pooled, alpha)
            q_genre[g] = q_g
            kl_row[g] = kl_divergence(p_play, q_g)

        for gi, g in enumerate(GENRE_ORDER):
            X_kl[k, gi] = kl_row[g] if np.isfinite(kl_row[g]) else 1e12

        pred = min(GENRE_ORDER, key=lambda g: kl_row[g])
        preds.append(pred)
        rows.append({"kl_to_genre_reference": kl_row, "argmin_kl_prediction": pred})

    return X_kl, preds, rows


def main() -> int:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description="Genre via KL of speech unigrams to genre pools (LOOCV).")
    ap.add_argument("--max-vocab", type=int, default=8000, help="Top unigrams from training fold")
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet/Laplace smoothing per vocabulary bucket (+ UNK)",
    )
    ap.add_argument(
        "--predictor",
        choices=("argmin", "logreg"),
        default="argmin",
        help="argmin KL to genre pool, or logistic regression on the 3 KL features",
    )
    ap.add_argument("--logreg-c", type=float, default=1.0)
    ap.add_argument(
        "--out-json",
        default=os.path.join(root_dir, "output", "genre_kl_speeches_loocv.json"),
    )
    ap.add_argument(
        "--out-plot",
        default=os.path.join(root_dir, "output", "genre_kl_speeches_loocv.svg"),
    )
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
        print("Not enough plays.", file=sys.stderr)
        return 1

    X_kl, pred_argmin, meta_rows = loocv_kl_features_and_argmin(
        docs,
        genres,
        max_vocab=args.max_vocab,
        alpha=args.alpha,
    )

    le = LabelEncoder()
    y = le.fit_transform(genres)
    labels = list(le.classes_)

    if args.predictor == "argmin":
        pred_labels = pred_argmin
        method = "argmin_g D_KL(P_play || Q_g) with LOOCV vocab and genre pools"
    else:
        # Higher value = closer to genre; avoids LR drowning in correlated large KL magnitudes.
        X_feat = -np.asarray(X_kl, dtype=np.float64)
        loo = LeaveOneOut()
        y_pred_idx = np.zeros(len(y), dtype=int)
        for tr, te in loo.split(X_feat):
            pipe = Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=4000,
                            C=float(args.logreg_c),
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            pipe.fit(X_feat[tr], y[tr])
            y_pred_idx[te] = pipe.predict(X_feat[te])
        pred_labels = le.inverse_transform(y_pred_idx)
        method = (
            "StandardScaler + LogisticRegression on [-KL(P||Q_g)]_g (3-D), LOOCV "
            f"(C={args.logreg_c})"
        )

    acc = float(accuracy_score(genres, pred_labels))
    macro_f1 = float(f1_score(genres, pred_labels, average="macro", labels=labels))
    report = classification_report(genres, pred_labels, labels=labels, zero_division=0)
    cm = confusion_matrix(genres, pred_labels, labels=labels)

    per_play = []
    for t, ns, g, p, row, klvec in zip(
        titles, n_sp, genres, pred_labels, meta_rows, X_kl,
    ):
        per_play.append(
            {
                "title": t,
                "n_speeches": ns,
                "true_genre": g,
                "predicted_genre": p,
                "correct": g == p,
                "kl_to_comedy": float(klvec[0]),
                "kl_to_history": float(klvec[1]),
                "kl_to_tragedy": float(klvec[2]),
                "kl_to_genre_reference": row["kl_to_genre_reference"],
                "argmin_kl_prediction": row["argmin_kl_prediction"],
            }
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    payload = {
        "method": method,
        "predictor": args.predictor,
        "smoothing_alpha": args.alpha,
        "max_vocab": args.max_vocab,
        "n_plays": len(docs),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_play": per_play,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="magma",
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
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True genre")
    ax.set_title(
        f"Genre from KL of speech unigrams (LOOCV)\n"
        f"{args.predictor} · α={args.alpha} · |V|={args.max_vocab} · "
        f"acc={acc:.2f} · macro-F1={macro_f1:.2f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(args.out_plot, bbox_inches="tight")
    plt.close()

    print(report)
    print(f"\nWrote {args.out_json}\nWrote {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
