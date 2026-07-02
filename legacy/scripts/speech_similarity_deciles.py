#!/usr/bin/env python3
"""Sample speech-pair examples from cosine-similarity deciles.

This uses the same Folger play list and XMLParser speech_interactions records
that genre_analysis.py uses for its BERT interaction features.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from genre_analysis import PLAYS  # noqa: E402
from xmlparser import XMLParser, resolve_model, slugify_transformer_model  # noqa: E402


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _speaker_name(speaker: str) -> str:
    if not speaker:
        return speaker
    if "_" in speaker:
        return speaker.rsplit("_", 1)[0]
    return speaker


def _use_ascii_apostrophes(value):
    """Replace typographic apostrophes with the ASCII apostrophe."""
    if not isinstance(value, str):
        return value
    return value.replace("\u2018", "'").replace("\u2019", "'")


def collect_interactions(
    model: str,
    batch_size: int,
    play: str | None = None,
) -> list[dict]:
    slug, _, _ = resolve_model(model)
    rows: list[dict] = []
    for xml_path, title, genre, year in PLAYS:
        if play and title.casefold() != play.casefold():
            continue
        path = ROOT / xml_path
        if not path.exists():
            print(f"Skipping missing XML: {xml_path}", flush=True)
            continue
        print(f"Parsing {title} [{model}]...", flush=True)
        parser = XMLParser(
            str(path),
            options={"co-oc", model},
            embedding_batch_size=batch_size,
        )
        parser.parse()
        for row in parser.speech_interactions:
            if row.get("model") != slug:
                continue
            enriched = dict(row)
            enriched["genre"] = genre
            enriched["year"] = year
            rows.append(enriched)
    return rows


def choose_decile_examples(df: pd.DataFrame, examples_per_decile: int) -> pd.DataFrame:
    sims = df["cosine_similarity"].astype(float).to_numpy()
    edges = np.quantile(sims, np.linspace(0, 1, 11))
    chosen = []
    used = set()

    for decile in range(1, 11):
        lo = edges[decile - 1]
        hi = edges[decile]
        if decile == 10:
            mask = (df["cosine_similarity"] >= lo) & (df["cosine_similarity"] <= hi)
        else:
            mask = (df["cosine_similarity"] >= lo) & (df["cosine_similarity"] < hi)
        bucket = df.loc[mask].copy()
        if bucket.empty:
            continue

        midpoint = (lo + hi) / 2.0
        bucket["decile"] = decile
        bucket["decile_min"] = lo
        bucket["decile_max"] = hi
        bucket["distance_to_midpoint"] = (bucket["cosine_similarity"] - midpoint).abs()
        bucket = bucket.sort_values(
            ["distance_to_midpoint", "play", "scene", "speaker1", "speaker2"]
        )
        n = 0
        for idx, row in bucket.iterrows():
            if idx in used:
                continue
            chosen.append(row)
            used.add(idx)
            n += 1
            if n >= examples_per_decile:
                break

    out = pd.DataFrame(chosen).drop(columns=["distance_to_midpoint"], errors="ignore")
    out["speaker1_short"] = out["speaker1"].map(_speaker_name)
    out["speaker2_short"] = out["speaker2"].map(_speaker_name)
    return out


def write_markdown(examples: pd.DataFrame, path: Path, max_chars: int) -> None:
    lines = [
        "# Speech Similarity Examples by Cosine-Similarity Decile",
        "",
        "Computed from adjacent-speaker BERT speech embeddings using the same",
        "`XMLParser.speech_interactions` pathway referenced by `src/genre_analysis.py`.",
        "",
    ]
    for _, row in examples.sort_values(["decile", "cosine_similarity"]).iterrows():
        lines.extend(
            [
                (
                    f"## Decile {int(row.decile)} "
                    f"({row.decile_min:.4f} to {row.decile_max:.4f})"
                ),
                "",
                (
                    f"- {row.play}, {row.scene}; {row.speaker1_short} -> "
                    f"{row.speaker2_short}; cosine={row.cosine_similarity:.4f}"
                ),
                f"- {row.speaker1_short}: {_shorten(row.text1, max_chars)}",
                f"- {row.speaker2_short}: {_shorten(row.text2, max_chars)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Use an existing speech_interactions CSV instead of recomputing embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--examples-per-decile", type=int, default=1)
    parser.add_argument("--max-text-chars", type=int, default=360)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--play",
        default=None,
        help="Restrict interactions and calculate deciles within one play.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_slug = slugify_transformer_model(args.model)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_qualifier = (
        f"_{slugify_transformer_model(args.play)}" if args.play else ""
    )
    interactions_path = output_dir / (
        f"speech_interactions_{model_slug}{output_qualifier}.csv"
    )
    if args.input_csv:
        interactions_path = Path(args.input_csv)
        if not interactions_path.is_absolute():
            interactions_path = ROOT / interactions_path
        df = pd.read_csv(interactions_path)
    else:
        rows = collect_interactions(args.model, args.batch_size, play=args.play)
        if not rows:
            raise SystemExit("No speech interactions found.")
        df = pd.DataFrame(rows)
        df.to_csv(interactions_path, index=False, quoting=csv.QUOTE_MINIMAL)

    if args.play:
        available_plays = sorted(df["play"].dropna().unique())
        play_mask = df["play"].str.casefold() == args.play.casefold()
        df = df.loc[play_mask].copy()
        if df.empty:
            available = ", ".join(available_plays)
            raise SystemExit(f"No interactions found for play {args.play!r}. {available}")

    df = df.sort_values("cosine_similarity").reset_index(drop=True)
    examples = choose_decile_examples(df, args.examples_per_decile)
    for column in examples.select_dtypes(include="object").columns:
        examples[column] = examples[column].map(_use_ascii_apostrophes)

    suffix = f"{args.examples_per_decile}x_" if args.examples_per_decile != 1 else ""
    examples_path = output_dir / (
        f"speech_similarity_decile_examples_{suffix}{model_slug}{output_qualifier}.csv"
    )
    examples.to_csv(examples_path, index=False, quoting=csv.QUOTE_MINIMAL)

    md_path = output_dir / (
        f"speech_similarity_decile_examples_{suffix}{model_slug}{output_qualifier}.md"
    )
    write_markdown(examples, md_path, args.max_text_chars)

    print(f"Read {len(df)} interactions from {interactions_path}")
    print(f"Wrote {len(examples)} examples to {examples_path}")
    print(f"Wrote Markdown report to {md_path}")


if __name__ == "__main__":
    main()
