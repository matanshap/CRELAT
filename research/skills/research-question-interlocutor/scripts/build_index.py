#!/usr/bin/env python3
"""Build SQLite FTS5 and optional MPNet semantic indexes."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np


def load_chunks(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("researcher_dir", type=Path)
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    chunks_path = args.researcher_dir / "index" / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    database = args.researcher_dir / "index" / "lexical.sqlite3"
    database.unlink(missing_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED, work_id, title, page UNINDEXED, text)")
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
            [(row["chunk_id"], row["work_id"], row["title"], row["page"], row["text"]) for row in chunks],
        )
    if args.semantic:
        if not os.environ.get("SLURM_JOB_ID"):
            raise SystemExit("Semantic index construction must run inside Slurm")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        vectors = model.encode([row["text"] for row in chunks], normalize_embeddings=True)
        np.save(args.researcher_dir / "index" / "embeddings.npy", vectors)
        (args.researcher_dir / "index" / "semantic_metadata.json").write_text(
            json.dumps({"model": "sentence-transformers/all-mpnet-base-v2", "chunk_ids": [row["chunk_id"] for row in chunks]}, indent=2),
            encoding="utf-8",
        )
    print(f"Indexed {len(chunks)} chunks")


if __name__ == "__main__":
    main()
