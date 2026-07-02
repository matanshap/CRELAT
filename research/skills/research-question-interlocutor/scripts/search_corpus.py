#!/usr/bin/env python3
"""Search a researcher corpus with FTS5 and optional semantic vectors."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("researcher_dir", type=Path)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    database = args.researcher_dir / "index" / "lexical.sqlite3"
    if not database.is_file():
        raise SystemExit(f"Missing index: {database}")
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT chunk_id, work_id, title, page, text, bm25(chunks) AS score FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
            (args.query, args.limit * 3),
        ).fetchall()
    candidates = {row["chunk_id"]: {**dict(row), "combined_score": 1.0 / (rank + 1)} for rank, row in enumerate(rows)}
    if args.semantic:
        if not os.environ.get("SLURM_JOB_ID"):
            raise SystemExit("Semantic query embedding must run inside Slurm")
        embeddings_path = args.researcher_dir / "index" / "embeddings.npy"
        metadata_path = args.researcher_dir / "index" / "semantic_metadata.json"
        if not embeddings_path.is_file() or not metadata_path.is_file():
            raise SystemExit("Semantic index is missing; run build_index.py --semantic")
        from sentence_transformers import SentenceTransformer

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vectors = np.load(embeddings_path)
        model = SentenceTransformer(metadata["model"])
        query = model.encode([args.query], normalize_embeddings=True)[0]
        scores = vectors @ query
        chunks = {
            row["chunk_id"]: row
            for row in (
                json.loads(line)
                for line in (args.researcher_dir / "index" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
        }
        for rank, index in enumerate(np.argsort(scores)[::-1][: args.limit * 3]):
            chunk_id = metadata["chunk_ids"][int(index)]
            row = chunks[chunk_id]
            candidate = candidates.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "work_id": row["work_id"],
                    "title": row["title"],
                    "page": row["page"],
                    "text": row["text"],
                    "score": None,
                    "combined_score": 0.0,
                },
            )
            candidate["combined_score"] += 1.25 / (rank + 1)

    ranked = sorted(candidates.values(), key=lambda row: row["combined_score"], reverse=True)
    diversified = []
    per_work = {}
    for row in ranked:
        if per_work.get(row["work_id"], 0) >= 3:
            continue
        diversified.append(row)
        per_work[row["work_id"]] = per_work.get(row["work_id"], 0) + 1
        if len(diversified) >= args.limit:
            break
    print(json.dumps(diversified, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
