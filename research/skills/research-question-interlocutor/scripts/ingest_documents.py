#!/usr/bin/env python3
"""Extract page-aware text and chunks from locally supplied researcher sources."""

import argparse
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import yaml
from pypdf import PdfReader


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self):
        return normalize_text(html.unescape(" ".join(self.parts)))


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunks(words, size=800, overlap=100):
    if size <= overlap:
        raise ValueError("chunk size must exceed overlap")
    start = 0
    while start < len(words):
        yield start, words[start : start + size]
        start += size - overlap


def emit_chunk(stream, metadata, work_id, source_label, checksum, chunk_index, offset, content):
    record = {
        "chunk_id": f"{work_id}:{source_label}:c{chunk_index}",
        "work_id": work_id,
        "title": metadata.get("title", work_id),
        "year": metadata.get("year"),
        "source_type": metadata.get("type"),
        "page": source_label,
        "word_offset": offset,
        "text": " ".join(content),
        "source_sha256": checksum,
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def ingest_text_source(source: Path, metadata, work_id: str, output: Path, size: int, overlap: int) -> int:
    raw = source.read_text(encoding="utf-8", errors="replace")
    if source.suffix.lower() in {".html", ".htm"}:
        parser = TextExtractor()
        parser.feed(raw)
        text = parser.text()
    else:
        text = normalize_text(raw)
    words = text.split()
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    count = 0
    with output.open("a", encoding="utf-8") as stream:
        for chunk_index, (offset, content) in enumerate(chunks(words, size, overlap)):
            emit_chunk(stream, metadata, work_id, "text", checksum, chunk_index, offset, content)
            count += 1
    return count


def ingest_pdf_source(source: Path, metadata, work_id: str, output: Path, size: int, overlap: int) -> int:
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    reader = PdfReader(source)
    count = 0
    with output.open("a", encoding="utf-8") as stream:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            words = text.split()
            for chunk_index, (offset, content) in enumerate(chunks(words, size, overlap)):
                emit_chunk(stream, metadata, work_id, f"p{page_number}", checksum, chunk_index, offset, content)
                count += 1
    return count


def ingest_work(work_dir: Path, output: Path, size: int, overlap: int) -> int:
    metadata_path = work_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return 0
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    work_id = metadata.get("id") or work_dir.name
    if (work_dir / "source.pdf").is_file():
        return ingest_pdf_source(work_dir / "source.pdf", metadata, work_id, output, size, overlap)
    for name in ("source.html", "source.htm", "source.txt", "source.md"):
        source = work_dir / name
        if source.is_file():
            return ingest_text_source(source, metadata, work_id, output, size, overlap)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("researcher_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()
    output = args.researcher_dir / "index" / "chunks.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    work_dirs = sorted(path.parent for path in (args.researcher_dir / "corpus").rglob("metadata.yaml"))
    total = sum(ingest_work(path, output, args.chunk_size, args.overlap) for path in work_dirs)
    print(f"Wrote {total} chunks to {output}")


if __name__ == "__main__":
    main()
