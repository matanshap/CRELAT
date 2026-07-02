#!/usr/bin/env python3
"""Inventory CRELAT intake files and report basic local extractability."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".rst",
    ".text",
    ".txt",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_summary(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"extractable": False, "reason": "text file is not utf-8 decodable"}
    return {
        "extractable": True,
        "chars": len(text),
        "lines": text.count("\n") + (1 if text else 0),
        "preview": " ".join(text.split())[:240],
    }


def pdf_summary(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"extractable": False, "reason": "pypdf is not installed"}
    try:
        reader = PdfReader(path)
    except Exception as error:  # pragma: no cover - parser-specific failures vary
        return {"extractable": False, "reason": f"pdf open failed: {type(error).__name__}"}
    chars = 0
    pages_with_text = 0
    preview_parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_with_text += 1
            if len(" ".join(preview_parts)) < 240:
                preview_parts.append(" ".join(text.split()))
        chars += len(text)
    return {
        "extractable": chars > 0,
        "pages": len(reader.pages),
        "pages_with_text": pages_with_text,
        "chars": chars,
        "preview": " ".join(preview_parts)[:240],
        "reason": None if chars > 0 else "no extractable text; scanned or image-only PDF likely",
    }


def classify(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        kind = "text"
        summary = text_summary(path)
    elif suffix == ".pdf":
        kind = "pdf"
        summary = pdf_summary(path)
    elif suffix in IMAGE_SUFFIXES:
        kind = "image"
        summary = {"extractable": False, "reason": "image OCR is not configured"}
    else:
        kind = "unsupported"
        summary = {"extractable": False, "reason": "unsupported file type"}
    return {"kind": kind, **summary}


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and not item.name.startswith("."))


def inventory(path: Path) -> dict[str, Any]:
    files = []
    for file_path in iter_files(path):
        stat = file_path.stat()
        record = {
            "path": str(file_path),
            "relative_path": str(file_path.relative_to(path)) if path.is_dir() else file_path.name,
            "suffix": file_path.suffix.lower(),
            "mime_type": mimetypes.guess_type(file_path.name)[0],
            "bytes": stat.st_size,
            "sha256": sha256_file(file_path),
        }
        record.update(classify(file_path))
        files.append(record)
    return {"root": str(path), "file_count": len(files), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.path.exists():
        raise SystemExit(f"not found: {args.path}")
    data = inventory(args.path)
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
