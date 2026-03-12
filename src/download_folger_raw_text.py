"""
Download raw text (.txt) of Shakespeare plays from Folger Digital Texts.
Uses the Folger TXT Complete zip (individual .txt URLs redirect to HTML).
Raises on failure; no silent fallbacks.
"""

import argparse
import os
import sys
import zipfile

import requests

FOLGER_TXT_ZIP_URL = "https://www.folgerdigitaltexts.org/download/txt/FolgerDigitalTexts_TXT_Complete.zip"

PLAY_CODE_TO_TXT = {
    "AWW": "AWW", "Ant": "Ant", "AYL": "AYL", "Err": "Err", "Cor": "Cor",
    "Cym": "Cym", "Ham": "Ham", "1H4": "1H4", "2H4": "2H4", "H5": "H5",
    "1H6": "1H6", "2H6": "2H6", "3H6": "3H6", "H8": "H8", "JC": "JC",
    "Jn": "Jn", "Lr": "Lr", "LLL": "LLL", "Luc": "Luc", "Mac": "Mac",
    "MM": "MM", "MV": "MV", "Wiv": "Wiv", "MND": "MND", "Ado": "Ado",
    "Oth": "Oth", "Per": "Per", "PhT": "PhT", "R2": "R2", "R3": "R3",
    "Rom": "Rom", "Son": "Son", "Shr": "Shr", "Tmp": "Tmp", "Tim": "Tim",
    "Tit": "Tit", "Tro": "Tro", "TN": "TN", "TGV": "TGV", "TNK": "TNK",
    "Ven": "Ven", "WT": "WT",
}


def download_raw_text(play_code, output_dir, force=False):
    """
    Download raw text for a play from Folger (via TXT Complete zip).
    Raises if download or extraction fails.

    Returns:
        Path to the local .txt file.
    """
    play_code = play_code.strip()
    txt_name = PLAY_CODE_TO_TXT.get(play_code, play_code) + ".txt"
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, txt_name)

    if os.path.exists(local_path) and not force:
        return local_path

    response = requests.get(FOLGER_TXT_ZIP_URL, timeout=120)
    response.raise_for_status()
    if response.headers.get("Content-Type", "").lower() not in (
        "application/zip",
        "application/x-zip-compressed",
    ):
        raise RuntimeError(
            f"Folger TXT zip URL returned unexpected Content-Type: {response.headers.get('Content-Type')}. "
            "Expected application/zip."
        )

    zip_path = os.path.join(output_dir, "_FolgerDigitalTexts_TXT_Complete.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if txt_name not in zf.namelist():
                raise RuntimeError(
                    f"Play code '{play_code}' (file {txt_name}) not found in Folger TXT zip. "
                    f"Available: {', '.join(sorted(zf.namelist()))}"
                )
            zf.extract(txt_name, path=output_dir)
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass

    extracted = os.path.join(output_dir, txt_name)
    if not os.path.exists(extracted):
        raise RuntimeError(f"Extraction failed: {extracted} not found after unzip.")
    return extracted


def ensure_raw_text_exists(play_code, output_dir="Data/raw_text"):
    """
    Ensure raw text exists for the play. Download from Folger if missing.
    Raises on failure (no fallback).

    Returns:
        Path to local .txt file.
    """
    txt_name = PLAY_CODE_TO_TXT.get(play_code, play_code) + ".txt"
    local_path = os.path.join(output_dir, txt_name)
    if os.path.exists(local_path):
        return local_path
    return download_raw_text(play_code, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download raw text of Shakespeare plays from Folger Digital Texts (TXT zip)."
    )
    parser.add_argument(
        "play_codes",
        nargs="*",
        default=list(PLAY_CODE_TO_TXT.keys()),
        help="Play codes to download (e.g. Ham MV). Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("Data", "raw_text"),
        help="Directory to save .txt files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if file exists.",
    )
    args = parser.parse_args()

    failed = []
    for code in args.play_codes:
        try:
            download_raw_text(code, args.output_dir, force=args.force)
        except Exception as e:
            failed.append((code, e))
            print(f"Failed {code}: {e}")

    if failed:
        print(f"Failed: {len(failed)}")
        return 1
    print(f"Downloaded: {len(args.play_codes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
