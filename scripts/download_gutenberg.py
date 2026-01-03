#!/usr/bin/env python
"""Download Project Gutenberg texts for a list of book IDs.

Usage:
    python download_gutenberg.py --ids-file ids.txt --output-dir D:\\Narrator\\gutenberg_texts

The IDs file should contain one Project Gutenberg numeric ID per line.
The script attempts several common URL patterns plus ZIP fallbacks, and
saves the resulting plain-text files into the requested output folder.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

import requests


USER_AGENT = "NarratorDownloader/1.0 (+https://www.gutenberg.org)"
TEXT_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt.utf8",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt.utf-8",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-8.txt",
]
ZIP_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.zip",
    "https://www.gutenberg.org/files/{id}/{id}.zip",
    "https://www.gutenberg.org/files/{id}/{id}-0.zip",
]
SLEEP_SECONDS = 0  # politeness delay between requests
TIMEOUT = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Gutenberg texts for a set of IDs.")
    parser.add_argument(
        "--ids-file",
        required=True,
        help="Path to a text file with one Project Gutenberg ID per line.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the downloaded .txt files will be stored.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs that already have a .txt file in the output directory.",
    )
    return parser.parse_args()


def load_ids(ids_path: Path) -> list[str]:
    ids = []
    with ids_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    if not ids:
        raise ValueError(f"No IDs found in {ids_path}")
    return sorted(set(ids), key=int)


def candidate_urls(book_id: str) -> Iterable[str]:
    for pattern in TEXT_PATTERNS:
        yield pattern.format(id=book_id)


def candidate_zips(book_id: str) -> Iterable[str]:
    for pattern in ZIP_PATTERNS:
        yield pattern.format(id=book_id)


def save_text(path: Path, book_id: str, text: str, source: str) -> None:
    path.write_text(text, encoding="utf-8", errors="ignore")
    print(f"Saved {book_id} ({len(text)} chars) from {source}")


def download_text(session: requests.Session, book_id: str) -> tuple[bool, str | None]:
    # Plain-text attempts
    for url in candidate_urls(book_id):
        try:
            resp = session.get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:
            print(f"  Text fetch failed for {book_id} at {url}: {exc}")
            continue
        if resp.status_code == 200 and "Project Gutenberg" in resp.text:
            return True, resp.text
    # ZIP fallbacks
    for url in candidate_zips(book_id):
        try:
            resp = session.get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:
            print(f"  Zip fetch failed for {book_id} at {url}: {exc}")
            continue
        if resp.status_code != 200:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
                for name in archive.namelist():
                    if not name.lower().endswith(".txt"):
                        continue
                    text = archive.read(name).decode("utf-8", errors="ignore")
                    if "Project Gutenberg" in text:
                        return True, text
        except zipfile.BadZipFile as exc:
            print(f"  Corrupt ZIP for {book_id} at {url}: {exc}")
    return False, None


def main() -> int:
    args = parse_args()
    ids_path = Path(args.ids_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        book_ids = load_ids(ids_path)
    except ValueError as exc:
        print(exc)
        return 1

    print(f"Attempting download for {len(book_ids)} Project Gutenberg IDs...")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    failures: list[str] = []
    downloaded = 0

    for book_id in book_ids:
        target_path = output_dir / f"{book_id}.txt"
        if args.resume and target_path.exists():
            print(f"Skipping {book_id}: already downloaded.")
            continue

        success, text = download_text(session, book_id)
        if success and text is not None:
            save_text(target_path, book_id, text, "Project Gutenberg")
            downloaded += 1
        else:
            failures.append(book_id)
        time.sleep(SLEEP_SECONDS)

    print(f"\nCompleted. Downloaded {downloaded} books to {output_dir}.")
    if failures:
        print(f"Failed to download {len(failures)} books:")
        print(failures)
        return 2
    print("All requested books were downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
