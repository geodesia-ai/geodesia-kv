"""Scarica uno split WikiText-2 senza dipendere da `datasets`/PyArrow.

L'harness usa un file di testo locale, ma il repository originale non indicava
come produrlo. Questo script legge le righe dall'API ufficiale Hugging Face
Datasets Server, conserva i ritorni a capo del dataset raw e scrive il file in
modo atomico.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
API = "https://datasets-server.huggingface.co/rows"


def fetch_rows(split: str = "validation", page_size: int = 100) -> str:
    offset = 0
    total = None
    chunks: list[str] = []
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
            payload = json.load(response)
        total = int(payload["num_rows_total"])
        rows = payload["rows"]
        chunks.extend(item["row"]["text"] for item in rows)
        offset += len(rows)
        if not rows and offset < total:
            raise RuntimeError(f"pagina vuota a offset {offset}, totale {total}")
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/wikitext2_valid.txt")
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        default="validation")
    args = parser.parse_args()

    out = Path(args.out)
    text = fetch_rows(args.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, out)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"{out}: {len(text):,} caratteri, sha256={digest}")


if __name__ == "__main__":
    main()
