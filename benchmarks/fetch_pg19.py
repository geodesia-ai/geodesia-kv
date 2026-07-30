"""Prepara un sottoinsieme riproducibile di PG-19 per benchmark long-context.

Scarica in streaming i primi libri di uno split fino a raggiungere almeno
``--min-chars``. Il testo non viene normalizzato; fra libri viene inserito solo
``\\n\\n``. Un manifest JSON conserva revisione, metadati, intervalli di
caratteri e hash di ogni libro, così le finestre possono essere ricostruite e
separate per documento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from datasets import load_dataset


DATASET = "emozilla/pg19"
REVISION = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"],
                        required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-chars", type=int, default=2_000_000)
    parser.add_argument("--max-books", type=int, default=50)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.min_chars <= 0 or args.max_books <= 0:
        raise SystemExit("--min-chars e --max-books devono essere positivi")

    stream = load_dataset(
        DATASET, split=args.split, streaming=True, revision=args.revision)
    pieces: list[str] = []
    books: list[dict] = []
    cursor = 0
    for index, row in enumerate(stream):
        if len(books) >= args.max_books:
            break
        text = row["text"]
        if pieces:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        books.append({
            "index": index,
            "short_book_title": row["short_book_title"],
            "publication_date": row["publication_date"],
            "url": row["url"],
            "char_start": start,
            "char_end": cursor,
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        if cursor >= args.min_chars:
            break

    if cursor < args.min_chars:
        raise RuntimeError(
            f"split insufficiente: {cursor:,} < {args.min_chars:,} caratteri")

    payload = "".join(pieces)
    out = Path(args.out)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "split": args.split,
        "separator": "\\n\\n",
        "chars": len(payload),
        "sha256": digest,
        "books": books,
    }
    atomic_write(out, payload)
    manifest_path = out.with_name(f"{out.name}.manifest.json")
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{out}: {len(books)} libri, {len(payload):,} caratteri, "
        f"sha256={digest}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
