"""Timing sample for the full OCR run: 24 random kept pages from the two scanned books (seed 42), in proportion to
their size. Reads the books and manifest.json, writes data/timing/timing_pages.pdf + timing_pages.json.

    uv run ingest/timing_pages.py

Kept = roles body + index + extra (blank, drop, toc and duplicate pages are never OCR'd). The 12 Phase 1 test pages
are left out, so the sample is fresh. Pages are named P01..P24 in sample order.
"""

import json
import random
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "timing"
N, SEED = 24, 42
BOOKS = {"park_psm": "K.pdf", "khurana_ophthal": "A.pdf"}


def kept_pages(book: dict) -> list[int]:
    return sorted(p for role in ("body", "index", "extra") for p in book["roles"].get(role, []))


if __name__ == "__main__":
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    test = {(r["file"], r["pdf_page"]) for r in json.loads((ROOT / "data/phase1/phase1_pages.json").read_text())}
    pools = {bid: [p for p in kept_pages(manifest[bid]) if (f, p) not in test] for bid, f in BOOKS.items()}
    total = sum(len(v) for v in pools.values())
    n_park = round(N * len(pools["park_psm"]) / total)
    rng = random.Random(SEED)
    picks = [("park_psm", p) for p in sorted(rng.sample(pools["park_psm"], n_park))] + \
            [("khurana_ophthal", p) for p in sorted(rng.sample(pools["khurana_ophthal"], N - n_park))]
    OUT.mkdir(parents=True, exist_ok=True)
    out, rows = pymupdf.open(), []
    for k, (bid, p) in enumerate(picks, start=1):
        out.insert_pdf(pymupdf.open(ROOT / "corpus" / BOOKS[bid]), from_page=p - 1, to_page=p - 1)
        rows.append({"page": f"P{k:02d}", "book_id": bid, "pdf_page": p})
    out.save(OUT / "timing_pages.pdf", garbage=4, deflate=True)
    (OUT / "timing_pages.json").write_text(json.dumps(rows, indent=1))
    print(f"kept pages: park {len(pools['park_psm'])}, khurana {len(pools['khurana_ophthal'])} (test pages excluded)")
    print(f"sample: {n_park} Park's + {N - n_park} Khurana -> data/timing/timing_pages.pdf "
          f"({(OUT / 'timing_pages.pdf').stat().st_size / 1e6:.1f} MB)")
    print(" ".join(f"{r['page']}={r['book_id'][:4]}:{r['pdf_page']}" for r in rows))
