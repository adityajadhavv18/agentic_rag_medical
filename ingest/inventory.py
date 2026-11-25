"""Phase 0 inventory: checksum, page count and the PDF-page → printed-page map for each book. Reads only.

    uv run ingest/inventory.py            # print the findings
    uv run ingest/inventory.py --write    # also write data/inventory_raw.json

How the page map is found: on each page, a bare number in the first or last 3 text lines is taken as the printed
page number, and offset = pdf_page - printed. Runs of pages that agree on one offset are reported as segments.
Pages without a text layer (Park's 513–812) give no number; their offset is filled in later from the scans.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
CORPUS, DATA = ROOT / "corpus", ROOT / "data"
NUM = re.compile(r"^\d{1,4}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def printed_number(page: pymupdf.Page) -> int | None:
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    for l in lines[:3] + lines[-3:]:
        if NUM.match(l):
            return int(l)
    return None


def segments(offsets: dict[int, int]) -> list[dict]:
    """Group pages into runs with the same offset; a run must hold ≥3 agreeing pages (noise filter)."""
    out, cur = [], None
    for pg in sorted(offsets):
        off = offsets[pg]
        if cur and cur["offset"] == off:
            cur["last_pdf"], cur["votes"] = pg, cur["votes"] + 1
        else:
            if cur and cur["votes"] >= 3:
                out.append(cur)
            cur = {"offset": off, "first_pdf": pg, "last_pdf": pg, "votes": 1}
    if cur and cur["votes"] >= 3:
        out.append(cur)
    return out


def inventory(pdf: Path) -> dict:
    doc = pymupdf.open(pdf)
    offsets, no_text = {}, 0
    for i, page in enumerate(doc, start=1):
        if len(page.get_text().split()) < 40:
            no_text += 1
        n = printed_number(page)
        if n is not None and 0 <= i - n <= 60:  # front matter is never longer than ~60 pages
            offsets[i] = i - n
    return {
        "file": pdf.name,
        "sha256": sha256(pdf),
        "bytes": pdf.stat().st_size,
        "pdf_pages": doc.page_count,
        "pages_without_text": no_text,
        "pages_with_number": len(offsets),
        "main_offset": Counter(offsets.values()).most_common(1)[0][0] if offsets else None,
        "offset_segments": segments(offsets),
    }


if __name__ == "__main__":
    rows = [inventory(p) for p in sorted(CORPUS.glob("*.pdf"))]
    for r in rows:
        print(f"\n{r['file']}  pages {r['pdf_pages']}  numbered {r['pages_with_number']}  "
              f"main offset {r['main_offset']}  sha {r['sha256'][:12]}")
        for s in r["offset_segments"]:
            print(f"   pdf {s['first_pdf']:>4}-{s['last_pdf']:<4} offset {s['offset']:>3}  ({s['votes']} pages agree)")
    if "--write" in sys.argv:
        DATA.mkdir(exist_ok=True)
        (DATA / "inventory_raw.json").write_text(json.dumps(rows, indent=1))
        print("\nwrote data/inventory_raw.json")
