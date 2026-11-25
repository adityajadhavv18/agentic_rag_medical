"""Phase 0: find blank pages and pages that appear twice inside one book (a page scanned twice shifts the page map
and would make duplicate chunks). Reads only.

    uv run ingest/duplicates.py            # print blanks and suspected duplicate pairs
    uv run ingest/duplicates.py --write    # also write data/duplicates.json

Blank: fewer than NO_TEXT words (watermarks removed) AND the picture is almost flat (little contrast) or the page
says "page left intentionally blank". Blanks are listed, never paired.

Duplicate, two signals (some pages have no text layer):
  - text: word-set overlap (Jaccard) >= TEXT_SIM between two pages that each have >= 60 words. OCR noise makes the
    two copies differ a little, so exact matching misses them (Park's 1010 vs 1013).
  - picture: page rendered tiny and grey, squeezed to a 16x16 "average hash"; within HAMMING bits counts only when
    neither page has a text layer (< NO_TEXT words), e.g. Park's 513–812.
"""

import json
import re
import statistics
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
CORPUS, DATA = ROOT / "corpus", ROOT / "data"
HAMMING, TEXT_SIM, MIN_WORDS, NO_TEXT, FLAT = 10, 0.6, 60, 10, 12
JUNK = re.compile(r"page left intentionally blank|mebooksfree\.com|edited by:- kranshmaha|<br>", re.I)


def picture(page: pymupdf.Page) -> tuple[int, float]:
    """(16x16 average hash, contrast) of the page picture."""
    zoom = 32 / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
    vals = list(pymupdf.Pixmap(pix, 16, 16, None).samples)
    mean = sum(vals) / len(vals)
    return sum(1 << i for i, v in enumerate(vals) if v > mean), statistics.pstdev(vals)


def words(page: pymupdf.Page) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", JUNK.sub(" ", page.get_text()).lower()))


def scan(pdf: Path) -> dict:
    doc = pymupdf.open(pdf)
    rows = []
    for i, page in enumerate(doc, start=1):
        h, contrast = picture(page)
        w = words(page)
        rows.append({"page": i, "hash": h, "flat": contrast < FLAT, "words": w})
    blanks = [r["page"] for r in rows if len(r["words"]) < NO_TEXT and
              (r["flat"] or "blank" in doc[r["page"] - 1].get_text().lower())]
    live = [r for r in rows if r["page"] not in blanks]
    pairs = []
    for x, a in enumerate(live):
        for b in live[x + 1:]:
            if len(a["words"]) >= MIN_WORDS and len(b["words"]) >= MIN_WORDS:
                sim = len(a["words"] & b["words"]) / len(a["words"] | b["words"])
                if sim >= TEXT_SIM:
                    pairs.append({"pdf_page": a["page"], "duplicate_pdf_page": b["page"], "signal": f"text {sim:.2f}"})
            elif len(a["words"]) < NO_TEXT and len(b["words"]) < NO_TEXT:
                d = bin(a["hash"] ^ b["hash"]).count("1")
                if d <= HAMMING:
                    pairs.append({"pdf_page": a["page"], "duplicate_pdf_page": b["page"], "signal": f"picture {d}"})
    return {"blank_pages": blanks, "duplicates": pairs}


if __name__ == "__main__":
    out = {}
    for pdf in sorted(CORPUS.glob("*.pdf")):
        r = out[pdf.name] = scan(pdf)
        print(f"\n{pdf.name}: {len(r['blank_pages'])} blank, {len(r['duplicates'])} suspected duplicate pairs")
        print(f"   blank: {r['blank_pages']}")
        for p in r["duplicates"]:
            print(f"   {p['pdf_page']:>5} = {p['duplicate_pdf_page']:<5} ({p['signal']})")
    if "--write" in sys.argv:
        DATA.mkdir(exist_ok=True)
        (DATA / "duplicates.json").write_text(json.dumps(out, indent=1))
        print("\nwrote data/duplicates.json")
