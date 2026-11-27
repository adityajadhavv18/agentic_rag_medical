"""Park's running headers, read with Apple Vision from the top strip of every kept page ($0, local, cached).

    uv run ingest/park_headers.py      # writes data/clean/toc_raw/park_headers.json {pdf_page: header text}

Park's prints "<page> <CHAPTER TITLE>" at the top of even pages and "<SECTION TITLE> <page>" on odd pages. olmOCR
leaves running headers out, and its plain-text title search can hit a page that only mentions a chapter title, so the
headers are the reliable way to place Park's chapters (and they name sections too).
"""

import json
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_local import apple_vision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "clean" / "toc_raw" / "park_headers.json"

if __name__ == "__main__":
    book = next(b for b in json.loads((ROOT / "manifest.json").read_text())["books"] if b["book_id"] == "park_psm")
    kept = sorted(p for r in ("body", "index", "extra") for p in book["roles"].get(r, []))
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    doc = pymupdf.open(ROOT / "corpus" / "K.pdf")
    for i, p in enumerate(kept):
        if str(p) in done:
            continue
        page = doc[p - 1]
        r = page.rect
        strip = page.get_pixmap(dpi=200, clip=pymupdf.Rect(0, 0, r.width, r.height * 0.09)).tobytes("png")
        done[str(p)] = apple_vision(strip).replace("\n", " | ")
        if i % 100 == 0:
            OUT.write_text(json.dumps(done, indent=0))
    OUT.write_text(json.dumps(done, indent=0))
    print(f"headers read for {len(done)} pages")
