"""Phase 1: cut the 12 hard test pages into one small PDF for the OCR bake-off. Reads the books, writes data/phase1/.

    uv run ingest/phase1_pages.py

Each page was picked for one difficulty (see WHY). The PDF goes to Kaggle as a PRIVATE dataset (copyrighted pages).
"""

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "phase1"
B = "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf"

PAGES = [  # (file, pdf page, why)
    ("K.pdf", 300, "Park's: dense two-column text; existing hidden text is poor"),
    ("K.pdf", 600, "Park's: curved camera photo from the no-text stretch (513-812)"),
    ("K.pdf", 633, "Park's: handwritten note + table"),
    ("K.pdf", 691, "Park's: pen underlines over a two-column list"),
    ("K.pdf", 808, "Park's: large numeric table, no-text stretch"),
    ("K.pdf", 1010, "Park's: chapter opening + flowchart, blurry photo"),
    ("A.pdf", 12, "Khurana: labelled anatomy diagram"),
    ("A.pdf", 335, "Khurana: dense clinical text"),
    ("A.pdf", 502, "Khurana: figure + definitions"),
    ("A.pdf", 643, "Khurana: back-of-book index (kept by decision)"),
    ("P_L_Dhingra.pdf", 39, "Dhingra (digital): table - control page"),
    (B, 118, "Bardale (digital): tables + bone figures - control page"),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    out, rows = pymupdf.open(), []
    for k, (f, p, why) in enumerate(PAGES, start=1):
        src = pymupdf.open(ROOT / "corpus" / f)
        out.insert_pdf(src, from_page=p - 1, to_page=p - 1)
        rows.append({"test_page": k, "file": f, "pdf_page": p, "why": why})
    out.save(OUT / "phase1_pages.pdf", garbage=4, deflate=True)
    (OUT / "phase1_pages.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} pages, {(OUT / 'phase1_pages.pdf').stat().st_size / 1e6:.1f} MB")
