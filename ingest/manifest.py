"""Phase 0: build and check manifest.json, the single source of truth about the 4 books.

    uv run ingest/manifest.py          # build, check, print a summary, write manifest.json

Everything in BOOKS was verified by eye on the page images (2026-11-25), helped by inventory.py (page numbers from
the text layer) and duplicates.py (repeated and blank pages). This script adds checksums and page counts, gives every
PDF page exactly one role, and checks that the printed page numbers of the kept pages run 1, 2, 3 ... with no gaps
except the pages listed as missing from the PDF.

Roles:
  body       chapters and appendices: OCR'd, chunked, indexed
  index      the book's back-of-book index: kept (his decision), chunked as type=index
  extra      kept reference pages outside chapters (Park's abbreviation list and CBME competency table)
  toc        table of contents: NOT indexed; used in Phase 3 as the checklist of chapters
  drop       cover, title, copyright, dedication, preface, acknowledgements (no medical content)
  blank      blank page inside the book
  duplicate  the same page photographed twice; the other copy is kept

The manifest holds no book text, only page numbers and facts, so it can go to git.
"""

import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

BOOKS = {
    "K.pdf": {
        "book_id": "park_psm", "title": "Park's Textbook of Preventive and Social Medicine", "author": "K. Park",
        "subject": "Community Medicine (PSM)", "edition": "28th", "publisher": "Banarsidas Bhanot",
        "source_type": "camera photos with a poor hidden OCR layer (missing on pdf 513-812)",
        # pdf page -> printed page: printed = pdf - offset, per segment
        "page_map": [(13, 635, 12), (636, 691, 11), (693, 1012, 12), (1014, 1073, 13)],
        "roles": {"drop": [(1, 6)], "toc": [(7, 8)], "extra": [(9, 12), (1062, 1063)],
                  "body": [(13, 691), (693, 1012), (1014, 1061)], "index": [(1064, 1073)],
                  "duplicate": [(692, 692), (1013, 1013)]},
        "duplicates": {"692": 691, "1013": 1010},
        "missing_printed_pages": [624],
        "notes": ["printed p.624 (low birth weight) is not in the PDF: pdf 635 = p.623, pdf 636 = p.625",
                  "pdf 692 repeats pdf 691 (p.680); pdf 1013 repeats pdf 1010 (p.998, sharper photo)",
                  "watermark 'Edited by:- KRANSHMAHA' on every page; 'Made with Xodo' on the cover",
                  "handwritten notes and underlines on some photos (e.g. pdf 633, 691): OCR must ignore them",
                  "pages are curved, skewed camera photos",
                  "pdf 9-12 = CBME competency table (codes CM1.1 ...); pdf 1062-1063 = abbreviations list"],
    },
    "A.pdf": {
        "book_id": "khurana_ophthal", "title": "Comprehensive Ophthalmology", "author": "A K Khurana",
        "subject": "Ophthalmology", "edition": "7th", "publisher": None,
        "source_type": "scanned pages with an acceptable hidden OCR layer (missing on 16 pages)",
        "page_map": [(7, 668, 6)],
        "roles": {"drop": [(1, 2)], "toc": [(3, 4)], "blank": [(5, 6), (8, 8), (26, 26), (28, 28), (62, 62),
                                                                (64, 64), (466, 466), (468, 468), (492, 492)],
                  "body": [(7, 7), (9, 25), (27, 27), (29, 61), (63, 63), (65, 465), (467, 467), (469, 491),
                           (493, 640)],
                  "index": [(641, 668)]},
        "duplicates": {}, "missing_printed_pages": [],
        "notes": ["no preface in the PDF", "blank pages show faint bleed-through only",
                  "Appendix I (clinical case sheet) sits inside chapter 23, kept as body"],
    },
    "P_L_Dhingra.pdf": {
        "book_id": "dhingra_ent", "title": "Diseases of Ear, Nose and Throat & Head and Neck Surgery",
        "author": "PL Dhingra", "subject": "ENT", "edition": "7th", "publisher": "Elsevier (RELX India)",
        "source_type": "digital text (24 image pages)",
        "page_map": [(16, 555, 15)],
        "roles": {"drop": [(1, 2), (4, 6), (8, 8), (10, 10)], "toc": [(12, 15)],
                  "body": [(16, 542)], "index": [(544, 555)]},
        "blank_text": "page left intentionally blank",  # blanks inside the book are found by this text
        "duplicates": {}, "missing_printed_pages": [],
        "notes": ["watermark 'mebooksfree.com' on every page",
                  "appendices pdf 522-542: I 'Memorable Nuggets for Rapid Review', II 'Instruments'"],
    },
    "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf": {
        "book_id": "bardale_fmt", "title": "Principles of Forensic Medicine and Toxicology", "author": "Rajesh Bardale",
        "subject": "Forensic Medicine and Toxicology", "edition": "© 2011 (edition not printed on the pages read)", "publisher": "Jaypee Brothers",
        "source_type": "digital text (1 image page)",
        "page_map": [(20, 583, 19)],
        "roles": {"drop": [(1, 3), (5, 6), (8, 8), (10, 10)], "toc": [(12, 19)],
                  "blank": [(4, 4), (7, 7), (9, 9), (11, 11), (21, 21), (429, 429), (431, 431), (573, 573)],
                  "body": [(20, 20), (22, 428), (430, 430), (432, 572)], "index": [(574, 583)]},
        "duplicates": {}, "missing_printed_pages": [],
        "notes": ["copyright 2011; legal sections refer to IPC/CrPC as of 2011 (India replaced them with BNS/BNSS "
                  "in 2024), so answers about law sections may be out of date"],
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expand(ranges) -> list[int]:
    return [p for a, b in ranges for p in range(a, b + 1)]


def printed(pdf_page: int, page_map) -> int | None:
    for a, b, off in page_map:
        if a <= pdf_page <= b:
            return pdf_page - off
    return None


def build(name: str, spec: dict) -> dict:
    doc = pymupdf.open(CORPUS / name)
    n = doc.page_count
    roles = {role: expand(r) for role, r in spec["roles"].items()}
    if "blank_text" in spec:  # Dhingra: blank pages are marked in the text; take them out of body
        blank = [p for p in roles["body"] + roles.get("index", [])
                 if spec["blank_text"] in doc[p - 1].get_text().lower()]
        roles["blank"] = sorted(set(roles.get("blank", [])) | set(blank))
        roles["body"] = [p for p in roles["body"] if p not in blank]
        roles["index"] = [p for p in roles["index"] if p not in blank]
        roles["drop"] = [p for p in roles["drop"] if p not in blank]
    # Dhingra/Bardale front-matter blanks not listed elsewhere count as blank
    seen = [p for ps in roles.values() for p in ps]
    unassigned = sorted(set(range(1, n + 1)) - set(seen))
    if spec.get("blank_text"):
        roles["blank"] = sorted(set(roles["blank"]) | set(unassigned))
        unassigned = []

    # --- checks ---
    seen = [p for ps in roles.values() for p in ps]
    assert not unassigned, f"{name}: pages with no role {unassigned}"
    assert len(seen) == len(set(seen)) == n, f"{name}: a page has two roles or is missing ({len(seen)} vs {n})"
    kept = sorted(roles["body"] + roles["index"] + roles.get("blank", []) + roles.get("extra", []))
    kept = [p for p in kept if printed(p, spec["page_map"]) is not None]
    nums = [printed(p, spec["page_map"]) for p in kept]
    gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b != a + 1]
    expected = [(m - 1, m + 1) for m in spec["missing_printed_pages"]]
    assert gaps == expected, f"{name}: printed-page gaps {gaps}, expected {expected}"

    return {
        "file": name, **{k: v for k, v in spec.items() if k not in ("roles", "page_map", "blank_text")},
        "sha256": sha256(CORPUS / name), "bytes": (CORPUS / name).stat().st_size, "pdf_pages": n,
        "page_map": [{"first_pdf": a, "last_pdf": b, "offset": o} for a, b, o in spec["page_map"]],
        "printed_range": [nums[0], nums[-1]],
        "role_counts": {r: len(ps) for r, ps in roles.items()},
        "roles": {r: sorted(ps) for r, ps in roles.items()},
    }


if __name__ == "__main__":
    out = {"version": "corpus_v1", "built": "2026-11-25", "books": [build(n, s) for n, s in BOOKS.items()]}
    (ROOT / "manifest.json").write_text(json.dumps(out, indent=1))
    for b in out["books"]:
        print(f"{b['book_id']:16} pdf {b['pdf_pages']:5}  printed {b['printed_range']}  {b['role_counts']}")
    tot = {}
    for b in out["books"]:
        for r, c in b["role_counts"].items():
            tot[r] = tot.get(r, 0) + c
    print("TOTAL", tot, "\nall checks passed, wrote manifest.json")
