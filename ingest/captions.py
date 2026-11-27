"""Phase 3, step B: find figure captions the OCR dropped, so the cleaning step can put them back ($0, local).

    caffeinate -i uv run ingest/captions.py     # writes data/clean/captions.json

olmOCR dropped ~1 in 5 figure captions (Phase 2 check). A caption is read from a second source and kept for
insertion only if the OCR text does not already contain it (fewer than KEEP_SHARE of its word pairs present):
  - Dhingra, Bardale: the PDF's own exact text ("Figure 65.3. ..." / "Fig. 4.21: ...");
  - Park's, Khurana: Apple Vision (macOS, free) on pages that can hold a figure (OCR shows a picture placeholder,
    mentions Fig/Figure, or olmOCR flagged a diagram), cached in data/clean/apple/<book>/p0001.txt.
Caption lines: the label ("FIG. 1", "Fig. 12.14", "Figure 65.3") plus the text after it on that line; Park's prints
the title on the next line, so a bare label takes the next line. Long captions are cut at 300 characters.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_local import apple_vision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OCR, OUT = ROOT / "data" / "ocr", ROOT / "data" / "clean" / "captions.json"
APPLE = ROOT / "data" / "clean" / "apple"
FILES = {"park_psm": "K.pdf", "khurana_ophthal": "A.pdf", "dhingra_ent": "P_L_Dhingra.pdf",
         "bardale_fmt": "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf"}
DIGITAL = {"dhingra_ent", "bardale_fmt"}
LABEL = re.compile(r"^\s*(FIG(?:URE)?\.?|Fig(?:ure)?\.?)\s*(\d+(?:\.\d+)?[A-Za-z]?)\s*[:.]?\s*(.*)$")
KEEP_SHARE, MAX_LEN = 0.6, 300


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", unicodedata.normalize("NFKC", s).lower())


def bigrams(t: list[str]) -> set:
    return set(zip(t, t[1:]))


def captions_in(text: str, digital: bool) -> list[dict]:
    lines = [l.strip() for l in unicodedata.normalize("NFKC", text).splitlines()]
    stop = re.compile(r"^(TABLE|Table|FIG|Fig|Figure|[•▪●·*-])")
    out = []
    for i, line in enumerate(lines):
        m = LABEL.match(line)
        if not m:
            continue
        body = m.group(3).strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if len(body) < 3:  # Park's: "FIG. 1" then the title on the next line - only if it looks like a short title
            body = nxt if nxt[:1].isupper() and len(nxt) <= 80 and not stop.match(nxt) else ""
        elif digital and nxt and not stop.match(nxt) and not body.endswith((".", ")")):
            body = f"{body} {nxt}"  # a caption that wraps onto a second line (exact text only)
        body = re.split(r"\s(?:FIG\.?|Fig\.|Figure|TABLE|Table)\s*\d", body)[0].strip()  # cut at the next label
        if len(toks(body)) >= 2:
            out.append({"label": f"Figure {m.group(2)}", "text": body[:MAX_LEN]})
    return out


def figure_pages(bid: str) -> list[int]:
    pages = []
    for f in sorted((OCR / bid).glob("p*.md")):
        t = f.read_text()
        meta = json.loads(f.with_suffix(".json").read_text())
        if "![" in t or re.search(r"\bFIG\b|\bFig\b|Figure", t) or meta.get("is_diagram"):
            pages.append(int(f.stem[1:]))
    return pages


def second_source(bid: str, doc, p: int) -> str:
    if bid in DIGITAL:
        return doc[p - 1].get_text()
    cache = APPLE / bid / f"p{p:04d}.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(apple_vision(doc[p - 1].get_pixmap(dpi=200).tobytes("png")))
    return cache.read_text()


if __name__ == "__main__":
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    result = {}
    for bid, pdf in FILES.items():
        doc = pymupdf.open(ROOT / "corpus" / pdf)
        kept = sorted(p for r in ("body", "index", "extra") for p in manifest[bid]["roles"].get(r, []))
        pages = kept if bid in DIGITAL else figure_pages(bid)
        found = present = 0
        missing = {}
        for p in pages:
            ocr_b = bigrams(toks((OCR / bid / f"p{p:04d}.md").read_text()))
            seen = set()
            for cap in captions_in(second_source(bid, doc, p), bid in DIGITAL):
                key = (cap["label"], cap["text"][:40])
                if key in seen:
                    continue
                seen.add(key)
                found += 1
                cb = bigrams(toks(cap["text"]))
                if cb and len(cb & ocr_b) / len(cb) >= KEEP_SHARE:
                    present += 1
                else:
                    missing.setdefault(str(p), []).append({**cap, "source": "text_layer" if bid in DIGITAL
                                                           else "apple_vision"})
        n_missing = sum(len(v) for v in missing.values())
        result[bid] = {"pages_read": len(pages), "captions_found": found, "already_in_ocr": present,
                       "to_insert": n_missing, "pages": missing}
        print(f"{bid:16} read {len(pages):4} pages: captions {found:4}, already in OCR {present:4} "
              f"({100 * present / max(found, 1):.0f}%), to insert {n_missing:4} on {len(missing)} pages", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote data/clean/captions.json")
