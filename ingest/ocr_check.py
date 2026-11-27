"""Checks after the full OCR run ($0, on the Mac). Run after `modal_full_ocr.py fetch`.

    uv run ingest/ocr_check.py

1. Complete: every kept page (manifest roles body/index/extra) has p0001.md + p0001.json in data/ocr/<book_id>/.
2. Flags: pages that never finished (hit the token limit twice), pages olmOCR thinks are rotated, and suspiciously
   short pages (under 200 characters for a body page: blank-ish or a lost page).
3. Proofreading Dhingra and Bardale against their exact digital text (the PDFs' own text layer):
   - words: share of the text layer's words (4+ letters) that the OCR also has;
   - numbers: share of the text layer's numbers (2+ digits, or decimals like 4.49) that the OCR also has.
   Before comparing, the text layer is made fair: unicode-normalised (ligatures), words split by a line-end hyphen
   rejoined ("compres-/sion"), and running headers / page numbers removed (olmOCR leaves them out on purpose): a line
   in the top or bottom 3 lines of a page that repeats on 3+ pages of the book is a header, and a bare number there is
   a page number. Pages under WORDS_FLAG / NUMBERS_FLAG are listed for a look.
   This also measures olmOCR's real accuracy on ~1,050 pages with a known exact answer (for the paper).
Writes data/ocr/check.json.
"""

import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OCR = ROOT / "data" / "ocr"
BOOK_FILES = {"park_psm": "K.pdf", "khurana_ophthal": "A.pdf", "dhingra_ent": "P_L_Dhingra.pdf",
              "bardale_fmt": "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf"}
DIGITAL = ("dhingra_ent", "bardale_fmt")
WORDS_FLAG, NUMBERS_FLAG, SHORT = 0.95, 0.90, 200
JUNK = re.compile(r"mebooksfree\.com", re.I)


def body_text(md: str) -> str:
    md = re.sub(r"\A---.*?---", " ", md, flags=re.S)  # olmOCR's YAML front matter
    return re.sub(r"</?[a-zA-Z][^<>]*>", " ", md)  # html tags of tables


def layer_lines(page) -> list[str]:
    text = unicodedata.normalize("NFKC", JUNK.sub(" ", page.get_text()))
    return [l.strip() for l in text.splitlines() if l.strip()]


def header_set(doc, pages: list[int]) -> set[str]:
    """Lines in the top/bottom 3 lines of a page that repeat on 3+ pages = running headers/footers."""
    c = Counter()
    for p in pages:
        lines = layer_lines(doc[p - 1])
        c.update(set(lines[:3] + lines[-3:]))
    return {l for l, n in c.items() if n >= 3}


def fair_layer(page, headers: set[str]) -> str:
    lines = layer_lines(page)
    zone = set(range(3)) | set(range(len(lines) - 3, len(lines)))
    keep = [l for i, l in enumerate(lines)
            if not (i in zone and (l in headers or re.fullmatch(r"[0-9ivxlc]{1,4}", l, flags=re.I)))]
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", "\n".join(keep))  # rejoin compres-/sion


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+\.\d+|\d{2,}", text))


if __name__ == "__main__":
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    report = {}
    for bid, pdf in BOOK_FILES.items():
        roles = manifest[bid]["roles"]
        kept = sorted(p for r in ("body", "index", "extra") for p in roles.get(r, []))
        d = OCR / bid
        missing = [p for p in kept if not (d / f"p{p:04d}.md").exists() or not (d / f"p{p:04d}.json").exists()]
        metas = {p: json.loads((d / f"p{p:04d}.json").read_text()) for p in kept if p not in missing}
        texts = {p: (d / f"p{p:04d}.md").read_text() for p in kept if p not in missing}
        body = set(roles.get("body", []))
        r = {"kept": len(kept), "missing": missing,
             "not_finished": [p for p, m in metas.items() if m.get("finish") != "stop"],
             "rotation_flag": [p for p, m in metas.items() if m.get("is_rotation_valid") is False],
             "short": [p for p, t in texts.items() if p in body and len(body_text(t).strip()) < SHORT],
             "out_tokens": sum(m.get("out_tokens", 0) for m in metas.values()),
             "retried": sum(m.get("attempts", 1) > 1 for m in metas.values())}
        if bid in DIGITAL:
            doc = pymupdf.open(ROOT / "corpus" / pdf)
            headers = header_set(doc, list(texts))
            wr, nr, low = {}, {}, []
            for p, t in texts.items():
                layer = fair_layer(doc[p - 1], headers)
                ocr = unicodedata.normalize("NFKC", body_text(t))
                lw, ow, ln, on = words(layer), words(ocr), numbers(layer), numbers(ocr)
                wr[p] = len(lw & ow) / len(lw) if lw else 1.0
                nr[p] = len(ln & on) / len(ln) if ln else 1.0
                if wr[p] < WORDS_FLAG or nr[p] < NUMBERS_FLAG:
                    low.append({"page": p, "words": round(wr[p], 3), "numbers": round(nr[p], 3),
                                "numbers_missing": sorted(ln - on)[:15]})
            r["proof"] = {"median_words": round(statistics.median(wr.values()), 3),
                          "median_numbers": round(statistics.median(nr.values()), 3),
                          "mean_words": round(statistics.mean(wr.values()), 3),
                          "mean_numbers": round(statistics.mean(nr.values()), 3),
                          "flagged": sorted(low, key=lambda x: (x["words"], x["numbers"]))}
        report[bid] = r
        print(f"{bid:16} kept {len(kept):5}  missing {len(missing):3}  not finished {len(r['not_finished']):3}  "
              f"rotation? {len(r['rotation_flag']):3}  short {len(r['short']):3}  retried {r['retried']:3}  "
              f"tokens {r['out_tokens']:,}")
        if "proof" in r:
            pr = r["proof"]
            print(f"{'':16} proofread vs exact text: words median {pr['median_words']} (mean {pr['mean_words']}), "
                  f"numbers median {pr['median_numbers']} (mean {pr['mean_numbers']}), flagged {len(pr['flagged'])}")
    (OCR / "check.json").write_text(json.dumps(report, indent=1))
    print("wrote data/ocr/check.json")
