"""Phase 1: score every OCR engine on the 12 test pages against data/phase1/answer_key.json ($0, no LLM).

    uv run ingest/phase1_score.py            # scores every folder in data/phase1/results/

Engines are folders data/phase1/results/<engine>/T01.md ... T12.md (+ optional log.json with seconds per page).
Unzip Kaggle's phase1_results.zip so its paddle/ and olmocr/ folders land in data/phase1/results/.

Checks (decision rule fixed before looking at any output):
  terms   each answer-key term must appear in the page's text. Matching ignores case, spaces, quote and dash
          styles, and markdown/html formatting, so only real reading errors count.
  rows    table cells that must sit in the SAME row: a markdown table line, an html <tr>, or one plain text line.
  absent  text that must NOT appear (the handwritten note on T3).
Winner: most terms right. Within 2 terms of each other → more rows right wins. Speed only breaks a tie after that.
"""

import html
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "data" / "phase1"
RESULTS = PHASE1 / "results"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", html.unescape(s)).lower()
    s = re.sub(r"[‘’`´]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‐-―−]", "-", s)
    s = re.sub(r"</?[a-z][a-z0-9]*\b[^<>]*>", " ", s)  # real html tags only (a bare "<6/60" is text, not a tag)
    s = re.sub(r"[*_#|]", "", s)            # markdown emphasis, headings, table pipes
    s = re.sub(r"\\", "", s)                # latex-style escapes like 50\%
    return re.sub(r"\s+", "", s)


def rows_of(text: str) -> list[str]:
    """Candidate table rows: html <tr> blocks, markdown table lines, and plain lines."""
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I)
    return trs + text.splitlines()


def score_page(text: str, key: dict) -> dict:
    flat = norm(text)
    terms = {t: norm(t) in flat for t in key["terms"]}
    rows_norm = [norm(r) for r in rows_of(text)]
    rows = {" | ".join(r): any(all(norm(c) in row for c in r) for row in rows_norm) for r in key.get("rows", [])}
    absent = {a: norm(a) in flat for a in key.get("absent", [])}
    return {"terms": terms, "rows": rows, "leaked": absent}


def score_engine(folder: Path, key: dict) -> dict:
    log = json.loads((folder / "log.json").read_text()) if (folder / "log.json").exists() else {"pages": {}}
    pages, tot = {}, {"terms": 0, "rows": 0, "leaked": 0, "missing_pages": 0, "seconds": []}
    for name, k in key.items():
        if name.startswith("_"):
            continue
        f = folder / f"T{int(name[1:]):02d}.md"
        text = f.read_text() if f.exists() else ""
        if not text.strip():
            tot["missing_pages"] += 1
        s = pages[name] = score_page(text, k)
        tot["terms"] += sum(s["terms"].values())
        tot["rows"] += sum(s["rows"].values())
        tot["leaked"] += sum(s["leaked"].values())
        sec = log["pages"].get(f"T{int(name[1:]):02d}", {}).get("seconds")
        if sec is not None:
            tot["seconds"].append(sec)
    tot["avg_seconds"] = round(sum(tot["seconds"]) / len(tot["seconds"]), 2) if tot["seconds"] else None
    del tot["seconds"]
    return {"total": tot, "pages": pages}


if __name__ == "__main__":
    key = json.loads((PHASE1 / "answer_key.json").read_text())
    n_terms = sum(len(v["terms"]) for k, v in key.items() if not k.startswith("_"))
    n_rows = sum(len(v.get("rows", [])) for k, v in key.items() if not k.startswith("_"))
    engines = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    out = {e.name: score_engine(e, key) for e in engines}
    (RESULTS / "score.json").write_text(json.dumps(out, indent=1))

    pages = [k for k in key if not k.startswith("_")]
    print(f"{'engine':14} {'terms/' + str(n_terms):>10} {'rows/' + str(n_rows):>8} {'leaked':>7} {'s/page':>7}   per page (terms right)")
    for name, r in sorted(out.items(), key=lambda x: (-x[1]["total"]["terms"], -x[1]["total"]["rows"])):
        t = r["total"]
        per = " ".join(f"{sum(r['pages'][p]['terms'].values()):>2}" for p in pages)
        print(f"{name:14} {t['terms']:>10} {t['rows']:>8} {t['leaked']:>7} {str(t['avg_seconds']):>7}   {per}")
    print(f"{'':14} {'':>10} {'':>8} {'':>7} {'':>7}   " + " ".join(f"{p[1:]:>2}" for p in pages))
    print(f"{'max':14} {'':>10} {'':>8} {'':>7} {'':>7}   " +
          " ".join(f"{len(key[p]['terms']):>2}" for p in pages))
