"""Phase 3, step C: turn the 2,768 OCR pages into clean, continuous, labelled blocks ($0, local).

    uv run ingest/clean.py        # needs data/clean/toc.json (toc.py) and data/clean/captions.json (captions.py)

Per page (step 1):
  - drop olmOCR's YAML front matter (its flags stay as metadata);
  - picture placeholders ![alt](page_..png) -> [Figure: alt]  (the fake file name goes; alt is often the caption);
  - inline math \\( .. \\) -> readable symbols (\\mu -> µ, \\leq -> ≤, ^3 -> ³, \\frac{1}{2} -> 1/2, ...);
  - bare page-number lines removed; HTML tables kept whole (they keep merged cells).
Captions (step 2): captions the OCR dropped (data/clean/captions.json) are added to their page as figure blocks.
Joining (step 3): a paragraph that ends mid-sentence at the bottom of a page is joined with the paragraph that
  continues it at the top of the next page (line-end hyphens rejoined); figures/tables between them stay after it.
Structure (step 4): chapter from data/clean/toc.json by PDF page; book section (Khurana/Dhingra/Bardale) from the
  contents; headings inside a chapter are detected from short heading-like lines (ALL CAPS -> level 2; numbered or
  Title Case -> level 3). For Dhingra and Bardale a heading must also be bold or larger than body text in the
  PDF's own fonts. Index pages -> type index; Park's competency table + abbreviations -> type extra.
Output (step 5): data/clean/blocks/<book_id>.jsonl (one block per paragraph / table / figure / heading) and
  data/clean/readable/<book_id>.md for reading by eye; checks -> data/clean/check.json.
"""

import html
import json
import random
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OCR, CLEAN = ROOT / "data" / "ocr", ROOT / "data" / "clean"
BOOKS = {"park_psm": ("Park's Textbook of Preventive and Social Medicine", "K.pdf"),
         "khurana_ophthal": ("Comprehensive Ophthalmology (Khurana)", "A.pdf"),
         "dhingra_ent": ("Diseases of Ear, Nose and Throat (Dhingra)", "P_L_Dhingra.pdf"),
         "bardale_fmt": ("Principles of Forensic Medicine and Toxicology (Bardale)",
                         "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf")}
DIGITAL = {"dhingra_ent", "bardale_fmt"}
PARK_EXTRA = {range(9, 13): "CBME competencies", range(1062, 1064): "Abbreviations"}

# ---- step 1: page cleaning ------------------------------------------------------------------------------------------
LATEX = {r"\mu": "µ", r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\times": "×", r"\pm": "±",
         r"\circ": "°", r"\degree": "°", r"\infty": "∞", r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←",
         r"\uparrow": "↑", r"\downarrow": "↓", r"\approx": "≈", r"\neq": "≠", r"\cdot": "·", r"\div": "÷",
         r"\%": "%", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\Delta": "Δ",
         r"\epsilon": "ε", r"\theta": "θ", r"\lambda": "λ", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ",
         r"\Sigma": "Σ", r"\tau": "τ", r"\phi": "φ", r"\chi": "χ", r"\omega": "ω", r"\Omega": "Ω", r"\sqrt": "√",
         r"\ldots": "…", r"\dots": "…", r"\quad": " ", r"\,": " ", r"\;": " ", r"\!": ""}
SUP, SUB = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"), str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")


def latex_to_text(m: re.Match) -> str:
    s = m.group(1)
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", s)  # 37^\circ -> 37°
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda x: x.group(1) + "\u0304", s)
    for k in sorted(LATEX, key=len, reverse=True):
        s = s.replace(k, LATEX[k])
    sup = lambda x: x.group(1).translate(SUP) if re.fullmatch(r"[0-9+-]+", x.group(1)) else f"^({x.group(1)})"
    sub = lambda x: x.group(1).translate(SUB) if re.fullmatch(r"[0-9+-]+", x.group(1)) else f"_({x.group(1)})"
    s = re.sub(r"\^\{([^{}]*)\}", sup, s)
    s = re.sub(r"\^([0-9A-Za-z+-])", sup, s)
    s = re.sub(r"_\{([^{}]*)\}", sub, s)
    s = re.sub(r"_([0-9A-Za-z])", sub, s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()


def clean_page(raw: str) -> str:
    t = re.sub(r"\A---.*?---\s*", "", raw, flags=re.S)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", lambda m: f"\n\n[Figure: {m.group(1).strip()}]\n\n" if m.group(1).strip()
               else "", t)
    t = re.sub(r"\\\((.+?)\\\)", latex_to_text, t, flags=re.S)
    t = re.sub(r"\\\[(.+?)\\\]", latex_to_text, t, flags=re.S)
    for k in sorted((k for k in LATEX if re.fullmatch(r"\\[A-Za-z]+", k)), key=len, reverse=True):
        t = re.sub(re.escape(k) + r"(?![A-Za-z])", LATEX[k], t)  # bare commands left outside \( \), e.g. in tables
    t = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", t)  # stray page numbers
    return t.strip()


# ---- step 3/4 helpers ------------------------------------------------------------------------------------------------
def page_items(text: str) -> list[dict]:
    items, pos = [], 0
    for m in re.finditer(r"<table\b.*?</table>", text, flags=re.S | re.I):
        items += paragraphs(text[pos:m.start()])
        items.append({"type": "table", "text": m.group(0).strip()})
        pos = m.end()
    return items + paragraphs(text[pos:])


def paragraphs(s: str) -> list[dict]:
    out = []
    for para in re.split(r"\n\s*\n", s):
        para = re.sub(r"<br\s*/?>", "\n", para).strip()  # olmOCR's line breaks in lists outside tables
        if para:
            out.append({"type": "figure" if re.fullmatch(r"\[Figure[^\]]*\]", para, flags=re.S) else "text",
                        "text": para})
    return out


def is_open(text: str) -> bool:
    """Ends mid-sentence (no closing punctuation)."""
    return not re.search(r"[.!?:;)\]\"'”’]\s*$", text)


def continues(text: str) -> bool:
    return bool(re.match(r"^[a-z(,;–-]", text))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", s).lower())


def heading_level(s: str) -> int:
    s = s.strip()
    if "\n" in s or not 3 <= len(s) <= 90 or re.match(r"^[a-z•\-*–·▪●(]", s):
        return 0
    if re.match(r"^(TABLE|Table|FIG|Fig|Figure|ANNEXURE|Annexure|Source|SOURCE|Note|NOTE)\b", s):
        return 0  # table/figure titles are not section headings
    if re.search(r"[.,;]$", s) and not re.fullmatch(r"[IVX0-9]+\.", s):
        return 0
    letters = re.sub(r"[^A-Za-z]", "", s)
    if len(letters) < 4 or len(letters) < 0.5 * len(s.replace(" ", "")):
        return 0
    if s.upper() == s:
        return 2
    words = s.split()
    if re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-H]\.)\s+[A-Z]", s) and len(words) <= 10:
        return 3
    caps = sum(w[0].isupper() for w in words if w[0].isalpha())
    if len(words) <= 8 and caps >= max(1, (len(words) + 1) // 2) and not s.endswith(":"):
        return 3
    if len(words) <= 6 and not s.endswith(":"):
        return 3  # sentence-case sub-heading on its own line ("Transmission of disease")
    return 0


def font_headings(doc, p: int) -> set[str]:
    """Digital books: normalised lines set in bold or bigger than the page's body text (PDF fonts)."""
    spans = [s for b in doc[p - 1].get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
             if s["text"].strip()]
    if not spans:
        return set()
    body = Counter(round(s["size"], 1) for s in spans).most_common(1)[0][0]
    lines = Counter()
    for b in doc[p - 1].get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            text = "".join(s["text"] for s in l["spans"]).strip()
            strong = all(s["size"] > body + 0.4 or s["flags"] & 16 or "Bold" in s["font"] for s in l["spans"]
                         if s["text"].strip())
            if text and strong:
                lines[norm(text)] += 1
    return set(lines)


class TableCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth, self.rows, self.bad = 0, 0, False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
        elif tag == "tr":
            self.rows += 1

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
            self.bad |= self.depth < 0


def table_ok(t: str) -> bool:
    p = TableCheck()
    p.feed(t)
    return not p.bad and p.depth == 0 and p.rows > 0


# ---- main ------------------------------------------------------------------------------------------------------------
def build(book: dict, toc: list[dict], captions: dict) -> tuple[list[dict], dict]:
    bid = book["book_id"]
    title, pdf = BOOKS[bid]
    doc = pymupdf.open(ROOT / "corpus" / pdf) if bid in DIGITAL else None
    roles = {r: set(v) for r, v in book["roles"].items()}
    kept = sorted(roles.get("body", set()) | roles.get("index", set()) | roles.get("extra", set()))
    printed = {p: p - seg["offset"] for seg in book["page_map"] for p in range(seg["first_pdf"], seg["last_pdf"] + 1)}
    chapters = sorted(toc, key=lambda c: c["pdf_start"])
    blocks, path, cur_ch, k = [], [None, None], None, 0
    stats = Counter()
    for p in kept:
        page = clean_page((OCR / bid / f"p{p:04d}.md").read_text())
        items = page_items(page)
        for cap in captions.get(str(p), []):
            items.append({"type": "figure", "text": f"[{cap['label']}: {cap['text']}]",
                          "source": f"caption:{cap['source']}"})
            stats["captions_inserted"] += 1
        if p in roles.get("index", set()):
            kind, ch = "index", {"n": None, "title": "Index", "section": None}
        elif p in roles.get("extra", set()):
            kind = "extra"
            ch = {"n": None, "title": next(v for r, v in PARK_EXTRA.items() if p in r), "section": None}
        else:
            kind = None
            ch = [c for c in chapters if c["pdf_start"] <= p][-1] if chapters[0]["pdf_start"] <= p else chapters[0]
        if ch is not cur_ch:
            cur_ch, path = ch, [None, None]
        strong = font_headings(doc, p) if doc else None
        for it in items:
            typ = kind or it["type"]
            text = it["text"]
            if typ == "text":
                lvl = heading_level(text)
                if lvl and norm(text)[:25] == norm(ch["title"])[:25]:
                    continue  # the chapter title itself: already in every block's path
                if lvl and strong is not None and norm(text) not in strong:
                    lvl = 0  # digital books: not bold/larger in the PDF -> ordinary line
                if lvl:
                    path = [text, None] if lvl == 2 else [path[0], text]
                    typ = "heading"
                    stats["headings"] += 1
                else:
                    prev = next((b for b in reversed(blocks) if b["type"] == "text"), None)
                    if (prev and prev["pdf_pages"][1] < p and prev["chapter_n"] == ch["n"] and is_open(prev["text"])
                            and continues(text) and all(b["type"] in ("figure", "table")
                                                        for b in blocks[blocks.index(prev) + 1:])):
                        joined = re.sub(r"(?<=[a-z])-$", "", prev["text"]) if prev["text"].endswith("-") else prev["text"] + " "
                        prev["text"] = joined + text
                        prev["pdf_pages"][1], prev["printed_pages"][1] = p, printed.get(p)
                        stats["joined_across_pages"] += 1
                        continue
            k += 1
            blocks.append({"id": f"{bid}-{k:05d}", "book_id": bid, "book": title, "chapter_n": ch["n"],
                           "chapter": ch["title"], "section": ch.get("section"),
                           "heading_path": [h for h in path if h], "type": typ, "text": text,
                           "pdf_pages": [p, p], "printed_pages": [printed.get(p), printed.get(p)],
                           "source": it.get("source", "ocr")})
            stats[typ] += 1
    return blocks, stats


def check(bid: str, blocks: list[dict], book: dict, toc: list[dict]) -> dict:
    kept = sorted(p for r in ("body", "index", "extra") for p in book["roles"].get(r, []))
    covered = set()
    for b in blocks:
        covered.update(range(b["pdf_pages"][0], b["pdf_pages"][1] + 1))
    ocr_tables = sum(len(re.findall(r"<table\b", (OCR / bid / f"p{p:04d}.md").read_text(), flags=re.I)) for p in kept)
    tables = [b for b in blocks if b["type"] == "table" or (b["type"] in ("index", "extra") and "<table" in b["text"])]
    out_tables = sum(len(re.findall(r"<table\b", b["text"], flags=re.I)) for b in blocks)
    leftovers = {
        "front_matter": sum("primary_language:" in b["text"] for b in blocks),
        "image_file_names": sum(bool(re.search(r"page_\d+_\d+_\d+_\d+\.png", b["text"])) for b in blocks),
        "latex": sum(bool(re.search(r"\\\(|\\\[|\\mu|\\frac|\\leq", b["text"])) for b in blocks),
        "watermarks": sum(bool(re.search(r"mebooksfree|KRANSHMAHA", b["text"], flags=re.I)) for b in blocks),
        "br_outside_tables": sum("<br" in b["text"] and "<table" not in b["text"] for b in blocks)}
    chapters_seen = [n for n in dict.fromkeys(b["chapter_n"] for b in blocks) if n is not None]
    return {"blocks": len(blocks), "pages_kept": len(kept), "pages_without_block": sorted(set(kept) - covered),
            "chapters_in_contents": len(toc), "chapters_with_blocks": len(chapters_seen),
            "chapters_in_order": chapters_seen == sorted(chapters_seen),
            "tables_in_ocr": ocr_tables, "tables_in_blocks": out_tables,
            "tables_broken": sum(not table_ok(b["text"]) for b in tables if "<table" in b["text"]),
            "leftovers": leftovers}


if __name__ == "__main__":
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    toc = json.loads((CLEAN / "toc.json").read_text())
    caps = json.loads((CLEAN / "captions.json").read_text())
    (CLEAN / "blocks").mkdir(parents=True, exist_ok=True)
    (CLEAN / "readable").mkdir(parents=True, exist_ok=True)
    report = {}
    for bid in BOOKS:
        blocks, stats = build(manifest[bid], toc[bid], caps[bid]["pages"])
        with (CLEAN / "blocks" / f"{bid}.jsonl").open("w") as f:
            for b in blocks:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
        with (CLEAN / "readable" / f"{bid}.md").open("w") as f:
            last = None
            for b in blocks:
                if (b["chapter"], tuple(b["heading_path"])) != last:
                    f.write(f"\n\n## {b['chapter']} › {' › '.join(b['heading_path'])}  (p.{b['printed_pages'][0]})\n\n")
                    last = (b["chapter"], tuple(b["heading_path"]))
                if b["type"] != "heading":
                    f.write(b["text"] + "\n\n")
        report[bid] = {"stats": dict(stats), "check": check(bid, blocks, manifest[bid], toc[bid])}
        c = report[bid]["check"]
        print(f"{bid:16} blocks {c['blocks']:5} | {dict(stats)}")
        print(f"{'':16} pages without a block {len(c['pages_without_block'])} | chapters {c['chapters_with_blocks']}/"
              f"{c['chapters_in_contents']} in order {c['chapters_in_order']} | tables {c['tables_in_blocks']}/"
              f"{c['tables_in_ocr']} broken {c['tables_broken']} | leftovers {c['leftovers']}")
    random.seed(7)
    sample = []
    for bid in BOOKS:
        blocks = [json.loads(l) for l in (CLEAN / "blocks" / f"{bid}.jsonl").open()]
        sample += random.sample([b for b in blocks if b["type"] == "text" and len(b["text"]) > 200], 3)
    report["eye_check_sample"] = [{k: b[k] for k in ("id", "chapter", "heading_path", "pdf_pages", "printed_pages")}
                                  | {"start": b["text"][:160]} for b in sample[:10]]
    (CLEAN / "check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("wrote data/clean/blocks/, data/clean/readable/, data/clean/check.json")
