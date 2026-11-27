"""Phase 3, step A: chapter lists for the 4 books, with each chapter's first PDF page, verified ($0).

    uv run ingest/toc.py          # writes data/clean/toc.json and prints the check

Chapter titles come from each book's contents pages:
  - Dhingra, Bardale: their exact digital text (contents pages 12-15 / 12-19);
  - Khurana: Apple Vision on contents pages 3-4 (titles and page numbers come out as two lists, paired in order);
  - Park's: titles read by eye from the scanned contents pages 7-8 (Apple Vision mixes its two columns and lost
    some titles); the page numbers that survive on those pages are kept as cross-checks.
Start pages: the contents page number (printed) is turned into a PDF page with the manifest's page map. Then the
start is CONFIRMED in the OCR text: the chapter title must appear near the top of that page (or the page before/
after, where the contents number points at the first sub-topic). Chapters without a usable number are found by
searching the OCR pages in order for the title near the top of a page.
Second reader: olmOCR often skips Park's decorative chapter banners, so when the OCR text does not show a title,
the page is read again - Dhingra/Bardale from their exact text layer, Park's/Khurana with Apple Vision (free, local,
cached in data/clean/toc_raw/pages/) - only for pages where the chapter can start (around its contents page number,
or between the neighbouring chapters' starts). Title searches use body pages only (an index page also lists titles).
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "clean" / "toc_raw"
OCR = ROOT / "data" / "ocr"
OUT = ROOT / "data" / "clean" / "toc.json"
TOP = 600  # characters from the top of a page where a chapter title must appear

PARK_TITLES = [  # read by eye from K.pdf pages 7-8 (contents), 2026-11-27
    "Man and Medicine: Towards Health for All", "Concept of Health and Disease",
    "Principles of Epidemiology and Epidemiologic Methods", "Screening for Disease",
    "Epidemiology of Communicable Diseases", "Epidemiology of Chronic Non-communicable Diseases and Conditions",
    "Health Programmes in India", "Demography and Family Planning",
    "Preventive Medicine in Obstetrics, Paediatrics and Geriatrics", "Health Care of the Community",
    "Nutrition and Health", "Medicine and Social Sciences", "Tribal Health in India", "Sustainable Development Goals",
    "Environment and Health", "Hospital Waste Management", "Disaster Management", "Occupational Health",
    "Genetics and Health", "Mental Health", "Health Information and Basic Medical Statistics",
    "Communication for Health Education", "Health Planning and Management",
    "Essential Medicines and Counterfeit Medicines", "International Health"]
# printed start pages that survive on Park's contents pages (Apple Vision lines like "13. TRIBAL HEALTH IN INDIA 836");
# chapter 3's "60" was printed on chapter 2's line by the column mix-up
PARK_PAGES = {1: 1, 2: 13, 3: 60, 4: 151, 5: 159, 13: 836, 15: 848, 16: 931, 18: 946, 19: 963, 20: 973, 23: 1012,
              24: 1028, 25: 1041}
KHURANA_SECTIONS = [(1, 2, "Section I: Anatomy and Physiology of Eye"), (3, 4, "Section II: Optics and Refraction"),
                    (5, 18, "Section III: Diseases of Eye and Ocular Adnexa"), (19, 20, "Section IV: Ocular Therapeutics"),
                    (21, 22, "Section V: Systemic and Community Ophthalmology"),
                    (23, 26, "Section VI: Practical Ophthalmology")]


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKC", s).lower())  # letters only: banners put the chapter number inside the title ("Tribal Health / 13 / in India")


def dhingra() -> list[dict]:
    lines = [l.strip() for l in (RAW / "dhingra.txt").read_text().splitlines()]
    chapters, section, i = [], None, 0
    while i < len(lines):
        if re.fullmatch(r"SECTION [IVXL]+", lines[i]):
            section = re.sub(r"\s+\d+$", "", lines[i + 1]).title()
        m = re.fullmatch(r"Chapter (\d+)", lines[i])
        if m:
            title, j = [], i + 1
            while j < len(lines):
                end = re.search(r"\s+(\d+)\s*$", lines[j])
                title.append(re.sub(r"\s+\d+\s*$", "", lines[j]))
                j += 1
                if end:
                    break
            chapters.append({"n": int(m.group(1)), "title": " ".join(title).strip(), "printed": int(end.group(1)),
                             "section": section})
            i = j
            continue
        i += 1
    return chapters


def bardale() -> list[dict]:
    chapters, section = [], None
    for line in (RAW / "bardale.txt").read_text().splitlines():
        s = re.match(r"\s*Section ([AB]): (.+)", line)
        if s:
            section = f"Section {s.group(1)}: {s.group(2).strip()}"
        m = re.match(r"\s*(\d+)\.\s+(.+?)\.{3,}\s*(\d+)\s*$", line)
        if m:
            chapters.append({"n": int(m.group(1)), "title": m.group(2).strip(), "printed": int(m.group(3)),
                             "section": section})
    return chapters


def khurana() -> list[dict]:
    lines = [l.strip() for l in (RAW / "khurana.txt").read_text().splitlines()]
    titles = [(int(m.group(1)), m.group(2)) for l in lines if (m := re.fullmatch(r"(\d+)\. (.+?)(?: (\d+))?", l))]
    inline = {int(m.group(1)): int(m.group(2)) for l in lines if (m := re.fullmatch(r"(\d+)\. .+ (\d+)", l))}
    loose = [int(l) for l in lines if re.fullmatch(r"\d{1,3}", l)]  # page numbers listed apart, in chapter order
    out = []
    for k, (n, t) in enumerate(titles):
        page = inline.get(n, loose[k] if k < len(loose) else None)
        sec = next(name for a, b, name in KHURANA_SECTIONS if a <= n <= b)
        out.append({"n": n, "title": re.sub(r" \d+$", "", t), "printed": page, "section": sec})
    return out


def park() -> list[dict]:
    return [{"n": n, "title": t, "printed": PARK_PAGES.get(n), "section": None}
            for n, t in enumerate(PARK_TITLES, start=1)]


# Park's running headers (ingest/park_headers.py) print a shortened title for two chapters
PARK_HEADER_ALIAS = {6: "noncommunicablediseases", 21: "medicalstatistics"}


def locate_park(book: dict, chapters: list[dict]) -> list[dict]:
    """Park's: olmOCR drops most chapter banners and plain title search hits pages that only mention a title, so
    chapters are placed from the running headers: a chapter starts on the LAST page with an empty header strip (the
    banner page) before the first page whose header names it. Checked against the contents numbers and OCR titles."""
    H = {int(k): v for k, v in json.loads((RAW / "park_headers.json").read_text()).items()}
    body = sorted(book["roles"]["body"])
    to_pdf = {p - seg["offset"]: p for seg in book["page_map"] for p in range(seg["first_pdf"], seg["last_pdf"] + 1)}
    cursor = body[0] - 1
    for c in chapters:
        key = PARK_HEADER_ALIAS.get(c["n"], norm(c["title"])[:22])
        hits = [p for p in body if p > cursor and key in norm(H.get(p, ""))]
        if c["n"] == 1:
            start = body[0]
        elif hits:
            window = [p for p in body if cursor < p <= hits[0]]
            titled = [p for p in window if norm(c["title"])[:28] in norm(re.sub(
                r"\A---.*?---", "", (OCR / "park_psm" / f"p{p:04d}.md").read_text(), flags=re.S))[:TOP]]
            empty = [p for p in window if len(norm(H.get(p, ""))) < 3]
            start = titled[0] if titled else (empty[-1] if empty else hits[0] - 1)  # a visible title wins
        else:
            start = to_pdf.get(c["printed"])
        top = norm(re.sub(r"\A---.*?---", "", (OCR / "park_psm" / f"p{start:04d}.md").read_text(), flags=re.S))[:TOP]
        guess = to_pdf.get(c["printed"]) if c["printed"] else None
        checks = []
        if guess:
            checks.append("contents number agrees" if abs(guess - start) <= 1 else f"CONTENTS SAYS pdf {guess}")
        checks.append("title in OCR" if norm(c["title"])[:28] in top else "title not in OCR (banner skipped)")
        c["pdf_start"], c["how"] = start, "running headers; " + ", ".join(checks)
        cursor = start
    return chapters


DIGITAL = {"dhingra_ent": "P_L_Dhingra.pdf", "bardale_fmt": "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf"}
SCANNED = {"park_psm": "K.pdf", "khurana_ophthal": "A.pdf"}


def second_reader(book_id: str, p: int) -> str:
    """Top of a page read by a second reader: exact text layer (digital books) or Apple Vision (scanned, cached)."""
    import pymupdf
    if book_id in DIGITAL:
        return norm(pymupdf.open(ROOT / "corpus" / DIGITAL[book_id])[p - 1].get_text())[:TOP]
    cache = RAW / "pages" / f"{book_id}_p{p:04d}.txt"
    if not cache.exists():
        from phase1_local import apple_vision
        cache.parent.mkdir(parents=True, exist_ok=True)
        page = pymupdf.open(ROOT / "corpus" / SCANNED[book_id])[p - 1]
        cache.write_text(apple_vision(page.get_pixmap(dpi=150).tobytes("png")))
    return norm(cache.read_text())[:TOP]


def locate(book: dict, chapters: list[dict]) -> list[dict]:
    """Find and confirm each chapter's first PDF page: OCR text first, then a second reader."""
    bid, pmap = book["book_id"], book["page_map"]
    body = sorted(book["roles"]["body"])
    tops = {p: norm(re.sub(r"\A---.*?---", "", (OCR / bid / f"p{p:04d}.md").read_text(), flags=re.S))[:TOP]
            for p in body}
    to_pdf = {p - seg["offset"]: p for seg in pmap for p in range(seg["first_pdf"], seg["last_pdf"] + 1)}
    for c in chapters:
        c["guess"] = to_pdf.get(c["printed"]) if c["printed"] else None
    for i, c in enumerate(chapters):  # pass 1: contents number (+-1) or title search, both on the OCR text
        key = norm(c["title"])[:28]
        g = c["guess"]
        near = [p for p in ((g, g - 1, g + 1) if g else ()) if p in tops and key in tops[p]]
        if near:
            c["pdf_start"], c["how"] = near[0], "contents page number, title confirmed in OCR"
            continue
        lo = max([x["pdf_start"] for x in chapters[:i] if x.get("pdf_start")] or [body[0]])
        hi = min([x["guess"] for x in chapters[i + 1:] if x["guess"]] or [body[-1] + 1])
        found = [p for p in body if lo <= p < hi and key in tops[p]]
        c["pdf_start"], c["how"] = (found[0], "title found in OCR") if found else (None, "")
    for i, c in enumerate(chapters):  # pass 2: second reader, only where pass 1 failed
        if c["pdf_start"]:
            continue
        key = norm(c["title"])[:28]
        g = c["guess"]
        lo = max([x["pdf_start"] for x in chapters[:i] if x.get("pdf_start")] or [body[0]])
        hi = min([x["pdf_start"] or x["guess"] or 10 ** 6 for x in chapters[i + 1:]] or [body[-1] + 1])
        window = [p for p in ((g, g - 1, g + 1) if g else range(lo + 1, hi)) if p in tops]
        hit = next((p for p in window if key in second_reader(bid, p)), None)
        if hit:
            c["pdf_start"], c["how"] = hit, "title confirmed by second reader"
        elif g:
            c["pdf_start"], c["how"] = g, "contents page number only (title NOT confirmed)"
        else:
            c["how"] = "NOT FOUND"
    for c in chapters:
        c.pop("guess", None)
    return chapters


if __name__ == "__main__":
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    out = {}
    for bid, fn in (("park_psm", park), ("khurana_ophthal", khurana), ("dhingra_ent", dhingra),
                    ("bardale_fmt", bardale)):
        ch = (locate_park if bid == "park_psm" else locate)(manifest[bid], fn())
        starts = [c["pdf_start"] for c in ch if c["pdf_start"]]
        ordered = all(a < b for a, b in zip(starts, starts[1:]))
        hows = {}
        for c in ch:
            hows[c["how"]] = hows.get(c["how"], 0) + 1
        print(f"{bid:16} {len(ch)} chapters, starts in order: {ordered} | {hows}")
        for c in ch:
            if "NOT" in c["how"] or "SAYS" in c["how"]:
                print(f"     check: ch {c['n']} {c['title'][:50]!r} printed {c['printed']} -> pdf {c['pdf_start']} ({c['how']})")
        out[bid] = ch
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote data/clean/toc.json")
