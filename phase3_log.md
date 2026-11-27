# Phase 3: Cleaning and structure (with the caption fix)

**Goal:** turn the 2,768 OCR pages into clean, continuous text blocks. Each block knows its book, chapter, section, sub-heading and printed page.

**Choices** (his, 2026-11-27):
- tables kept as HTML (merged cells survive);
- math turned into readable symbols;
- olmOCR's picture descriptions kept as `[Figure: …]`;
- captions the OCR dropped are put back.

Everything runs locally for $0.

## Steps and scripts

| Step | Script | What it does |
|---|---|---|
| A. Chapters | `ingest/toc.py`, `ingest/park_headers.py` | Chapter lists from each contents page and each chapter's first page, verified (below) |
| B. Captions | `ingest/captions.py` | Finds captions the OCR dropped: exact text layer (Dhingra, Bardale) or Apple Vision on figure pages (Park's, Khurana) |
| C. Cleaning | `ingest/clean.py` | Cleans pages, adds captions, joins pages, attaches structure, writes blocks and runs the checks |

## A. Chapters: 194 of 194 placed, in order, verified

| Book | Chapters | How they were placed and verified |
|---|---:|---|
| Khurana | 26 | contents page number, title confirmed in the OCR text (26/26) |
| Dhingra | 96 | contents number + OCR title (95); 1 confirmed from the exact text layer |
| Bardale | 47 | contents number + OCR title (47/47) |
| Park's | 25 | running headers (see below); 14/14 agree with the surviving contents numbers |

**Park's needed a different method.**
- olmOCR skips most of its decorative chapter banners.
- Apple Vision mixes the two columns of its contents pages, so several titles and page numbers were lost. The 25 titles were read by eye from the scan.
- A plain title search placed 2 chapters wrong, on pages that only mention the title (e.g. "Nutrition and Health" at PDF 701 instead of 749).
- Fix: Apple Vision read the header strip of all 1,063 pages. A chapter starts on the page showing its title (OCR), or else on the last page with an empty header strip (the banner page) before the first page whose running header names it.
- Two chapters print a shortened header ("NON-COMMUNICABLE DISEASES", "… MEDICAL STATISTICS").
- One banner puts the chapter number inside the title ("Tribal Health / 13 / in India"), so title matching uses letters only.

## B + C. Results

| | Park's | Khurana | Dhingra | Bardale | Total |
|---|---:|---:|---:|---:|---:|
| Blocks | 16,360 | 6,594 | 5,847 | 5,484 | **34,285** |
| Text paragraphs | 12,416 | 5,863 | 4,838 | 3,820 | |
| Headings | 3,232 | 371 | 655 | 1,087 | |
| Tables (HTML, all valid) | 566 | 65 | 122 | 342 | **1,095** |
| Figure blocks | 108 | 238 | 191 | 204 | |
| Captions added | 8 | 50 | 66 | 109 | **233** |
| Paragraphs joined across a page break | 360 | 136 | 52 | 60 | **608** |

## Checks (all passed)

1. **Chapters:** every chapter in every contents list has blocks, in order (25, 26, 96, 47).
2. **No lost pages:** every kept page contributes to at least one block, on 0 of 2,768 pages missing.
3. **Tables:** the number of tables in the blocks equals the number in the OCR (1,095), and 0 are broken.
4. **Leftovers:** 0 olmOCR front matter, 0 image file names, 0 math code, 0 watermarks, 0 `<br>` outside tables.
5. **Caption coverage**, re-measured on the cleaned text. A caption counts as present if ≥ 60% of its first 8 words' pairs are on its page. The reference is each PDF's own text layer: exact for Dhingra and Bardale; olmOCR's own hidden layer, with typos, for Khurana.

   | Book | OCR (Phase 2) | After Phase 3 |
   |---|---:|---:|
   | Dhingra | 78% | **99%** (447/450) |
   | Bardale | 81% | **94%** (154/164) |
   | Khurana | ~74% | **~80%** (rough: reference has OCR typos; new captions come from Apple Vision) |

   Park's has no usable reference; Apple Vision found 140 captions there, 132 of them already in the OCR text.
6. **Eye check:** 10 random blocks. One (Park's PDF 307) was compared word for word with the scan: the text and printed page are right. Its section path was wrong ("Agent factors" instead of "Transmission of disease") because sentence-case sub-headings weren't recognised. Fixed: short standalone lines are now headings.

## Problems and fixes

| Problem | Fix |
|---|---|
| Park's contents pages: Apple Vision mixed the columns, losing titles and page numbers | Titles read by eye from the scan; chapter starts found from running headers |
| Plain title search hit pages that only mention a chapter title | Park's: running-header rule; other books: contents number ± 1 with the title on the page |
| A chapter number printed inside a banner title | Title matching on letters only |
| Caption extraction grabbed the next line (a body paragraph or the next table title) | Wrap only on the exact text layer; stop at the next Table/Fig/bullet; a bare "FIG. 1" only takes a short title-like next line |
| `37^\circ` → "37^°" | `^\circ` turned into "°" before superscripts |
| "TABLE 3" and "Source : (18)" detected as section headings | Table, figure, annexure, source and note lines are never headings |
| `<br>` used as a line break outside tables; bare `\leq` inside a table | `<br>` → newline outside tables; bare LaTeX commands replaced |
| Caption checker artifacts (glued watermark, body-text figure references, hyphen splits, two captions on one line) | Checker fixed before reporting coverage |

## Known limits (carried forward)

- Labels inside diagrams (e.g. vein names on an anatomy figure) are not in the text. That's the optional figure-description step, only if golden questions need it.
- Bardale's own text layer splits some words around the lost "fi" glyph ("Rifl ing", "decomposit ion") in a few inserted captions.
- Some heading noise remains: Khurana's "Section Outline" divider pages, and an occasional stray figure label ("MYOPIA 10m").
- Park's printed page 624 is missing from the PDF itself.

## Outputs (git-ignored: book text)

- `data/clean/blocks/<book_id>.jsonl`: one block per paragraph, table, figure or heading. Fields: `id`, `book_id`, `book`, `chapter_n`, `chapter`, `section`, `heading_path`, `type`, `text`, `pdf_pages`, `printed_pages`, `source`.
- `data/clean/readable/<book_id>.md`: the same text, readable by eye.
- `data/clean/toc.json`, `captions.json`, `check.json`, `caption_coverage.json`.
