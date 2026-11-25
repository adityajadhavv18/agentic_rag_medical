# Plan: OCR → Cleaning → Chunking → Ingestion (medical corpus)

**Goal:** turn 4 MBBS textbooks into clean, well-labelled chunks in Qdrant, without losing pages, tables, headings or page numbers along the way.

**The one idea behind this plan:** the corpus is a building's foundation. A bad OCR, a lost heading or a missing page can't be fixed later by a clever retriever. Every phase ends with a **check**, and we don't move on until the check passes.

## The books (measured 2026-11-25)

| File | Book | Pages | Type | Needs OCR? |
|---|---|---:|---|---|
| `K.pdf` | Park's, Preventive & Social Medicine (PSM) | 1,073 | Scanned. Hidden text layer is poor, and missing on pages 513–812 | **Yes, all pages** |
| `A.pdf` | Khurana, Ophthalmology | 668 | Scanned. Hidden text layer mostly fine (16 pages missing) | **Yes** (decided in Phase 1) |
| `P_L_Dhingra.pdf` | Dhingra, ENT | 556 | Digital text (24 image pages) | Only the image pages |
| `Rajesh_Bardale_…pdf` | Bardale, Forensic Medicine | 583 | Digital text (1 image page) | Only the image page |

Total 2,880 pages, about 2.6M tokens, so roughly **10,000–12,000 chunks** at 300 tokens per chunk with 50 overlap.

## Things we must not miss (the checklist)

1. **Copyright:** the PDFs and every text derived from them stay out of git (`.gitignore` before the first commit). Kaggle datasets stay **private**. Only code, questions and scores get published.
2. **Pin the files:** record a SHA-256 checksum for each PDF, so the paper can say exactly which files were used.
3. **Page numbers:** keep both the PDF page and the **printed page** for every paragraph. Citations need the printed page ("Park's p. 214").
4. **Headings:** keep Book › Chapter › Section for every paragraph. This is the free "header stamp" that worked so well in cricket (r4j).
5. **Tables:** keep them whole, as Markdown tables. Never cut a table in the middle of a row.
6. **Figures:** keep the caption and any label text as a "figure" block, flagged so we can add figure descriptions later if needed.
7. **No lost pages:** every page must produce text or be on a short, explained "blank/skipped" list.
8. **Same ruler:** same tokenizer (bge), same chunk size (300/50) and same retrievers as the cricket builds, so results can be compared.
9. **Stable chunk IDs:** a re-run gives the same IDs, so golden-set evidence never breaks.
10. **Versioned output:** the OCR output, clean text, chunks and Qdrant collection each get a version name. Redoing one step doesn't silently change the others.

## Phase 0: Setup and inventory ($0, on the Mac)

- `git init`, then `.gitignore` for `corpus/`, `data/` and any extracted text **before** the first commit.
- `manifest.json`: for each book, its file name, title, subject, edition/year (from the title page), page count and SHA-256.
- **Page map:** find where the printed page 1 starts in each PDF (front matter offset), and list the front matter, index and appendix ranges.
- **Scope decision (yours):** what we keep. My suggestion is to keep the main chapters, tables and appendices with real content, and drop the cover, preface, contents and index. The table of contents is kept separately as a **checklist** for Phase 3.

**Check:** the manifest is complete; the page offset is confirmed on 3 random pages per book (PDF page → printed number).

## Phase 1: OCR bake-off on 12 test pages (Kaggle T4, ~free)

- I cut **12 hard pages** into one small test PDF:
  - Park's: a table, a flowchart, a dense two-column page, 2 pages from the blank stretch (513–812)
  - Khurana: a labelled eye diagram, a "key points" box
  - Dhingra and Bardale: one page each, as a control, since we know their real text
  - one page with running headers and footers
- **Candidates:**
  - **PaddleOCR-VL** (1 T4)
  - **olmOCR-2** (split across 2 T4s, if it fits)
  - **Apple Vision** (on the Mac, as a baseline)
  - the **existing hidden text layer** (as the "do nothing" option)
- **Scoring:**
  - I make a checklist of about 80 medical terms and numbers visible on those pages (drug names, doses, rates, figure labels), and count how many come out exactly right.
  - Is each table intact (rows and columns correct)? Are headings marked as headings? Is the reading order right on two-column pages?
  - Seconds per page.
- **Decision rule (fixed before we look):** highest term accuracy wins. If two are within 2 terms of each other, the one with better tables wins. Speed only breaks ties.
- We also decide here whether Dhingra and Bardale use their own digital text (likely) or the same OCR, for uniform Markdown headings and tables.

**Check:** you look at the side-by-side output for 3–4 pages and agree with the pick.

## Phase 2: Full OCR run (Kaggle T4, ~1–3 GPU hours)

- Upload Park's and Khurana as a **private** Kaggle dataset, then run the winning engine.
- Save **one output file per page** (Markdown + JSON) and skip pages already done. If a Kaggle session dies, it resumes instead of restarting.
- Keep a log for every page: words found, time and errors.
- **Automatic quality gate per page:**
  - too few words for a text page
  - too many non-words (checked against a medical word list)
  - broken table syntax

  Flagged pages are retried with the runner-up engine.

**Check:** 0 missing pages; every flagged page is fixed or explained; you spot-check 20 random pages against the scans.

## Phase 3: Cleaning and structure ($0, on the Mac)

- Remove running headers, footers and page numbers (they repeat on every page and pollute search).
- Fix line-break hyphens ("super- ficial" → "superficial"), join paragraphs split across pages, fix ligatures and odd spaces, and remove `<br>`-type junk.
- Build the heading tree Book › Chapter › Section › Subsection from the OCR headings, **checked against each book's table of contents**.
- Output one file per book as a list of **blocks**:
  `{book, chapter, section, heading_path, printed_page_start, printed_page_end, type: text|table|figure|list|box, text}`

**Checks:**
- Every chapter in the table of contents is found, in order, with no extras.
- Page coverage: every kept page is used by at least one block.
- 10 random blocks read correctly against the scan.
- Tables: each one parses as a valid table.

## Phase 4: Chunking ($0, on the Mac)

- **Text:** 300 bge tokens with 50 overlap, **inside one section only** (never crossing a heading), same as cricket.
- **Tables:** kept whole if under about 500 tokens. Otherwise split by rows, **repeating the header row** in every piece.
- **Figures:** own chunk (caption + labels), marked `type=figure`.
- **Header stamp** on every chunk: `Park's PSM › Epidemiology of Communicable Diseases › Tuberculosis › Treatment`.
- **Payload:** book, subject, chapter, section, printed pages, type, version.
- **Stable ID:** book + chapter + section + position (hash).

**Checks:**
- Chunk count and size spread (how many are tiny, how many are full).
- Text kept: all characters in the clean blocks appear in the chunks (overlap aside).
- No chunk crosses a section.
- 10 random chunks read sensibly on their own.

## Phase 5: Golden set (before any retrieval scoring)

- Questions are written **from the clean text**, each with an evidence quote that must be found word-for-word in a chunk (like `check_golden.py`).
- **Mix:**
  - simple facts
  - exact terms (drug, dose, act / IPC section)
  - table lookups
  - figure questions
  - multi-part
  - **cross-book** (e.g. forensic + PSM)
  - no-answer
- Tune/test split; the test set is used once, only when you say.
- **Source (yours):** I write them, you write some, or real MBBS exam-style questions with answers checked against the books.

**Check:** every evidence quote is found in exactly the chunk we expect.

## Phase 6: Ingestion into Qdrant ($0, local or Kaggle)

- New collection, e.g. `medical_mbbs_v1`: dense bge + BM25 + SPLADE as named vectors in one point per chunk (same as the cricket stack), with the payload from Phase 4.
- Embedding about 12k chunks with SPLADE on the M1 may take about 30–60 min. If it's too slow, we embed on the Kaggle T4 and upload the vectors.
- Save a Qdrant snapshot, so a rebuild isn't needed after a crash.

**Checks:**
- Point count = chunk count.
- Filters work (e.g. only Bardale).
- 10 sanity questions find the right book and chapter.
- **Retrieval-only score** on the tune golden set (recall@5, MRR). This is the baseline every later architecture is compared against.

## After this plan (each needs a separate yes, because of cost)

- Contextual notes for each chunk: about $3.5 with gpt-4o-mini.
- Knowledge graph extraction: about $3.
- Figure descriptions with a vision model: only if golden questions show answers that exist only in figures.
- Then the agentic pipeline on top (the best of the 8 builds).

## Decisions I need from you

1. **Scope:** keep main chapters + tables + appendices, and drop the cover, preface, contents and index?
2. **Dhingra and Bardale:** use their digital text, or decide in Phase 1 after seeing both?
3. **Golden questions:** who writes them (me / you / exam-style)?
4. **Go for Phase 0** (setup + inventory, free)?
