# Phase 2: full OCR run (all 4 books)

**Engine:** olmOCR-2 (Phase 1 winner) in fast mode:
- model `allenai/olmOCR-2-7B-1025-FP8`
- vLLM 0.34.0
- pictures 1288 px on the longest side
- olmOCR's v4 prompt, max 8,192 tokens
- one retry at temperature 0.3 if a page does not finish

**Why all 4 books** (decided 2026-11-27):
- Park's: needs OCR (no text, or bad text).
- Khurana: clear gain (existing text 22/32 terms on its test pages, OCR 27/32).
- Dhingra and Bardale: already have exact digital text, but it flattens tables. Bardale has tables on 35% of its pages (191), Dhingra on 3% (16).

A free digital-PDF converter (pymupdf4llm) kept exact words but scrambled 1 of 3 tables on the Bardale test page. OCR for all 4 gives one format, intact tables, and lets the exact text of the two digital books proofread the OCR.

## Timing test before the full run (`ingest/timing_pages.py`, `ingest/timing_check.py`)

**Pages:**
- 24 random kept pages (seed 42; 15 Park's, 9 Khurana)
- the 12 scored Phase 1 test pages

All 36 ran in one Modal container (L40S).

| | Slow mode (transformers, bf16, 1 page at a time) | Fast mode (vLLM, FP8, 24 at a time) |
|---|---|---|
| Scored test pages | 93/101 terms, 13/13 table rows | **94/101, 13/13** |
| Speed | 96 s/page | **2.2 s/page** (24 pages in 53 s, incl. the slowest page's tail) |
| Failed / retried / cut off | – | 0 / 0 / 0 of 36 |

- **Cross-check against Apple Vision on the 24 random pages:** median word coverage 0.97, lowest 0.86. The only gaps were running headers, which olmOCR leaves out on purpose.
- **Projection:** ~$2–3.40 for the full run, against ~$37–59 in slow mode. The test itself cost ~$0.16.

## Full run (`ingest/modal_full_ocr.py`)

**Design** (keeps running if the Mac sleeps or shuts down):
- a deployed Modal app;
- `start` uploads the PDFs once, hands 16 batches of up to 200 pages to Modal, and exits;
- each GPU starts vLLM once and keeps 32 pages in flight;
- every page is saved on the Modal volume as `ocr/<book>/pNNNN.md` + `.json`, committed every 20 pages or 60 s;
- a batch skips pages already saved, so a rerun never OCRs a page twice;
- Modal retries a crashed batch up to 2 times; at most 4 GPUs.

**Result (2026-11-27):**

| | |
|---|---|
| Pages | **2,768 / 2,768**: Park's 1,063, Khurana 654, Dhingra 491, Bardale 560 |
| GPUs | 4 × NVIDIA L40S, booted in 110–170 s |
| Time | 12:04 → 12:18 (~14 min), ~230 pages/min in total, ~1 page/s per GPU |
| Failures | 0 failed batches, 0 missing pages, 0 cut off, 0 rotated, 2 pages retried successfully |
| Output tokens | 3.06 M |
| Cost | ~0.98 GPU-hours ≈ **$1.92** (+ small CPU/RAM charge) |
| Files | `data/ocr/<book>/pNNNN.md` + `.json`, 28 MB (git-ignored); a second copy stays on the Modal volume |

## Checks (`ingest/ocr_check.py` → `data/ocr/check.json`)

- **Complete:** every kept page has its text and details file.
- **11 "short" pages:** section-title pages and figure-only pages. Not lost content.
- **Proofreading against the exact digital text.** To make the comparison fair, the text layer was:
  - unicode-normalised,
  - had line-end hyphen splits rejoined,
  - had repeated header/footer lines and page numbers removed.

| Book | Words matched (median / mean) | Numbers matched (median / mean) |
|---|---|---|
| Dhingra (491 pages) | 0.996 / 0.990 | 1.000 / 0.937 |
| Bardale (560 pages) | 0.972 / 0.963 | 1.000 / 0.963 |

The remaining gaps are mostly:
- text inside figures (axes, labels: olmOCR leaves diagram text out by design);
- breakages in the PDFs' own text: the "fi" letter pair is lost, so the text layer says "rearm" where the page shows "firearm", and OCR is right;
- a few header lines the filter missed.

## Open issue: dropped figure captions

olmOCR drops about 1 in 5 figure captions, mostly on pages full of photos or X-rays. A caption counted as kept if ≥ 60% of its word pairs appear in the OCR text.

| Book | Captions found | Kept | Pages with a dropped caption |
|---|---:|---:|---:|
| Dhingra | 434 | 338 (78%) | 79 |
| Bardale | 163 | 132 (81%) | 28 |
| Khurana (rough: reference text is itself OCR) | 561 | 414 (74%) | 115 |

Example: Dhingra PDF p381 lost all 4 of its captions. Park's cannot be measured this way, because its hidden text is too poor.

**Proposed fix, in Phase 3 cleaning, not by re-running OCR:**
- Dhingra and Bardale: re-insert missing captions from their exact text layer (free).
- Park's and Khurana: take caption lines from Apple Vision on figure pages (free), with PaddleOCR-VL as a fallback (~$0.50).

## Problems and fixes

| Problem | Fix |
|---|---|
| A key clash in the timing log: the batch page count overwrote the per-page details | Renamed to `n_pages`. Local logs were rebuilt from the run printout; per-page token counts for that test are lost, totals kept |
| Proofreading first showed ~92% words / 75% numbers | Not OCR errors: running headers, page numbers and hyphen splits in the text layer. Fixed in the comparison |
| Caption check first showed 13–31% kept | Too strict: required 5 caption words back-to-back, but "(1)", "(2)" sit between them. Switched to the word-pair rule |
| `fetch` read the 5,536 files one by one (~2 s each, would take hours) | Switched to `modal volume get` (all files in 74 s) |
