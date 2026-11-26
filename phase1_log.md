# Phase 1: OCR bake-off (12 hard test pages)

**Goal:** pick the OCR engine for the two scanned books (Park's PSM, Khurana Ophthalmology) before the full run.

**Test set:** 12 pages, each chosen for one difficulty:
- Park's ×6: dense two-column text, a curved photo, handwriting + a table, pen underlines, a big numeric table, a blurry flowchart
- Khurana ×4: a labelled diagram, dense text, a table, the index
- Dhingra and Bardale ×1 each: control pages (digital text)

**Answer key:** read by eye from the scans, or from the digital text on the control pages:
- 101 terms
- 13 checks that table cells landed in the same row
- 1 "must not appear" check: the handwritten note

Matching ignores case, spaces, quote/dash style and markdown/html.

**Decision rule** (fixed before looking at any output): most terms right wins. If two engines are within 2 terms, more table rows wins. Speed only breaks a tie after that.

## Results

| Engine | Terms (of 101) | Table rows (of 13) | Handwriting leaked | Seconds / page | Where |
|---|---:|---:|---:|---:|---|
| **olmOCR-2 7B** (allenai/olmOCR-2-7B-1025, bf16) | **93** | **13** | no | 96 | Modal L4, transformers, 1 page at a time |
| PaddleOCR-VL 1.6 | 89 | 13 | no | **2.4** | Modal A10G/L4, layout on PaddlePaddle + VL model on vLLM |
| Apple Vision (macOS, accurate) | 77 | 5 | yes | 1.3 | Mac M1, free |
| Existing hidden text layer | 56 | 1 | no | 0 | in the PDFs |

**Winner by the rule: olmOCR-2** (+4 terms). The 96 s/page is the slow one-page-at-a-time mode; the full run should use vLLM (to be timed on 20–30 pages first).

**Where they differ:**
- PaddleOCR-VL turns flowcharts and diagrams into picture placeholders and loses their words. It also missed a faint table title and the first line of the index page.
- olmOCR-2 leaves out page headers and decorative chapter banners.
- Neither reads labels inside anatomy diagrams (3/8 each; Apple Vision 7/8). That is the planned "figure descriptions" step.
- Both kept every table row and ignored the handwriting.

**Control pages (Dhingra, Bardale):** the digital text layer gets every word (8/8, 8/8) but flattens tables (0/2 rows on the Bardale table). OCR keeps the rows (2/2). Whether to use the text layer, OCR, or text layer + OCR on table pages is still open.

## Problems and fixes along the way

| Problem | Fix |
|---|---|
| Kaggle free T4 queue (position 224 → 142, days) | Moved to Modal (no queue, $30/month credit), driven from the Mac with `uv run modal run` |
| Modal's PyPI mirror timed out on `paddlepaddle-gpu` | Install from Paddle's own index, real PyPI for its dependencies |
| One failed image build stopped both models | Each model is its own Modal app |
| olmOCR: `AutoVideoProcessor requires Torchvision` | Added `torchvision==0.22.1` (matches torch 2.7.1) |
| Mac idle sleep → Modal stopped the app, 9 finished pages lost | Every page saved to a Modal Volume as it finishes, reruns skip done pages, `--detach`, `caffeinate` |
| No free L4 GPUs | PaddleOCR-VL may use L4, A10G or L40S |
| PaddleOCR-VL "hung": GPU 0 %, one CPU core 100 %, no page in 15+ min | Stack dumps (`modal_paddle_diag.py`) showed it was generating text token by token in Python on the native engine: a light page took 10 s, a dense page 25+ min. Fix: serve the VL model with vLLM in its own virtualenv |
| vLLM: `Failed to find C compiler` (Triton) | Added `build-essential` to the image |
| vLLM rejected PaddleOCR's per-request image settings | `--trust-request-mm-kwargs` (only client is our own pipeline on localhost) |
| Scorer bug: tag-stripping regex deleted text after a bare `<` (e.g. "<6/60") | Strip real html tags only; olmOCR 87 → 92 before the key fix |
| Answer key error: I misread "Khichidi" as "Khichdi" | Key corrected from a zoomed scan (both models were right) |

**Modal spend for Phase 1** (including failed attempts): about $3–4 of the $30 monthly credit.

## Files

- `ingest/phase1_pages.py`: cuts the 12 test pages into one PDF
- `ingest/phase1_local.py`: Apple Vision + text-layer baselines
- `ingest/modal_ocr.py`: PaddleOCR-VL and olmOCR-2 on Modal
- `ingest/modal_paddle_diag.py`: the stack-dump diagnosis
- `ingest/phase1_score.py`: the scorer
- `kaggle/phase1_ocr.ipynb`: the Kaggle notebook (not used: queue)
- Book pages, the answer key and all OCR output stay in the git-ignored `data/phase1/`, because they contain copyrighted text.
