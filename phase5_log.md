# Phase 5: Golden set

**Goal:** an exam paper with an answer key, used to score every later retrieval and answering step. Each answer points to the exact passage in the book, so a script can check whether retrieval found it.

**Choices** (his, 2026-11-28):
- 100 questions, split 60 tune / 40 test;
- kept in git-ignored `data/golden/`, because the evidence quotes are copyrighted book text;
- no questions on Bardale's IPC/CrPC section numbers (2011 book, before the BNS of 2024).

Questions were written in-session, from the clean chunk text only. $0.

## Method

| Step | Script | What it does |
|---|---|---|
| 1. Draw passages | `ingest/golden_pool.py` | Seeded random draw (seed 20261128), so questions come from random places, not from topics the writer knows |
| 2. Write | (by hand) | One question per passage, phrased as a student would ask, answered only from the book's text |
| 3. Merge + split | `ingest/golden_split.py` | Drafts → `golden_mbbs_all_v1.jsonl` → seeded 60/40 split, balanced by category and book |
| 4. Check | `ingest/check_golden.py` | Schema, evidence found and unique, cross-book, mix; writes `golden_map.json` |

**How passages were drawn:**
- **Text:** chunks of ≥ 150 tokens, at most one per chapter until the pool is full (~1.5× the quota).
- **Tables:** first pieces with ≥ 60 plain tokens.
- **Figures:** chunks with a caption or figure description of ≥ 8 words beyond its label. This rule was added after the first draw: 4 of Khurana's 5 captions only named the picture ("Schiotz tonometer"), so there was nothing to ask. Figures are shuffled last, so the text and table draws stayed byte-identical. The pool was raised to 3× + 2 for the same reason, and the figure questions first written for Bardale and Dhingra were replaced from the new draw.

**Taking and skipping passages.** Passages were taken in pool order. One was skipped only if it gave nothing to ask, and nothing else was hand-picked.

| | Text | Tables | Figures |
|---|---|---|---|
| Park's | 0 | 1 (milk table: OCR columns scrambled) | 2 (chart titles only) |
| Khurana | 1 (chapter outline list) | 1 (steroid table: the answer rows hold only common drug names, so no quote can pin the right table piece) | 2 (names only) |
| Dhingra | 0 | 0 | 0 |
| Bardale | 2 (a reference list; IPC exceptions, i.e. law) | 0 | 3 (names only) |

**Cross-book and no-answer questions** were written from targeted searches of the chunks:
- **Cross-book:** 10 pairs, each stating one fact from each book. Pairs: Park's+Dhingra 3, Park's+Khurana 2, Park's+Bardale 2, Khurana+Dhingra 2, Khurana+Bardale 1.
- **No-answer:** 6 outside the books' scope (e.g. Gleason score, SGLT2 inhibitors) and 6 near-misses, where the topic is covered but the fact isn't (e.g. the BNS section for murder, the CFTR mutation name, balloon sinuplasty, the price of anti-snake venom). The key terms were confirmed absent from all 12,591 chunks by regex search. An mpox-vaccine candidate was dropped because Park's discusses mpox vaccination.

## Format

- It is the eval harness schema, unchanged (`id, question_type, source_doc, question, ground_truth, evidence`), so the harness scores it as is.
- The harness has 4 types (simple, exact_term, multi_part, no_answer). The category is the id prefix, as in cricket:

| Prefix | Category |
|---|---|
| `sim` | simple fact |
| `ex` | exact term, number or dose |
| `tab` | table lookup |
| `fig` | figure or caption |
| `mp` | multi-part |
| `xb` | cross-book |
| `na` | no-answer |

**Evidence quotes:**
- Each is copied word for word from the chunk `text` (the field the LLM reads; tables keep their HTML).
- Parts that must all be retrieved are joined with ` ... `.
- A fact the book states in two places is written as two wordings joined by ` || `, each pinned to its own place (4 items).
- Table evidence is the text of one cell, so it never crosses HTML tags.

**`source_doc`** is the book id. A cross-book item lists its main book; evidence recall still checks the parts from both books, but the harness's book-hit metric sees only one.

## Results

| Category | All | Tune | Test |
|---|---:|---:|---:|
| simple | 25 | 15 | 10 |
| exact term | 15 | 9 | 6 |
| table | 15 | 9 | 6 |
| figure | 8 | 5 | 3 |
| multi-part | 15 | 9 | 6 |
| cross-book | 10 | 6 | 4 |
| no-answer | 12 | 7 | 5 |
| **Total** | **100** | **60** | **40** |

| Book (`source_doc`) | Tune | Test |
|---|---:|---:|
| Park's | 19 | 13 |
| Khurana | 13 | 7 |
| Dhingra | 11 | 8 |
| Bardale | 10 | 7 |
| none (no-answer) | 7 | 5 |

**Other counts:**
- Harness types: tune has 23 simple, 15 exact_term, 15 multi_part and 7 no_answer; test has 14, 11, 10 and 5.
- 88 answerable items have 115 evidence parts: 99 in text chunks, 16 in table chunks.
- In 15 of the 25 multi-part items, all parts sit in one chunk; the other 10 need two chunks (including all 10 cross-book items).

## Checks (all passed)

1. **Schema:** all 100 items load with `eval_harness.load_golden` (ids unique, `source_doc` present exactly when answerable).
2. **Found:** every evidence wording is inside a chunk's text in the item's book, matched exactly as the harness matches (whitespace collapsed, lowercase).
3. **Unique:** every wording sits in one place in the corpus (all chunks holding it share a block), so a wrong chunk can't count as a hit.
4. **Cross-book:** every `xb` item draws its parts from two books.
5. **Mix and split:** as in the tables above. Each book is about 60/40 between tune and test.

## Problems and fixes

| Problem | Fix |
|---|---|
| 4 of Khurana's 5 figure candidates only named the picture | Explicit rule (caption ≥ 8 words); figure pool re-drawn (text and tables unchanged) and raised to 3× + 2 |
| 6 quotes also matched a second place in the same book (e.g. "end of 1st week" in body text; an outbreak sentence in another chapter) | Quotes lengthened, or both places listed as ` \|\| ` wordings when both state the fact; the checker now tests uniqueness per wording |
| "Level VI" would also match inside "Level VII" (substring match) | Question moved to a unique cell |
| Round-robin split put Park's surplus into test (Park's 20, Bardale 3) | Items spread proportionally per book before cutting at 60%. Fixed before any run on either split |

## Known limits

- One writer (me). Questions come from random draws, but the wording and answers are mine; you may want to spot-check some.
- The figure questions answer from captions or olmOCR's figure descriptions, not from labels inside diagrams (those aren't in the text).
- One Bardale caption quote keeps the PDF's own split word ("part icles"), since evidence must match the chunk exactly.
- Park's 28th edition numbers (e.g. sex ratio at birth 907, NFHS-5) are the reference answers, even where newer data exists.
- **For Phase 6:** a pipeline must return chunk `text` (not `embed_text`) as its contexts, or table evidence won't match.

## Outputs (git-ignored: book text)

- `data/golden/golden_mbbs_tune_v1.jsonl` (60), `golden_mbbs_test_v1.jsonl` (40), `golden_mbbs_all_v1.jsonl` (100)
- `data/golden/golden_map.json`: id → expected chunk ids, book, PDF and printed pages per evidence part
- `data/golden/draft_*.jsonl`: the hand-written drafts the split is built from
- `data/golden/pool/<book>.md`: the drawn passages

**The test split has never been run. It is used once, only when you say so.**
