# Phase 4: Chunking

**Goal:** cut the 34,285 clean blocks into search-sized chunks. Each chunk knows exactly where it came from (book, PDF page, printed page, chapter, heading path), so a UI can open the source page and an answer can cite "Park's p. 295".

**Choices** (his, 2026-11-27):
- tables: the chunk keeps the HTML for the LLM, but a plain `Header: value | …` version is embedded;
- sections under 80 tokens are merged with a neighbour under the same parent heading;
- 300/50 bge tokens, the same as the cricket builds.

Script: `ingest/chunk.py`. $0, local, about 17 s.

## Rules

| What | Rule |
|---|---|
| Text | 300 `BAAI/bge-small-en-v1.5` tokens with 50 overlap. A window slices the original text by token offsets (bge lowercases). It never crosses a section or chapter. Figure and caption blocks stay in their section's text. |
| Tiny sections | A section under 80 text tokens joins its neighbour (a sibling, or its parent's intro) if the merged text fits in one chunk (≤ 300). The chunk's path is the parent heading; each sub-heading stays inline as a line. |
| Tables | Kept whole if the plain version is ≤ 350 tokens. Otherwise they are split by rows, and every piece repeats the title and header rows. Rows tied by a rowspan stay together. Fallbacks, only when a piece is still too big: one row rebuilt from the expanded grid, then one cell, then 300/50 windows of a huge cell. |
| Table titles | The block just before a table becomes its title line if it starts with "Table N" (≤ 60 tokens) or is one short line (≤ 40 tokens). |
| Index / front matter | Index pages → `type=index` chunks (300/50 over each book's index). Park's competency table and abbreviations → `type=extra`. |
| Left out | 16 "SECTION OUTLINE" divider blocks (Khurana 6, Dhingra 10). These are mini contents lists that would match many questions by mistake. |
| Header stamp | `Book › Chapter › heading path`, e.g. `Park's PSM › Epidemiology of Communicable Diseases › CHOLERA`. Median 20 tokens, max 49. |

**Two text fields per chunk.**
- `text` is what the LLM reads and what golden evidence is matched against.
- `embed_text` = stamp + plain body. This is the only text the retrievers index.

The plain body has its HTML removed: `<sup>` → `^`, `<sub>` is joined, and other tags become spaces. Only real tags are removed, so "<6/60" and "< 5 years" stay.

**Payload per chunk:**
- `chunk_id`: a hash of book + first block + type + position, stable across re-runs
- `version`, `book_id`, `book`, `pdf_file`, `subject`, `chapter_n`, `chapter`, `heading_path`, `stamp`, `type`
- `pdf_pages [start, end]` and `printed_pages [start, end]`
- `block_ids`, `n_tokens`
- text chunks also get `has_figure` and `merged_sections`
- table chunks also get `table_piece [i, n]`, `table_rows` and `table_header_rows`

There are no on-page coordinates (olmOCR gives none).

## Results

| | Park's | Khurana | Dhingra | Bardale | Total |
|---|---:|---:|---:|---:|---:|
| Chunks | 6,600 | 2,321 | 1,805 | 1,865 | **12,591** |
| Text | 5,475 | 2,091 | 1,581 | 1,459 | 10,606 |
| Table | 1,052 | 88 | 151 | 374 | 1,665 |
| Index | 46 | 142 | 73 | 32 | 293 |
| Extra | 27 | – | – | – | 27 |
| Sections | 3,102 | 344 | 711 | 1,062 | 5,219 |
| Tiny sections (< 80 tok) | 468 | 23 | 128 | 368 | 987 |
| Sections merged (→ units) | 610 → 273 | 17 → 8 | 155 → 69 | 462 → 185 | 1,244 → 535 |
| Text chunks still < 80 tok | 282 | 44 | 83 | 96 | 505 (4.8%) |

**Sizes (body tokens):**
- Text: median 300, 10th percentile 103. 6,051 chunks are full; short ones are section ends.
- Tables: median 206, max 350.
- Embed text (stamp + body): max 385 tokens. None exceed 412, so there is room for a ~100-token contextual note under bge's 512 limit.

**Tables:**
- 1,094 tables. 237 are split, into 826 pieces.
- 107 have no header row (lists set as tables), so there is nothing to repeat.
- The other table block (Park's PDF 711) is a truncated table that Phase 3 typed as text, so it is chunked as text.

**Other counts:**
- 699 text chunks include a figure or caption.
- 18 `extra` chunks have no printed page (front matter).
- Output SHA-256 `787147ca…4d44`; a re-run gives the identical file.

## Checks (all passed)

1. **Coverage:**
   - all 28,929 content blocks are in at least one chunk (16 outline blocks left out on purpose);
   - every table body row is in some piece;
   - text windows cover their whole stream (asserted for every window).
2. **No section crossing:** 0 chunks whose blocks fall outside the chunk's chapter or heading path.
3. **Tables:**
   - 0 invalid HTML pieces;
   - 0 pieces missing their header rows;
   - 0 pieces over 350 plain tokens;
   - 0 embed texts with leftover tags.
4. **Size spread:** see Results.
5. **Eye check, 10 random chunks** (seed 42, `data/chunks/sample.md`): 9 text, 1 table.
   - Pages, stamps and slices are right.
   - The table piece carries its "Table 32.60" title, and its plain version reads well.
   - Windows starting mid-sentence are the 50-token overlap, the same as in cricket.
   - This check found the missing two-line Park's titles (fixed below).
6. **Stable IDs:** all 12,591 IDs are unique; two runs give a byte-identical file.

## Problems and fixes

| Problem | Fix |
|---|---|
| bge's `tokenizer.json` truncates at 512, so long blocks were under-counted | `no_truncation()`, as in cricket |
| A rowspan ran past a table's last row (index error) | Rowspan reach capped at the last row |
| 29 table pieces over the limit (max 2,128 tokens): one-cell "page in a box" rows, or big rowspan groups | Fallbacks: one grid row → one cell → 300/50 windows of a huge cell; pieces packed by exact token count |
| A one-column table's 16 `<td>` rows were all taken as "title" rows, leaving no room for the body (block lost) | Title rows must be `<th>`; an assert now fails loudly if the title and header leave no room |
| The first tag stripper treated any `<…>` as a tag and ate spans like "<140 … >90" (tiny chunks 505 → 562) | Only real tags (`<letter…>`) are removed: the same bug as the Phase 1 scorer |
| "10<sup>3</sup>" would become "10 3" | `<sup>` → `^`, `<sub>`/`<b>`/`<i>` joined |
| A window edge cut a tag of the truncated Park's table in half | Half-tags at window edges are dropped in the embed text |
| Park's two-line titles ("TABLE 23" + title line) were not attached | Blocks starting "Table N" (≤ 60 tokens) count as titles. Tables with a "Table N" title in their chunk: Park's 44 → 152 of 565. The other ~375 Park's tables print no numbered title (boxed lists, continued tables); their stamp gives the context. |

## Known limits (carried forward)

- Phase 3's heading noise shows up in some stamps, e.g. a table under "› Respiratory System" when it belongs to "Diagnosis of poisoning".
- 10 empty `<html></html>` pairs from olmOCR remain in the LLM text (stripped from the embed text).
- Table footnotes and sources after a table ("Source : (6)") stay in the section text, not in the table chunk.
- No on-page coordinates, so highlighting in the UI needs a later step.

## Outputs (git-ignored: book text)

- `data/chunks/chunks_v1.jsonl`: 12,591 chunks.
- `data/chunks/check.json`: per-book counts and every check.
- `data/chunks/sample.md`: the 10 eye-check chunks.
