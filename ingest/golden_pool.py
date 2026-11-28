"""Phase 5, step 1: draw the passages the golden questions are written from (seeded, so the draw is reproducible).

    uv run ingest/golden_pool.py      # -> data/golden/pool/<book_id>.md (book text: git-ignored)

Questions come from random places, not from topics the writer happens to know:
  * text:   chunks shuffled with SEED, keeping the first one from each chapter not yet used (spreads over chapters);
            only chunks with >= 150 tokens (a section end with half a sentence makes a poor question)
  * table:  first pieces of tables with >= 60 plain tokens, shuffled with SEED
  * figure: text chunks holding a caption or figure description of >= 8 words beyond its label ("Fig. 2.6 ...",
            "[Figure: ...]"), shuffled with SEED. A caption that only names the picture ("Schiotz tonometer")
            gives nothing to ask (rule added after the first draw: 4 of Khurana's 5 captions were names only).
The pool is ~1.5x the quota (figures 3x + 2: many captions still give nothing to ask) so an unusable passage (OCR noise,
a bare list of references, a caption that only names the picture) can be skipped, in pool order; skips are
counted in the golden log. Quotas per book are in QUOTA (his mix, 2026-11-28).
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks" / "chunks_v1.jsonl"
OUT = ROOT / "data" / "golden" / "pool"
SEED = 20261128
# book: (text questions = simple + exact + multi-part, tables, figures)
QUOTA = {"park_psm": (22, 7, 1), "khurana_ophthal": (12, 2, 3), "dhingra_ent": (12, 3, 2), "bardale_fmt": (9, 3, 2)}
CAPTION = re.compile(r"^(?:FIGS?\.?|Figs?\.?|Figure)\s*[\d.]+[A-Z]?(?:\s*(?:to|and)\s*[\d.]+[A-Z]?)?[:.]?\s*(.*)$|^\[Figure[^:]*:\s*(.*?)\]?$", re.M)


def rich_caption(text: str) -> bool:
    """A printed caption or figure description with something to ask about: >= 8 words beyond the figure label."""
    return any(len((m.group(1) or m.group(2) or "").split()) >= 8 for m in CAPTION.finditer(text))


def pool(book: str, chunks: list[dict]) -> dict[str, list[dict]]:
    n_text, n_table, n_fig = QUOTA[book]
    rng = random.Random(f"{SEED}-{book}")
    text = [c for c in chunks if c["type"] == "text" and c["n_tokens"] >= 150]
    rng.shuffle(text)
    picked, chapters = [], set()
    for c in text:
        if c["chapter_n"] not in chapters:
            picked.append(c)
            chapters.add(c["chapter_n"])
        if len(picked) == round(n_text * 1.5):
            break
    if len(picked) < round(n_text * 1.5):  # fewer chapters than needed: allow repeats
        picked += [c for c in text if c not in picked][:round(n_text * 1.5) - len(picked)]
    tables = [c for c in chunks if c["type"] == "table" and c["table_piece"][0] == 1 and c["n_tokens"] >= 60]
    rng.shuffle(tables)
    figs = [c for c in chunks if c["type"] == "text" and c["has_figure"] and rich_caption(c["text"])]
    rng.shuffle(figs)
    return {"text": picked, "table": tables[:round(n_table * 1.5) + 1], "figure": figs[:3 * n_fig + 2]}


def main():
    chunks = [json.loads(line) for line in CHUNKS.open()]
    OUT.mkdir(parents=True, exist_ok=True)
    for book in QUOTA:
        p = pool(book, [c for c in chunks if c["book_id"] == book])
        lines = []
        for kind, cs in p.items():
            for i, c in enumerate(cs, 1):
                body = c["embed_text"] if kind == "table" else c["text"]
                lines.append(f"## {kind} {i} · {c['chunk_id']} · printed {c['printed_pages']}\n{c['stamp']}\n\n{body}\n")
        (OUT / f"{book}.md").write_text("\n".join(lines))
        print(book, {k: len(v) for k, v in p.items()})


if __name__ == "__main__":
    main()
