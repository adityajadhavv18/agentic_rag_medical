"""Phase 5, step 4: check the golden sets against chunks_v1.

    uv run ingest/check_golden.py              # tune + test
    uv run ingest/check_golden.py all          # the merged file, before splitting

Every item loads under the eval harness schema (eval_harness.load_golden). Its evidence is matched exactly the way the
harness scores it (metrics._evidence_parts: whitespace collapsed, lowercase; parts joined by " ... " or " / " must
ALL be retrieved; " || " lists other wordings of one part). For each part:
  - FOUND: at least one wording sits inside one chunk's text (what the LLM reads), in the item's book;
  - UNIQUE: all chunks holding a wording share a block (one place in the book: window overlap or a table's
    repeated title is fine). Otherwise a wrong chunk could count as a hit. A fact the book states in two places is
    written as two wordings joined by " || ", each pinned to its own place; every wording must be found;
  - cross-book items (xb-) must draw their parts from at least two books.
no_answer items carry no evidence and no source_doc (the schema enforces it).
Writes data/golden/golden_map.json: id -> expected chunk ids, books, pdf and printed pages per part, so a UI or a
paper table can point at the exact source. Prints the mix by category, type and book.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval_harness import load_golden

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
CHUNKS = ROOT / "data" / "chunks" / "chunks_v1.jsonl"


def norm(text: str) -> str:  # same as eval_harness.metrics._norm
    return re.sub(r"\s+", " ", text).strip().lower()


def parts(evidence: str | None) -> list[list[str]]:  # same as eval_harness.metrics._evidence_parts
    return [[a for a in p.split(" || ") if a.strip()] for p in re.split(r" \.\.\. | / ", evidence or "") if p.strip()]


def main():
    names = ["all"] if sys.argv[1:] == ["all"] else ["tune", "test"]
    chunks = [json.loads(line) for line in CHUNKS.open()]
    normed = [norm(c["text"]) for c in chunks]
    problems, mapping, rows = [], {}, []
    for name in names:
        items = load_golden(str(GOLDEN / f"golden_mbbs_{name}_v1.jsonl"))
        for it in items:
            rows.append((name, it))
            if it.question_type == "no_answer":
                continue
            entry, books_used = [], set()
            for k, alts in enumerate(parts(it.evidence), 1):
                where = []
                for a in alts:  # each wording must be found, and in one place only
                    hits = [i for i, t in enumerate(normed) if norm(a) in t]
                    if not it.id.startswith("xb-"):
                        hits = [i for i in hits if chunks[i]["book_id"] == it.source_doc]
                    if not hits:
                        problems.append(f"{it.id} part {k}: NOT FOUND {a[:70]!r}")
                        continue
                    if not set.intersection(*[set(chunks[i]["block_ids"]) for i in hits]):
                        places = sorted({(chunks[i]["book_id"], tuple(chunks[i]["printed_pages"] or [])) for i in hits})
                        problems.append(f"{it.id} part {k}: NOT UNIQUE ({len(hits)} chunks: {places[:4]}) {a[:50]!r}")
                    where += hits
                if not where:
                    continue
                books_used |= {chunks[i]["book_id"] for i in where}
                entry.append({"part": k, "chunk_ids": [chunks[i]["chunk_id"] for i in where],
                              "book_id": chunks[where[0]]["book_id"], "pdf_pages": chunks[where[0]]["pdf_pages"],
                              "printed_pages": chunks[where[0]]["printed_pages"]})
            if it.id.startswith("xb-") and len(books_used) < 2:
                problems.append(f"{it.id}: cross-book item uses only {books_used}")
            mapping[it.id] = {"split": name, "source_doc": it.source_doc, "parts": entry}
    (GOLDEN / "golden_map.json").write_text(json.dumps(mapping, indent=1, ensure_ascii=False))

    print(f"items: {len(rows)}  ({', '.join(f'{n} {sum(r[0] == n for r in rows)}' for n in names)})")
    for name in names:
        its = [it for n, it in rows if n == name]
        print(f"\n[{name}] category:", dict(Counter(it.id.split("-")[0] for it in its)))
        print(f"[{name}] type:    ", dict(Counter(it.question_type for it in its)))
        print(f"[{name}] book:    ", dict(Counter(it.source_doc or "none" for it in its)))
    one_chunk = sum(1 for m in mapping.values() if len(m["parts"]) > 1
                    and set.intersection(*[set(p["chunk_ids"]) for p in m["parts"]]))
    multi = sum(1 for m in mapping.values() if len(m["parts"]) > 1)
    print(f"\nmulti-part items whose parts all sit in one chunk: {one_chunk}/{multi}")
    print(f"\nproblems: {len(problems)}")
    for p in problems:
        print("  " + p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
