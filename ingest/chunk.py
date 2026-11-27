"""Phase 4: clean blocks -> chunks (chunks_v1). $0, local.

    uv run ingest/chunk.py            # writes data/chunks/chunks_v1.jsonl + check.json + sample.md

Rules (his choices, 2026-11-27):
  * Text: 300 bge tokens, 50 overlap, inside one section only (same ruler as the cricket builds). The section's
    text and figure blocks form one stream; windows slice the ORIGINAL text by token offsets (bge lowercases).
  * Tiny sections (< 80 text tokens) are merged with a neighbouring section under the same parent heading
    (a sibling, or the parent's own intro + its first child), only while the merged text fits in one chunk
    (<= 300). The merged chunk's path is the parent; each member's sub-heading is kept inline as a line.
  * Tables: chunk `text` keeps the HTML (for the LLM); `embed_text` holds a plain "Header: value | ..." version.
    A table stays whole if its plain version is <= 350 tokens; otherwise it is split by rows, every piece repeating
    the title and header rows. Rows tied together by a rowspan are never separated.
  * Index pages -> their own `type=index` chunks (300/50 over the book's index). Park's front-matter extras
    (competency table, abbreviations) -> `type=extra`. "SECTION OUTLINE" divider pages (mini contents lists)
    are left out.
  * Every chunk: stable chunk_id, book/pdf mapping, pdf_pages + printed_pages [start, end], chapter, heading_path,
    stamp, type, block_ids.

The chunk `text` is what the LLM reads and what golden evidence is matched against. `embed_text` = stamp + plain
body, the only thing the retrievers index (Phase 6). Contextual notes later go in front of the body there.
"""

import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from functools import cache
from html.parser import HTMLParser
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = ROOT / "data" / "clean" / "blocks"
OUT = ROOT / "data" / "chunks"
VERSION = "chunks_v1"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS, OVERLAP_TOKENS = 300, 50
TINY_TOKENS = 80
TABLE_WHOLE_TOKENS = 350  # measured on the plain version (what gets embedded)
BOOK_ORDER = ["park_psm", "khurana_ophthal", "dhingra_ent", "bardale_fmt"]
SHORT = {"park_psm": "Park's PSM", "khurana_ophthal": "Khurana Ophthalmology",
         "dhingra_ent": "Dhingra ENT", "bardale_fmt": "Bardale Forensic Medicine"}
SEP = " › "


@cache
def tokenizer() -> Tokenizer:
    tok = Tokenizer.from_pretrained(EMBED_MODEL)
    tok.no_truncation()  # bge's tokenizer.json truncates at 512 by default
    return tok


def ntok(s: str) -> int:
    return len(tokenizer().encode(s, add_special_tokens=False).ids)


# ---------------------------------------------------------------- tables

TR = re.compile(r"<tr\b[^>]*>.*?</tr>", re.S)
CELL = re.compile(r"<(t[hd])\b([^>]*)>(.*?)</\1>", re.S)
TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")  # real tags only: "<6/60" or "< 5 years" stay


def strip_tags(s: str) -> str:
    """10<sup>3</sup> -> 10^3, H<sub>2</sub>O -> H2O, <b>x</b> -> x; any other tag -> a space."""
    s = re.sub(r"<sup\b[^>]*>", "^", s)
    s = re.sub(r"</?(?:sub|sup|b|i|u|em|strong)\b[^>]*>", "", s)
    return TAG.sub(" ", s)


def cell_text(inner: str) -> str:
    s = re.sub(r"<br\s*/?>|</li>", "; ", inner)
    s = html.unescape(strip_tags(s))
    s = re.sub(r"\s+", " ", s).strip(" ;")
    return re.sub(r"(;\s*)+", "; ", s)


def span(attrs: str, name: str) -> int:
    m = re.search(rf'{name}\s*=\s*"?(\d+)', attrs)
    return max(1, int(m.group(1))) if m else 1


def parse_table(t: str) -> dict:
    """Raw <tr> strings plus an expanded grid (rowspan/colspan copied into every cell they cover)."""
    open_tag = re.match(r"\s*(<table\b[^>]*>)", t).group(1)
    trs = TR.findall(t)
    rows = [[(kind, cell_text(inner), span(a, "colspan"), span(a, "rowspan")) for kind, a, inner in CELL.findall(tr)]
            for tr in trs]
    grid, pending = [], {}  # pending[col] = (cell, rows_left)
    reach = []  # last row index each row's rowspans reach into
    for r, cells in enumerate(rows):
        line, col, it, far = [], 0, iter(cells), r
        while True:
            if col in pending:
                cell, left = pending[col]
                line.append(cell)
                pending[col] = (cell, left - 1) if left > 1 else None
                if pending[col] is None:
                    del pending[col]
                col += 1
                continue
            c = next(it, None)
            if c is None:
                if pending and col <= max(pending):
                    col += 1  # gap before a carried cell (ragged row)
                    line.append(None)
                    continue
                break
            kind, text, cs, rs = c
            for k in range(cs):
                line.append((kind, text))
                if rs > 1:
                    pending[col + k] = ((kind, text), rs - 1)
            far = max(far, r + rs - 1)
            col += cs
        grid.append(line)
        reach.append(far)
    width = max((len(g) for g in grid), default=0)
    reach = [min(x, len(rows) - 1) for x in reach]  # a rowspan running past the last row

    title = 0  # leading full-width single-cell rows: "TABLE 3<br>Health-related ..." (not in one-column tables)
    while width > 1 and title < len(rows) - 1 and len(rows[title]) == 1 and rows[title][0][0] == "th" \
            and rows[title][0][2] >= width:
        title += 1
    head = title  # then leading all-<th> rows, keeping at least one body row
    while head < len(rows) - 1 and rows[head] and all(c[0] == "th" for c in rows[head]):
        head += 1
    while head < len(rows) - 1 and any(reach[r] >= head for r in range(title, head)):
        head += 1  # a header cell spanning down into the next row pulls that row into the header
    return {"open": open_tag, "trs": trs, "grid": grid, "reach": reach, "title": title, "head": head, "width": width}


def column_labels(p: dict) -> list[str]:
    labels = []
    for c in range(p["width"]):
        parts = []
        for row in p["grid"][p["title"]:p["head"]]:
            v = row[c][1] if c < len(row) and row[c] else ""
            if v and (not parts or parts[-1] != v):
                parts.append(v)
        labels.append(" / ".join(parts))
    return labels


def plain_rows(p: dict, idx: range) -> list[str]:
    labels = column_labels(p)
    out = []
    for r in idx:
        vals, seen = [], set()
        for c, cell in enumerate(p["grid"][r]):
            if not cell or not cell[1]:
                continue
            key = (labels[c] if c < len(labels) else "", cell[1])
            if key in seen:  # a colspan copy
                continue
            seen.add(key)
            vals.append(f"{key[0]}: {key[1]}" if key[0] and key[0] != key[1] else key[1])
        if vals:
            out.append(" | ".join(vals))
    return out


def plain_table(p: dict, idx: range, caption: str) -> str:
    lines = [caption] if caption else []
    lines += [p["grid"][r][0][1] for r in range(p["title"]) if p["grid"][r] and p["grid"][r][0]]
    return "\n".join(lines + plain_rows(p, idx))


def row_groups(p: dict) -> list[range]:
    """Body rows packed into atoms that no rowspan crosses."""
    groups, r, n = [], p["head"], len(p["trs"])
    while r < n:
        end = r
        k = r
        while k <= end:
            end = max(end, p["reach"][k])
            k += 1
        groups.append(range(r, min(end, n - 1) + 1))
        r = end + 1
    return groups


def grid_tr(cells) -> str:
    """One expanded grid row as plain <tr> HTML (merged cells already copied, so no spans needed)."""
    return "<tr>" + "".join(f"<{k}>{html.escape(v)}</{k}>" for k, v in cells) + "</tr>"


def table_units(p: dict, caption: str) -> list[dict]:
    """Smallest pieces the body may be cut into, coarse to fine, each no bigger than the limit where possible:
    a rowspan group (original HTML) -> one grid row -> one cell -> 300/50 token windows of one huge cell."""
    labels, limit = column_labels(p), TABLE_WHOLE_TOKENS - ntok(plain_table(p, range(0), caption))
    assert limit > 2 * OVERLAP_TOKENS, f"title + header rows leave no room for the body ({limit} tokens)"
    units = []
    for g in row_groups(p):
        plain = plain_rows(p, g)
        if ntok("\n".join(plain)) <= limit:
            units.append({"trs": [p["trs"][r] for r in g], "plain": plain, "rows": list(g)})
            continue
        for r in g:
            row_plain = plain_rows(p, range(r, r + 1))
            cells = [c for c in p["grid"][r] if c]
            if ntok("\n".join(row_plain)) <= limit:
                units.append({"trs": [grid_tr(cells)], "plain": row_plain, "rows": [r]})
                continue
            for c, cell in enumerate(p["grid"][r]):
                if not cell or not cell[1] or (c and p["grid"][r][c - 1] == cell):
                    continue
                label = labels[c] if c < len(labels) and labels[c] != cell[1] else ""
                text = f"{label}: {cell[1]}" if label else cell[1]
                offs = tokenizer().encode(text, add_special_tokens=False).offsets
                step, size = limit - OVERLAP_TOKENS, limit
                for s in range(0, len(offs), step):
                    e = min(s + size, len(offs))
                    part = text[offs[s][0]:offs[e - 1][1]]
                    units.append({"trs": [grid_tr([(cell[0], part)])], "plain": [part], "rows": [r]})
                    if e == len(offs):
                        break
    return units


def table_pieces(text: str, caption: str) -> list[tuple[str, str, list[int]]]:
    """[(html piece, plain piece, body row indices)]: whole, or split by rows with title + header rows repeated."""
    p = parse_table(text)
    body = range(p["head"], len(p["trs"]))
    whole_plain = plain_table(p, body, caption)
    lead = f"{caption}\n" if caption else ""
    if ntok(whole_plain) <= TABLE_WHOLE_TOKENS:
        return [(lead + text, whole_plain, list(body))]
    head = [p["open"], *p["trs"][:p["head"]]]
    fixed = plain_table(p, range(0), caption)

    def render(group):
        plain = "\n".join(([fixed] if fixed else []) + [x for u in group for x in u["plain"]])
        return (lead + "\n".join(head + [tr for u in group for tr in u["trs"]] + ["</table>"]), plain,
                sorted({r for u in group for r in u["rows"]}))

    pieces, cur = [], []
    for u in table_units(p, caption):
        if cur and ntok(render(cur + [u])[1]) > TABLE_WHOLE_TOKENS:
            pieces.append(render(cur))
            cur = []
        cur.append(u)
    if cur:
        pieces.append(render(cur))
    return pieces


class HTMLCheck(HTMLParser):
    VOID = {"br", "img", "hr"}

    def __init__(self):
        super().__init__()
        self.stack, self.ok = [], True

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.ok = False
        else:
            self.stack.pop()


def valid_html(s: str) -> bool:
    h = HTMLCheck()
    h.feed(s[s.index("<table"):])
    h.close()
    return h.ok and not h.stack


# ---------------------------------------------------------------- sections and windows

def load_blocks(book_id: str) -> list[dict]:
    return [json.loads(line) for line in (BLOCKS / f"{book_id}.jsonl").open()]


def is_outline(b: dict) -> bool:
    return b["text"].lstrip().upper().startswith("SECTION OUTLINE")


def is_table(b: dict) -> bool:
    return b["type"] in ("table", "extra") and b["text"].lstrip().startswith("<table") \
        and b["text"].rstrip().endswith("</table>")


def sections(blocks: list[dict]) -> list[dict]:
    """Maximal runs of content blocks with the same (chapter, heading path). Headings live in the path."""
    out = []
    for b in blocks:
        if b["type"] in ("heading", "index", "extra") or is_outline(b):
            continue
        key = (b["chapter_n"], tuple(b["heading_path"]))
        if not out or out[-1]["key"] != key:
            out.append({"key": key, "chapter_n": b["chapter_n"], "chapter": b["chapter"],
                        "path": tuple(b["heading_path"]), "blocks": []})
        out[-1]["blocks"].append(b)
    for s in out:
        s["text_tokens"] = sum(ntok(b["text"]) for b in s["blocks"] if not is_table(b))
    return out


def merge_tiny(secs: list[dict]) -> list[dict]:
    """Units of one or more sections. A tiny section joins a neighbour under the same parent heading while the
    merged text still fits in one chunk."""
    units = []
    for s in secs:
        u = units[-1] if units else None
        if u and u["chapter_n"] == s["chapter_n"] and s["text_tokens"] and u["text_tokens"] \
                and (u["tiny"] or s["text_tokens"] < TINY_TOKENS) \
                and u["text_tokens"] + s["text_tokens"] + ntok(s["path"][-1] if s["path"] else "") <= CHUNK_TOKENS:
            # the joined path P: s must sit directly under P, and every member already in u too
            cands = [u["path"]] if len(u["members"]) > 1 else [u["path"], u["path"][:-1]]
            P = next((c for c in cands if len(s["path"]) == len(c) + 1 and s["path"][:len(c)] == c), None)
            if P is not None:
                u["path"] = P
                u["members"].append(s)
                u["text_tokens"] += s["text_tokens"]
                u["tiny"] = u["text_tokens"] < TINY_TOKENS
                continue
        units.append({"chapter_n": s["chapter_n"], "chapter": s["chapter"], "path": s["path"], "members": [s],
                      "text_tokens": s["text_tokens"], "tiny": 0 < s["text_tokens"] < TINY_TOKENS})
    return units


def windows(pieces: list[tuple[str, dict]]) -> list[dict]:
    """pieces: (text, block) in order, joined by newlines into one stream. 300/50 windows over bge tokens,
    each sliced from the original text; returns text, block list and token count per window."""
    stream, spans = "", []  # spans: (char start, char end, block)
    for text, b in pieces:
        if stream:
            stream += "\n"
        spans.append((len(stream), len(stream) + len(text), b))
        stream += text
    offsets = tokenizer().encode(stream, add_special_tokens=False).offsets
    out, step = [], CHUNK_TOKENS - OVERLAP_TOKENS
    covered = 0
    for start in range(0, len(offsets), step):
        end = min(start + CHUNK_TOKENS, len(offsets))
        lo, hi = offsets[start][0], offsets[end - 1][1]
        assert lo <= covered or not stream[covered:lo].strip(), "window gap"  # coverage by construction
        covered = max(covered, hi)
        blocks = [b for s, e, b in spans if s < hi and e > lo and b is not None]
        out.append({"text": stream[lo:hi], "blocks": list({id(b): b for b in blocks}.values()), "n_tokens": end - start})
        if end == len(offsets):
            break
    assert not stream[covered:].strip(), "tail not covered"
    return out


# ---------------------------------------------------------------- chunks

def page_range(blocks: list[dict], key: str):
    vals = [p for b in blocks if b[key] for p in b[key] if p is not None]
    return [min(vals), max(vals)] if vals else None


def chunk_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def make_chunk(meta: dict, ctype: str, chapter_n, chapter, path, text, plain, blocks, key, **extra) -> dict:
    stamp = SEP.join([SHORT[meta["book_id"]], *([chapter] if chapter else []), *path])
    return {"chunk_id": chunk_id(meta["book_id"], *key), "version": VERSION, "book_id": meta["book_id"],
            "book": meta["title"], "pdf_file": meta["file"], "subject": meta["subject"], "chapter_n": chapter_n,
            "chapter": chapter, "heading_path": list(path), "stamp": stamp, "type": ctype, "text": text,
            "embed_text": f"{stamp}\n{plain}", "n_tokens": ntok(plain),
            "pdf_pages": page_range(blocks, "pdf_pages"), "printed_pages": page_range(blocks, "printed_pages"),
            "block_ids": [b["id"] for b in blocks], **extra}


def plain_text(s: str) -> str:
    """Embed version of a text stream: strip stray HTML (one truncated Park's table is typed text)."""
    if not TAG.search(s):
        return s
    s = re.sub(r"</?[A-Za-z][^<>]*$", "", re.sub(r"^[^<>]*>", "", s))  # half a tag at a window edge
    return re.sub(r"[ \t]+", " ", html.unescape(strip_tags(s)))


def table_caption(blocks: list[dict], i: int, p: dict) -> str:
    """The block right before a table without its own title row is its title if it starts with "Table N"
    (up to 60 tokens, any lines) or is one short line (up to 40 tokens)."""
    if p["title"] or i == 0:
        return ""
    prev = blocks[i - 1]
    t = prev["text"].strip()
    if prev["type"] != "text":
        return ""
    if re.match(r"(?:TABLE|Table)\s*[\dIVX]", t) and ntok(t) <= 60:  # Park's: "TABLE 23\nInfant mortality rate ..."
        return t
    return t if "\n" not in t and ntok(t) <= 40 else ""


def book_chunks(meta: dict) -> tuple[list[dict], dict]:
    blocks = load_blocks(meta["book_id"])
    secs = sections(blocks)
    units = merge_tiny(secs)
    chunks, stats = [], Counter()
    for u in units:
        multi = len(u["members"]) > 1
        stats["merged_units"] += multi
        stats["sections_merged"] += multi * len(u["members"])
        uid = u["members"][0]["blocks"][0]["id"]
        pieces = []
        for m in u["members"]:
            if multi and m["path"] != u["path"] and m["text_tokens"]:
                pieces.append((m["path"][-1], None))  # the member's sub-heading, inline
            for i, b in enumerate(m["blocks"]):
                if is_table(b):
                    p = parse_table(b["text"])
                    cap = table_caption(m["blocks"], i, p)
                    parts = table_pieces(b["text"], cap)
                    for k, (h, plain, rows) in enumerate(parts):
                        chunks.append(make_chunk(meta, "table", m["chapter_n"], m["chapter"], m["path"], h, plain,
                                                 [b], (b["id"], "table", k), table_piece=[k + 1, len(parts)],
                                                 table_rows=rows, table_header_rows=p["head"] - p["title"]))
                else:
                    pieces.append((b["text"], b))
        if any(b for _, b in pieces):
            for k, w in enumerate(windows(pieces)):
                chunks.append(make_chunk(meta, "text", u["chapter_n"], u["chapter"], u["path"], w["text"],
                                         plain_text(w["text"]), w["blocks"], (uid, "text", k),
                                         has_figure=any(b["type"] == "figure" for b in w["blocks"]),
                                         merged_sections=len(u["members"]) if multi else 1))
    for ctype in ("index", "extra"):
        group = [b for b in blocks if b["type"] == ctype]
        if not group:
            continue
        label = "Index" if ctype == "index" else "Front matter"
        for b in [b for b in group if is_table(b)]:
            parts = table_pieces(b["text"], "")
            p = parse_table(b["text"])
            for k, (h, plain, rows) in enumerate(parts):
                chunks.append(make_chunk(meta, ctype, None, None, (label,), h, plain, [b], (b["id"], ctype, k),
                                         table_piece=[k + 1, len(parts)], table_rows=rows,
                                         table_header_rows=p["head"] - p["title"]))
        rest = [(b["text"], b) for b in group if not is_table(b)]
        if rest:
            for k, w in enumerate(windows(rest)):
                chunks.append(make_chunk(meta, ctype, None, None, (label,), w["text"], w["text"], w["blocks"],
                                         (rest[0][1]["id"], ctype, k)))
    stats["sections"] = len(secs)
    stats["tiny_sections"] = sum(0 < s["text_tokens"] < TINY_TOKENS for s in secs)
    stats["outline_blocks_left_out"] = sum(is_outline(b) for b in blocks)
    return chunks, stats, blocks, secs


# ---------------------------------------------------------------- checks

def check(chunks: list[dict], all_blocks: dict, all_secs: dict) -> dict:
    out = {}
    in_chunk = Counter(bid for c in chunks for bid in c["block_ids"])
    by_id = {b["id"]: b for bl in all_blocks.values() for b in bl}

    # 1. every content block lands in >= 1 chunk
    content = [b for b in by_id.values() if b["type"] != "heading" and not is_outline(b)]
    missing = [b["id"] for b in content if not in_chunk[b["id"]]]
    out["1_blocks_covered"] = {"content_blocks": len(content), "missing": len(missing), "missing_ids": missing[:20],
                               "left_out_outline": sum(is_outline(b) for b in by_id.values())}
    # table rows all kept
    rows_lost = []
    for bid in {c["block_ids"][0] for c in chunks if "table_rows" in c}:
        p = parse_table(by_id[bid]["text"])
        kept = {r for c in chunks if c.get("table_rows") is not None and c["block_ids"][0] == bid
                for r in c["table_rows"]}
        if kept != set(range(p["head"], len(p["trs"]))):
            rows_lost.append(bid)
    out["1_table_rows_lost"] = rows_lost

    # 2. no chunk crosses a section: one chapter, every block's path under the chunk's path
    cross = [c["chunk_id"] for c in chunks if c["type"] in ("text", "table") and any(
        by_id[b]["chapter_n"] != c["chapter_n"] or by_id[b]["heading_path"][:len(c["heading_path"])] != c["heading_path"]
        for b in c["block_ids"])]
    out["2_cross_section"] = {"chunks": len(cross), "ids": cross[:20]}

    # 3. table pieces: valid HTML, header rows repeated, no tags left in the embed text
    tabs = [c for c in chunks if "table_piece" in c]
    bad_html = [c["chunk_id"] for c in tabs if not valid_html(c["text"])]
    no_head = []
    for c in tabs:
        p = parse_table(by_id[c["block_ids"][0]]["text"])
        head_trs = p["trs"][:p["head"]]
        if any(tr not in c["text"] for tr in head_trs):
            no_head.append(c["chunk_id"])
    split = [c for c in tabs if c["table_piece"][1] > 1]
    out["3_tables"] = {"table_chunks": len(tabs), "tables": len({c["block_ids"][0] for c in tabs}),
                       "tables_split": len({c["block_ids"][0] for c in split}), "split_pieces": len(split),
                       "invalid_html": bad_html, "missing_header_rows": no_head,
                       "tables_without_header_row": len({c["block_ids"][0] for c in tabs if not c["table_header_rows"]}),
                       "pieces_over_limit": sum(c["n_tokens"] > TABLE_WHOLE_TOKENS for c in tabs),
                       "embed_with_tags": sum(bool(re.search(r"</?t[rdh]\b", c["embed_text"])) for c in chunks)}

    # 4. sizes
    def spread(xs):
        xs = sorted(xs)
        q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
        return {"n": len(xs), "min": xs[0], "p10": q(.1), "median": q(.5), "p90": q(.9), "max": xs[-1],
                "under_80": sum(x < 80 for x in xs), "full_300": sum(x >= CHUNK_TOKENS for x in xs)}

    out["4_sizes"] = {t: spread([c["n_tokens"] for c in chunks if c["type"] == t])
                      for t in ("text", "table", "index", "extra")}
    embed = [ntok(c["embed_text"]) for c in chunks]
    out["4_embed_tokens"] = {"max": max(embed), "over_412": sum(e > 412 for e in embed),
                             "over_512": sum(e > 512 for e in embed)}
    out["4_stamp_tokens"] = spread([ntok(c["stamp"]) for c in chunks])
    out["unique_ids"] = len({c["chunk_id"] for c in chunks}) == len(chunks)
    return out


def main():
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    OUT.mkdir(parents=True, exist_ok=True)
    chunks, per_book, all_blocks, all_secs = [], {}, {}, {}
    for book_id in BOOK_ORDER:
        cs, stats, blocks, secs = book_chunks(manifest[book_id])
        chunks += cs
        all_blocks[book_id], all_secs[book_id] = blocks, secs
        per_book[book_id] = {"chunks": len(cs), **Counter(c["type"] for c in cs), **stats,
                             "tiny_text_chunks": sum(c["type"] == "text" and c["n_tokens"] < TINY_TOKENS for c in cs)}
        print(book_id, per_book[book_id])
    with (OUT / f"{VERSION}.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    result = {"version": VERSION, "total_chunks": len(chunks), "per_book": per_book,
              "checks": check(chunks, all_blocks, all_secs)}
    (OUT / "check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["checks"], indent=2))
    rng = random.Random(42)  # 10 random chunks to read by eye
    sample = rng.sample(chunks, 10)
    (OUT / "sample.md").write_text("\n\n---\n\n".join(
        f"### {c['chunk_id']} · {c['type']} · pdf {c['pdf_pages']} · printed {c['printed_pages']} · {c['n_tokens']} tok\n"
        f"**{c['stamp']}**\n\n{c['text']}\n\n_embed:_ {c['embed_text'][:600]}" for c in sample))


if __name__ == "__main__":
    main()
