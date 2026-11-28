"""Phase 5, step 3: merge the hand-written drafts and split them into tune and test (seeded, before any retrieval run).

    uv run ingest/golden_split.py     # data/golden/draft_*.jsonl -> golden_mbbs_{all,tune,test}_v1.jsonl

The split keeps every category in the same 60/40 ratio and spreads books evenly inside each category: items of one
category are shuffled per book with SEED, each book's items are spaced evenly through the category's order, and the
first 60% go to tune. (A first version dealt round-robin, which sent Park's surplus to test: test had Park's 20,
Bardale 3. Fixed before any run on either split.) The category is the id prefix (sim, ex, tab, fig, mp, xb, na), as
the cricket sets did ("which-01").
The test split is used once, only when he says so.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
SEED = 20261128
TUNE_SHARE = 0.6
CATEGORIES = ["sim", "ex", "tab", "fig", "mp", "xb", "na"]


def main():
    items = [json.loads(line) for f in sorted(GOLDEN.glob("draft_*.jsonl")) for line in f.open() if line.strip()]
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    by_cat = defaultdict(lambda: defaultdict(list))
    for i in items:
        by_cat[i["id"].split("-")[0]][i["id"].split("-")[1] if i["id"].count("-") == 2 else "-"].append(i)
    assert set(by_cat) == set(CATEGORIES), sorted(by_cat)
    rng = random.Random(SEED)
    tune, test = [], []
    for cat in CATEGORIES:
        books = by_cat[cat]
        for b in sorted(books):
            books[b].sort(key=lambda i: i["id"])
            rng.shuffle(books[b])
        # each book's items spread evenly over the category's order (item r of n sits at (r + 0.5) / n), so every
        # book lands ~60% in tune; plain round-robin pushed the biggest book's surplus into test
        dealt = sorted(((r + 0.5) / len(books[b]), b, i) for b in sorted(books) for r, i in enumerate(books[b]))
        dealt = [i for _, _, i in sorted(dealt, key=lambda x: (x[0], x[1]))]
        k = round(len(dealt) * TUNE_SHARE)
        tune += dealt[:k]
        test += dealt[k:]
    key = lambda i: (CATEGORIES.index(i["id"].split("-")[0]), i["id"])
    for name, rows in [("all", items), ("tune", tune), ("test", test)]:
        with (GOLDEN / f"golden_mbbs_{name}_v1.jsonl").open("w") as f:
            for i in sorted(rows, key=key):
                f.write(json.dumps(i, ensure_ascii=False) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
