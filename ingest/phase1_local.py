"""Phase 1, local baselines on the Mac ($0): run Apple Vision OCR and dump the existing hidden text layer for the
12 test pages, so they can be scored next to the Kaggle models.

    uv run ingest/phase1_local.py

Writes data/phase1/results/apple_vision/T01.md ... and data/phase1/results/text_layer/T01.md ..., each with log.json.
Apple Vision = the text recognizer built into macOS (VNRecognizeTextRequest, "accurate" level, language correction on).
It returns lines of text only: no tables, no headings, so it is a plain-text baseline.
"""

import json
import time
from pathlib import Path

import pymupdf
import Quartz
import Vision
from Foundation import NSData

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "data" / "phase1"
OUT = PHASE1 / "results"


def apple_vision(png: bytes) -> str:
    data = NSData.dataWithBytes_length_(png, len(png))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(err)
    obs = [{"x0": o.boundingBox().origin.x, "x1": o.boundingBox().origin.x + o.boundingBox().size.width,
            "top": 1 - (o.boundingBox().origin.y + o.boundingBox().size.height),  # Vision's y runs bottom-up
            "text": o.topCandidates_(1)[0].string()} for o in req.results()]
    return reading_order(obs)


def reading_order(obs: list[dict]) -> str:
    """Put Vision's text pieces back in reading order: find the empty vertical gutters between columns, read each
    column top to bottom, and keep pieces on one visual line together (left to right). Pieces wider than 55% of the
    page (headings, captions) span columns: those above the columns come first, the rest last."""
    narrow = [o for o in obs if o["x1"] - o["x0"] < 0.55]
    covered = [False] * 100
    for o in narrow:
        for b in range(int(o["x0"] * 100), min(int(o["x1"] * 100) + 1, 100)):
            covered[b] = True
    cuts, b = [], 10
    while b < 90:  # a gutter = a run of >= 2 empty bins away from the page edges
        if not covered[b]:
            e = b
            while e < 90 and not covered[e]:
                e += 1
            if e - b >= 2:
                cuts.append((b + e) / 200)
            b = e
        b += 1

    def column(o):
        mid = (o["x0"] + o["x1"]) / 2
        return sum(mid > c for c in cuts)

    def lines(items):  # group pieces whose tops are within 0.8% of the page height, then read left to right
        items = sorted(items, key=lambda o: o["top"])
        out, cur = [], []
        for o in items:
            if cur and o["top"] - cur[0]["top"] > 0.008:
                out.append(cur)
                cur = []
            cur.append(o)
        if cur:
            out.append(cur)
        return [" ".join(p["text"] for p in sorted(line, key=lambda p: p["x0"])) for line in out]

    wide = [o for o in obs if o["x1"] - o["x0"] >= 0.55]
    first_top = min((o["top"] for o in narrow), default=1)
    head = [o for o in wide if o["top"] < first_top]
    tail = [o for o in wide if o["top"] >= first_top]
    cols = [lines([o for o in narrow if column(o) == c]) for c in range(len(cuts) + 1)]
    return "\n".join(lines(head) + [l for col in cols for l in col] + lines(tail))


def run(name: str, fn) -> None:
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PHASE1 / "phase1_pages.pdf")
    log = {"model": name, "pages": {}}
    for i, page in enumerate(doc, start=1):
        t = time.time()
        text = fn(page)
        (dst / f"T{i:02d}.md").write_text(text)
        log["pages"][f"T{i:02d}"] = {"seconds": round(time.time() - t, 2), "chars": len(text)}
    (dst / "log.json").write_text(json.dumps(log, indent=1))
    secs = [v["seconds"] for v in log["pages"].values()]
    print(f"{name}: {len(secs)} pages, avg {sum(secs) / len(secs):.2f} s/page")


if __name__ == "__main__":
    run("text_layer", lambda page: page.get_text())
    run("apple_vision", lambda page: apple_vision(page.get_pixmap(dpi=200).tobytes("png")))
