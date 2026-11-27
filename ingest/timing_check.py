"""Check the olmOCR-2 fast-mode timing run and project the cost of the full OCR run ($0, on the Mac).

    uv run ingest/timing_check.py

1. Apple Vision reads the same 24 timing pages (free, local) as an independent second opinion.
2. Coverage per page: share of Apple Vision's longer words (5+ letters) that also appear in olmOCR's text. There is no
   answer key for these pages, so this only catches gross failures (a skipped column, a blank or truncated page);
   Apple Vision itself misreads some words, so ~0.85-0.95 is normal and pages under 0.80 are flagged for a look.
3. Projection: full-run GPU time = pages to OCR x measured seconds per page (+ model start-up), times the GPU price.
"""

import json
import re
from pathlib import Path

import pymupdf

from phase1_local import apple_vision

ROOT = Path(__file__).resolve().parent.parent
TIMING = ROOT / "data" / "timing"
RES = TIMING / "results"
PRICE_PER_HOUR = {"L4": 0.80, "L40S": 1.95}  # Modal list prices, GPU only (CPU/RAM add a few %); check the dashboard
FLAG = 0.80


def words(text: str) -> set[str]:
    text = re.sub(r"\A---.*?---", " ", text, flags=re.S)  # olmOCR's YAML front matter
    text = re.sub(r"<[^<>]*>", " ", text)
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def gpu_price(gpu_name: str) -> tuple[str, float]:
    for key in sorted(PRICE_PER_HOUR, key=len, reverse=True):  # "L40S" before "L4"
        if key in gpu_name.replace(" ", ""):
            return key, PRICE_PER_HOUR[key]
    raise ValueError(f"no price for GPU {gpu_name!r}")


def kept(book: dict) -> int:
    return sum(len(book["roles"].get(r, [])) for r in ("body", "index", "extra"))


if __name__ == "__main__":
    pages = json.loads((TIMING / "timing_pages.json").read_text())
    av = RES / "apple_vision"
    av.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(TIMING / "timing_pages.pdf")
    for i, page in enumerate(doc, start=1):
        if not (av / f"P{i:02d}.md").exists():
            (av / f"P{i:02d}.md").write_text(apple_vision(page.get_pixmap(dpi=200).tobytes("png")))

    log = json.loads((RES / "olmocr_vllm" / "log.json").read_text())
    summary = json.loads((RES / "summary.json").read_text())
    print(f"{'page':5} {'book:pdf page':22} {'latency s':>9} {'out tok':>8} {'finish':>7} {'coverage':>9}")
    covs = []
    for row in pages:
        name = row["page"]
        meta = log["pages"].get(name, {})
        f = RES / "olmocr_vllm" / f"{name}.md"
        a, o = words((av / f"{name}.md").read_text()), words(f.read_text()) if f.exists() else set()
        cov = len(a & o) / len(a) if a else 1.0
        covs.append(cov)
        flag = "  <-- look" if cov < FLAG or meta.get("finish", "stop") != "stop" else ""
        print(f"{name:5} {row['book_id'] + ':' + str(row['pdf_page']):22} {meta.get('seconds', '-'):>9} "
              f"{meta.get('out_tokens', '-'):>8} {meta.get('finish', 'n/a'):>7} {cov:>9.2f}{flag}")
    print(f"median coverage {sorted(covs)[len(covs) // 2]:.2f}, flagged {sum(c < FLAG for c in covs)}")

    t = summary["batches"]["timing"]
    t["n_pages"] = t.get("n_pages", t.get("pages"))  # first run's summary called the count "pages"
    gpu_key, price = gpu_price(summary["gpu"])
    s_per_page = t["wall_seconds"] / t["pages_ok"]
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    scanned = kept(manifest["park_psm"]) + kept(manifest["khurana_ophthal"])
    digital = kept(manifest["dhingra_ent"]) + kept(manifest["bardale_fmt"])
    start_h = summary["server_ready_seconds"] / 3600
    print(f"\nGPU {summary['gpu']} ({gpu_key} ${price:.2f}/h), {summary['engine']}, concurrency {summary['concurrency']}")
    print(f"start-up (download/load/compile) {summary['server_ready_seconds']:.0f} s; container total "
          f"{summary['container_seconds']:.0f} s -> this test ~${summary['container_seconds'] / 3600 * price:.2f}")
    print(f"timing batch: {t['pages_ok']}/{t['n_pages']} pages in {t['wall_seconds']:.0f} s = {s_per_page:.1f} s/page "
          f"(includes the slowest page's tail, so it is pessimistic); {t['out_tokens'] / t['wall_seconds']:.0f} "
          f"output tokens/s; retried {t['retried']}, still at max tokens {t['hit_max_tokens']}")
    for label, n in (("Park's + Khurana (scanned)", scanned), ("+ Dhingra + Bardale too", scanned + digital)):
        hours = n * s_per_page / 3600 + start_h
        print(f"projection {label:28} {n:5} pages: {hours:5.2f} GPU-h -> ${hours * price:5.2f} "
              f"(vs one-page-at-a-time ~${n * 96 / 3600 * PRICE_PER_HOUR['L4']:.0f})")
