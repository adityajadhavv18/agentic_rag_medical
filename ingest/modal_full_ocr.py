"""Full OCR run: all 4 books with olmOCR-2 (FP8) on vLLM, on Modal GPUs. Keeps running if the Mac sleeps or shuts down.

    uv run modal deploy ingest/modal_full_ocr.py            # 1. put the app on Modal (redo after code changes)
    caffeinate -i uv run ingest/modal_full_ocr.py start      # 2. upload the PDFs (first time) + hand batches to Modal
    uv run ingest/modal_full_ocr.py start --dry              #    plan + estimate only, nothing sent
    uv run ingest/modal_full_ocr.py status                   # pages done per book, GPUs, rate, finish estimate
    uv run ingest/modal_full_ocr.py watch --until-pages 100  # wait until N pages are saved, then print status
    uv run ingest/modal_full_ocr.py fetch                    # copy finished pages to data/ocr/ (only new files)

Why the Mac can sleep: `start` uploads the PDFs, *spawns* one job per batch of BATCH pages on the deployed app, and
exits. The jobs run on Modal's servers; nothing waits on the Mac.

Never OCR a page twice: each page is saved on the volume as ocr/<book_id>/p0001.md (+ p0001.json with tokens,
finish reason, rotation flags), committed every 20 pages or 60 s. Every batch first skips pages already saved, so
`start` again (after any failure) only does what is missing. Modal retries a crashed batch up to 2 times (it
resumes, it does not restart). `start` refuses to launch while batches from the last start are still running, so
two GPUs never work on the same pages.

Same settings as the verified timing test (2026-11-27: 94/101 terms on the 12 test pages, 2.2 s/page on an L40S):
allenai/olmOCR-2-7B-1025-FP8, vLLM 0.34.0, pictures 1288 px on the longest side, olmOCR's v4 prompt, max 8192
tokens; a page that does not finish is retried once at temperature 0.3, then saved and flagged.
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
LOCAL_OUT = ROOT / "data" / "ocr"
APP_NAME, VOL_NAME, MNT = "mbbs-ocr-full", "mbbs-ocr", "/vol"
MODEL, VLLM_VERSION = "allenai/olmOCR-2-7B-1025-FP8", "0.34.0"
BATCH, IN_FLIGHT, MAX_GPUS = 200, 32, 4
MAX_TOKENS, TEMPS, LONGEST = 8192, (0.1, 0.3), 1288
SEC_PER_PAGE_L40S, BOOT_MIN = 2.2, 3  # measured in the timing test (pessimistic: includes the slowest page's tail)
BOOK_FILES = {"park_psm": "K.pdf", "khurana_ophthal": "A.pdf", "dhingra_ent": "P_L_Dhingra.pdf",
              "bardale_fmt": "Rajesh_Bardale_Principles_of_Forensic_Me_261010_104043.pdf"}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "build-essential")  # Triton (inside vLLM) needs gcc at startup
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "pymupdf", "pyyaml", index_url="https://pypi.org/simple")
    .uv_pip_install("olmocr>=0.4.0", extra_options="--no-deps")  # only its prompt builder is used
    .env({"HF_HOME": f"{MNT}/hf", "HF_HUB_OFFLINE": "1"})  # weights already cached on the volume by the timing test
)
VLLM_CMD = ["vllm", "serve", MODEL, "--served-model-name", "olmocr", "--max-model-len", "16384",
            "--gpu-memory-utilization", "0.90", "--port", "8000"]


def start_vllm(port: int = 8000, timeout: int = 1200):
    """Start `vllm serve` in the background and wait until it answers; raise with its log tail if it dies."""
    import urllib.request
    log = open("/tmp/vllm.log", "w")
    proc = subprocess.Popen(VLLM_CMD, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited:\n" + open("/tmp/vllm.log").read()[-4000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return proc
        except Exception:
            if time.time() > deadline:
                raise RuntimeError("vLLM not ready in time:\n" + open("/tmp/vllm.log").read()[-4000:])
            time.sleep(5)


def front_matter(text: str) -> dict:
    """olmOCR starts every page with YAML: primary_language, is_rotation_valid, rotation_correction, is_table..."""
    import yaml
    m = re.match(r"\A---\s*\n(.*?)\n---", text, flags=re.S)
    if not m:
        return {}
    try:
        d = yaml.safe_load(m.group(1))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


@app.cls(image=image, gpu=["L40S", "L4"], volumes={MNT: vol}, timeout=3 * 3600, max_containers=MAX_GPUS,
         scaledown_window=60, retries=modal.Retries(max_retries=2, initial_delay=10.0))
class OCR:
    @modal.enter()
    def boot(self):
        import os
        from openai import OpenAI
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
        t0 = time.time()
        self.gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
        self.server = start_vllm()
        self.boot_seconds = round(time.time() - t0, 1)
        self.client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=1800)
        self.prompt = build_no_anchoring_v4_yaml_prompt()
        boot_dir = Path(MNT) / "ocr" / "_boot"
        boot_dir.mkdir(parents=True, exist_ok=True)
        task = os.environ.get("MODAL_TASK_ID", f"t{int(t0)}")
        (boot_dir / f"{task}.json").write_text(json.dumps(
            {"task": task, "gpu": self.gpu, "boot_seconds": self.boot_seconds, "ready_at": time.time()}))
        vol.commit()
        print(f"BOOT {task} {self.gpu} ready in {self.boot_seconds}s", flush=True)

    def ensure_server(self):
        """Restart vLLM if it died since the last batch (otherwise every page of every later batch would fail)."""
        import urllib.request
        try:
            assert self.server.poll() is None
            urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10)
        except Exception:
            print("vLLM not answering: restarting it", flush=True)
            self.server = start_vllm()

    def ocr(self, png: bytes) -> tuple[str, dict]:
        import base64
        t = time.time()
        b64 = base64.b64encode(png).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
        for attempt, temp in enumerate(TEMPS, start=1):
            r = self.client.chat.completions.create(model="olmocr", messages=messages, temperature=temp,
                                                    max_tokens=MAX_TOKENS)
            if r.choices[0].finish_reason == "stop":
                break
        c = r.choices[0]
        return c.message.content or "", {"seconds": round(time.time() - t, 1), "finish": c.finish_reason,
                                         "attempts": attempt, "in_tokens": r.usage.prompt_tokens,
                                         "out_tokens": r.usage.completion_tokens}

    @modal.method()
    def batch(self, book_id: str, pdf_name: str, pages: list[int]) -> dict:
        import pymupdf
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        vol.reload()
        self.ensure_server()
        out = Path(MNT) / "ocr" / book_id
        out.mkdir(parents=True, exist_ok=True)
        done = {int(f.stem[1:]) for f in out.glob("p*.md")}
        todo = [p for p in pages if p not in done]
        stats = {"book_id": book_id, "first": pages[0], "last": pages[-1], "asked": len(pages),
                 "already_done": len(pages) - len(todo), "ok": 0, "errors": {}, "not_finished": [],
                 "rotation_flag": [], "gpu": self.gpu, "boot_seconds": self.boot_seconds, "started": time.time()}
        doc = pymupdf.open(f"{MNT}/pdfs/{pdf_name}")  # rendering stays on this thread (PyMuPDF isn't thread-safe)
        state = {"since_commit": 0, "last_commit": time.time()}

        def save_finished(pending: dict) -> None:
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for f in finished:
                p = pending.pop(f)
                try:
                    text, meta = f.result()
                    fm = front_matter(text)
                    meta.update(page=p, book_id=book_id, model=MODEL, vllm=VLLM_VERSION, gpu=self.gpu,
                                **{k: fm.get(k) for k in ("primary_language", "is_rotation_valid",
                                                          "rotation_correction", "is_table", "is_diagram")})
                    (out / f"p{p:04d}.md").write_text(text)
                    (out / f"p{p:04d}.json").write_text(json.dumps(meta))
                    stats["ok"] += 1
                    if meta["finish"] != "stop":
                        stats["not_finished"].append(p)
                    if fm.get("is_rotation_valid") is False:
                        stats["rotation_flag"].append(p)
                except Exception as e:  # no file is written, so the next `start` redoes this page
                    stats["errors"][str(p)] = repr(e)[:300]
                state["since_commit"] += 1
            if state["since_commit"] >= 20 or time.time() - state["last_commit"] > 60:
                vol.commit()
                state.update(since_commit=0, last_commit=time.time())

        with ThreadPoolExecutor(IN_FLIGHT) as ex:
            pending = {}
            for p in todo:
                page = doc[p - 1]
                zoom = LONGEST / max(page.rect.width, page.rect.height)
                png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
                pending[ex.submit(self.ocr, png)] = p
                while len(pending) >= 2 * IN_FLIGHT:  # keep the GPU fed without holding every picture in memory
                    save_finished(pending)
            while pending:
                save_finished(pending)
        stats["seconds"] = round(time.time() - stats["started"], 1)
        runs = Path(MNT) / "ocr" / "_batches"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{book_id}_{pages[0]:04d}-{pages[-1]:04d}.json").write_text(json.dumps(stats, indent=1))
        vol.commit()
        return stats


# ---- local commands (run on the Mac) -------------------------------------------------------------------------------

def kept_pages() -> dict[str, list[int]]:
    manifest = {b["book_id"]: b for b in json.loads((ROOT / "manifest.json").read_text())["books"]}
    return {bid: sorted(p for r in ("body", "index", "extra") for p in manifest[bid]["roles"].get(r, []))
            for bid in BOOK_FILES}


def remote_files(path: str) -> list:
    try:
        return vol.listdir(path)
    except Exception:  # folder does not exist yet
        return []


def remote_done(book_id: str) -> dict[int, float]:
    """page -> save time (mtime) for every page already on the volume."""
    return {int(Path(e.path).stem[1:]): e.mtime for e in remote_files(f"ocr/{book_id}") if e.path.endswith(".md")}


def plan() -> list[tuple[str, str, list[int]]]:
    batches = []
    for bid, pages in kept_pages().items():
        done = remote_done(bid)
        todo = [p for p in pages if p not in done]
        batches += [(bid, BOOK_FILES[bid], todo[i:i + BATCH]) for i in range(0, len(todo), BATCH)]
    return batches


def last_run() -> dict | None:
    runs = sorted((LOCAL_OUT / "runs").glob("*.json"))
    return json.loads(runs[-1].read_text()) if runs else None


def call_states(run: dict) -> dict[str, int]:
    states = {"done": 0, "running_or_queued": 0, "failed": 0}
    for c in run["calls"]:
        try:
            modal.FunctionCall.from_id(c["call_id"]).get(timeout=0)
            states["done"] += 1
        except TimeoutError:
            states["running_or_queued"] += 1
        except Exception:
            states["failed"] += 1
    return states


def upload_pdfs() -> None:
    have = {Path(e.path).name: e.size for e in remote_files("pdfs")}
    missing = [f for f in BOOK_FILES.values() if have.get(f) != (ROOT / "corpus" / f).stat().st_size]
    if not missing:
        print("PDFs already on Modal")
        return
    mb = sum((ROOT / "corpus" / f).stat().st_size for f in missing) / 1e6
    print(f"uploading {len(missing)} PDFs ({mb:.0f} MB) - keep the Mac awake until this finishes", flush=True)
    t = time.time()
    with vol.batch_upload(force=True) as up:
        for f in missing:
            up.put_file(ROOT / "corpus" / f, f"pdfs/{f}")
    print(f"upload done in {time.time() - t:.0f} s", flush=True)


def start(dry: bool) -> None:
    run = last_run()
    if run and not dry and call_states(run)["running_or_queued"]:
        sys.exit("batches from the last start are still running: wait for them (status), or stop the app first")
    batches = plan()
    n = sum(len(b[2]) for b in batches)
    gpus = min(MAX_GPUS, len(batches)) or 1
    print(f"to OCR: {n} pages in {len(batches)} batches of <= {BATCH}: " +
          ", ".join(f"{bid} {sum(len(b[2]) for b in batches if b[0] == bid)}" for bid in BOOK_FILES))
    print(f"estimate with {gpus} L40S GPUs: ~{n * SEC_PER_PAGE_L40S / gpus / 60 + BOOT_MIN:.0f} min, "
          f"~${(n * SEC_PER_PAGE_L40S / 3600 + gpus * BOOT_MIN / 60) * 1.95:.2f} (fewer or slower GPUs: longer, "
          f"similar cost)")
    if dry or not batches:
        return
    upload_pdfs()
    worker = modal.Cls.from_name(APP_NAME, "OCR")()
    calls = []
    for bid, f, pages in batches:
        c = worker.batch.spawn(bid, f, pages)
        calls.append({"call_id": c.object_id, "book_id": bid, "first": pages[0], "last": pages[-1], "n": len(pages)})
    LOCAL_OUT.joinpath("runs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (LOCAL_OUT / "runs" / f"{stamp}.json").write_text(json.dumps(
        {"started": time.time(), "pages": n, "calls": calls}, indent=1))
    print(f"handed {len(calls)} batches to Modal at {stamp}. The Mac can sleep now; check with: status")


def status(quiet: bool = False) -> dict:
    kept = kept_pages()
    run = last_run()
    t0 = run["started"] if run else 0
    total = sum(len(v) for v in kept.values())
    done_all, new_times = 0, []
    lines = []
    for bid, pages in kept.items():
        d = remote_done(bid)
        done = len(set(pages) & set(d))
        done_all += done
        new_times += [m for m in d.values() if m >= t0]
        lines.append(f"  {bid:16} {done:5}/{len(pages):<5}")
    boots = [json.loads(b"".join(vol.read_file(e.path))) for e in remote_files("ocr/_boot")
             if e.mtime >= t0 - 60]
    eta = ""
    if len(new_times) >= 40:
        rate = (len(new_times) - 1) / max(max(new_times) - min(new_times), 1)  # pages per second, all GPUs together
        left = total - done_all
        eta = f"rate {rate * 60:.0f} pages/min -> ~{left / rate / 60:.0f} min left (finish ~{time.strftime('%H:%M', time.localtime(time.time() + left / rate))})"
    out = {"done": done_all, "total": total, "boots": boots, "eta": eta,
           "calls": call_states(run) if run else {}}
    if not quiet:
        print(f"pages saved: {done_all}/{total}")
        print("\n".join(lines))
        print(f"GPUs booted this run: {len(boots)} " +
              ", ".join(f"{b['gpu']} ({b['boot_seconds']:.0f}s)" for b in boots))
        print(f"batches: {out['calls']}")
        print(eta or "rate: not enough pages yet")
    return out


def watch(until_pages: int, max_minutes: int) -> None:
    t = time.time()
    while time.time() - t < max_minutes * 60:
        s = status(quiet=True)
        if s["done"] >= min(until_pages, s["total"]) or not s["calls"].get("running_or_queued", 1):
            break
        time.sleep(30)
    status()


def fetch() -> None:
    """Download the whole ocr/ folder with `modal volume get` (parallel: 5,536 files in ~75 s; reading the files one
    by one through the Python client took ~2 s each), then move each folder into data/ocr/."""
    import shutil
    tmp = ROOT / "data" / "_dl"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", VOL_NAME, "ocr", str(tmp)], check=True)
    for folder in (tmp / "ocr").iterdir():
        dst = LOCAL_OUT / folder.name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(folder), str(dst))
        print(f"{folder.name:16} {len(list(dst.iterdir()))} files")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        start(dry="--dry" in sys.argv)
    elif cmd == "status":
        status()
    elif cmd == "watch":
        args = sys.argv[2:]
        until = int(args[args.index("--until-pages") + 1]) if "--until-pages" in args else 10 ** 9
        minutes = int(args[args.index("--max-minutes") + 1]) if "--max-minutes" in args else 120
        watch(until, minutes)
    elif cmd == "fetch":
        fetch()
    else:
        sys.exit(__doc__)
