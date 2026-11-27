"""Phase 1 OCR bake-off on Modal (L4 GPU, no queue). Replaces the Kaggle notebook.

    caffeinate -i uv run modal run --detach ingest/modal_ocr.py::paddle    # PaddleOCR-VL 1.6
    caffeinate -i uv run modal run --detach ingest/modal_ocr.py::olmocr    # olmOCR-2 7B (transformers, 1 page at a time)
    caffeinate -i uv run modal run --detach ingest/modal_ocr.py::olmocr_vllm   # olmOCR-2 FP8 on vLLM: 12 test + 24 timing
    uv run ingest/modal_ocr.py fetch paddle|olmocr|olmocr_vllm             # copy results from the volume

Each model is its own Modal app, so a failed image build for one never blocks the other.

Surviving a dropped connection (first runs died when the Mac went to idle sleep and Modal stopped the app):
  - every page is written to the Modal Volume "mbbs-ocr" (phase1/<engine>/T01.md + log.json) the moment it is done;
  - a rerun skips pages already on the volume;
  - --detach keeps the job running on Modal if the Mac disconnects; caffeinate keeps the Mac awake;
  - olmOCR weights are cached on the volume (HF_HOME), so reruns don't download 16 GB again.
The volume holds OCR'd book text: private to this Modal account; delete it when the project no longer needs it
(uv run modal volume delete mbbs-ocr).

Why L4 and not T4: L4 (Ada, 24 GB) supports bf16, so olmOCR-2 runs in its native format on one GPU, and PaddleOCR-VL
is not on the "CC 7.x not recommended" list. Cost: about $0.80 per GPU hour (Modal's $30/month free credit).
"""

import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT / "data" / "phase1"
VOL_NAME, MNT = "mbbs-ocr", "/vol"
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
paddle_app = modal.App("mbbs-ocr-paddle")
olmo_app = modal.App("mbbs-ocr-olmocr")

paddle_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    # Paddle's own index first (Modal's PyPI mirror times out on paddlepaddle-gpu), real PyPI for its dependencies
    .uv_pip_install("paddlepaddle-gpu==3.2.1", index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
                    extra_index_url="https://pypi.org/simple", extra_options="--index-strategy unsafe-best-match")
    .uv_pip_install("paddleocr[doc-parser]>=3.4.0", "pymupdf")
)

olmo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("torch==2.7.1", "torchvision==0.22.1", "transformers>=4.51,<5", "accelerate", "pillow",
                    "pymupdf", "pyyaml")
    .uv_pip_install("olmocr>=0.4.0", extra_options="--no-deps")  # only its prompt builder is used
    .env({"HF_HOME": f"{MNT}/hf"})
)


def pages_as_png(pdf_bytes: bytes, dpi: int | None = None, longest: int | None = None) -> list[bytes]:
    import pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for page in doc:
        zoom = longest / max(page.rect.width, page.rect.height) if longest else dpi / 72
        out.append(page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png"))
    return out


def run_pages(engine: str, model_name: str, pngs: list[bytes], load, read_page) -> None:
    """Shared loop: skip pages already on the volume, OCR the rest, write each page + log the moment it is done."""
    import traceback
    out = Path(MNT) / "phase1" / engine
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"model": model_name, "pages": {}}
    todo = [i for i in range(1, len(pngs) + 1) if not (out / f"T{i:02d}.md").exists()]
    print(f"{engine}: {len(pngs) - len(todo)} pages already done, {len(todo)} to go", flush=True)
    if not todo:
        return
    t0 = time.time()
    state = load()
    log["load_seconds"] = round(time.time() - t0, 1)
    for i in todo:
        name, t = f"T{i:02d}", time.time()
        try:
            text, extra = read_page(state, pngs[i - 1], name)
            (out / f"{name}.md").write_text(text)
            log["pages"][name] = {"seconds": round(time.time() - t, 1), "chars": len(text), **extra}
        except Exception as e:
            log["pages"][name] = {"error": repr(e), "trace": traceback.format_exc()[-1500:]}
        log_path.write_text(json.dumps(log, indent=1))
        vol.commit()
        print(name, log["pages"][name].get("seconds", log["pages"][name].get("error")), flush=True)


# PaddleOCR-VL's own PaddlePaddle engine writes text token by token in plain Python: GPU ~0%, one CPU core 100%,
# a light page in 10 s but a dense Park's page for 25+ min (found 2026-11-26 with modal_paddle_diag.py stack dumps:
# stuck in generation/utils.py greedy_search). The docs call that engine "for rapid validation" and point production
# at a VLM inference service. So the VL model is served by vLLM (recipe flags from docs.vllm.ai PaddleOCR-VL recipe),
# in its OWN virtualenv (vLLM's and PaddlePaddle's dependencies conflict), and the pipeline (layout model on
# PaddlePaddle) sends it each text block over HTTP. vLLM needs compute capability >= 8.0: L4, A10G, L40S all qualify.
PADDLE_VL_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
paddle_vllm_image = paddle_image.run_commands(
    "python -m venv /opt/vllm",
    "/opt/vllm/bin/pip install -q --index-url https://pypi.org/simple uv",
    "/opt/vllm/bin/uv pip install --python /opt/vllm/bin/python --index-url https://pypi.org/simple 'vllm>=0.11.1'",
).apt_install("build-essential")  # Triton (inside vLLM) compiles small C launchers at startup: needs gcc


PADDLE_VLLM_CMD = [
    "/opt/vllm/bin/vllm", "serve", PADDLE_VL_MODEL, "--served-model-name", PADDLE_VL_MODEL, "--trust-remote-code",
    "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0",
    "--gpu-memory-utilization", "0.5", "--port", "8000",
    # PaddleOCR's client sends per-request image-size settings (mm_processor_kwargs); recent vLLM rejects them
    # unless trusted. Safe here: the only client is our own pipeline, on localhost, in the same container.
    "--trust-request-mm-kwargs"]


def start_vllm(cmd: list[str], port: int = 8000, timeout: int = 1200):
    """Start `vllm serve` (cmd) in the background and wait until it answers; raise with its log tail if it dies."""
    import os
    import subprocess
    import urllib.request
    env = dict(os.environ, HF_HOME=f"{MNT}/hf")
    log = open("/tmp/vllm.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
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


@paddle_app.function(image=paddle_vllm_image, gpu=["L4", "A10G", "L40S"], timeout=3600, volumes={MNT: vol})
def run_paddle(pdf_bytes: bytes) -> None:
    import glob
    import subprocess
    import tempfile
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()

    def load():
        start_vllm(PADDLE_VLLM_CMD)  # before PaddlePaddle touches the GPU, so vLLM sees the free memory it asks for
        vol.commit()  # keep the downloaded VL weights for reruns
        from paddleocr import PaddleOCRVL
        pipe = PaddleOCRVL(pipeline_version="v1.6", vl_rec_backend="vllm-server",
                           vl_rec_server_url="http://127.0.0.1:8000/v1", vl_rec_api_model_name=PADDLE_VL_MODEL)
        return {"pipe": pipe, "work": tempfile.mkdtemp()}

    def read_page(s, png, name):
        src = f"{s['work']}/{name}.png"
        open(src, "wb").write(png)
        for res in s["pipe"].predict(src):
            res.save_to_markdown(save_path=f"{s['work']}/{name}")
        return "\n".join(open(f).read() for f in sorted(glob.glob(f"{s['work']}/{name}/*.md"))), {}

    run_pages("paddle", f"PaddleOCR-VL v1.6 (layout on PaddlePaddle + VL on vLLM server, {gpu})",
              pages_as_png(pdf_bytes, dpi=200), load, read_page)


@olmo_app.function(image=olmo_image, gpu="L4", timeout=3600, volumes={MNT: vol})
def run_olmocr(pdf_bytes: bytes) -> None:
    import base64
    import io

    def load():
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-2-7B-1025", torch_dtype=torch.bfloat16, device_map="cuda").eval()
        vol.commit()  # keep the downloaded weights on the volume for reruns
        return {"model": model, "proc": AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct"),
                "prompt": build_no_anchoring_v4_yaml_prompt()}

    def read_page(s, png, name):
        import torch
        from PIL import Image
        b64 = base64.b64encode(png).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": s["prompt"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
        text = s["proc"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = s["proc"](text=[text], images=[Image.open(io.BytesIO(png))], padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            out = s["model"].generate(**inputs, temperature=0.1, do_sample=True, max_new_tokens=4096)
        new = out[:, inputs["input_ids"].shape[1]:]
        return s["proc"].tokenizer.batch_decode(new, skip_special_tokens=True)[0], {"new_tokens": int(new.shape[1])}

    run_pages("olmocr", "olmOCR-2-7B-1025 bf16 transformers (L4)", pages_as_png(pdf_bytes, longest=1288),
              load, read_page)  # 1288 px longest side, as the model card says


# ---- olmOCR-2 fast mode: vLLM + FP8 -------------------------------------------------------------------------------
# The transformers run above reads one page at a time: 96 s/page on an L4, i.e. ~46 GPU-hours (~$37) for the ~1,717
# scanned pages. Here vLLM serves the FP8 copy of the model (allenai/olmOCR-2-7B-1025-FP8, the olmOCR README's serving
# example: `vllm serve ... --max-model-len 16384`) and reads many pages at once. Same prompt, same 1288 px pictures as
# the transformers run, so only the engine and the number format change. FP8 runs natively on compute capability 8.9
# (L4, L40S); A10G (8.6) is left out on purpose.
TIMING = ROOT / "data" / "timing"
OLMO_FP8 = "allenai/olmOCR-2-7B-1025-FP8"
olmo_vllm_app = modal.App("mbbs-ocr-olmocr-vllm")
olmo_vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "build-essential")  # Triton (inside vLLM) needs gcc at startup
    .uv_pip_install("vllm>=0.11.1", "pymupdf", "pyyaml", index_url="https://pypi.org/simple")
    .uv_pip_install("olmocr>=0.4.0", extra_options="--no-deps")  # only its prompt builder is used
    .env({"HF_HOME": f"{MNT}/hf"})
)
OLMO_VLLM_CMD = ["vllm", "serve", OLMO_FP8, "--served-model-name", "olmocr", "--max-model-len", "16384",
                 "--gpu-memory-utilization", "0.90", "--port", "8000"]
MAX_TOKENS, TEMPS = 8192, (0.1, 0.3)  # a page that stops at MAX_TOKENS (usually a repeat loop) is retried once warmer


@olmo_vllm_app.function(image=olmo_vllm_image, gpu=["L4", "L40S"], timeout=3600, volumes={MNT: vol})
def run_olmocr_vllm(batches: dict, concurrency: int = 24) -> dict:
    """batches: {tag: (page-name prefix, pdf bytes)}. Each batch is sent all at once (up to `concurrency` pages in
    flight); every page is written to olmocr_vllm/<tag>/ as it finishes; timings and token counts go to log.json."""
    import base64
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import vllm
    from openai import OpenAI
    from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    t_start = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    start_vllm(OLMO_VLLM_CMD)
    vol.commit()  # keep the FP8 weights on the volume for the full run
    ready = time.time() - t_start
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=1800)
    prompt = build_no_anchoring_v4_yaml_prompt()

    def ocr(png: bytes):
        t = time.time()
        b64 = base64.b64encode(png).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
        for attempt, temp in enumerate(TEMPS, start=1):
            r = client.chat.completions.create(model="olmocr", messages=messages, temperature=temp,
                                               max_tokens=MAX_TOKENS)
            if r.choices[0].finish_reason == "stop":
                break
        c = r.choices[0]
        return c.message.content or "", {"seconds": round(time.time() - t, 1), "finish": c.finish_reason,
                                         "attempts": attempt, "in_tokens": r.usage.prompt_tokens,
                                         "out_tokens": r.usage.completion_tokens}

    summary = {"model": OLMO_FP8, "engine": f"vLLM {vllm.__version__}", "gpu": gpu, "concurrency": concurrency,
               "server_ready_seconds": round(ready, 1), "batches": {}}
    for tag, (prefix, pdf_bytes) in batches.items():
        out = Path(MNT) / "olmocr_vllm" / tag
        out.mkdir(parents=True, exist_ok=True)
        pngs = pages_as_png(pdf_bytes, longest=1288)
        log, t0 = {"model": f"{OLMO_FP8} on {summary['engine']} ({gpu})", "pages": {}}, time.time()
        with ThreadPoolExecutor(min(concurrency, len(pngs))) as ex:
            futs = {ex.submit(ocr, png): f"{prefix}{i:02d}" for i, png in enumerate(pngs, start=1)}
            for f in as_completed(futs):
                name = futs[f]
                try:
                    text, meta = f.result()
                    (out / f"{name}.md").write_text(text)
                    log["pages"][name] = {"chars": len(text), **meta}
                except Exception as e:
                    log["pages"][name] = {"error": repr(e)}
                print(name, log["pages"][name].get("seconds", log["pages"][name].get("error")), flush=True)
        ok = [v for v in log["pages"].values() if "seconds" in v]
        stats = {"n_pages": len(pngs), "pages_ok": len(ok), "wall_seconds": round(time.time() - t0, 1),
                 "out_tokens": sum(v["out_tokens"] for v in ok), "retried": sum(v["attempts"] > 1 for v in ok),
                 "hit_max_tokens": sum(v["finish"] != "stop" for v in ok)}
        log.update(stats, load_seconds=round(ready, 1))
        (out / "log.json").write_text(json.dumps(log, indent=1))
        vol.commit()
        summary["batches"][tag] = stats
    summary["container_seconds"] = round(time.time() - t_start, 1)
    (Path(MNT) / "olmocr_vllm" / "summary.json").write_text(json.dumps(summary, indent=1))
    vol.commit()
    return summary


def fetch_dir(remote: str, local: Path) -> int:
    """Copy the files (not sub-folders) of a volume folder into a local folder; returns how many."""
    local.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in vol.listdir(remote):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        (local / Path(entry.path).name).write_bytes(b"".join(vol.read_file(entry.path)))
        n += 1
    return n


def fetch_olmocr_vllm() -> None:
    a = fetch_dir("olmocr_vllm/phase1", PHASE1 / "results" / "olmocr_vllm")
    b = fetch_dir("olmocr_vllm/timing", TIMING / "results" / "olmocr_vllm")
    fetch_dir("olmocr_vllm", TIMING / "results")  # summary.json
    print(f"olmocr_vllm: fetched {a} files (12 test pages) + {b} files (24 timing pages)")


def fetch(engine: str) -> None:
    """Copy phase1/<engine>/ from the volume into data/phase1/results/<engine>/."""
    if engine == "olmocr_vllm":
        return fetch_olmocr_vllm()
    dst = PHASE1 / "results" / engine
    n = fetch_dir(f"phase1/{engine}", dst)
    log = json.loads((dst / "log.json").read_text())
    ok = [v for v in log["pages"].values() if "seconds" in v]
    print(f"{engine}: fetched {n} files, {len(ok)} pages ok, load {log.get('load_seconds')} s, "
          f"avg {sum(v['seconds'] for v in ok) / max(len(ok), 1):.1f} s/page")


@paddle_app.local_entrypoint()
def paddle():
    run_paddle.remote((PHASE1 / "phase1_pages.pdf").read_bytes())
    fetch("paddle")


@olmo_app.local_entrypoint()
def olmocr():
    run_olmocr.remote((PHASE1 / "phase1_pages.pdf").read_bytes())
    fetch("olmocr")


@olmo_vllm_app.local_entrypoint()
def olmocr_vllm(concurrency: int = 24):
    """12 scored test pages (quality vs the transformers run) + 24 random pages (timing), one container."""
    summary = run_olmocr_vllm.remote({"phase1": ("T", (PHASE1 / "phase1_pages.pdf").read_bytes()),
                                      "timing": ("P", (TIMING / "timing_pages.pdf").read_bytes())}, concurrency)
    fetch_olmocr_vllm()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "fetch":
    fetch(sys.argv[2])
