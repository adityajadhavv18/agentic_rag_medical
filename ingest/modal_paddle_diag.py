"""Diagnose why PaddleOCR-VL hangs on Modal (GPU 0%, one CPU core 100%, no page finished).

    caffeinate -i uv run modal run ingest/modal_paddle_diag.py

One page (T07, a light diagram page), with: paddle's own GPU check, timing of model load, and a Python stack dump
every 90 s (faulthandler) so the log shows exactly where it is stuck.
"""

import time
from pathlib import Path

import modal

from modal_ocr import PHASE1, pages_as_png, paddle_image

app = modal.App("mbbs-ocr-paddle-diag")


@app.function(image=paddle_image.add_local_python_source("modal_ocr"), gpu=["L4", "A10G", "L40S"], timeout=1500)
def diag(png: bytes) -> str:
    import faulthandler
    import subprocess
    import sys
    faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout, flush=True)
    import paddle
    print("paddle", paddle.__version__, "cuda compiled:", paddle.device.is_compiled_with_cuda(),
          "device:", paddle.device.get_device(), "cuda", paddle.version.cuda(), "cudnn", paddle.version.cudnn(),
          flush=True)
    paddle.utils.run_check()
    t = time.time()
    x = paddle.randn([4096, 4096]); y = paddle.matmul(x, x); float(y.sum())
    print(f"4096 matmul on {x.place}: {time.time() - t:.2f}s", flush=True)
    from paddleocr import PaddleOCRVL
    t = time.time()
    pipe = PaddleOCRVL(pipeline_version="v1.6")
    print(f"pipeline load {time.time() - t:.1f}s", flush=True)
    open("/tmp/p.png", "wb").write(png)
    t = time.time()
    out = list(pipe.predict("/tmp/p.png"))
    print(f"predict 1 page {time.time() - t:.1f}s", flush=True)
    out[0].save_to_markdown(save_path="/tmp/o")
    import glob
    return "\n".join(open(f).read() for f in glob.glob("/tmp/o/*.md"))


@app.local_entrypoint()
def main(page: int = 7):
    png = pages_as_png((PHASE1 / "phase1_pages.pdf").read_bytes(), dpi=200)[page - 1]
    print(diag.remote(png)[:1500])
