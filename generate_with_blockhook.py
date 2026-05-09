"""
Phase 1 driver for per-block TRT pipeline.

Reuses generate_with_hook.py for all arg parsing and model setup, swapping
the hook so we capture per-block inputs instead of exporting the stem.

Single-sample by default; multi-sample via env var:
    CAPTURE_INDICES=0,3,6,9,12 sh trt_blocks_phase1.sh
captures call indices 0,3,6,9,12 for each block (good for entropy-style
calibration over a 16-step diffusion run).

Snapshots land in /workspace/s2v_trt/blocks/.
"""
import os
import runpy

import trt_conv.export_stem as _export_stem
from trt_conv.capture_blocks import install_block_capture_hooks


def _install_block_capture(model, onnx_path=None, opset=17):
    raw = os.environ.get("CAPTURE_INDICES", "0")
    indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    install_block_capture_hooks(
        model,
        save_dir="/workspace/s2v_trt/blocks",
        capture_indices=indices,
    )


_export_stem.install_export_hook = _install_block_capture

runpy.run_module("generate_with_hook", run_name="__main__")
