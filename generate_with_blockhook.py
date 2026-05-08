"""
Phase 1 driver for per-block TRT pipeline.

Reuses generate_with_hook.py's full argument parsing and model setup, but
swaps the hook from "export the whole stem" to "capture each block's first
inputs to disk". Once every block has fired once, capture_blocks raises
SystemExit and we stop.

Snapshots land in /workspace/s2v_trt/blocks/block_NN.pt.
"""
import runpy

import trt_conv.export_stem as _export_stem
from trt_conv.capture_blocks import install_block_capture_hooks


def _install_block_capture(model, onnx_path=None, opset=17):
    # `onnx_path` is ignored — kept in the signature so we drop into
    # generate_with_hook.py's existing call site without further edits.
    install_block_capture_hooks(model, save_dir="/workspace/s2v_trt/blocks")


_export_stem.install_export_hook = _install_block_capture

runpy.run_module("generate_with_hook", run_name="__main__")
