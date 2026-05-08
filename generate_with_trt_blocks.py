"""
Phase 5 driver: run inference with each transformer block delegated to a
per-block TRT engine. The audio_injector stays in PyTorch.

Reuses generate_with_hook.py for everything (CLI parsing, model setup,
denoising loop) and only swaps the install hook.
"""
import runpy

import trt_conv.export_stem as _export_stem
from trt_conv.block_runner import install_block_runners


_runners_holder = []


def _install_trt_block_runners(model, onnx_path=None, opset=17):
    runners = install_block_runners(
        model,
        engines_dir="/workspace/s2v_trt/blocks",
        device="cuda:0",
    )
    _runners_holder.extend(runners)


_export_stem.install_export_hook = _install_trt_block_runners

runpy.run_module("generate_with_hook", run_name="__main__")
