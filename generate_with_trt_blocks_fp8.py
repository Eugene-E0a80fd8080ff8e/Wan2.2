"""
Phase 5 driver (fp8): inference with each transformer block delegated to a
per-block FP8 TRT engine. Audio injector stays in PyTorch.
"""
import runpy

import trt_conv.export_stem as _export_stem
from trt_conv.block_runner import install_block_runners
from trt_conv.load_s2v_no_blocks import install_no_blocks_loader


_runners_holder = []


def _install_trt_block_runners(model, onnx_path=None, opset=17):
    runners = install_block_runners(
        model,
        engines_dir="/workspace/s2v_trt/blocks",
        device="cuda:0",
        engine_suffix=".fp8.trt",
    )
    _runners_holder.extend(runners)


install_no_blocks_loader()
_export_stem.install_export_hook = _install_trt_block_runners

runpy.run_module("generate_with_hook", run_name="__main__")
