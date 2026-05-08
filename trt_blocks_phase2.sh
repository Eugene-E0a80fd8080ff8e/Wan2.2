#!/bin/bash
# Phase 2 (per-block): export each block's ONNX from its captured inputs.
# Skips blocks whose ONNX already exists.
set -e

PYTHONPATH=/workspace/Wan2.2 python -m trt_conv.export_blocks \
    --ckpt ./Wan2.2-S2V-14B/ \
    --inputs_dir /workspace/s2v_trt/blocks \
    --onnx_dir /workspace/s2v_trt/blocks \
    --dtype bfloat16
