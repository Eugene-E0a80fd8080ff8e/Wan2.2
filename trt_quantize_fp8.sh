#!/bin/bash
# Phase 6 (fp8): quantize stem.onnx (bf16) to stem.fp8.onnx via in-memory fp32 staging.
# Stages a fp32 copy of the ONNX in /dev/shm so disk usage stays low.

set -e

mkdir -p /workspace/s2v_trt

PYTHONPATH=/workspace/Wan2.2 python -m trt_conv.quantize_fp8_via_fp32 \
    --onnx /workspace/s2v_trt/stem.onnx \
    --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
    --out /workspace/s2v_trt/stem.fp8.onnx \
    --workdir /dev/shm
