#!/bin/bash
# Phase 3 fp8 (per-block): for every block_NN.onnx in BLOCKS_DIR, produce
#   block_NN.fp8.onnx  (via ModelOpt FP8 quantization, fp16 staging)
#   block_NN.fp8.trt   (via trtexec --stronglyTyped --bf16)
# Skips any block that already has the final .fp8.trt engine.
set -e

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

BLOCKS_DIR=/workspace/s2v_trt/blocks
WORKDIR=/workspace/workdir

mkdir -p "$BLOCKS_DIR" "$WORKDIR"

for onnx in "$BLOCKS_DIR"/block_[0-9][0-9].onnx; do
    base=$(basename "$onnx" .onnx)

    snap="$BLOCKS_DIR/$base.pt"
    fp8_onnx="$BLOCKS_DIR/$base.fp8.onnx"
    fp8_engine="$BLOCKS_DIR/$base.fp8.trt"

    if [ -f "$fp8_engine" ]; then
        echo "[phase3_fp8] $base: engine already exists, skipping"
        continue
    fi

    if [ ! -f "$fp8_onnx" ]; then
        echo "[phase3_fp8] $base: quantizing to fp8 onnx"
        PYTHONPATH=/workspace/Wan2.2 python -m trt_conv.quantize_block_fp8 \
            --onnx "$onnx" \
            --inputs "$snap" \
            --out "$fp8_onnx" \
            --workdir "$WORKDIR"
    else
        echo "[phase3_fp8] $base: fp8 onnx exists, reusing"
    fi

    echo "[phase3_fp8] $base: building fp8 engine"
    trtexec --onnx="$fp8_onnx" \
            --saveEngine="$fp8_engine" \
            --stronglyTyped --bf16 \
            --memPoolSize=workspace:8192 \
            2>&1 | tee "$BLOCKS_DIR/$base.fp8.build.log"
done
