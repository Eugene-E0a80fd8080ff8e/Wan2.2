#!/bin/bash
# Phase 3 (per-block): build a bf16 TRT engine for each block ONNX.
# Skips blocks whose .trt already exists.
set -e

BLOCKS_DIR=/workspace/s2v_trt/blocks

for onnx in "$BLOCKS_DIR"/block_*.onnx; do
    name=$(basename "$onnx" .onnx)
    engine="$BLOCKS_DIR/$name.trt"
    if [ -f "$engine" ]; then
        echo "[trt_blocks_phase3] $name: engine already exists, skipping"
        continue
    fi
    echo "[trt_blocks_phase3] $name: building engine"
    trtexec --onnx="$onnx" \
            --saveEngine="$engine" \
            --bf16 \
            --memPoolSize=workspace:8192 \
            2>&1 | tee "$BLOCKS_DIR/$name.build.log"
done
