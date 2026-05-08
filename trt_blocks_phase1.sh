#!/bin/bash
# Phase 1 (per-block): capture each transformer block's first-call inputs.
# Writes /workspace/s2v_trt/blocks/block_NN.pt for N=0..(num_blocks-1)
# then exits.
set -e

mkdir -p /workspace/s2v_trt/blocks

python generate_with_blockhook.py --task s2v-14B --size 480*480 \
    --ckpt_dir ./Wan2.2-S2V-14B/ \
    --offload_model True \
    --convert_model_dtype \
    --prompt "The girl smiles and talks to the camera." \
    --sample_steps 16 \
    --image "face1.jpg" --audio "input2.wav"
