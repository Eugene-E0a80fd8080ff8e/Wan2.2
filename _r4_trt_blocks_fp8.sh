#!/bin/bash
# Run inference with per-block FP8 TRT engines + PyTorch audio_injector.
set -e

python generate_with_trt_blocks_fp8.py --task s2v-14B --size 480*480 \
    --ckpt_dir ./Wan2.2-S2V-14B/ \
    --offload_model True \
    --convert_model_dtype \
    --prompt "The girl smiles and talks to the camera." \
    --sample_steps 16 \
    --image "face1.jpg" --audio "input2.wav"
