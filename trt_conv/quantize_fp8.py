"""
Quantize stem.onnx to FP8 using TensorRT-ModelOpt.

Produces a new ONNX file with FP4 weights and Q/DQ nodes inserted, which
trtexec can then compile into an FP4 TRT engine.

Usage:
    python -m trt_conv.quantize_fp8 \
        --onnx /workspace/s2v_trt/stem.onnx \
        --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
        --out /workspace/s2v_trt/stem.fp8.onnx

Then build the engine:
    trtexec --onnx=/workspace/s2v_trt/stem.fp8.onnx \
            --saveEngine=/workspace/s2v_trt/stem.fp8.trt \
            --stronglyTyped --bf16 \
            --memPoolSize=workspace:8192
"""
import argparse

import numpy as np
import torch


def _bf16_safe_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.detach().cpu().numpy()


class _SingleSampleReader:
    """Minimal CalibrationDataReader: yields the captured snapshot once."""

    def __init__(self, feed):
        self._feed = feed
        self._done = False

    def get_next(self):
        if self._done:
            return None
        self._done = True
        return self._feed

    def rewind(self):
        self._done = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="input stem.onnx (bf16)")
    ap.add_argument("--inputs", required=True, help="captured .pt from Phase 1")
    ap.add_argument("--out", required=True, help="output fp8 ONNX path")
    args = ap.parse_args()

    from modelopt.onnx.quantization import quantize

    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
    tensor_input_names = [
        "x", "e", "seq_lens", "freqs", "context",
        "merged_audio_emb", "audio_emb_global",
    ]
    feed = {}
    for name in tensor_input_names:
        v = snap[name]
        if not isinstance(v, torch.Tensor):
            continue
        feed[name] = _bf16_safe_numpy(v)
        print(f"[quantize_fp8] {name:22s} {feed[name].shape} {feed[name].dtype}")

    reader = _SingleSampleReader(feed)

    print(f"[quantize_fp8] quantizing {args.onnx} → {args.out} (FP8)")
    quantize(
        onnx_path=args.onnx,
        quantize_mode="fp8",
        calibration_data_reader=reader,
        calibration_method="max",
        output_path=args.out,
    )
    print(f"[quantize_fp8] wrote {args.out}")


if __name__ == "__main__":
    main()
