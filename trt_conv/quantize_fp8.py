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

import ml_dtypes
import numpy as np
import onnx
import torch


# ONNX tensor elem_type enum values we care about.
_ONNX_FLOAT = 1
_ONNX_INT64 = 7
_ONNX_BFLOAT16 = 16
_ONNX_FLOAT16 = 10


def _cast_to_graph_dtype(arr: np.ndarray, elem_type: int) -> np.ndarray:
    if elem_type == _ONNX_BFLOAT16:
        return arr.astype(np.float32).astype(ml_dtypes.bfloat16)
    if elem_type == _ONNX_FLOAT16:
        return arr.astype(np.float16)
    if elem_type == _ONNX_FLOAT:
        return arr.astype(np.float32)
    if elem_type == _ONNX_INT64:
        return arr.astype(np.int64)
    return arr


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.detach().cpu().numpy()


def _read_input_dtypes(onnx_path: str) -> dict[str, int]:
    m = onnx.load(onnx_path, load_external_data=False)
    return {inp.name: inp.type.tensor_type.elem_type for inp in m.graph.input}


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

    def get_first(self):
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
    from modelopt.onnx.quantization import graph_utils as _gu
    from modelopt.onnx.quantization import quantize as _q_mod

    # Skip MHA-exclusion analysis: it runs the bf16 graph through ORT which
    # fails because ORT's Python API can't take bf16 inputs cleanly. This only
    # affects whether certain MHA-adjacent nodes get quantized; benign to skip.
    _stub = lambda *a, **kw: []
    _gu.find_nodes_from_mha_to_exclude = _stub
    _q_mod.find_nodes_from_mha_to_exclude = _stub

    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
    graph_dtypes = _read_input_dtypes(args.onnx)
    tensor_input_names = [
        "x", "e", "seq_lens", "freqs", "context",
        "merged_audio_emb", "audio_emb_global",
    ]
    feed = {}
    for name in tensor_input_names:
        v = snap[name]
        if not isinstance(v, torch.Tensor):
            continue
        arr = _tensor_to_numpy(v)
        elem_type = graph_dtypes.get(name)
        if elem_type is not None:
            arr = _cast_to_graph_dtype(arr, elem_type)
        feed[name] = arr
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
