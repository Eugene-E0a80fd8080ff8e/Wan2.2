"""
FP8 quantization for ONE per-block ONNX, via in-memory bf16 -> fp16 staging.

Same trick as quantize_fp8_via_fp32.py but scoped to a single block, so the
ORT calibration session stays small enough to fit easily on 96 GB.

Usage:
    python -m trt_conv.quantize_block_fp8 \\
        --onnx /workspace/s2v_trt/blocks/block_05.onnx \\
        --inputs /workspace/s2v_trt/blocks/block_05.pt \\
        --out /workspace/s2v_trt/blocks/block_05.fp8.onnx \\
        --workdir /workspace/workdir
"""
import argparse
import gc
import importlib
from pathlib import Path

import onnx
import torch

from trt_conv.quantize_fp8_via_fp32 import (
    _convert_graph_bf16_to_fp16,
    _hoist_constants_to_initializers,
    _tensor_to_graph_numpy,
)


# Block ONNX inputs (from export_blocks.py BLOCK_TENSOR_INPUT_NAMES)
BLOCK_TENSOR_INPUT_NAMES = ["x", "e_tensor", "seq_lens", "freqs", "context"]


def _build_feed(snap, graph_dtypes):
    """Map block snapshot keys -> ONNX input names. e_tensor comes from e[0]."""
    e_list = snap["e"]
    e_tensor = e_list[0] if isinstance(e_list, (list, tuple)) else e_list

    sources = {
        "x": snap["x"],
        "e_tensor": e_tensor,
        "seq_lens": snap["seq_lens"],
        "freqs": snap["freqs"],
        "context": snap["context"],
    }

    feed = {}
    for name in BLOCK_TENSOR_INPUT_NAMES:
        v = sources[name]
        if not isinstance(v, torch.Tensor):
            raise RuntimeError(f"snapshot {name} is not a Tensor: {type(v).__name__}")
        feed[name] = _tensor_to_graph_numpy(v, graph_dtypes.get(name, onnx.TensorProto.FLOAT))
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="input bf16 block ONNX")
    ap.add_argument("--inputs", required=True, help="block_NN.pt snapshot")
    ap.add_argument("--out", required=True, help="output fp8 ONNX path")
    ap.add_argument("--workdir", default="/workspace/workdir",
                    help="staging dir for fp16 ONNX + ModelOpt scratch files")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    stem = onnx_path.stem  # e.g. "block_05"
    fp16_onnx = workdir / f"{stem}.fp16.onnx"
    fp16_data = f"{stem}.fp16.onnx.data"
    fp16_data_path = workdir / fp16_data

    if fp16_onnx.exists() and fp16_data_path.exists():
        print(f"[block_fp8] reusing existing fp16 staging {fp16_onnx}")
    else:
        print(f"[block_fp8] loading {onnx_path}")
        model = onnx.load(str(onnx_path), load_external_data=True)

        print(f"[block_fp8] converting bf16 -> fp16 in memory")
        n = _convert_graph_bf16_to_fp16(model.graph)
        print(f"[block_fp8] converted {n} bf16 entities")

        print(f"[block_fp8] hoisting Constant nodes to initializers")
        n = _hoist_constants_to_initializers(model.graph)
        print(f"[block_fp8] hoisted {n} Constant nodes")

        print(f"[block_fp8] writing fp16 staging onnx to {fp16_onnx}")
        onnx.save(
            model,
            str(fp16_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=fp16_data,
            size_threshold=1024,
        )
        del model
        gc.collect()

    print(f"[block_fp8] building calibration feed from {args.inputs}")
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
    staging_meta = onnx.load(str(fp16_onnx), load_external_data=False)
    graph_dtypes = {
        inp.name: inp.type.tensor_type.elem_type for inp in staging_meta.graph.input
    }
    del staging_meta
    feed = _build_feed(snap, graph_dtypes)
    for name, arr in feed.items():
        print(f"[block_fp8] {name:12s} {arr.shape} {arr.dtype}")

    # Stub out MHA-exclusion analysis: it runs the full block as one ORT session
    # capturing every per-layer activation, which can OOM. Skipping has the same
    # benign trade-off we made for the stem — MHA-adjacent ops get quantized
    # like everything else.
    _q_mod = importlib.import_module("modelopt.onnx.quantization.quantize")
    _q_mod.find_nodes_from_mha_to_exclude = lambda *a, **kw: []
    quantize = _q_mod.quantize

    print(f"[block_fp8] quantizing -> {args.out} (FP8)")
    quantize(
        onnx_path=str(fp16_onnx),
        quantize_mode="fp8",
        calibration_data=feed,
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        output_path=args.out,
        use_external_data_format=True,
        calibrate_per_node=True,
    )
    print(f"[block_fp8] wrote {args.out}")


if __name__ == "__main__":
    main()
