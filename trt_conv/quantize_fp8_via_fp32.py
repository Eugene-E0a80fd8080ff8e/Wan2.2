"""
FP8 quantization via an in-memory bf16 -> fp32 conversion of stem.onnx.

The straight FP8 path (trt_conv/quantize_fp8.py) failed because ModelOpt's
calibration runs ORT on the bf16 graph, and ORT can't accept bf16 numpy feeds
cleanly. This script sidesteps that by first promoting the ONNX to fp32 in
memory, writing it to a workdir (default /dev/shm so it doesn't touch disk),
and only then handing it to ModelOpt.

The fp32 staging file is large (~56 GB stem weights + ~28 GB external data).
Make sure /dev/shm has room (vast.ai usually does), or pass --workdir to
override.

Usage:
    python -m trt_conv.quantize_fp8_via_fp32 \\
        --onnx /workspace/s2v_trt/stem.onnx \\
        --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \\
        --out /workspace/s2v_trt/stem.fp8.onnx
"""
import argparse
import gc
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import torch


def _convert_tensor_bf16_to_fp16(tp: onnx.TensorProto) -> None:
    if tp.data_type != onnx.TensorProto.BFLOAT16:
        return
    if tp.raw_data:
        arr_bf16 = np.frombuffer(tp.raw_data, dtype=ml_dtypes.bfloat16)
        arr_fp16 = arr_bf16.astype(np.float16)
        tp.raw_data = arr_fp16.tobytes()
    elif len(tp.int32_data) > 0:
        arr_u16 = np.array(tp.int32_data, dtype=np.uint32).astype(np.uint16)
        arr_bf16 = arr_u16.view(ml_dtypes.bfloat16)
        arr_fp16 = arr_bf16.astype(np.float16)
        del tp.int32_data[:]
        tp.raw_data = arr_fp16.tobytes()
    tp.data_type = onnx.TensorProto.FLOAT16


def _convert_value_info_bf16_to_fp16(vi: onnx.ValueInfoProto) -> None:
    if vi.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16


def _hoist_constants_to_initializers(g: onnx.GraphProto) -> int:
    """Move Constant nodes with a `value` tensor attr into graph.initializer.

    ModelOpt's save_as_external_data only externalizes initializers (not
    Constant op `value` attributes), so leaving them as Constant nodes makes
    the proto exceed the 2 GB wire-format limit when the model has many large
    Constants. Hoisting them to initializers lets ModelOpt externalize them.
    """
    n_hoisted = 0
    keep_nodes = []
    new_inits = []
    for node in g.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            value_tp = None
            for attr in node.attribute:
                if attr.name == "value":
                    value_tp = attr.t
                    break
            if value_tp is not None:
                init = onnx.TensorProto()
                init.CopyFrom(value_tp)
                init.name = node.output[0]
                new_inits.append(init)
                n_hoisted += 1
                continue
        keep_nodes.append(node)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                n_hoisted += _hoist_constants_to_initializers(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    n_hoisted += _hoist_constants_to_initializers(sg)
    del g.node[:]
    g.node.extend(keep_nodes)
    g.initializer.extend(new_inits)
    return n_hoisted


def _convert_graph_bf16_to_fp16(g: onnx.GraphProto) -> int:
    n_changed = 0
    for init in g.initializer:
        if init.data_type == onnx.TensorProto.BFLOAT16:
            _convert_tensor_bf16_to_fp16(init)
            n_changed += 1
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
            _convert_value_info_bf16_to_fp16(vi)
            n_changed += 1
    for node in g.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.BFLOAT16:
                    attr.i = onnx.TensorProto.FLOAT16
                    n_changed += 1
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.BFLOAT16:
                    _convert_tensor_bf16_to_fp16(attr.t)
                    n_changed += 1
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                n_changed += _convert_graph_bf16_to_fp16(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    n_changed += _convert_graph_bf16_to_fp16(sg)
    return n_changed


def _tensor_to_graph_numpy(t: torch.Tensor, graph_dtype: int) -> np.ndarray:
    """Convert a torch tensor to numpy with a dtype matching the ONNX graph input."""
    if graph_dtype == onnx.TensorProto.FLOAT16:
        return t.detach().cpu().to(torch.float16).numpy()
    if graph_dtype == onnx.TensorProto.FLOAT:
        return t.detach().cpu().to(torch.float32).numpy()
    if graph_dtype == onnx.TensorProto.INT64:
        return t.detach().cpu().to(torch.int64).numpy()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="input bf16 stem.onnx")
    ap.add_argument("--inputs", required=True, help="captured .pt from Phase 1")
    ap.add_argument("--out", required=True, help="output fp8 ONNX path")
    ap.add_argument("--workdir", default="/dev/shm",
                    help="where to stage the fp32 ONNX (RAM-backed; default /dev/shm)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = workdir / "stem.fp32.onnx"
    fp32_data = "stem.fp32.onnx.data"
    fp32_data_path = workdir / fp32_data

    if fp32_onnx.exists() and fp32_data_path.exists():
        print(f"[fp8_via_fp32] reusing existing fp16 staging {fp32_onnx} "
              f"(+ {fp32_data_path.stat().st_size / 1e9:.1f} GB sidecar)")
    else:
        print(f"[fp8_via_fp32] loading {args.onnx} (with external data)")
        model = onnx.load(args.onnx, load_external_data=True)

        print(f"[fp8_via_fp32] converting bf16 -> fp16 in memory")
        n_changed = _convert_graph_bf16_to_fp16(model.graph)
        print(f"[fp8_via_fp32] converted {n_changed} bf16 entities")

        print(f"[fp8_via_fp32] hoisting Constant nodes to graph initializers")
        n_hoisted = _hoist_constants_to_initializers(model.graph)
        print(f"[fp8_via_fp32] hoisted {n_hoisted} Constant nodes")

        print(f"[fp8_via_fp32] writing fp16 staging onnx to {fp32_onnx}")
        onnx.save(
            model,
            str(fp32_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=fp32_data,
            size_threshold=1024,
        )
        del model
        gc.collect()

    print(f"[fp8_via_fp32] building calibration feed from {args.inputs}")
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
    tensor_input_names = [
        "x", "e", "seq_lens", "freqs", "context",
        "merged_audio_emb", "audio_emb_global",
    ]
    staging_meta = onnx.load(str(fp32_onnx), load_external_data=False)
    graph_dtypes = {
        inp.name: inp.type.tensor_type.elem_type for inp in staging_meta.graph.input
    }
    del staging_meta
    feed = {}
    for name in tensor_input_names:
        v = snap[name]
        if not isinstance(v, torch.Tensor):
            continue
        arr = _tensor_to_graph_numpy(v, graph_dtypes.get(name, onnx.TensorProto.FLOAT))
        feed[name] = arr
        print(f"[fp8_via_fp32] {name:22s} {arr.shape} {arr.dtype}")

    # Stub out find_nodes_from_mha_to_exclude before invoking quantize.
    # That function runs an ORT session over the full fp32 model with extended
    # outputs (every per-layer activation kept resident) which OOMs on 96 GB.
    # Skipping it just means MHA-adjacent nodes get quantized like everything
    # else — a known-benign tradeoff for this graph.
    import importlib
    _q_mod = importlib.import_module("modelopt.onnx.quantization.quantize")
    _q_mod.find_nodes_from_mha_to_exclude = lambda *a, **kw: []
    quantize = _q_mod.quantize

    print(f"[fp8_via_fp32] quantizing {fp32_onnx} -> {args.out} (FP8)")
    quantize(
        onnx_path=str(fp32_onnx),
        quantize_mode="fp8",
        calibration_data=feed,
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        output_path=args.out,
        use_external_data_format=True,
        calibrate_per_node=True,
    )
    print(f"[fp8_via_fp32] wrote {args.out}")

    try:
        os.remove(str(fp32_onnx))
        os.remove(str(workdir / fp32_data))
        print(f"[fp8_via_fp32] cleaned up {fp32_onnx} and external data")
    except OSError as e:
        print(f"[fp8_via_fp32] cleanup warning: {e}")


if __name__ == "__main__":
    main()
