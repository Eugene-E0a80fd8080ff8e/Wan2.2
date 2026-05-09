"""
FP8 quantization for ONE per-block ONNX, via in-memory bf16 -> fp32 staging.

Same trick as quantize_fp8_via_fp32.py but scoped to a single block. fp32
(rather than fp16) keeps numerics stable — per-block calibration only needs
~45 GB peak which fits comfortably on a 96 GB card.

Usage:
    python -m trt_conv.quantize_block_fp8 \\
        --onnx /workspace/s2v_trt/blocks/block_05.onnx \\
        --inputs /workspace/s2v_trt/blocks/block_05.pt \\
        --out /workspace/s2v_trt/blocks/block_05.fp8.onnx \\
        --workdir /workspace/workdir
"""
import argparse
import gc
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import torch

from trt_conv.quantize_fp8_via_fp32 import _hoist_constants_to_initializers


# Block ONNX inputs (from export_blocks.py BLOCK_TENSOR_INPUT_NAMES)
BLOCK_TENSOR_INPUT_NAMES = ["x", "e_tensor", "seq_lens", "freqs", "context"]


def _convert_tensor_bf16_to_fp32(tp: onnx.TensorProto) -> None:
    if tp.data_type != onnx.TensorProto.BFLOAT16:
        return
    if tp.raw_data:
        arr_bf16 = np.frombuffer(tp.raw_data, dtype=ml_dtypes.bfloat16)
        tp.raw_data = arr_bf16.astype(np.float32).tobytes()
    elif len(tp.int32_data) > 0:
        arr_u16 = np.array(tp.int32_data, dtype=np.uint32).astype(np.uint16)
        arr_bf16 = arr_u16.view(ml_dtypes.bfloat16)
        del tp.int32_data[:]
        tp.raw_data = arr_bf16.astype(np.float32).tobytes()
    tp.data_type = onnx.TensorProto.FLOAT


def _convert_graph_bf16_to_fp32(g: onnx.GraphProto) -> int:
    n = 0
    for init in g.initializer:
        if init.data_type == onnx.TensorProto.BFLOAT16:
            _convert_tensor_bf16_to_fp32(init)
            n += 1
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            n += 1
    for node in g.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.BFLOAT16:
                    attr.i = onnx.TensorProto.FLOAT
                    n += 1
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.BFLOAT16:
                    _convert_tensor_bf16_to_fp32(attr.t)
                    n += 1
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                n += _convert_graph_bf16_to_fp32(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    n += _convert_graph_bf16_to_fp32(sg)
    return n


def _tensor_to_numpy(t: torch.Tensor, graph_dtype: int) -> np.ndarray:
    if graph_dtype == onnx.TensorProto.FLOAT:
        return t.detach().cpu().to(torch.float32).numpy()
    if graph_dtype == onnx.TensorProto.FLOAT16:
        return t.detach().cpu().to(torch.float16).numpy()
    if graph_dtype == onnx.TensorProto.INT64:
        return t.detach().cpu().to(torch.int64).numpy()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.detach().cpu().numpy()


def _find_attention_bmms(graph: onnx.GraphProto) -> list[str]:
    """Return names of MatMul nodes whose inputs are both activations.

    These are the attention BMMs (Q@K^T and softmax@V) that NVIDIA's
    automatic MHA exclusion identifies. Skipping FP8 on these preserves
    quality. Projection/FFN MatMuls (which have a weight initializer)
    keep getting FP8.

    Heuristic: a tensor is "a weight" if it's an initializer or Constant
    output, possibly via pass-through ops (Cast/Reshape/Transpose/etc.).
    """
    init_names = {init.name for init in graph.initializer}
    const_outputs = {o for n in graph.node if n.op_type == "Constant" for o in n.output}
    weight_tensors = init_names | const_outputs

    producer = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n

    PASS_THROUGH = {"Cast", "Reshape", "Transpose", "Identity",
                    "Squeeze", "Unsqueeze", "Expand"}

    def traces_to_weight(name, depth=12):
        if name in weight_tensors:
            return True
        if depth <= 0:
            return False
        n = producer.get(name)
        if n is None:
            return False
        if n.op_type in PASS_THROUGH and n.input:
            return traces_to_weight(n.input[0], depth - 1)
        return False

    bmms = []
    for n in graph.node:
        if n.op_type != "MatMul":
            continue
        if len(n.input) < 2:
            continue
        if not traces_to_weight(n.input[0]) and not traces_to_weight(n.input[1]):
            bmms.append(n.name or f"<unnamed at output {n.output[0]}>")
    return bmms


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
        feed[name] = _tensor_to_numpy(v, graph_dtypes.get(name, onnx.TensorProto.FLOAT))
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="input bf16 block ONNX")
    ap.add_argument("--inputs", required=True, nargs="+",
                    help="block snapshot file(s); pass multiple for multi-sample calibration")
    ap.add_argument("--out", required=True, help="output fp8 ONNX path")
    ap.add_argument("--workdir", default="/workspace/workdir",
                    help="staging dir for fp32 ONNX + ModelOpt scratch files")
    ap.add_argument("--calibration_method", default="entropy",
                    choices=["max", "entropy", "percentile"])
    ap.add_argument("--skip_mha_exclude", action="store_true",
                    help="skip MHA exclusion analysis (avoids TRT engine build OOM)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    stem = onnx_path.stem  # e.g. "block_05"
    fp32_onnx = workdir / f"{stem}.fp32.onnx"
    fp32_data = f"{stem}.fp32.onnx.data"
    fp32_data_path = workdir / fp32_data

    if fp32_onnx.exists() and fp32_data_path.exists():
        print(f"[block_fp8] reusing existing fp32 staging {fp32_onnx}")
    else:
        print(f"[block_fp8] loading {onnx_path}")
        model = onnx.load(str(onnx_path), load_external_data=True)

        print(f"[block_fp8] converting bf16 -> fp32 in memory")
        n = _convert_graph_bf16_to_fp32(model.graph)
        print(f"[block_fp8] converted {n} bf16 entities")

        print(f"[block_fp8] hoisting Constant nodes to initializers")
        n = _hoist_constants_to_initializers(model.graph)
        print(f"[block_fp8] hoisted {n} Constant nodes")

        print(f"[block_fp8] writing fp32 staging onnx to {fp32_onnx}")
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

    staging_meta = onnx.load(str(fp32_onnx), load_external_data=False)
    graph_dtypes = {
        inp.name: inp.type.tensor_type.elem_type for inp in staging_meta.graph.input
    }
    manual_mha_excludes = _find_attention_bmms(staging_meta.graph)
    del staging_meta
    print(f"[block_fp8] manual MHA-bmm exclusions: {manual_mha_excludes}")

    print(f"[block_fp8] building {len(args.inputs)} calibration sample(s)")
    feeds = []
    for path in args.inputs:
        snap = torch.load(path, map_location="cpu", weights_only=False)
        feeds.append(_build_feed(snap, graph_dtypes))
        print(f"[block_fp8]   loaded {path}")
    for name, arr in feeds[0].items():
        print(f"[block_fp8] {name:12s} {arr.shape} {arr.dtype}  (×{len(feeds)} samples)")

    # MHA exclusion analysis is enabled — per-block scope is small enough to
    # fit, and TRT EP (in calibration_eps below) gives it flash attention so
    # the seq×seq×heads buffer never materializes. The exclusion keeps ops
    # adjacent to attention (softmax inputs, etc.) at higher precision, which
    # FP8 quantization quality depends on.
    from modelopt.onnx.quantization import quantize

    if args.skip_mha_exclude:
        # quantize.py imported find_nodes_from_mha_to_exclude into its own
        # namespace, so we patch THAT — patching graph_utils alone has no
        # effect on the bound reference in quantize.py.
        import importlib
        _q_mod = importlib.import_module("modelopt.onnx.quantization.quantize")
        _q_mod.find_nodes_from_mha_to_exclude = lambda *a, **kw: []
        print("[block_fp8] MHA exclusion DISABLED (quality tradeoff)")

    # ModelOpt's quantize() builds an ORT calibration reader from a single
    # `calibration_data` dict. To feed multiple samples we monkey-patch
    # fp8.quantize_static to swap in our own multi-sample reader.
    import inspect

    import modelopt.onnx.quantization.fp8 as _fp8
    _orig_quantize_static = _fp8.quantize_static
    _orig_sig = inspect.signature(_orig_quantize_static)

    class _MultiSampleReader:
        def __init__(self, samples):
            self.samples = samples
            self.idx = 0
        def get_next(self):
            if self.idx >= len(self.samples):
                return None
            s = self.samples[self.idx]
            self.idx += 1
            return s
        def rewind(self):
            self.idx = 0

    reader = _MultiSampleReader(feeds)

    def _patched_quantize_static(*p_args, **p_kwargs):
        bound = _orig_sig.bind(*p_args, **p_kwargs)
        bound.arguments["calibration_data_reader"] = reader
        return _orig_quantize_static(**bound.arguments)

    _fp8.quantize_static = _patched_quantize_static

    nodes_to_exclude = manual_mha_excludes if args.skip_mha_exclude else None

    print(f"[block_fp8] quantizing -> {args.out} "
          f"(FP8, method={args.calibration_method}, samples={len(feeds)}, "
          f"manual_excludes={len(manual_mha_excludes) if nodes_to_exclude else 0})")
    quantize(
        onnx_path=str(fp32_onnx),
        quantize_mode="fp8",
        calibration_data=feeds[0],  # placeholder; replaced by our reader
        calibration_method=args.calibration_method,
        # CUDA EP only. TRT EP would JIT-compile a giant fused engine that
        # asks for ~89 GB on this graph; per-node CUDA calibration sidesteps
        # that by keeping only one MatMul's activation alive at a time.
        calibration_eps=["cuda:0", "cpu"],
        output_path=args.out,
        use_external_data_format=True,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_per_node=True,
    )
    print(f"[block_fp8] wrote {args.out}")


if __name__ == "__main__":
    main()
