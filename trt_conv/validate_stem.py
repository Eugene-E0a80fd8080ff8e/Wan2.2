"""
Phase 3: numerical validation of stem.onnx against the PyTorch stem.

Loads the same captured inputs used for export, runs:
  - PyTorch stem on GPU,
  - ONNX stem via onnxruntime (CUDA if available, else CPU),
and reports max / mean abs-diff per output.

Usage:
    python -m trt_conv.validate_stem --ckpt ./Wan2.2-S2V-14B/ \
        --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
        --onnx /workspace/s2v_trt/stem.onnx
"""
import argparse
import gc

import numpy as np
import torch

from trt_conv.export_stem import ALL_POSITIONAL_NAMES, TENSOR_INPUT_NAMES


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.detach().cpu().numpy()


def run_pytorch(args, snap, dtype, device):
    print(f"[validate] loading pytorch model from {args.ckpt}")
    from wan.modules.s2v.model_s2v import WanModel_S2V
    model = WanModel_S2V.from_pretrained(
        args.ckpt, torch_dtype=dtype, device_map=str(device))

    trim_attrs = [
        "patch_embedding", "text_embedding", "time_embedding", "time_projection",
        "head", "casual_audio_encoder", "trainable_cond_mask",
        "motioner", "zip_motion_out", "frame_packer", "cond_encoder",
        "token_freqs",
    ]
    for attr in trim_attrs:
        if hasattr(model, attr):
            delattr(model, attr)

    stem = model.stem
    stem.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    positional = []
    for name in ALL_POSITIONAL_NAMES:
        v = snap[name]
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        positional.append(v)

    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        out = stem(*positional)

    out_cpu = out.detach().cpu()
    del model, stem, positional, out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_cpu


def run_onnx(args, snap):
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    print(f"[validate] onnxruntime providers: {providers}")
    sess = ort.InferenceSession(args.onnx, providers=providers)

    expected = {i.name for i in sess.get_inputs()}
    print(f"[validate] onnx graph inputs: {sorted(expected)}")

    feed = {}
    for name in TENSOR_INPUT_NAMES:
        if name not in expected:
            print(f"[validate] warn: {name} absent from onnx graph (folded?)")
            continue
        v = snap[name]
        if not isinstance(v, torch.Tensor):
            raise RuntimeError(f"snap[{name}] is {type(v)}, expected Tensor")
        feed[name] = _to_numpy(v)

    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feed)
    return torch.from_numpy(outs[0])


def compare(a: torch.Tensor, b: torch.Tensor):
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        print(f"[validate] SHAPE MISMATCH: pt={tuple(a.shape)} onnx={tuple(b.shape)}")
        return
    diff = (a - b).abs()
    rel = diff / (a.abs().clamp(min=1e-6))
    print(f"[validate] shape        = {tuple(a.shape)}")
    print(f"[validate] max abs diff = {diff.max().item():.6g}")
    print(f"[validate] mean abs diff= {diff.mean().item():.6g}")
    print(f"[validate] max rel diff = {rel.max().item():.6g}")
    print(f"[validate] pt  stats    : min={a.min().item():.4g} max={a.max().item():.4g} "
          f"mean={a.mean().item():.4g}")
    print(f"[validate] onnx stats   : min={b.min().item():.4g} max={b.max().item():.4g} "
          f"mean={b.mean().item():.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print(f"[validate] loading inputs from {args.inputs}")
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)

    pt_out = run_pytorch(args, snap, dtype, device)
    onnx_out = run_onnx(args, snap)
    compare(pt_out, onnx_out)


if __name__ == "__main__":
    main()
