"""
Phase 2: export S2VBlockStem to ONNX from a captured input snapshot.

Loads the model in a fresh process (so no live forward frame is holding
~15 GB of intermediate activations), trims it down to just the stem's
owned params (blocks + audio_injector), then runs torch.onnx.export on GPU.

Usage:
    python -m trt_conv.run_export --ckpt ./Wan2.2-S2V-14B/ \
        --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
        --onnx /workspace/s2v_trt/stem.onnx
"""
import argparse
import gc

import torch

from trt_conv.export_stem import ALL_POSITIONAL_NAMES, TENSOR_INPUT_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to Wan2.2-S2V-14B checkpoint dir")
    ap.add_argument("--inputs", required=True, help="captured inputs .pt from Phase 1")
    ap.add_argument("--onnx", required=True, help="output ONNX path")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print(f"[run_export] loading model from {args.ckpt}")
    from wan.modules.s2v.model_s2v import WanModel_S2V
    model = WanModel_S2V.from_pretrained(
        args.ckpt, torch_dtype=dtype, device_map=str(device))

    # Trim everything not needed for stem export. The stem only needs
    # `blocks` and `audio_injector` (both referenced by stem via object.__setattr__).
    trim_attrs = [
        "patch_embedding", "text_embedding", "time_embedding", "time_projection",
        "head", "casual_audio_encoder", "trainable_cond_mask",
        "motioner", "zip_motion_out", "frame_packer", "cond_encoder",
        "token_freqs", "freqs",
    ]
    for attr in trim_attrs:
        if hasattr(model, attr):
            print(f"[run_export] deleting model.{attr}")
            delattr(model, attr)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[run_export] GPU mem after trim: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

    print(f"[run_export] loading inputs from {args.inputs}")
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)

    positional = []
    for name in ALL_POSITIONAL_NAMES:
        v = snap[name]
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        positional.append(v)
    positional = tuple(positional)

    print("[run_export] input summary:")
    for n, v in zip(ALL_POSITIONAL_NAMES, positional):
        if isinstance(v, torch.Tensor):
            print(f"  {n:22s} Tensor {tuple(v.shape)} {v.dtype} dev={v.device}")
        else:
            print(f"  {n:22s} {type(v).__name__} {v!r}")

    stem = model.stem
    stem.eval()
    for p in stem.parameters():
        p.requires_grad_(False)

    print(f"[run_export] exporting to {args.onnx} (opset={args.opset})")
    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        torch.onnx.export(
            stem,
            positional,
            args.onnx,
            input_names=TENSOR_INPUT_NAMES,
            output_names=["out"],
            opset_version=args.opset,
            do_constant_folding=False,
            dynamic_axes=None,
        )
    print(f"[run_export] wrote {args.onnx}")


if __name__ == "__main__":
    main()
