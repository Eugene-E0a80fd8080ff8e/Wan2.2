"""
Per-block ONNX export. Loads WanModel_S2V once, iterates over blocks, exports
each with its captured inputs. Skips blocks whose ONNX already exists, so this
is safe to re-run.

Usage:
    python -m trt_conv.export_blocks \\
        --ckpt ./Wan2.2-S2V-14B/ \\
        --inputs_dir /workspace/s2v_trt/blocks \\
        --onnx_dir   /workspace/s2v_trt/blocks
"""
import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn


# Tensor inputs we expose to the ONNX tracer. Everything else (seg_idx,
# grid_sizes, context_lens) is baked as a constant at export time.
BLOCK_TENSOR_INPUT_NAMES = ["x", "e_tensor", "seq_lens", "freqs", "context"]


class BlockONNXWrapper(nn.Module):
    """Presents only real tensor inputs; bakes the static structural args."""

    def __init__(self, block, seg_idx, grid_sizes, context_lens):
        super().__init__()
        object.__setattr__(self, "block", block)
        if not isinstance(seg_idx, torch.Tensor):
            seg_idx = torch.tensor(seg_idx)
        self.seg_idx = seg_idx
        self.grid_sizes = grid_sizes
        self.context_lens = context_lens

    def forward(self, x, e_tensor, seq_lens, freqs, context):
        e = [e_tensor, self.seg_idx]
        return self.block(x, e, seq_lens, self.grid_sizes,
                          freqs, context, self.context_lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Wan2.2-S2V-14B checkpoint dir")
    ap.add_argument("--inputs_dir", required=True,
                    help="dir with block_NN.pt snapshots from Phase 1")
    ap.add_argument("--onnx_dir", required=True,
                    help="output dir for block_NN.onnx files")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--blocks", default="all",
                    help="'all' or comma-separated indices, e.g. '0,5,12'")
    args = ap.parse_args()

    inputs_dir = Path(args.inputs_dir)
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    snap_paths = sorted(inputs_dir.glob("block_*.pt"))
    if args.blocks == "all":
        block_ids = [int(p.stem.split("_")[1]) for p in snap_paths]
    else:
        block_ids = [int(x.strip()) for x in args.blocks.split(",")]

    todo = [i for i in block_ids
            if not (onnx_dir / f"block_{i:02d}.onnx").exists()]
    if not todo:
        print("[export_blocks] all requested blocks already exported. nothing to do.")
        return
    print(f"[export_blocks] will export blocks: {todo}")

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    print(f"[export_blocks] loading model from {args.ckpt}")
    from wan.modules.s2v.model_s2v import WanModel_S2V
    model = WanModel_S2V.from_pretrained(
        args.ckpt, torch_dtype=dtype, device_map=str(device))

    trim_attrs = [
        "patch_embedding", "text_embedding", "time_embedding", "time_projection",
        "head", "casual_audio_encoder", "trainable_cond_mask",
        "motioner", "zip_motion_out", "frame_packer", "cond_encoder",
        "token_freqs", "freqs", "audio_injector", "stem",
    ]
    for attr in trim_attrs:
        if hasattr(model, attr):
            delattr(model, attr)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[export_blocks] GPU mem after trim: "
              f"{(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")

    for p in model.parameters():
        p.requires_grad_(False)

    blocks = model.blocks

    for idx in todo:
        snap_path = inputs_dir / f"block_{idx:02d}.pt"
        out_path = onnx_dir / f"block_{idx:02d}.onnx"
        print(f"[export_blocks] block {idx:2d}: loading {snap_path}")
        snap = torch.load(snap_path, map_location=device, weights_only=False)

        e_list = snap["e"]
        e_tensor, seg_idx = e_list[0], e_list[1]
        if isinstance(e_tensor, torch.Tensor):
            e_tensor = e_tensor.to(device)
        if isinstance(seg_idx, torch.Tensor):
            seg_idx = seg_idx.to(device)

        block = blocks[idx]
        block.eval()

        wrapper = BlockONNXWrapper(
            block,
            seg_idx=seg_idx,
            grid_sizes=snap["grid_sizes"],
            context_lens=snap["context_lens"],
        )
        wrapper.eval()

        wrapper_inputs = (
            snap["x"].to(device),
            e_tensor,
            snap["seq_lens"].to(device),
            snap["freqs"].to(device),
            snap["context"].to(device),
        )

        print(f"[export_blocks] block {idx:2d}: exporting -> {out_path}")
        with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=dtype,
                enabled=(dtype != torch.float32)):
            torch.onnx.export(
                wrapper,
                wrapper_inputs,
                str(out_path),
                input_names=BLOCK_TENSOR_INPUT_NAMES,
                output_names=["out"],
                opset_version=args.opset,
                do_constant_folding=False,
                dynamic_axes=None,
            )
        print(f"[export_blocks] block {idx:2d}: wrote {out_path}")

        del snap, wrapper, wrapper_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
