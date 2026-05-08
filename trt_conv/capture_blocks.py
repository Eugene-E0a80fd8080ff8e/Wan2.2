"""
Per-block input capture: hook each model.blocks[i].forward and save its
inputs on first call. After all blocks have fired once, exit.

Usage (from a generate-style driver):
    from trt_conv.capture_blocks import install_block_capture_hooks
    install_block_capture_hooks(noise_model, save_dir="/workspace/s2v_trt/blocks")

Each block's first call writes <save_dir>/block_<NN>.pt with this dict:
    {
        "x":            Tensor (B, S, C)        bf16
        "e":            [Tensor (B, 6, C) fp32, Tensor () int]
        "seq_lens":     Tensor (B,) int64
        "grid_sizes":   list[tuple]   (baked at export time)
        "freqs":        Tensor                  fp32
        "context":      Tensor (B, 512, C)      bf16
        "context_lens": None / list             (baked at export time)
    }
"""
from pathlib import Path

import torch


BLOCK_POSITIONAL_NAMES = [
    "x", "e", "seq_lens", "grid_sizes", "freqs", "context", "context_lens",
]


def _bind_block_args(args, kwargs):
    bound = dict(zip(BLOCK_POSITIONAL_NAMES[:len(args)], args))
    bound.update(kwargs)
    missing = [n for n in BLOCK_POSITIONAL_NAMES if n not in bound]
    if missing:
        raise RuntimeError(f"block call missing args: {missing}")
    return tuple(bound[n] for n in BLOCK_POSITIONAL_NAMES)


def _to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, list):
        return [_to_cpu(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_to_cpu(x) for x in v)
    return v


def install_block_capture_hooks(model, save_dir):
    """Patch each block's forward to dump its first-call inputs to disk.

    Once every block has fired once, raise SystemExit so the driver halts
    after one denoising step instead of running the whole video.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    blocks = model.blocks
    n_blocks = len(blocks)
    captured = [False] * n_blocks

    def make_hook(idx, original):
        def hooked(*args, **kwargs):
            if not captured[idx]:
                positional = _bind_block_args(args, kwargs)
                snap = {n: _to_cpu(v)
                        for n, v in zip(BLOCK_POSITIONAL_NAMES, positional)}
                out_path = save_dir / f"block_{idx:02d}.pt"
                torch.save(snap, out_path)
                captured[idx] = True
                print(f"[capture_blocks] block {idx:2d} -> {out_path}")
                if all(captured):
                    print(f"[capture_blocks] all {n_blocks} blocks captured. exiting.")
                    raise SystemExit(0)
            return original(*args, **kwargs)
        return hooked

    for i, blk in enumerate(blocks):
        blk.forward = make_hook(i, blk.forward)

    print(f"[capture_blocks] hooks installed on {n_blocks} blocks; "
          f"snapshots will go to {save_dir}")
