"""
Per-block input capture: hook each model.blocks[i].forward and save its
inputs on selected calls. Once every block has captured all its requested
samples, raise SystemExit.

Single-sample mode (default): captures the first call only — writes
    <save_dir>/block_<NN>.pt
This is what export_blocks.py reads for ONNX export.

Multi-sample mode (capture_indices=[0, 3, 6, 9, 12]): captures the listed
call indices for each block — writes
    <save_dir>/block_<NN>_sample_<II>.pt        (II = 0..N-1)
plus block_<NN>.pt as a copy of sample 0, so single-sample consumers
(export_blocks.py) keep working.

For 16-step diffusion without CFG, call_index == diffusion_step. With CFG,
each step does 2 calls so call_index = 2*step (cond) or 2*step+1 (uncond).
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


def install_block_capture_hooks(model, save_dir, capture_indices=(0,)):
    """Patch each block's forward to dump inputs at the requested call indices.

    Once every block has captured every requested sample, raise SystemExit.

    capture_indices: tuple/list of call indices to capture (0-based). The
    block's i-th invocation is captured if i appears in this list. Default
    (0,) preserves the single-sample behavior used by export_blocks.py.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    capture_indices = tuple(sorted(set(int(i) for i in capture_indices)))
    n_samples = len(capture_indices)
    multi = n_samples > 1
    blocks = model.blocks
    n_blocks = len(blocks)

    call_counts = [0] * n_blocks
    captured = [[False] * n_samples for _ in range(n_blocks)]

    def make_hook(idx, original):
        def hooked(*args, **kwargs):
            n = call_counts[idx]
            call_counts[idx] = n + 1
            if n in capture_indices:
                sample_idx = capture_indices.index(n)
                positional = _bind_block_args(args, kwargs)
                snap = {nm: _to_cpu(v)
                        for nm, v in zip(BLOCK_POSITIONAL_NAMES, positional)}
                if multi:
                    out_path = save_dir / f"block_{idx:02d}_sample_{sample_idx:02d}.pt"
                    torch.save(snap, out_path)
                    if sample_idx == 0:
                        # keep block_NN.pt = sample 0 so export_blocks.py works unchanged
                        torch.save(snap, save_dir / f"block_{idx:02d}.pt")
                else:
                    out_path = save_dir / f"block_{idx:02d}.pt"
                    torch.save(snap, out_path)
                captured[idx][sample_idx] = True
                print(f"[capture_blocks] block {idx:2d} call {n:2d} "
                      f"(sample {sample_idx}) -> {out_path}")
                if all(all(row) for row in captured):
                    print(f"[capture_blocks] all {n_blocks} blocks × "
                          f"{n_samples} samples captured. exiting.")
                    raise SystemExit(0)
            return original(*args, **kwargs)
        return hooked

    for i, blk in enumerate(blocks):
        blk.forward = make_hook(i, blk.forward)

    print(f"[capture_blocks] hooks installed on {n_blocks} blocks; "
          f"capture call indices {list(capture_indices)}; "
          f"snapshots will go to {save_dir}")
