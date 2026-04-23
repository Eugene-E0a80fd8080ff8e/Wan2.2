"""
Export S2VBlockStem to ONNX by hooking a real inference forward.

The stem's inputs depend on the outer forward pass (pre_compute_freqs, e0, etc).
Rather than reconstruct them, we monkey-patch `model.stem.forward` so that on
the first call it:
  1. saves the inputs (for later numerical validation),
  2. runs `torch.onnx.export` with them,
  3. raises SystemExit.

Usage
-----
Set the env var STEM_EXPORT_ONNX=/path/to/stem.onnx before launching whatever
script normally drives inference (e.g. generate.py). Import and call
`install_export_hook(noise_model)` right after the model is built but before
the first denoising step. The first stem call will export and exit.

The input-capture .pt file is written next to the ONNX at
`<onnx_path>.inputs.pt` and is used by validate_stem.py.
"""
import os
from pathlib import Path

import torch


TENSOR_INPUT_NAMES = [
    "x",
    "e",
    "seg_idx",
    "seq_lens",
    "freqs",
    "context",
    "original_seq_len",
    "merged_audio_emb",
    "audio_emb_global",
]

ALL_POSITIONAL_NAMES = [
    "x",
    "e",
    "seg_idx",
    "seq_lens",
    "grid_sizes",
    "freqs",
    "context",
    "context_lens",
    "original_seq_len",
    "merged_audio_emb",
    "audio_emb_global",
]


def _bind_call_args(args, kwargs):
    bound = dict(zip(ALL_POSITIONAL_NAMES[:len(args)], args))
    bound.update(kwargs)
    missing = [n for n in ALL_POSITIONAL_NAMES if n not in bound]
    if missing:
        raise RuntimeError(f"stem call missing args: {missing}")
    return tuple(bound[n] for n in ALL_POSITIONAL_NAMES)


def _describe(name, val):
    if isinstance(val, torch.Tensor):
        return f"  {name:22s} Tensor {tuple(val.shape)} {val.dtype} dev={val.device}"
    return f"  {name:22s} {type(val).__name__}  {val!r}"


def install_export_hook(model, onnx_path=None, opset=17):
    """Patch `model.stem.forward` so the first call exports ONNX and exits.

    Parameters
    ----------
    model       : WanModel_S2V instance (must already have `.stem`)
    onnx_path   : output path for the ONNX file.  Defaults to env
                  STEM_EXPORT_ONNX or "stem.onnx" in cwd.
    opset       : ONNX opset version (default 17; SDPA needs >= 14).
    """
    onnx_path = onnx_path or os.environ.get("STEM_EXPORT_ONNX", "stem.onnx")
    onnx_path = str(Path(onnx_path).resolve())
    inputs_pt = onnx_path + ".inputs.pt"

    stem = model.stem
    original_forward = stem.forward

    def hooked(*args, **kwargs):
        positional = _bind_call_args(args, kwargs)

        print("[export_stem] captured stem inputs:")
        for n, v in zip(ALL_POSITIONAL_NAMES, positional):
            print(_describe(n, v))

        cpu_inputs = {
            n: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for n, v in zip(ALL_POSITIONAL_NAMES, positional)
        }
        torch.save(cpu_inputs, inputs_pt)
        print(f"[export_stem] saved inputs snapshot to {inputs_pt}")

        print(f"[export_stem] exporting ONNX to {onnx_path} (opset={opset})")
        stem.eval()
        with torch.inference_mode():
            torch.onnx.export(
                stem,
                positional,
                onnx_path,
                input_names=TENSOR_INPUT_NAMES,
                output_names=["out"],
                opset_version=opset,
                do_constant_folding=False,
                dynamic_axes=None,
            )
        print(f"[export_stem] wrote {onnx_path}")
        raise SystemExit(0)

    stem.forward = hooked
    print(f"[export_stem] hook installed; ONNX will be written to {onnx_path}")
