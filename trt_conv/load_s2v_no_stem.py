"""
Load WanModel_S2V without ever allocating the stem-owned modules
(`blocks` and `audio_injector`). Their weights live in the TRT engine,
so keeping a PyTorch copy just burns VRAM.

Strategy
--------
1. Use `accelerate.init_empty_weights()` to construct the model on the
   "meta" device — parameters exist as zero-memory meta tensors.
2. Stream tensors from the checkpoint's safetensors shards and
   materialize only the ones we need (skip keys starting with `blocks.`
   or `audio_injector.`).
3. Delete the meta-only `blocks` / `audio_injector` submodules so
   `_configure_model`'s `.to(device)` walk skips them entirely.

`install_no_stem_loader()` monkey-patches `WanModel_S2V.from_pretrained`
to use this loader, so the normal `wan.WanS2V(...)` init picks it up
without further changes.
"""
import glob
import logging
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open


SKIP_PREFIXES = ("blocks.", "audio_injector.")


def _should_skip(key: str) -> bool:
    return any(key.startswith(p) for p in SKIP_PREFIXES)


def load_s2v_no_stem(checkpoint_dir: str,
                     torch_dtype: torch.dtype,
                     device: str | torch.device):
    """Return a WanModel_S2V with everything except blocks/audio_injector."""
    from wan.modules.s2v.model_s2v import WanModel_S2V

    device = torch.device(device)
    ckpt = Path(checkpoint_dir)

    config, _ = WanModel_S2V.load_config(str(ckpt), return_unused_kwargs=True)
    print(f"[no_stem] building model structure on meta device")
    with init_empty_weights():
        model = WanModel_S2V.from_config(config)

    shard_files = sorted(glob.glob(str(ckpt / "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"no safetensors under {ckpt}")
    print(f"[no_stem] {len(shard_files)} shard(s); loading non-stem tensors")

    loaded = 0
    skipped = 0
    for shard in shard_files:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _should_skip(key):
                    skipped += 1
                    continue
                t = f.get_tensor(key)
                if t.is_floating_point():
                    t = t.to(torch_dtype)
                set_module_tensor_to_device(model, key, device, value=t)
                loaded += 1
    print(f"[no_stem] loaded {loaded} tensors, skipped {skipped} stem tensors")

    # Drop meta-only modules entirely so downstream `.to(device)` etc. don't
    # try to walk them.
    for attr in ("blocks", "audio_injector"):
        if hasattr(model, attr):
            print(f"[no_stem] deleting model.{attr}")
            delattr(model, attr)

    # Sever the stem's back-references (they were set via object.__setattr__
    # so they bypass normal nn.Module cleanup).
    if hasattr(model, "stem"):
        for attr in ("blocks", "audio_injector"):
            if hasattr(model.stem, attr):
                object.__setattr__(model.stem, attr, None)

    # Verify: no param should still be on meta device.
    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    meta_bufs = [n for n, b in model.named_buffers() if b.is_meta]
    if meta_params or meta_bufs:
        raise RuntimeError(
            f"still on meta device after load: params={meta_params[:5]} "
            f"bufs={meta_bufs[:5]}")

    free, total = torch.cuda.mem_get_info(device)
    print(f"[no_stem] GPU mem after load: "
          f"{(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")
    return model


def install_no_stem_loader():
    """Monkey-patch WanModel_S2V.from_pretrained to skip stem weights."""
    from wan.modules.s2v import model_s2v

    original = model_s2v.WanModel_S2V.from_pretrained

    @classmethod
    def patched(cls, pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        device_map = kwargs.get("device_map", "cuda:0")
        if isinstance(device_map, dict):
            raise NotImplementedError("dict device_map not supported here")
        return load_s2v_no_stem(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device_map,
        )

    model_s2v.WanModel_S2V.from_pretrained = patched
    logging.info("[no_stem] patched WanModel_S2V.from_pretrained")
