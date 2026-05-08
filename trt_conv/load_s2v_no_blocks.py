"""
Load WanModel_S2V without allocating per-block transformer weights.

Each transformer block's weights live in its own TRT engine, so keeping a
PyTorch copy of all 40 blocks (~28 GB bf16) burns VRAM that we then can't
spend on the engines themselves. The audio_injector stays in PyTorch (we
chose option 1: injector remains a normal nn.Module).

Differences vs load_s2v_no_stem.py:
  - Skips only `blocks.*` keys, NOT `audio_injector.*`.
  - Keeps `model.blocks` (with meta-tensor params) so the block_runner
    patcher can walk it and replace each .forward with a TRT runner.
"""
import glob
import logging
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open


SKIP_PREFIXES = ("blocks.",)


def _should_skip(key: str) -> bool:
    return any(key.startswith(p) for p in SKIP_PREFIXES)


def load_s2v_no_blocks(checkpoint_dir: str,
                       torch_dtype: torch.dtype,
                       device: str | torch.device):
    from wan.modules.s2v.model_s2v import WanModel_S2V

    device = torch.device(device)
    ckpt = Path(checkpoint_dir)

    config, _ = WanModel_S2V.load_config(str(ckpt), return_unused_kwargs=True)
    print(f"[no_blocks] building model structure on meta device")
    with init_empty_weights():
        model = WanModel_S2V.from_config(config)

    shard_files = sorted(glob.glob(str(ckpt / "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"no safetensors under {ckpt}")
    print(f"[no_blocks] {len(shard_files)} shard(s); loading non-block tensors")

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
    print(f"[no_blocks] loaded {loaded} tensors, skipped {skipped} block tensors")

    # Verify: only block params should still be on meta. Everything else must
    # have been materialized.
    bad = []
    for n, p in model.named_parameters():
        if p.is_meta and not n.startswith("blocks."):
            bad.append(n)
    for n, b in model.named_buffers():
        if b.is_meta and not n.startswith("blocks."):
            bad.append(n)
    if bad:
        raise RuntimeError(
            f"non-block params/buffers still on meta after load: {bad[:5]}")

    free, total = torch.cuda.mem_get_info(device)
    print(f"[no_blocks] GPU mem after load: "
          f"{(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")
    return model


def install_no_blocks_loader():
    """Monkey-patch WanModel_S2V.from_pretrained to skip block weights."""
    from wan.modules.s2v import model_s2v

    @classmethod
    def patched(cls, pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        device_map = kwargs.get("device_map", "cuda:0")
        if isinstance(device_map, dict):
            raise NotImplementedError("dict device_map not supported here")
        return load_s2v_no_blocks(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device_map,
        )

    model_s2v.WanModel_S2V.from_pretrained = patched
    logging.info("[no_blocks] patched WanModel_S2V.from_pretrained")
