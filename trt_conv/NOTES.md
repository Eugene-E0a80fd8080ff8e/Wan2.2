# S2V → TensorRT Conversion Notes

Log of everything we did to get the 40 transformer blocks of Wan2.2 S2V-14B running
on TensorRT, why each step was necessary, and where each artifact lives.

## Goal

The `noise_model` in `WanS2V` runs 40 `WanS2VAttentionBlock` modules per denoising
step; they dominate wall time. Convert that block stack to a single TRT engine,
keep the rest of the pipeline (tokenizer, VAE, pre/post-processing, scheduler) in
PyTorch. Targets: (a) lower latency, (b) eventually smaller memory via fp8/fp4.

## Architecture choice: the "stem"

Rather than exporting `WanModel_S2V` whole, we introduced a thin wrapper,
`S2VBlockStem`, that owns only the loop over `blocks` and the post-loop audio
injection. Everything outside the stem (patch/text/time embeddings, audio
encoder, head) stays in PyTorch.

Stem input/output boundary = exactly what crosses the 40-block stack. That
boundary becomes the ONNX/TRT contract.

`stem.blocks` and `stem.audio_injector` are attached via `object.__setattr__` so
they are *not* registered submodules of the stem — they keep being owned by the
top-level model. This lets us delete them later without breaking
`nn.Module.state_dict`.

## ONNX-export blockers we had to fix

These are in `wan/modules/s2v/model_s2v.py`:

1. **Python list input `e = [tensor, int]`**: ONNX inputs must be tensors. We
   split into two: `e_tensor` (fp32 modulation) and `seg_idx` (int64 scalar).
   The stem reassembles `[e, seg_idx]` before calling each block, which keeps
   the block code unchanged.

2. **`flash_attention`**: not an ONNX-registered op. Replaced with
   `F.scaled_dot_product_attention` (SDPA) both in self-attn and in the cross-attn
   (text + audio_injector cross-attns), binding the new forward via
   `types.MethodType` so pretrained weights keep working.

3. **`amp.autocast` inside blocks**: the ONNX tracer ignores autocast contexts,
   so math that relied on it would trace incorrectly. Replaced the three
   `@amp.autocast(enabled=False)` regions with explicit `.float()` casts on the
   tensors that need fp32 math (modulation, self-attn gating, FFN gating).

4. **In-place slice assignment** (`x[:, :original_seq_len] = ...`): not traceable.
   Replaced with `torch.cat([head, x[:, original_seq_len:]], dim=1)`.

5. **`rope_apply`** complex rotations were rewritten as fp32 real-pair math
   (also from the previous session; doesn't rely on `torch.view_as_complex`).

## Phase 1: capture real inputs

`trt_conv/export_stem.py`:`install_export_hook(model)` monkey-patches
`model.stem.forward` so the first real call:

1. prints the tensors it received,
2. saves them (CPU copy) to `<onnx>.inputs.pt`,
3. raises `SystemExit`.

Why: the stem's inputs depend on the outer `WanModel_S2V.forward` setup — `e0`,
`seq_lens`, `grid_sizes`, rope `freqs`, audio embeddings. Reconstructing those
by hand is fragile. Capturing them from a real forward is robust.

`generate_with_hook.py` is a copy of `generate.py` that calls
`install_export_hook` right after the noise model is built. Running it normally
until Phase 1 exits writes `/workspace/s2v_trt/stem.onnx.inputs.pt`.

## Phase 2: export to ONNX in a fresh process

`trt_conv/run_export.py`:

- Reloads the checkpoint (so no live forward-pass activations are holding VRAM).
- Trims the model to stem-owned modules only (`blocks` + `audio_injector`). All
  other attrs (`patch_embedding`, `text_embedding`, `head`, etc.) are
  `delattr`-ed.
- Loads the captured snapshot, preserving each tensor's original dtype (the
  snapshot is correct by construction from the live forward; don't re-cast).
- Sets `requires_grad_(False)` on *every* parameter the stem touches, including
  the hidden-via-`object.__setattr__` `blocks` and `audio_injector`. Required
  because ONNX tracing refuses to fold `nn.Parameter` constants that still
  require grad.
- Wraps the stem in `StemONNXWrapper` whose forward takes only the real tensor
  inputs (`x, e, seq_lens, freqs, context, merged_audio_emb, audio_emb_global`);
  the `.item()`'d scalars (`seg_idx`, `original_seq_len`) and non-tensor args
  (`grid_sizes`, `context_lens`) are baked into the wrapper and don't cross the
  ONNX boundary. This was the fix for mis-aligned `input_names` when the raw
  stem was exported directly.
- Runs `torch.onnx.export` under `torch.autocast(cuda, bf16)` so float32
  tensors (`e`, `freqs`) get demoted to bf16 at matmul boundaries matching
  runtime behavior.

Produces `stem.onnx` (~30 GB — 40 blocks × ~700 MB each).

## Phase 3: why we abandoned onnxruntime validation

Wrote `validate_stem.py` first. Hit two walls:

1. ORT's Python API accepts numpy arrays, and **numpy has no bf16**. Worked
   around with `OrtValue` + dlpack — worked, but...
2. ORT's CUDA EP has weak bf16 support. It injects Cast-to-fp32 nodes around
   unsupported ops; one materialized a 43 GB buffer that OOMed the 44 GB L40S.

ORT is not in our deployment path anyway. Deleted the file, validated via TRT
directly (see below).

## Phase 4: TRT engine build

`trtexec` ships with TensorRT and parses ONNX:

```
trtexec --onnx=/workspace/s2v_trt/stem.onnx \
        --saveEngine=/workspace/s2v_trt/stem.trt \
        --bf16 \
        --memPoolSize=workspace:8192
```

- `--bf16` lets TRT pick bf16 kernels where profitable.
- `--memPoolSize=workspace:8192` = 8 GB workspace for auto-tuner.
- Fixed shapes (we didn't pass `dynamic_axes` at export). Engine is bound to
  exactly the captured shapes — fine for S2V because every clip has the same
  frames/resolution per run.

Build time: a few minutes. Produced `/workspace/s2v_trt/stem.trt`.

## Phase 5: inference wrapper

`trt_conv/stem_runner.py`:`StemTRTRunner`

- Loads engine, creates execution context.
- `forward(*positional)` accepts the same 11 args the PyTorch stem takes; uses
  only the 7 real tensor ones (`TRT_TENSOR_INPUT_NAMES`), ignores the baked
  scalars/lists.
- Binds input tensors directly via `set_tensor_address(data_ptr)` — no H2D
  copies, tensors stay in GPU memory.
- Allocates output with `torch.empty` on the same device.
- `execute_async_v3` on the current CUDA stream.

A numerical sanity run (captured inputs, PyTorch stem vs TRT stem) showed:
- mean abs diff = 0.0175 on values in [-11, 113],
- 99.125% within 0.1, 99.999% within 1.0.

Good enough; remaining error is bf16 accumulation and SDPA-vs-flash kernel
differences. Denoising is robust to this.

## Phase 6: wiring TRT into generation

`generate_with_trt.py` = copy of `generate_with_hook.py`, but in the s2v branch
it installs a `StemTRTRunner` and does `wan_s2v.noise_model.stem.forward = trt_runner.forward`.

Run the same CLI args as `generate.py`. The 40 blocks now run on TRT; rest stays
in PyTorch.

## Phase 7: stop loading stem weights twice

Problem after phase 6: peak VRAM doubled. `WanModel_S2V.from_pretrained` still
loads the full 28 GB of block/audio_injector weights into PyTorch, and the TRT
engine has its own copy. Total working set > 60 GB, forced an upgrade from
L40S to RTX PRO 6000.

`trt_conv/load_s2v_no_stem.py`:`load_s2v_no_stem()`:

- `accelerate.init_empty_weights()` builds the model with every parameter as a
  zero-memory "meta" tensor.
- Walks each safetensors shard, skips any key starting with `blocks.` or
  `audio_injector.`, and `set_module_tensor_to_device(...)` materializes
  just the kept tensors directly on the GPU.
- Deletes the meta-only `blocks` and `audio_injector` submodules so later
  `.to(device)` walks don't touch them.
- Severs the stem's `object.__setattr__` back-references to those modules.

`install_no_stem_loader()` monkey-patches
`WanModel_S2V.from_pretrained` → `load_s2v_no_stem`, so `wan.WanS2V(...)` picks
it up transparently. `generate_with_trt.py` calls it before constructing the
S2V pipeline.

Result: PyTorch never holds a copy of the block/audio_injector weights.

## Phase 8: attempted fp8 quantization (still pending)

Motivation: fp8 weights are 4× smaller than fp32, 2× smaller than bf16, and
Blackwell has native fp8 matmul.

Tried `trtexec --fp8 --bf16` on the bf16 ONNX — no-op. Without Q/DQ nodes in
the ONNX, TRT has no quantization info and keeps bf16 weights. Engine stayed
~30 GB and speed was identical.

Real fp8 needs QDQ insertion via NVIDIA TensorRT-ModelOpt
(`trt_conv/quantize_fp8.py`). ModelOpt calibration runs the bf16 ONNX through
onnxruntime to collect activation statistics; two blockers hit so far:

1. `modelopt.onnx.quantization.quantize()` exposes only `int8` and `fp8`
   modes; `fp4` fails with "Invalid quantization mode choice". FP4 lives in
   `modelopt.onnx.quantization.int4.py` — unexplored.
2. ORT-CUDA EP failed to initialize because `libcudnn_adv*.so*` wasn't on
   `LD_LIBRARY_PATH`. Likely fix: install/locate `libcudnn9-cuda-12` libs and
   export `LD_LIBRARY_PATH`. Once ORT-CUDA runs, calibration may still OOM on
   smaller GPUs (the old bf16-Cast buffer issue); RTX PRO 6000 with 97 GB
   should be fine.

TODO for the fp8 path:
1. `find / -name libcudnn_adv* 2>/dev/null` → adjust LD_LIBRARY_PATH.
2. `python -m trt_conv.quantize_fp8 …` → produces `stem.fp8.onnx` with QDQ.
3. `trtexec --onnx=stem.fp8.onnx --stronglyTyped --bf16 --saveEngine=stem.fp8.trt`.
4. Use `generate_with_trt_fp8.py` (already written, just points at `stem.fp8.trt`).

FP4 (not yet attempted): would use `modelopt.onnx.quantization.int4` module;
NVFP4 is a group-scaled 4-bit format supported on Blackwell.

## File inventory

Under `trt_conv/`:

- `export_stem.py` — Phase 1 hook (input capture).
- `run_export.py` — Phase 2 ONNX export (uses `StemONNXWrapper`).
- `stem_runner.py` — `StemTRTRunner` and a standalone numerical smoke test.
- `load_s2v_no_stem.py` — lean loader + monkey-patch installer.
- `quantize_fp4.py` — unused (fp4 mode not in modelopt.onnx).
- `quantize_fp8.py` — fp8 QDQ quantization (pending cuDNN fix).

At project root:

- `generate_with_hook.py` — Phase 1 driver.
- `generate_with_trt.py` — bf16 TRT inference.
- `generate_with_trt_fp8.py` — fp8 TRT inference (pending engine).

Artifacts under `/workspace/s2v_trt/`:

- `stem.onnx` + external data files — must stay together. ~30 GB.
- `stem.onnx.inputs.pt` — captured Phase 1 inputs; needed to re-run Phase 2.
- `stem.trt` — working bf16 engine.
- `stem.fp8.trt` — currently bogus (no-op --fp8 build); needs Phase 8.
- `trt_build*.log` — disposable.

## Gotchas for the future

- **External data files next to `stem.onnx` must not be deleted.** Torch
  exports big tensors as sidecar files; TRT opens them lazily. If they go
  missing, errors look like `Failed to open file: _Constant_*`.
- **`stem.onnx.inputs.pt` is needed for any re-export.** Cheap to keep.
- **Resolution or frames per clip change ⇒ re-export ONNX with
  `dynamic_axes` and re-build TRT with optimization profiles.** Current setup
  is fixed-shape.
- **Stem assertion `assert e[0].dtype == torch.float32`** is intentional —
  modulation math needs fp32. If you change the stem's input dtype contract,
  update this.
- **Always sync edited files to the remote before running.** `stem_runner.py`
  etc. live in the git repo; vast.ai runs over sshfs/rsync, stale copies
  cause confusing stack traces pointing at lines that no longer exist.
