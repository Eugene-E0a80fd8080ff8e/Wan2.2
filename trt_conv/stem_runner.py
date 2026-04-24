"""
StemTRTRunner — drop-in replacement for `model.stem.forward` backed by a
TensorRT engine built from stem.onnx.

Usage
-----
    runner = StemTRTRunner("/workspace/s2v_trt/stem.trt")
    # replace the pytorch stem forward:
    model.stem.forward = runner.forward

The runner accepts the same 11 positional args the pytorch stem receives
(`x, e, seg_idx, seq_lens, grid_sizes, freqs, context, context_lens,
 original_seq_len, merged_audio_emb, audio_emb_global`) and returns a torch
bf16 tensor on the same CUDA device as the inputs. The non-tensor args
(`grid_sizes`, `context_lens`) and the .item()'d scalars (`seg_idx`,
`original_seq_len`) were baked into the engine at export time and are ignored
here — we pass only the 7 real tensor inputs the engine expects.
"""
from pathlib import Path

import tensorrt as trt
import torch


TRT_TENSOR_INPUT_NAMES = [
    "x", "e", "seq_lens", "freqs", "context",
    "merged_audio_emb", "audio_emb_global",
]


class StemTRTRunner:
    def __init__(self, engine_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine at {engine_path}")
        self.context = self.engine.create_execution_context()

        self.io_names = [self.engine.get_tensor_name(i)
                         for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in self.io_names
                            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.io_names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        missing = [n for n in TRT_TENSOR_INPUT_NAMES if n not in self.input_names]
        if missing:
            raise RuntimeError(f"engine missing inputs: {missing}; has {self.input_names}")
        print(f"[StemTRT] loaded {engine_path}")
        print(f"[StemTRT] inputs : {self.input_names}")
        print(f"[StemTRT] outputs: {self.output_names}")

    @staticmethod
    def _trt_dtype_to_torch(d):
        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }[d]

    def _bind_input(self, name, tensor):
        expected_shape = tuple(self.engine.get_tensor_shape(name))
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"shape mismatch for {name}: got {tuple(tensor.shape)}, "
                f"engine wants {expected_shape}")
        expected_dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype)
        tensor = tensor.contiguous().to(self.device)
        self.context.set_tensor_address(name, tensor.data_ptr())
        return tensor  # hold ref so storage stays alive

    def _alloc_output(self, name):
        shape = tuple(self.context.get_tensor_shape(name))
        dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        t = torch.empty(shape, dtype=dtype, device=self.device)
        self.context.set_tensor_address(name, t.data_ptr())
        return t

    def forward(self, x, e, seg_idx, seq_lens, grid_sizes, freqs, context,
                context_lens, original_seq_len, merged_audio_emb, audio_emb_global):
        feed = {
            "x": x,
            "e": e,
            "seq_lens": seq_lens,
            "freqs": freqs,
            "context": context,
            "merged_audio_emb": merged_audio_emb,
            "audio_emb_global": audio_emb_global,
        }
        held = [self._bind_input(n, feed[n]) for n in TRT_TENSOR_INPUT_NAMES]

        outs = {n: self._alloc_output(n) for n in self.output_names}

        stream = torch.cuda.current_stream(self.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        # Caller is expected to synchronize (or consume on same stream)
        del held
        if len(outs) == 1:
            return next(iter(outs.values()))
        return outs


if __name__ == "__main__":
    import argparse, gc
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--inputs", required=True, help="captured .pt from Phase 1")
    ap.add_argument("--ckpt", required=True, help="pytorch ckpt dir for comparison")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
    positional_names = [
        "x", "e", "seg_idx", "seq_lens", "grid_sizes", "freqs", "context",
        "context_lens", "original_seq_len", "merged_audio_emb", "audio_emb_global",
    ]
    positional = []
    for n in positional_names:
        v = snap[n]
        if isinstance(v, torch.Tensor):
            v = v.to(device)
        positional.append(v)

    # --- TRT ---
    runner = StemTRTRunner(args.engine, device=args.device)
    trt_out = runner.forward(*positional)
    torch.cuda.synchronize(device)
    print(f"[test] trt output: shape={tuple(trt_out.shape)} dtype={trt_out.dtype}")

    del runner
    gc.collect()
    torch.cuda.empty_cache()

    # --- PyTorch ---
    from wan.modules.s2v.model_s2v import WanModel_S2V
    model = WanModel_S2V.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map=str(device))
    for attr in ["patch_embedding", "text_embedding", "time_embedding",
                 "time_projection", "head", "casual_audio_encoder",
                 "trainable_cond_mask", "motioner", "zip_motion_out",
                 "frame_packer", "cond_encoder", "token_freqs"]:
        if hasattr(model, attr):
            delattr(model, attr)
    stem = model.stem
    stem.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16):
        pt_out = stem(*positional)

    a = trt_out.float().cpu()
    b = pt_out.float().cpu()
    diff = (a - b).abs()
    print(f"[test] shape        = {tuple(a.shape)}")
    print(f"[test] max abs diff = {diff.max().item():.6g}")
    print(f"[test] mean abs diff= {diff.mean().item():.6g}")
    print(f"[test] trt stats    : min={a.min().item():.4g} max={a.max().item():.4g}")
    print(f"[test] pt  stats    : min={b.min().item():.4g} max={b.max().item():.4g}")
