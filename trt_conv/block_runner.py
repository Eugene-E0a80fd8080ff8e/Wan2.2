"""
Per-block TRT runners + a patcher that swaps every model.blocks[i].forward
for the matching engine.

The audio_injector stays in PyTorch — only the transformer blocks are
delegated to TRT. Block engines accept (x, e_tensor, seq_lens, freqs, context)
and return out.
"""
from pathlib import Path

import tensorrt as trt
import torch


BLOCK_TRT_INPUTS = ["x", "e_tensor", "seq_lens", "freqs", "context"]


class BlockTRTRunner:
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

        missing = [n for n in BLOCK_TRT_INPUTS if n not in self.input_names]
        if missing:
            raise RuntimeError(
                f"engine {engine_path} missing inputs: {missing}; "
                f"has {self.input_names}")

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
        return tensor

    def _alloc_output(self, name):
        shape = tuple(self.context.get_tensor_shape(name))
        dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
        t = torch.empty(shape, dtype=dtype, device=self.device)
        self.context.set_tensor_address(name, t.data_ptr())
        return t

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        # e is [e_tensor, seg_idx]; seg_idx, grid_sizes, context_lens are baked
        e_tensor = e[0] if isinstance(e, (list, tuple)) else e
        feed = {
            "x": x,
            "e_tensor": e_tensor,
            "seq_lens": seq_lens,
            "freqs": freqs,
            "context": context,
        }
        held = [self._bind_input(n, feed[n]) for n in BLOCK_TRT_INPUTS]
        outs = {n: self._alloc_output(n) for n in self.output_names}

        stream = torch.cuda.current_stream(self.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        del held
        if len(outs) == 1:
            return next(iter(outs.values()))
        return outs


def install_block_runners(model, engines_dir, device="cuda:0"):
    """Replace each model.blocks[i].forward with a BlockTRTRunner.

    Skips any block whose engine file is missing — those keep PyTorch forward,
    so a partial conversion still produces a working model.
    """
    engines_dir = Path(engines_dir)
    n_blocks = len(model.blocks)
    runners = []
    n_replaced = 0

    for i, blk in enumerate(model.blocks):
        engine_path = engines_dir / f"block_{i:02d}.trt"
        if not engine_path.exists():
            print(f"[block_runners] block {i:2d}: no engine ({engine_path}); "
                  f"keeping PyTorch forward")
            continue
        runner = BlockTRTRunner(str(engine_path), device=device)
        runners.append(runner)
        blk.forward = runner.forward
        n_replaced += 1

    print(f"[block_runners] replaced {n_replaced}/{n_blocks} block forwards "
          f"with TRT engines")
    return runners
