"""
Per-block TRT runners + a patcher that swaps every model.blocks[i].forward
for the matching engine.

The audio_injector stays in PyTorch — only the transformer blocks are
delegated to TRT. Block engines accept (x, e_tensor, seq_lens, freqs, context)
and return out.

Memory: each block engine needs ~3 GB of "device memory" (per-context
activation scratch). With 40 blocks that's >100 GB, blowing the GPU. We
allocate ONE buffer sized to max(engine.device_memory_size) and have every
context point at it — sequential execution means there's no aliasing risk.
"""
import subprocess
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn as nn


BLOCK_TRT_INPUTS = ["x", "e_tensor", "seq_lens", "freqs", "context"]


class _TRTBlockStub(nn.Module):
    """Stand-in nn.Module with no parameters that delegates to a TRT runner.

    Replaces meta-tensor PyTorch blocks so .to(device) walks find nothing to
    move on this block.
    """

    def __init__(self, runner):
        super().__init__()
        object.__setattr__(self, "_runner", runner)

    def forward(self, *args, **kwargs):
        return self._runner.forward(*args, **kwargs)


def _smi(tag):
    """Print a one-line nvidia-smi summary (mem used / total, util)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        used, total, util = [s.strip() for s in out.split(",")]
        print(f"[smi] {tag}: {used}/{total} MiB ({util}% util)")
    except Exception as e:
        print(f"[smi] {tag}: nvidia-smi failed: {e}")


class BlockTRTRunner:
    def __init__(self, engine_path: str, device: str = "cuda:0",
                 shared_device_memory_ptr: int = 0):
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine at {engine_path}")
        if shared_device_memory_ptr:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = shared_device_memory_ptr
        else:
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


def install_block_runners(model, engines_dir, device="cuda:0",
                          engine_suffix=".trt"):
    """Replace each model.blocks[i].forward with a BlockTRTRunner.

    Skips any block whose engine file is missing — those keep PyTorch forward,
    so a partial conversion still produces a working model.

    All runners share one device-memory buffer to avoid OOM; sized to
    max(engine.device_memory_size_v2). Sequential execution makes sharing safe.
    """
    engines_dir = Path(engines_dir)
    n_blocks = len(model.blocks)
    dev = torch.device(device)

    _smi("after WanModel_S2V load")

    # Pass 1: deserialize engines (without contexts) and find the max scratch
    # size needed. Engines themselves still take VRAM (~700 MB each).
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engines = {}
    max_scratch = 0
    for i, blk in enumerate(model.blocks):
        engine_path = engines_dir / f"block_{i:02d}{engine_suffix}"
        if not engine_path.exists():
            continue
        if i == 0:
            _smi("before loading first TRT block")
        with open(engine_path, "rb") as f:
            eng = runtime.deserialize_cuda_engine(f.read())
        if eng is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        engines[i] = eng
        max_scratch = max(max_scratch, eng.device_memory_size_v2)
        if i == 0:
            _smi("after loading first TRT block")
        if i == 9:
            _smi("after loading tenth TRT block")

    print(f"[block_runners] max engine scratch = {max_scratch / 1e9:.2f} GB; "
          f"allocating one shared buffer")
    shared = torch.empty(max_scratch, dtype=torch.uint8, device=dev)
    shared_ptr = shared.data_ptr()
    _smi("after shared scratch alloc")

    # Pass 2: build runners that all reuse the shared buffer, and replace each
    # PyTorch block with a parameter-free stub so `.to(device)` ignores them.
    runners = []
    n_replaced = 0
    for i in range(n_blocks):
        if i not in engines:
            print(f"[block_runners] block {i:2d}: no engine; keeping PyTorch block")
            continue
        runner = BlockTRTRunner.__new__(BlockTRTRunner)
        runner.device = dev
        runner.engine = engines[i]
        runner.context = engines[i].create_execution_context_without_device_memory()
        runner.context.device_memory = shared_ptr
        runner.io_names = [runner.engine.get_tensor_name(j)
                           for j in range(runner.engine.num_io_tensors)]
        runner.input_names = [n for n in runner.io_names
                              if runner.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        runner.output_names = [n for n in runner.io_names
                               if runner.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        missing = [n for n in BLOCK_TRT_INPUTS if n not in runner.input_names]
        if missing:
            raise RuntimeError(
                f"engine block_{i:02d} missing inputs: {missing}; "
                f"has {runner.input_names}")
        runners.append(runner)
        model.blocks[i] = _TRTBlockStub(runner)
        n_replaced += 1

    # Hold the shared buffer alive on the first runner so it doesn't get GC'd.
    if runners:
        runners[0]._shared_scratch = shared

    _smi("after all engines loaded")
    print(f"[block_runners] replaced {n_replaced}/{n_blocks} block forwards "
          f"with TRT engines")
    return runners
