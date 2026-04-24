
trtexec --onnx=/workspace/s2v_trt/stem.onnx \
        --saveEngine=/workspace/s2v_trt/stem.trt \
        --bf16 \
        --memPoolSize=workspace:8192 \
        --verbose 2>&1 | tee /workspace/s2v_trt/trt_build.log