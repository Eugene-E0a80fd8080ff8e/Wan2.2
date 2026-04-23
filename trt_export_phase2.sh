
python -m trt_conv.run_export \
  --ckpt ./Wan2.2-S2V-14B/ \
  --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
  --onnx /workspace/s2v_trt/stem.onnx