

python generate_with_trt.py --task s2v-14B --size 480*480 \
	--ckpt_dir ./Wan2.2-S2V-14B/ \
	--offload_model True \
	--convert_model_dtype \
	--prompt "The girl smiles and talks to the camera."  \
	--sample_steps 16 \
	--image "face1.jpg" --audio "input2.wav"


#python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
# Without setting --num_clip, the generated video length will automatically adjust based on the input audio length
