
---
[2026-04-18 16:36:28]
what is this project ?

---
[2026-04-18 16:41:08]
okay.
lets work with speech2video.py

I want the embedding that come out of T5EncoderModel to be memoized on disk . So that if embedding is available, the model would not even load.

---
[2026-04-18 17:02:26]
thanks!
now please do the same for VAE input.  just in case the input is the same. 

make sure that by the time VAE decoder is needed , it is loaded. so, lazy-load for VAE

---
[2026-04-18 17:08:03]
Can we do the same for audio input ?

---
[2026-04-19 15:50:29]
<ide_selection>The user selected the lines 252 to 252 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
WanS2VAttentionBlock

This may or may not be related to the current task.</ide_selection>
I want to sinplify after_transformer_block .

I want toremove some branches of code, assuming some values of config.  to ensure that config is exactly what I assumed, each change should produce assert in a proper place.

let start from:
self.use_context_parallel is False
self.enbale_adain = True
self.adain_mode = "attn_norm"

---
[2026-04-19 16:30:17]
<ide_selection>The user selected the lines 601 to 601 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
after_transformer_block

This may or may not be related to the current task.</ide_selection>
please move this check ( if block_idx in self.audio_injector.injected_block_id.keys(): ) to be out of after_transformer_block
to inside of this cycle:
        for idx, block in enumerate(self.blocks):


also move asserts in there

---
[2026-04-19 16:32:48]
I want to estimate how pure after_transformer_block function is.

please list all the inputs that it takes
do not do any changes

---
[2026-04-19 16:49:53]
please make it pure by providing it with explicit arguments:
an instance from from self.audio_injector.injector_adain_layers (should be picked from the array at that cycle above)
an instance from from self.audio_injector.injector ( also should be picked from the array at that cycle above)
self.merged_audio_emb
self.audio_emb_global

- please review my proposition. this supposed to dissolve some complexity away from
- please confirm that self.original_seq_len is unchanged during the inference, so technically pure
- do we ever access hidden_states with other than  [:, :self.original_seq_len] ? can we just pass it in and out ?

---
[2026-04-19 16:53:14]
please apply these changes

---
[2026-04-20 14:27:15]
<ide_selection>The user selected the lines 823 to 834 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if idx in self.audio_injector.injected_block_id.keys():
                aid = self.audio_injector.injected_block_id[idx]
                x[:, :self.original_seq_len] = self.after_transformer_block(
                    x[:, :self.original_seq_len].clone(),
                    self.audio_injector.injector_adain_layers[aid],
                    self.audio_injector.injector[aid],
                    self.merged_audio_emb,
                    self.audio_emb_global,
                )


This may or may not be related to the current task.</ide_selection>
okay, great.

now this block of code:

        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if idx in self.audio_injector.injected_block_id.keys():
                aid = self.audio_injector.injected_block_id[idx]
                x[:, :self.original_seq_len] = self.after_transformer_block(
                    x[:, :self.original_seq_len].clone(),
                    self.audio_injector.injector_adain_layers[aid],
                    self.audio_injector.injector[aid],
                    self.merged_audio_emb,
                    self.audio_emb_global,
                )


It receives some input and x , processes it and outputs x. 

I want to save it to a separate neural network. Then I want to convert it to TensorRT. maybe even quantize it. 

But first, I just want to separate the one big network which I have now onto three (or four) parts:

1. audio processor
2. the part that preceeds the mentioned block
3. this block
4. the following block

I want to load them as a separate files. 
I believe I need a tool which will separate the current network onto four pieces. 
Also, this source code needs to be modified so when the network is loaded, all the respective weights is placed under its places. 

Is this even a good idea?

---
[2026-04-20 14:35:14]
<ide_selection>The user selected the lines 173 to 173 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
rope_apply

This may or may not be related to the current task.</ide_selection>
I only want to convert to trt the third block. the rest will stay as is and use the new trt block in place of current stack of blocks and audio injection things.

do I use rope_apply somewhere inside third block ?

---
[2026-04-20 14:37:27]
can you change float64 to float32? 
 is there a lot of work to convert view_as_complex to be pairs of float32 ?

---
[2026-04-20 15:19:59]
go on

---
[2026-04-20 17:33:03]
<ide_selection>The user selected the lines 168 to 168 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
rope_apply

This may or may not be related to the current task.</ide_selection>
so, any obstacles left for TensorRT ?

again. I do not want to Convert to trt anythong other than that stem of transformers from the middle.

I belieev the first step should be to savef it to separate files.  then I will convert the stem to ONNX, which will open a path to trt

---
[2026-04-20 18:03:09]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
1. what is that seg_idx ??  is it somehow related to multi-gpu training/inference? I do not have that
2. yes. remove that
3,4 . what is that e anyway ?
5. please make an estimation on seq_len. also make a print(f"{seq_len}") in a proper place.  does seq_len stays constant during entire run ?
6. hmm config.py says that s2v_14B.transformer.num_layers = 40  . not 32. please double check. am I misunderstand something ?

---
[2026-04-20 18:13:40]
> For a 480×832 @ ~20 latent frames video (common Wan config): 20 × 240 × 416 / 4 ≈ 500k
Have you account for VAE ?

---
[2026-04-20 18:16:30]
<ide_selection>The user selected the lines 819 to 820 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
print(f"[stem] x.shape={tuple(x.shape)}  original_seq_len={int(self.original_seq_len)}  seg_idx={e0[1]}")
        

This may or may not be related to the current task.</ide_selection>
okay.  for a 480x480 :
[stem] x.shape=(1, 16464, 5120)  original_seq_len=15680  seg_idx=15680

please check if your calculations are correct

---
[2026-04-20 18:20:23]
good.
now this :
1. Extract the stem as an nn.Module subclass — S2VBlockStem — owning self.blocks + self.audio_injector. In the same file, same checkpoint. Wire WanModel_S2V.forward to call it.

---
[2026-04-20 18:27:50]
<ide_selection>The user selected the lines 279 to 279 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
S2VBlockStem

This may or may not be related to the current task.</ide_selection>
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Traceback (most recent call last):
  File "/workspace/q/Wan2.2/generate.py", line 575, in <module>
    generate(args)
  File "/workspace/q/Wan2.2/generate.py", line 484, in generate
    wan_s2v = wan.WanS2V(
              ^^^^^^^^^^^
  File "/workspace/q/Wan2.2/wan/speech2video.py", line 127, in __init__
    self.noise_model = WanModel_S2V.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/diffusers/models/modeling_utils.py", line 1344, in from_pretrained
    dispatch_model(model, **device_map_kwargs)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/big_modeling.py", line 512, in dispatch_model
    model.to(device)
  File "/usr/local/lib/python3.12/dist-packages/diffusers/models/modeling_utils.py", line 1451, in to
    return super().to(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1369, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 928, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 928, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 928, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 955, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1362, in convert
    raise NotImplementedError(
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.

---
[2026-04-20 18:32:56]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
Traceback (most recent call last):
  File "/workspace/q/Wan2.2/generate.py", line 575, in <module>
    generate(args)
  File "/workspace/q/Wan2.2/generate.py", line 484, in generate
    wan_s2v = wan.WanS2V(
              ^^^^^^^^^^^
  File "/workspace/q/Wan2.2/wan/speech2video.py", line 127, in __init__
    self.noise_model = WanModel_S2V.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/diffusers/models/modeling_utils.py", line 1344, in from_pretrained
    dispatch_model(model, **device_map_kwargs)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/big_modeling.py", line 512, in dispatch_model
    model.to(device)
  File "/usr/local/lib/python3.12/dist-packages/diffusers/models/modeling_utils.py", line 1451, in to
    return super().to(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1369, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 928, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 928, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 928, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 955, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1362, in convert
    raise NotImplementedError(
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.

---
[2026-04-21 23:21:01]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
okay. what do we do next?
do not do that yet .

---
[2026-04-21 23:24:50]
> Confirm the meta-tensor fix works
I confirm. it does work just like before.

please note that the current computer is not capable of cuda, onnx and ,ulti-gigabyte models . so, I would do the testing manually. so, please do not run the code you produce

---
[2026-04-21 23:26:08]
list the blockers

---
[2026-04-21 23:29:21]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
good. go on. do 1+3

---
[2026-04-21 23:41:06]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
okay. now deal with flash attention.

---
[2026-04-21 23:47:03]
<ide_selection>The user selected the lines 98 to 98 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
bfloat

This may or may not be related to the current task.</ide_selection>
this is interesting. what is amp.autocast  ?  
what does it casts ? isn't everything is in bfloat16 by now ?

---
[2026-04-21 23:48:42]
okay. go on

---
[2026-04-22 00:07:57]
good!

are we ready to plan onnx splicing , loading and inference ?
do not do TensorRT yet.  Lets first test if stem separation worked.

are we ready to make a plan?

---
[2026-04-23 11:46:56]
go on

---
[2026-04-23 15:14:17]
go on

---
[2026-04-23 15:36:46]
...
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 857, in forward
    x = self.stem(
        ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 292, in forward
    x = block(x, **kwargs)
        ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 213, in forward
    y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 166, in forward
    x = flash_attention(
        ^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/attention.py", line 112, in flash_attention
    assert FLASH_ATTN_2_AVAILABLE
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError

---
[2026-04-23 16:02:50]
<ide_selection>The user selected the lines 98 to 98 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
bfloat

This may or may not be related to the current task.</ide_selection>
okay. it is good.

I have copied generate.py to a new file "generate_with_hook.py" and added a hook there.

The video generates normally, but I see no new files in /workspace/s2v_trt/

---
[2026-04-23 16:06:31]
could you please add (1) and (2) into files ? and I will just copy files to the remote and run there

---
[2026-04-23 16:09:09]
root@C.35464689:/Wan2.2$ ./_r3.sh 
[2026-04-23 09:08:20,545] INFO: Generation job args: Namespace(task='s2v-14B', size='480*480', frame_num=81, ckpt_dir='./Wan2.2-S2V-14B/', offload_model=True, ulysses_size=1, t5_fsdp=False, t5_cpu=False, dit_fsdp=False, save_file=None, prompt='The girl smiles and talks to the camera.', use_prompt_extend=False, prompt_extend_method='local_qwen', prompt_extend_model=None, prompt_extend_target_lang='zh', base_seed=2372618062170313350, image='face1.jpg', sample_solver='unipc', sample_steps=16, sample_shift=3, sample_guide_scale=4.5, convert_model_dtype=True, src_root_path=None, refert_num=77, replace_flag=False, use_relighting_lora=False, num_clip=None, audio='input2.wav', enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None, pose_video=None, start_from_ref=False, infer_frames=80)
[2026-04-23 09:08:20,545] INFO: Generation model config: {'__name__': 'Config: Wan S2V 14B', 't5_model': 'umt5_xxl', 't5_dtype': torch.bfloat16, 'text_len': 512, 'param_dtype': torch.bfloat16, 'num_train_timesteps': 1000, 'sample_fps': 16, 'sample_neg_prompt': '画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑
  的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', 'frame_num': 81, 't5_checkpoint': 'models_t5_umt5-xxl-enc-bf16.pth', 't5_tokenizer': 'google/umt5-xxl', 'vae_checkpoint': 'Wan2.1_VAE.pth', 'vae_stride': (4, 8, 8), 'wav2vec': 'wav2vec2-large-xlsr-53-english', 'num_heads': 40, 'transformer': {'__name__': 'Config: Transformer config for WanModel_S2V', 'patch_size': (1, 2, 2), 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'num_heads': 40, 'num_layers': 40, 'window_size': (-1, -1), 'qk_norm': True, 'cross_attn_norm': True, 'eps': 1e-06, 'enable_adain': True, 'adain_mode': 'attn_norm', 'audio_inject_layers': [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39], 'zero_init': True, 'zero_timestep': True, 'enable_motioner': False, 'add_last_motion': True, 'trainable_token': False, 'enable_tsm': False, 'enable_framepack': True, 'framepack_drop_mode': 'padd', 'audio_dim': 1024, 'motion_frames': 73, 'cond_dim': 16}, 'drop_first_motion': True, 'sample_shift': 3, 'sample_steps': 40, 'sample_guide_scale': 4.5}
[2026-04-23 09:08:20,545] INFO: Input prompt: The girl smiles and talks to the camera.
[2026-04-23 09:08:20,565] INFO: Input image: face1.jpg
[2026-04-23 09:08:20,565] INFO: Creating WanS2V pipeline.
[2026-04-23 09:08:20,565] INFO: Creating WanModel from ./Wan2.2-S2V-14B/
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:07<00:00,  1.93s/it]
[2026-04-23 09:08:28,663] INFO: Generating video ...
[2026-04-23 09:08:28,694] INFO: Audio embedding cache hit: ./Wan2.2-S2V-14B/audio_embed_cache/076b5fd7dca482623ead89949b92d7794f4acd4d51097d8f664211db6637aaac.pt
[2026-04-23 09:08:28,920] INFO: VAE encode cache hit: ./Wan2.2-S2V-14B/vae_embed_cache/enc_93c572113f5741e0091f41a9012fe691d2df09f3c0122202989c350474281d8b.pt
[2026-04-23 09:08:29,763] INFO: VAE encode cache hit: ./Wan2.2-S2V-14B/vae_embed_cache/enc_7e8d7dde18769f2eb4521d86a9d22f28a1f89fa656638418150642b725cba33f.pt
[2026-04-23 09:08:31,189] INFO: VAE encode cache hit: ./Wan2.2-S2V-14B/vae_embed_cache/enc_706b635b0ae2b519b8bf8d1326c9a1c1b0f18d0041e960df191f008cd5d6432a.pt
[2026-04-23 09:08:31,244] INFO: T5 embedding cache hit: ./Wan2.2-S2V-14B/t5_embed_cache/20dde52690cca817e53f6abae8a346dd574d349ca070ebb5acbc6d2d84935fa0.pt
[2026-04-23 09:08:31,245] INFO: T5 embedding cache hit: ./Wan2.2-S2V-14B/t5_embed_cache/514e72c797f750e49a2acf8fde9b5516cdb4b428daca8cbd5f865b5d395e6c37.pt
  0%|                                                                                                                                              | 0/16 [00:00<?, ?it/s]
[stem] x.shape=(1, 16464, 5120)  original_seq_len=15680  seg_idx=15680
[diag] S2VBlockStem.forward called, self.forward id: 139911480415616, qualname: S2VBlockStem.forward
[stem] x.shape=(1, 16464, 5120)  original_seq_len=15680  seg_idx=15680
[diag] S2VBlockStem.forward called, self.forward id: 139911476137856, qualname: S2VBlockStem.forward
  6%|████████▍                                                                                                                             | 1/16 [00:14<03:36, 14.44s/it]
[stem] x.shape=(1, 16464, 5120)  original_seq_len=15680  seg_idx=15680
[diag] S2VBlockStem.forward called, self.forward id: 139911476103296, qualname: S2VBlockStem.forward

---
[2026-04-23 16:15:55]
ah yeah. I forgot to update _r3.sh itself.
the hook is working now.

  File "/usr/local/lib/python3.12/dist-packages/torch/jit/_trace.py", line 1501, in _get_trace_graph
    outs = ONNXTracedModule(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/jit/_trace.py", line 138, in forward
    graph, _out = torch._C._create_graph_by_tracing(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/jit/_trace.py", line 129, in wrapper
    outs.append(self.inner(*trace_inputs))
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/trt_conv/export_stem.py", line 103, in hooked
    torch.onnx.export(
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/__init__.py", line 399, in export
    export(
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 522, in export
    _export(
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 1381, in _export
    assert GLOBALS.in_onnx_export is False
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

---
[2026-04-23 16:20:33]
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/trt_conv/export_stem.py", line 107, in hooked
    torch.onnx.export(
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/__init__.py", line 399, in export
    export(
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 522, in export
    _export(
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 1460, in _export
    graph, params_dict, torch_out = _model_to_graph(
                                    ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 1084, in _model_to_graph
    graph = _optimize_graph(
            ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 659, in _optimize_graph
    graph = _C._jit_pass_onnx(graph, operator_export_type)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/utils.py", line 1729, in _run_symbolic_function
    return symbolic_fn(graph_context, *inputs, **attrs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/symbolic_opset9.py", line 6555, in prim_constant
    return g.op("Constant", value_t=symbolic_helper._node_get(node, "value"))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/_internal/jit_utils.py", line 93, in op
    return _add_op(self, opname, *raw_args, outputs=outputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/_internal/jit_utils.py", line 248, in _add_op
    node = _create_node(
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/onnx/_internal/jit_utils.py", line 309, in _create_node
    _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 50.00 MiB. GPU 0 has a total capacity of 44.40 GiB of which 25.31 MiB is free. Process 612885 has 44.37 GiB memory in use. Of the allocated memory 43.42 GiB is allocated by PyTorch, and 458.52 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

---
[2026-04-23 16:24:25]
...
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 238, in forward
    y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 164, in forward
    q, k, v = qkv_fn(x)
              ^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 159, in qkv_fn
    q = self.norm_q(self.q(x)).view(b, s, n, d)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16

---
[2026-04-23 16:26:28]
> Now everything entering the tracer is float32

wait what?  float32 will double memory requirements.

can we use bfloat16 for everything?

---
[2026-04-23 16:30:08]
...
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/model.py", line 98, in forward
    return super().forward(x.float()).type_as(x)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/normalization.py", line 217, in forward
    return F.layer_norm(
           ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py", line 2910, in layer_norm
    return torch.layer_norm(
           ^^^^^^^^^^^^^^^^^
RuntimeError: mixed dtype (CPU): expect parameter to have scalar type of Float

Can we come back to doing it on GPU, but just clean up after every export ?

---
[2026-04-23 16:37:53]
[2026-04-23 09:35:41,168] INFO: Audio embedding cache hit: ./Wan2.2-S2V-14B/audio_embed_cache/076b5fd7dca482623ead89949b92d7794f4acd4d51097d8f664211db6637aaac.pt
[2026-04-23 09:35:41,502] INFO: VAE encode cache hit: ./Wan2.2-S2V-14B/vae_embed_cache/enc_93c572113f5741e0091f41a9012fe691d2df09f3c0122202989c350474281d8b.pt
[2026-04-23 09:35:42,250] INFO: VAE encode cache hit: ./Wan2.2-S2V-14B/vae_embed_cache/enc_7e8d7dde18769f2eb4521d86a9d22f28a1f89fa656638418150642b725cba33f.pt
[2026-04-23 09:35:43,582] INFO: VAE encode cache hit: ./Wan2.2-S2V-14B/vae_embed_cache/enc_706b635b0ae2b519b8bf8d1326c9a1c1b0f18d0041e960df191f008cd5d6432a.pt
[2026-04-23 09:35:43,633] INFO: T5 embedding cache hit: ./Wan2.2-S2V-14B/t5_embed_cache/20dde52690cca817e53f6abae8a346dd574d349ca070ebb5acbc6d2d84935fa0.pt
[2026-04-23 09:35:43,635] INFO: T5 embedding cache hit: ./Wan2.2-S2V-14B/t5_embed_cache/514e72c797f750e49a2acf8fde9b5516cdb4b428daca8cbd5f865b5d395e6c37.pt
  0%|                                                                                                                                              | 0/16 [00:00<?, ?it/s]
[stem] x.shape=(1, 16464, 5120)  original_seq_len=15680  seg_idx=15680
[export_stem] captured stem inputs:
  x                      Tensor (1, 16464, 5120) torch.bfloat16 dev=cuda:0
  e                      Tensor (1, 6, 2, 5120) torch.float32 dev=cuda:0
  seg_idx                Tensor () torch.int64 dev=cuda:0
  seq_lens               Tensor (1,) torch.int64 dev=cpu
  grid_sizes             list  [[tensor([[0, 0, 0]]), tensor([[20, 28, 28]]), tensor([[20, 28, 28]])], [tensor([[30,  0,  0]]), tensor([[31, 28, 28]]), tensor([[ 1, 28, 28]])]]
  freqs                  Tensor (1, 16464, 40, 64, 2) torch.float32 dev=cuda:0
  context                Tensor (1, 512, 5120) torch.bfloat16 dev=cuda:0
  context_lens           NoneType  None
  original_seq_len       Tensor () torch.int64 dev=cpu
  merged_audio_emb       Tensor (1, 20, 5, 5120) torch.float32 dev=cuda:0
  audio_emb_global       Tensor (1, 20, 1, 5120) torch.float32 dev=cuda:0
[export_stem] saved inputs snapshot to /workspace/s2v_trt/stem.onnx.inputs.pt
[export_stem] Phase 1 done. Now run:
  python -m trt_conv.run_export --ckpt <checkpoint_dir> --inputs /workspace/s2v_trt/stem.onnx.inputs.pt --onnx /workspace/s2v_trt/stem.onnx
  0%|                                                                                                                                              | 0/16 [00:02<?, ?it/s]
root@C.35464689:/Wan2.2$ ls -lah /workspace/s2v_trt/
total 490M
drwxr-xr-x 2 root root   41 Apr 23 09:15 .
drwxrwxrwx 1 root root   85 Apr 23 08:55 ..
-rw-r--r-- 1 root root 490M Apr 23 09:35 stem.onnx.inputs.pt
root@C.35464689:/Wan2.2$ 


why stem is just 490 MB ?  is this correct ?

---
[2026-04-23 16:39:48]
ah. okay.  this is phase 2:

  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 319, in forward
    x = block(x, **kwargs)
        ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1741, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 221, in forward
    assert e[0].dtype == torch.float32
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

---
[2026-04-23 16:47:57]
running phase2 :  

File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 226, in forward
    modulation = self.modulation.unsqueeze(2)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient

---
[2026-04-23 16:50:43]
File "/Wan2.2/wan/modules/s2v/model_s2v.py", line 226, in forward
    modulation = self.modulation.unsqueeze(2)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Cannot insert a Tensor that requires grad as a constant. Consider making it a parameter or input, or detaching the gradient

---
[2026-04-24 10:34:59]
well, it have finished:

[run_export] exporting to /workspace/s2v_trt/stem.onnx (opset=17)
[diag] S2VBlockStem.forward called, self.forward id: 140231571081536, qualname: S2VBlockStem.forward
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:222: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  seg_idx = e[1].item()
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:223: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  seg_idx = min(max(0, seg_idx), x.size(1))
[run_export] wrote /workspace/s2v_trt/stem.onnx
root@C.35504872:/workspace/Wan2.2$ 


Are those tracer warnings okay ?

---
[2026-04-24 10:37:42]
okay. whatever.  we have stem.onnx now:
[run_export] wrote /workspace/s2v_trt/stem.onnx

lets move on

---
[2026-04-24 10:43:05]
[validate] loading inputs from /workspace/s2v_trt/stem.onnx.inputs.pt
[validate] loading pytorch model from ./Wan2.2-S2V-14B/
[diag] S2VBlockStem.forward called, self.forward id: 139850184558464, qualname: S2VBlockStem.forward
odules/s2v/motioner.py:41: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:61: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:74: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)

Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:01<00:03,  1.22s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [00:02<00:02,  1.22s/it]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:03<00:01,  1.25s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:04<00:00,  1.12it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:04<00:00,  1.02s/it]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 135, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 129, in main
    pt_out = run_pytorch(args, snap, dtype, device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 58, in run_pytorch
    out = stem(*positional)
          ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/wan/modules/s2v/model_s2v.py", line 319, in forward
    x = block(x, **kwargs)
        ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/wan/modules/s2v/model_s2v.py", line 238, in forward
    y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/wan/modules/s2v/model_s2v.py", line 164, in forward
    q, k, v = qkv_fn(x)
              ^^^^^^^^^
  File "/workspace/Wan2.2/wan/modules/s2v/model_s2v.py", line 159, in qkv_fn
    q = self.norm_q(self.q(x)).view(b, s, n, d)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16

---
[2026-04-24 10:44:51]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
please apply

---
[2026-04-24 10:47:31]
...
ckpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:01<00:03,  1.22s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [00:02<00:02,  1.22s/it]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:03<00:01,  1.21s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.15it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.00it/s]
[0;93m2026-04-24 03:45:42.825482063 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_2'[m
[0;93m2026-04-24 03:45:42.863544554 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_7'[m
[0;93m2026-04-24 03:46:05.831482112 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_7'[m
[0;93m2026-04-24 03:46:05.831561089 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_2'[m
[0;93m2026-04-24 03:46:06.789341827 [W:onnxruntime:, transformer_memcpy.cc:111 ApplyImpl] 91 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.[m
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 136, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 131, in main
    onnx_out = run_onnx(args, snap)
               ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 91, in run_onnx
    outs = sess.run(out_names, feed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 317, in run
    self._validate_input(list(input_feed.keys()))
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 299, in _validate_input
    raise ValueError(
ValueError: Required inputs (['onnx::Split_10', 'onnx::MatMul_11', 'onnx::Unsqueeze_12', 'merged_audio_emb.1', 'tensor.5']) are missing from input feed (['x', 'e', 'seq_lens']).

---
[2026-04-24 11:02:19]
I have regenerated the phase2.

now this error:
...
[0;93m2026-04-24 04:00:42.759103068 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_412'[m
[0;93m2026-04-24 04:00:42.759110840 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_422'[m
[0;93m2026-04-24 04:00:42.759117139 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_427'[m
[0;93m2026-04-24 04:00:42.759125341 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_437'[m
[0;93m2026-04-24 04:00:42.759134785 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_447'[m
[0;93m2026-04-24 04:00:42.759143538 [W:onnxruntime:, constant_folding.cc:278 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/Sqrt_457'[m
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 136, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 131, in main
    onnx_out = run_onnx(args, snap)
               ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 91, in run_onnx
    outs = sess.run(out_names, feed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 321, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))

---
[2026-04-24 11:08:14]
...
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 136, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 131, in main
    onnx_out = run_onnx(args, snap)
               ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 91, in run_onnx
    outs = sess.run(out_names, feed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 321, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))

---
[2026-04-24 11:14:56]
<ide_selection>The user selected the lines 71 to 71 from /home/eugene/prj26/videogen1/Wan2.2/trt_conv/validate_stem.py:
dlpa

This may or may not be related to the current task.</ide_selection>
...
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 143, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 138, in main
    onnx_out = run_onnx(args, snap)
               ^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/validate_stem.py", line 92, in run_onnx
    ov = ort.OrtValue.from_dlpack(to_dlpack(t), False)
         ^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: type object 'OrtValue' has no attribute 'from_dlpack'

---
[2026-04-24 11:16:40]
root@C.35504872:/workspace/Wan2.2$ python -c "import onnxruntime; print(onnxruntime.__version__)"
1.25.0

---
[2026-04-24 11:21:15]
<ide_selection>The user selected the lines 88 to 88 from /home/eugene/prj26/videogen1/Wan2.2/trt_conv/validate_stem.py:
dlpa

This may or may not be related to the current task.</ide_selection>
RuntimeError: Error in execution: Non-zero status code returned while running Cast node. Name:'/Cast_18' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:358 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*) Failed to allocate memory for requested buffer of size 43370127360

---
[2026-04-24 11:23:10]
why do we have ort anyway ?  isnt it just to fix a single float32 convertion ?

---
[2026-04-24 11:24:36]
can't we just agree on interfaces ?
lets say we going to use bfloat16 in stem's input, output , and adjacent interfaces ?

---
[2026-04-24 11:28:39]
<ide_selection>The user selected the lines 221 to 221 from /home/eugene/prj26/videogen1/Wan2.2/wan/modules/s2v/model_s2v.py:
assert e[0].dtype == torch.float32

This may or may not be related to the current task.</ide_selection>
wait. I see line lines:

assert e[0].dtype == torch.float32
seg_idx = e[1].item()

how seg_idx could be int64 , if e is fp32 ?
is e even a tensor ?

---
[2026-04-24 11:30:36]
okay. you use ort , dlpack .

what do you use for?  for type conversion? don't we have a well-defined interface now ?

---
[2026-04-24 11:32:16]
okay. please do

---
[2026-04-24 11:37:01]
> no dynamic axes

does this affects ability to produce different resolutions ?

---
[2026-04-24 11:39:00]
okay. this is for resulotion.  how about audio length ? will it be fixed as well ?

---
[2026-04-24 11:40:53]
okay. good. lets have fixed resolution

---
[2026-04-24 11:49:55]
...
[04/24/2026-04:49:03] [I] === Performance summary ===
[04/24/2026-04:49:03] [I] Throughput: 0.366471 qps
[04/24/2026-04:49:03] [I] Latency: min = 2493.97 ms, max = 2495.86 ms, mean = 2495.25 ms, median = 2495.38 ms, percentile(90%) = 2495.78 ms, percentile(95%) = 2495.86 ms, percentile(99%) = 2495.86 ms
[04/24/2026-04:49:03] [I] Enqueue Time: min = 2396.5 ms, max = 2400.42 ms, mean = 2397.9 ms, median = 2397.75 ms, percentile(90%) = 2398.35 ms, percentile(95%) = 2400.42 ms, percentile(99%) = 2400.42 ms
[04/24/2026-04:49:03] [I] H2D Latency: min = 8.94361 ms, max = 8.96094 ms, mean = 8.95073 ms, median = 8.94971 ms, percentile(90%) = 8.95703 ms, percentile(95%) = 8.96094 ms, percentile(99%) = 8.96094 ms
[04/24/2026-04:49:03] [I] GPU Compute Time: min = 2478.99 ms, max = 2480.87 ms, mean = 2480.22 ms, median = 2480.3 ms, percentile(90%) = 2480.81 ms, percentile(95%) = 2480.87 ms, percentile(99%) = 2480.87 ms
[04/24/2026-04:49:03] [I] D2H Latency: min = 6.02734 ms, max = 6.32812 ms, mean = 6.08481 ms, median = 6.0293 ms, percentile(90%) = 6.23242 ms, percentile(95%) = 6.32812 ms, percentile(99%) = 6.32812 ms
[04/24/2026-04:49:03] [I] Total Host Walltime: 27.2873 s
[04/24/2026-04:49:03] [I] Total GPU Compute Time: 24.8022 s
[04/24/2026-04:49:03] [W] * Throughput may be bound by Enqueue Time rather than GPU Compute and the GPU may be under-utilized.
[04/24/2026-04:49:03] [W]   If not already in use, --useCudaGraph (utilize CUDA graphs where possible) may increase the throughput.
[04/24/2026-04:49:03] [I] Explanations of the performance metrics are printed in the verbose logs.
[04/24/2026-04:49:03] [V] 
[04/24/2026-04:49:03] [V] === Explanations of the performance metrics ===
[04/24/2026-04:49:03] [V] Total Host Walltime: the host walltime from when the first query (after warmups) is enqueued to when the last query is completed.
[04/24/2026-04:49:03] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.
[04/24/2026-04:49:03] [V] Total GPU Compute Time: the summation of the GPU Compute Time of all the queries. If this is significantly shorter than Total Host Walltime, the GPU may be under-utilized because of host-side overheads or data transfers.
[04/24/2026-04:49:03] [V] Throughput: the observed throughput computed by dividing the number of queries by the Total Host Walltime. If this is significantly lower than the reciprocal of GPU Compute Time, the GPU may be under-utilized because of host-side overheads or data transfers.
[04/24/2026-04:49:03] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
[04/24/2026-04:49:03] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
[04/24/2026-04:49:03] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
[04/24/2026-04:49:03] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.
[04/24/2026-04:49:03] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v101000] [b31] # trtexec --onnx=/workspace/s2v_trt/stem.onnx --saveEngine=/workspace/s2v_trt/stem.trt --bf16 --memPoolSize=workspace:8192 --verbose


okay, seems like it finished.

---
[2026-04-24 11:52:42]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/stem_runner.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
root@C.35504872:/workspace/Wan2.2$ python -m trt_conv.stem_runner \
    --engine /workspace/s2v_trt/stem.trt \
    --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
    --ckpt ./Wan2.2-S2V-14B/
[StemTRT] loaded /workspace/s2v_trt/stem.trt
[StemTRT] inputs : ['x', 'e', 'seq_lens', 'freqs', 'context', 'merged_audio_emb', 'audio_emb_global']
[StemTRT] outputs: ['out']
[04/24/2026-04:51:59] [TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.
[test] trt output: shape=(1, 16464, 5120) dtype=torch.float32
/workspace/Wan2.2/wan/modules/s2v/motioner.py:30: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
/workspace/Wan2.2/wan/modules/s2v/motioner.py:41: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:61: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:74: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.00s/it]
[diag] S2VBlockStem.forward called, self.forward id: 140376862030400, qualname: S2VBlockStem.forward
[test] shape        = (1, 16464, 5120)
[test] max abs diff = 3.98837
[test] mean abs diff= 0.0175179
[test] trt stats    : min=-11.62 max=113
[test] pt  stats    : min=-11.7 max=113
root@C.35504872:/workspace/Wan2.2$

---
[2026-04-24 11:58:44]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/stem_runner.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
> Quick additional check — what fraction of elements are "close"?

[diag] S2VBlockStem.forward called, self.forward id: 140215026020864, qualname: S2VBlockStem.forward
[test] shape        = (1, 16464, 5120)
[test] max abs diff = 3.98837
[test] mean abs diff= 0.0175179
[test] trt stats    : min=-11.62 max=113
[test] pt  stats    : min=-11.7 max=113
[test] within 0.01: 43.311%
[test] within  0.1: 99.125%
[test] within  1.0: 99.999%

---
[2026-04-24 12:00:14]
please preserve  generate_with_hook.py and instead create
 generate_with_trt.py no need for flag

---
[2026-04-24 12:07:42]
It works! Thank you!

I believe generate_with_trt.py still loads the entire S2V weights, including stem, twice. and now it takes > 60 gigs . I had to switch from L40S to RTX PRO 6000

Can we make a custom loader, which would skip stem part from the original weights ?

---
[2026-04-24 12:15:01]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/generate_with_trt.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
thanks. this works, but this still uses a lot of memory. In fact, the peak memory consumption should be the same
Can you load only the needed parts ?
If you need, please you may prepare a different file to load, by picking parts from the current

---
[2026-04-24 12:25:13]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/generate_with_trt.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
good . it works!

I also want to test fp4 trt, if possible.
lets not overwrite existing tools and files

---
[2026-04-24 12:30:05]
> trtexec --onnx=/workspace/s2v_trt/stem.onnx \
        --saveEngine=/workspace/s2v_trt/stem.fp4.trt \
        --fp4 --bf16 \
        --memPoolSize=workspace:8192 \
        --verbose 2>&1 | tee /workspace/s2v_trt/trt_build_fp4.log

...
[04/24/2026-05:29:37] [E] Unknown option: --fp4

---
[2026-04-24 12:33:59]
root@C.35504872:/workspace/Wan2.2$ python -c "import modelopt.onnx.quantization as q; print(q.__file__)"
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/__init__.py
root@C.35504872:/workspace/Wan2.2$

---
[2026-04-24 12:35:34]
/usr/bin/python: No module named trt_conv.quantize_fp4

---
[2026-04-24 12:37:32]
h yeah. thanks.

---

root@C.35504872:/workspace/Wan2.2$ python -m trt_conv.quantize_fp4     --onnx /workspace/s2v_trt/stem.onnx     --inputs /workspace/s2v_trt/stem.onnx.inputs.pt     --out /workspace/s2v_trt/stem.fp4.onnx
[quantize_fp4] x                      (1, 16464, 5120) float32
[quantize_fp4] e                      (1, 6, 2, 5120) float32
[quantize_fp4] seq_lens               (1,) int64
[quantize_fp4] freqs                  (1, 16464, 40, 64, 2) float32
[quantize_fp4] context                (1, 512, 5120) float32
[quantize_fp4] merged_audio_emb       (1, 20, 5, 5120) float32
[quantize_fp4] audio_emb_global       (1, 20, 1, 5120) float32
[quantize_fp4] quantizing /workspace/s2v_trt/stem.onnx → /workspace/s2v_trt/stem.fp4.onnx (NVFP4)
[W] colored module is not installed, will not use colors when logging. To enable colors, please install the colored module: python3 -m pip install colored
[W] Could not convert: BFLOAT16 to a corresponding NumPy type. The original ONNX type will be preserved. 
[04/24/2026-05:36:30] [TRT] [W] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).
INFO:root:Model /workspace/s2v_trt/stem.onnx with opset_version 17 is loaded.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp4.py", line 84, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp4.py", line 73, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 405, in quantize
    raise RuntimeError(f"Invalid quantization mode choice: {quantize_mode}")
RuntimeError: Invalid quantization mode choice: fp4

---
[2026-04-24 12:39:04]
root@C.35504872:/workspace/Wan2.2$ grep -n "Invalid quantization mode\|quantize_mode\s*=\|quantize_modes" /usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py | head
273:            If quantize_mode == 'fp8' and mha_accumulation_dtype == 'fp32', Cast nodes will be added to
353:    # (2) else when quantize_mode == "int8", if seq_len > 512, don't add Q/DQ layers to
355:    # (2) else when quantize_mode == "fp8", if head_size > 256 or head_size <= 8
370:        quantize_func = quantize_int8 if quantize_mode == "int8" else quantize_fp8
371:        default_calibration_method = "entropy" if quantize_mode == "int8" else "max"
405:        raise RuntimeError(f"Invalid quantization mode choice: {quantize_mode}")

---
[2026-04-24 12:40:10]
root@C.35504872:/workspace/Wan2.2$ ls /usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/
grep -rn "fp4\|nvfp4\|FP4\|NVFP4" /usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ | head -30
__init__.py  __pycache__     extensions.py  graph_utils.py  int4.py  operators.py     ort_utils.py     qdq_utils.py    quantize.py  trt_utils.py
__main__.py  calib_utils.py  fp8.py         gs_patching.py  int8.py  ort_patching.py  partitioning.py  quant_utils.py  src
grep: /usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/__pycache__/qdq_utils.cpython-312.pyc: binary file matches
grep: /usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/__pycache__/quant_utils.cpython-312.pyc: binary file matches
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:601:def replace_fp4qdq_with_2dq(
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:709:def fp4qdq_to_2dq(onnx_model: onnx.onnx_pb.ModelProto) -> onnx.onnx_pb.ModelProto:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:710:    """Convert FP32/FP16 weights of the given ONNX model to FP4 weights and scaling factors.
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:712:    TRT_FP4QDQ nodes will get removed from the weights and have two DQ nodes with those converted FP4
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:781:    print("Post-processing TRT_FP4QDQ nodes for TRT deployment...")
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:783:    fp4_qdq_nodes = [node for node in graph.node if node.op_type == "TRT_FP4QDQ"]
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:785:    for node in fp4_qdq_nodes:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/qdq_utils.py:801:        replace_fp4qdq_with_2dq(
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quant_utils.py:164:    """Converting a tensor to a quantized format based on NVFP4 quantization."""
root@C.35504872:/workspace/Wan2.2$

---
[2026-04-24 12:41:24]
ah. okay. lets try fp8 first

---
[2026-04-24 12:48:25]
...
[W] Could not convert: BFLOAT16 to a corresponding NumPy type. The original ONNX type will be preserved. 
[04/24/2026-05:45:53] [TRT] [W] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).
INFO:root:Model /workspace/s2v_trt/stem.onnx with opset_version 17 is loaded.
WARNING:root:Failed to enable ORT with CUDA EP: 'libcudnn_adv*.so* is not accessible in LD_LIBRARY_PATH! Please make sure that the path to that library is in the env var to use the CUDA or TensorRT EP and ensure that the correct version is available. Versioning compatibility can be checked at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.'
INFO:root:Successfully imported the `tensorrt` python package with version 10.10.0.31.
WARNING:root:Failed to enable ORT with TensorRT EP: 'libcudnn_adv*.so* is not accessible in LD_LIBRARY_PATH! Please make sure that the path to that library is in the env var to use the CUDA or TensorRT EP and ensure that the correct version is available. Versioning compatibility can be checked at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.'
INFO:root:Successfully enabled 1 EPs for ORT: ['CPUExecutionProvider']
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 84, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 73, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 632, in get_extended_model_outputs
    session = create_inference_session(extended_model.SerializeToString(), calibration_eps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_utils.py", line 170, in create_inference_session
    return ort.InferenceSession(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 465, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 537, in _create_inference_session
    sess.initialize_session(providers, provider_options, disabled_optimizers)
onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for MatMul(13) node with name '/MatMul'
root@C.35504872:/workspace/Wan2.2$

---
[2026-04-24 12:50:16]
> libs aren't on the LD path

should I just install something ?

---
[2026-04-24 13:09:56]
[04/24/2026-05:55:57] [E] Saving engine to file failed.
okay I am running trtexec --onnx=/workspace/s2v_trt/stem.onnx \
        --saveEngine=/workspace/s2v_trt/stem.fp8.trt \
        --fp8 --bf16 \
        --memPoolSize=workspace:8192 \
        --verbose 2>&1 | tee /workspace/s2v_trt/trt_build_fp8.log
---

[04/24/2026-05:55:57] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v101000] [b31] # trtexec --onnx=/workspace/s2v_trt/stem.onnx --saveEngine=/workspace/s2v_trt/stem.fp8.trt --fp8 --bf16 --memPoolSize=workspace:8192 --verbose
root@C.35504872:/workspace/Wan2.2$ df -h
Filesystem                         Size  Used Avail Use% Mounted on
overlay                            160G  160G  3.2M 100% /
tmpfs                               64M     0   64M   0% /dev
shm                                113G     0  113G   0% /dev/shm
/dev/nvme1n1p3                     3.5T  384G  3.2T  11% /etc/hosts
/dev/mapper/ubuntu--vg-ubuntu--lv   98G   23G   71G  25% /usr/bin/nvidia-smi
tmpfs                              4.0K  4.0K     0 100% /run/nvidia-ctk-hook3b4294e4-9099-493f-93c1-e5cde75a362c
tmpfs                              567G     0  567G   0% /proc/acpi
tmpfs                              567G     0  567G   0% /proc/scsi
tmpfs                              567G     0  567G   0% /sys/firmware
tmpfs                              567G     0  567G   0% /sys/devices/virtual/powercap
root@C.35504872:/workspace/Wan2.2$ 

---
this need a clean up.  what may I delete from /workspace/s2v_trt/ ?

---
[2026-04-24 13:30:24]
okay. all done.
I am running:
python generate_with_trt_fp8.py --task s2v-14B --size 480*480 \
	--ckpt_dir ./Wan2.2-S2V-14B/ \
	--offload_model True \
	--convert_model_dtype \
	--prompt "The girl smiles and talks to the camera."  \
	--sample_steps 20 \
	--image "face1.jpg" --audio "input2.wav"
---
but I see no speed improvement over bfloat16. it is exactly the same.

also the size of stem.fp8.trt is 30G , so it did not decreased compared to bfloat16.

---
[2026-04-24 13:32:21]
is this okay ?

root@C.35504872:/workspace/Wan2.2$ dpkg -l | grep cudnn
ii  libcudnn9-cuda-12                    9.10.1.4-1                        amd64        cuDNN runtime libraries for CUDA 12.9
ii  libcudnn9-dev-cuda-12                9.10.1.4-1                        amd64        cuDNN development libraries for CUDA 12.9
ii  libcudnn9-headers-cuda-12            9.10.1.4-1                        amd64        cuDNN header files for CUDA 12.9
root@C.35504872:/workspace/Wan2.2$ nvidia-smi
Fri Apr 24 06:31:45 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:06:00.0 Off |                    0 |
| N/A   30C    P8             34W /  600W |       0MiB /  97887MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
root@C.35504872:/workspace/Wan2.2$ 

can we go with 12.9 ?

---
[2026-04-24 13:45:12]
2026-04-24 06:44:55.109810596 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 06:44:55 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 06:44:55.110218418 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 06:44:55   ERROR] WeightsContext.cpp:178: Failed to open file: _Constant_100_attr__value
2026-04-24 06:44:55.110261992 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 06:44:55   ERROR] In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer

---
[2026-04-24 13:47:04]
may I just rm -rf /workspace/s2v_trt/
and then do

python -m trt_conv.run_export \
  --ckpt ./Wan2.2-S2V-14B/ \
  --inputs /workspace/s2v_trt/stem.onnx.inputs.pt \
  --onnx /workspace/s2v_trt/stem.onnx

?

---
[2026-04-24 13:48:48]
Can you please save all the steps we did into a file, for future reference ?
With motivation and explanation of every step

---
[2026-04-24 14:04:44]
I regeneratad everything, but:

2026-04-24 07:04:12.553423958 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:04:12 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:04:12.553822537 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:12   ERROR] WeightsContext.cpp:178: Failed to open file: _Constant_100_attr__value
2026-04-24 07:04:12.553868154 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:12   ERROR] In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer
2026-04-24 07:04:13.255296420 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:04:13 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:04:13.255700848 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:13   ERROR] WeightsContext.cpp:178: Failed to open file: _Constant_100_attr__value
2026-04-24 07:04:13.255744803 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:13   ERROR] In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer
2026-04-24 07:04:13.954166824 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:04:13 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:04:13.954576259 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:13   ERROR] WeightsContext.cpp:178: Failed to open file: _Constant_100_attr__value
2026-04-24 07:04:13.954621937 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:13   ERROR] In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer
2026-04-24 07:04:14.656949318 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:04:14 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:04:14.657354337 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:14   ERROR] WeightsContext.cpp:178: Failed to open file: _Constant_100_attr__value
2026-04-24 07:04:14.657399133 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-04-24 07:04:14   ERROR] In node -1 with name:  and operator:  (parseGraph): INVALID_GRAPH: Failed to import initialzer

---
[2026-04-24 14:08:20]
root@C.35504872:/workspace/Wan2.2$ ls /workspace/s2v_trt/ | head -20
ls /workspace/s2v_trt/ | wc -l
find / -name "_Constant_100_attr__value" 2>/dev/null
Constant_10715_attr__value
Constant_10716_attr__value
Constant_11563_attr__value
Constant_11564_attr__value
Constant_12174_attr__value
Constant_12175_attr__value
Constant_12785_attr__value
Constant_12786_attr__value
Constant_13396_attr__value
Constant_13397_attr__value
Constant_14239_attr__value
Constant_14240_attr__value
Constant_14850_attr__value
Constant_14851_attr__value
Constant_15461_attr__value
Constant_15462_attr__value
Constant_16072_attr__value
Constant_16073_attr__value
Constant_16915_attr__value
Constant_16916_attr__value
1450
/workspace/s2v_trt/_Constant_100_attr__value
root@C.35504872:/workspace/Wan2.2$ 


---

root@C.35504872:/workspace/Wan2.2$ cd /workspace/s2v_trt && python -m trt_conv.quantize_fp8 \
    --onnx stem.onnx --inputs stem.onnx.inputs.pt --out stem.fp8.onnx
/usr/bin/python: Error while finding module specification for 'trt_conv.quantize_fp8' (ModuleNotFoundError: No module named 'trt_conv')
root@C.35504872:/workspace/s2v_trt$ 

okay. how do I let python know where is trt_conv.quantize_fp8 ?

---
[2026-04-24 14:12:17]
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).
INFO:root:Model stem.onnx with opset_version 17 is loaded.
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Successfully imported the `tensorrt` python package with version 10.10.0.31.
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Successfully enabled 3 EPs for ORT: ['CPUExecutionProvider', ('CUDAExecutionProvider', {'device_id': 0}), 'TensorrtExecutionProvider']
2026-04-24 07:10:04.240766032 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:10:04 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:10:28.344447510 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 1944 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 84, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 73, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 636, in get_extended_model_outputs
    outputs = session.run(extended_model_output_names, inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))

---
[2026-04-24 14:19:14]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 115, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 104, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 636, in get_extended_model_outputs
    outputs = session.run(extended_model_output_names, inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))

---
[2026-04-24 14:19:50]
$ sed -n '600,640p' /usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py
            Path to the original onnx model.
        extended_model:
            The onnx model with some intermediate tensors marked as model outputs.
        use_external_data_format:
            If not None, this path will be used to store the weights of the quantized model.
        intermediate_generated_files:
            List of intermediate generated files that will be deleted after quantization.
        calibration_data_reader:
            Calibration data reader for running inference.
        calibration_eps:
            Priority order for the execution providers (EP) to calibrate the model.
            Any subset of ['cuda:x', 'cpu', 'trt'], where 'x' is the device id.

    Returns: a map with each output name pointed to the corresponding output numpy ndarray.
    """
    # Get the first calibration input data.
    inputs = calibration_data_reader.get_first()

    # Initialize ORT session.
    if use_external_data_format:
        extended_onnx_path = f"{onnx_path[:-5]}.extended.onnx"
        extended_model_external_data_path = f"{onnx_path[:-5]}.extended.onnx_data"
        onnx.save_model(
            extended_model,
            extended_onnx_path,
            save_as_external_data=True,
            location=os.path.basename(extended_model_external_data_path),
        )
        intermediate_generated_files.append(extended_onnx_path)
        intermediate_generated_files.append(extended_model_external_data_path)
        session = create_inference_session(extended_onnx_path, calibration_eps)
    else:
        session = create_inference_session(extended_model.SerializeToString(), calibration_eps)

    # Run extended model's inference.
    extended_model_output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(extended_model_output_names, inputs)
    output_map = {name: output for name, output in zip(extended_model_output_names, outputs)}

    return output_map

---
[2026-04-24 14:22:42]
2026-04-24 07:21:36.789792249 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:21:36 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:22:00.832613235 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 1944 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 118, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 107, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 636, in get_extended_model_outputs
    outputs = session.run(extended_model_output_names, inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))

---
[2026-04-24 14:26:43]
2026-04-24 07:24:42.379293332 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:24:42 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:25:06.359530630 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 1944 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 124, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 113, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 636, in get_extended_model_outputs
    outputs = session.run(extended_model_output_names, inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))
root@C.35504872:/workspace/s2v_trt$

---
[2026-04-24 14:31:40]
2026-04-24 07:28:32.718750871 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-04-24 07:28:32 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-04-24 07:28:56.550053432 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 1944 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 127, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8.py", line 116, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 636, in get_extended_model_outputs
    outputs = session.run(extended_model_output_names, inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(bfloat16))
---
do not do anything. explain me, what are those input, and what are those matched to ?

---
[2026-05-08 11:50:27]
Окей, I'm returning to this project after a pause. 

Could you please remind me which file is phase one? 

I have tried to run trt_export_phase2.sh and got this :

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/run_export.py", line 146, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/run_export.py", line 87, in main
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1479, in load
    with _open_file_like(f, "rb") as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 759, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 740, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/s2v_trt/stem.onnx.inputs.pt'

---
[2026-05-08 12:18:00]
root@C.36327693:/workspace/Wan2.2$ sh trt_validate_phase3.sh 
/usr/bin/python: No module named trt_conv.validate_stem
root@C.36327693:/workspace/Wan2.2$

---
[2026-05-08 12:19:17]
Would it be correct to just delete trt_validate_phase3.sh and rename trtexec1.sh to trtexec1_phase3.sh  ?

---
[2026-05-08 12:21:07]
1. please update trt_validate_phase3.sh with stem_runner

2. I have created trt_export_phase1.sh  please check it

---
[2026-05-08 12:22:32]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/generate_with_trt.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
root@C.36327693:/workspace/Wan2.2$ sh trt_validate_phase3.sh 
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/stem_runner.py", line 136, in <module>
    runner = StemTRTRunner(args.engine, device=args.device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/stem_runner.py", line 36, in __init__
    with open(engine_path, "rb") as f:
         ^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/s2v_trt/stem.trt'
root@C.36327693:/workspace/Wan2.2$ 


---

At what point do we create stem.trt?

---
[2026-05-08 12:44:30]
I can see that stem TRT is 30 gigabytes. 

This feels like bfloat16 (or float16). Is it so?

---
[2026-05-08 12:48:28]
Окей, let's make a test run with bfloat16, and then we will continue to making fp8.

How do I do a test run?

---
[2026-05-08 12:58:25]
okay it works

Now let's move on to creating FP8. Where did we stop? I remember something was not working. Or maybe I'm wrong.

---
[2026-05-08 13:06:11]
I suggest we do this conversion in a simplier way. 

lets take stem.trt and convert it to float32 in memory, it would be about 60 gigs , but it is okay, as taday I am renting RTX PRO 6000 with 96 gigs onboard.

And then, starting from that float32, we will do fp8 conversion. 

do not save float32 version to the storage , as there is only 50 gigs left now.

What do you think about this plan?

---
[2026-05-08 13:07:58]
please do quantize_fp8_via_fp32.py
(and make .sh starter script)

---
[2026-05-08 13:31:39]
root@C.36327693:/workspace/Wan2.2$ sh trt_quantize_fp8.sh 
[fp8_via_fp32] loading /workspace/s2v_trt/stem.onnx (with external data)
[fp8_via_fp32] converting bf16 -> fp32 in memory
[fp8_via_fp32] converted 1871 bf16 entities
[fp8_via_fp32] writing fp32 staging onnx to /dev/shm/stem.fp32.onnx
[fp8_via_fp32] building calibration feed from /workspace/s2v_trt/stem.onnx.inputs.pt
[fp8_via_fp32] x                      (1, 16464, 5120) float32
[fp8_via_fp32] e                      (1, 6, 2, 5120) float32
[fp8_via_fp32] seq_lens               (1,) int64
[fp8_via_fp32] freqs                  (1, 16464, 40, 64, 2) float32
[fp8_via_fp32] context                (1, 512, 5120) float32
[fp8_via_fp32] merged_audio_emb       (1, 20, 5, 5120) float32
[fp8_via_fp32] audio_emb_global       (1, 20, 1, 5120) float32
[fp8_via_fp32] quantizing /dev/shm/stem.fp32.onnx -> /workspace/s2v_trt/stem.fp8.onnx (FP8)
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).
INFO:root:Model /dev/shm/stem.fp32.onnx with opset_version 17 is loaded.
WARNING:root:Failed to enable ORT with CUDA EP: 'libcudnn_adv*.so* is not accessible in LD_LIBRARY_PATH! Please make sure that the path to that library is in the env var to use the CUDA or TensorRT EP and ensure that the correct version is available. Versioning compatibility can be checked at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.'
INFO:root:Successfully imported the `tensorrt` python package with version 10.10.0.31.
WARNING:root:Failed to enable ORT with TensorRT EP: 'libcudnn_adv*.so* is not accessible in LD_LIBRARY_PATH! Please make sure that the path to that library is in the env var to use the CUDA or TensorRT EP and ensure that the correct version is available. Versioning compatibility can be checked at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.'
INFO:root:Successfully enabled 1 EPs for ORT: ['CPUExecutionProvider']
Killed
root@C.36327693:/workspace/Wan2.2$ 

was it oom ?  where to look at ?

---
[2026-05-08 13:36:18]
root@C.36327693:/workspace/Wan2.2$ ls /workspace/s2v_trt/ | head
Constant_10715_attr__value
Constant_10716_attr__value
Constant_11563_attr__value
Constant_11564_attr__value
Constant_12174_attr__value
Constant_12175_attr__value
Constant_12785_attr__value
Constant_12786_attr__value
Constant_13396_attr__value
Constant_13397_attr__value
root@C.36327693:/workspace/Wan2.2$

do we still need all that Constant_ files , if we already have onnx ?

---
[2026-05-08 13:49:38]
...
[fp8_via_fp32] merged_audio_emb       (1, 20, 5, 5120) float32                                                                                                            
[fp8_via_fp32] audio_emb_global       (1, 20, 1, 5120) float32                                                                                                            
[fp8_via_fp32] quantizing /workspace/workdir/stem.fp32.onnx -> /workspace/s2v_trt/stem.fp8.onnx (FP8)                                                                     
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' a
nd TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so'
 TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).                                                                                    
INFO:root:Model /workspace/workdir/stem.fp32.onnx with opset_version 17 is loaded.                                                                                        
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https
://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.                                                                                      
INFO:root:Successfully imported the `tensorrt` python package with version 10.10.0.31.                                                                                    
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https
://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.                                                                                      
INFO:root:Successfully enabled 3 EPs for ORT: ['CPUExecutionProvider', ('CUDAExecutionProvider', {'device_id': 0}), 'TensorrtExecutionProvider']
*************** EP Error ***************
EP Error narrowing_error when using ['CPUExecutionProvider', ('CUDAExecutionProvider', {'device_id': 0}), 'TensorrtExecutionProvider']
Falling back to ['CUDAExecutionProvider', 'CPUExecutionProvider'] and retrying.
****************************************
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 465, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 528, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: narrowing_error

---
[2026-05-08 13:53:41]
> The standard workaround is to keep weights in external data files

Let's go the standard way. How do we do this?

---
[2026-05-08 13:59:10]
for grep : 
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/__main__.py:109:        "--use_external_data_format",
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/__main__.py:235:        use_external_data_format=args.use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/fp8.py:201:    use_external_data_format: bool = True,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/fp8.py:219:    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/fp8.py:290:        use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/fp8.py:294:    if use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:588:    use_external_data_format: bool,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:603:        use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:619:    if use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:625:            save_as_external_data=True,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:632:        session = create_inference_session(extended_model.SerializeToString(), calibration_eps)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:644:    use_external_data_format: bool = False,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:659:        use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:674:    model = onnx.load(onnx_path, load_external_data=use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:695:        use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:718:    use_external_data_format: bool = False,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:738:        use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:759:    model = onnx.load(onnx_path, load_external_data=use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py:781:            use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:421:    use_external_data_format: bool,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:453:    save_onnx(augmented_model, augmented_onnx_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:546:        if use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:902:    use_external_data_format: bool,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:954:    save_onnx(augmented_model, augmented_onnx_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:1204:        if use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:1217:    use_external_data_format: bool = True,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:1243:        use_external_data_format: If True, save tensors to external file(s) for quantized model.
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:1283:        onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:1305:            use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int4.py:1317:            use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int8.py:126:    use_external_data_format: bool = True,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int8.py:142:    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int8.py:197:            use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int8.py:242:        use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/int8.py:252:    if use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:264:    """Create an ORT InferenceSession."""
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:293:    calibrator.infer_session = ort.InferenceSession(
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:559:    use_external_data_format=False,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:574:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:589:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:603:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:618:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:645:    use_external_data_format=False,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:716:                save_as_external_data=True,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:727:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py:770:    quantizer.model.save_model_to_file(model_output, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_utils.py:167:    """Create an ORT InferenceSession."""
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_utils.py:170:    return ort.InferenceSession(
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:73:    use_external_data_format: bool,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:90:        use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:95:        save_onnx(onnx_model, onnx_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:123:        save_onnx(onnx_model, onnx_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:142:                save_onnx(onnx_model, onnx_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:169:        save_onnx(onnx_model, onnx_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:204:    use_external_data_format: bool = False,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:250:        use_external_data_format:
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:331:        use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:359:        use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:383:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:398:            use_external_data_format=use_external_data_format,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py:416:        save_onnx(onnx_model, output_path, use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/trt_utils.py:127:    use_external_data_format: bool = False,
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/trt_utils.py:136:        use_external_data_format: If True, separate data path will be used to store the weights of the quantized model.
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/trt_utils.py:149:    onnx_model = onnx.load(onnx_path, load_external_data=use_external_data_format)
/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/trt_utils.py:173:                onnx_model, onnx_path_static_shapes, save_as_external_data=use_external_data_format


---

$ python -c "import inspect; from modelopt.onnx.quantization import quantize; print(inspect.signature(quantize))"
(onnx_path: str, quantize_mode: str = 'int8', calibration_data: Union[numpy.ndarray, dict[str, numpy.ndarray]] = None, calibration_method: str = None, calibration_cache_path: str = None, calibration_shapes: str = None, calibration_eps: list[str] = ['cpu', 'cuda:0', 'trt'], op_types_to_quantize: list[str] = None, op_types_to_exclude: list[str] = None, nodes_to_quantize: list[str] = None, nodes_to_exclude: list[str] = None, use_external_data_format: bool = False, keep_intermediate_files: bool = False, output_path: str = None, verbose: bool = False, trt_plugins: str = None, trt_plugins_precision: list[str] = None, high_precision_dtype: str = None, mha_accumulation_dtype: str = 'fp16', disable_mha_qdq: bool = False, dq_only: bool = True, block_size: Optional[int] = None, use_zero_point: bool = False, passes: list[str] = None, simplify: bool = False, **kwargs: Any) -> None

---
[2026-05-08 14:21:14]
> WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).


btw, where is TensorRT's lib/   ?

---
[2026-05-08 14:50:03]
INFO:root:Successfully enabled 3 EPs for ORT: ['CPUExecutionProvider', ('CUDAExecutionProvider', {'device_id': 0}), 'TensorrtExecutionProvider']
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 181, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 162, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 630, in get_extended_model_outputs
    session = create_inference_session(extended_onnx_path, calibration_eps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_utils.py", line 170, in create_inference_session
    return ort.InferenceSession(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 465, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 526, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf: [ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from /workspace/workdir/stem.fp32.extended.onnx failed:Protobuf parsing failed.
root@C.36327693:/workspace/Wan2.2$

---
[2026-05-08 14:56:15]
root@C.36327693:/workspace/Wan2.2$ sh trt_quantize_fp8.sh 
[fp8_via_fp32] loading /workspace/s2v_trt/stem.onnx (with external data)
[fp8_via_fp32] converting bf16 -> fp32 in memory
...

What exactly do we convert bf16 -> fp32 , if stem.onnx is float32 ?

---
[2026-05-08 14:56:54]
then why stem.onnx is 63 gigs ?

---
[2026-05-08 14:57:18]
53

---
[2026-05-08 16:21:44]
...
[fp8_via_fp32] quantizing /workspace/workdir/stem.fp32.onnx -> /workspace/s2v_trt/stem.fp8.onnx (FP8)
[05/08/2026-08:59:39] [TRT] [W] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).
INFO:root:Model /workspace/workdir/stem.fp32.onnx with opset_version 17 is loaded.
INFO:root:Model is cloned to /workspace/s2v_trt/stem.fp32_named.onnx after naming the nodes.
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Successfully imported the `tensorrt` python package with version 10.10.0.31.
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Successfully enabled 3 EPs for ORT: ['CPUExecutionProvider', ('CUDAExecutionProvider', {'device_id': 0}), 'TensorrtExecutionProvider']
2026-05-08 09:03:50.954027191 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-05-08 09:03:50 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 223, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 204, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 372, in quantize
    onnx_model = quantize_func(
                 ^^^^^^^^^^^^^^
TypeError: modelopt.onnx.quantization.fp8.quantize() got multiple values for keyword argument 'calibration_data_reader'

---
By the way, I have significantly increased the size of storage available, so we don't have to try to squeeze everything into memory. Let's keep it simple.

---
[2026-05-08 16:58:31]
<ide_selection>The user selected the lines 165 to 165 from /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py:
name

This may or may not be related to the current task.</ide_selection>
again:

OSError: [Errno 28] No space left on device

---

why do I have 3 new copies of 59g onnx_data ??

$ ls -lh /workspace/s2v_trt/*.onnx_data /workspace/workdir/*.onnx.data
-rw-r--r-- 1 root root 60G May  8 09:34 /workspace/s2v_trt/stem.fp32_named.extended.onnx_data
-rw-r--r-- 1 root root 60G May  8 09:32 /workspace/s2v_trt/stem.fp32_named.onnx_data
-rw-r--r-- 1 root root 60G May  8 09:29 /workspace/workdir/stem.fp32.onnx.data

what does trt_conv.quantize_fp8_via_fp32 actually does ?

---
[2026-05-08 17:05:55]
<ide_selection>The user selected the lines 154 to 154 from /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py:
str(fp32_onnx)

This may or may not be related to the current task.</ide_selection>
modify quantize_fp8_via_fp32.py so it would not recreate str(fp32_onnx) if it is already exists

---
[2026-05-08 17:10:18]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
does it even uses gpu ?

|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:41:00.0 Off |                  Off |
| 30%   44C    P8             14W /  300W |     564MiB /  97887MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

---
[2026-05-08 17:21:06]
<ide_selection>The user selected the lines 186 to 186 from /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py:
modelopt.onnx.quantization import quantize

This may or may not be related to the current task.</ide_selection>
please make use of calibrate_per_node=True

---
[2026-05-08 17:40:36]
INFO:root:Successfully enabled 2 EPs for ORT: [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
2026-05-08 10:36:02.149003441 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 3908 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2026-05-08 10:36:23.841475984 [E:onnxruntime:, sequential_executor.cc:516 ExecuteKernel] Non-zero status code returned while running Transpose node. Name:'/Transpose_211' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 104857600

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 210, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 189, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 636, in get_extended_model_outputs
    outputs = session.run(extended_model_output_names, inputs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Transpose node. Name:'/Transpose_211' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 104857600

---
[2026-05-08 17:44:42]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 217, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 193, in main
    quantize = _q_mod.quantize
               ^^^^^^^^^^^^^^^
AttributeError: 'function' object has no attribute 'quantize'

---
[2026-05-08 17:46:10]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 217, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 193, in main
    quantize = _q_mod.quantize
               ^^^^^^^^^^^^^^^
AttributeError: 'function' object has no attribute 'quantize'

---
[2026-05-08 18:04:38]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
2026-05-08 11:03:29.902407039 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 3908 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2026-05-08 11:03:49.203406493 [E:onnxruntime:, sequential_executor.cc:516 ExecuteKernel] Non-zero status code returned while running MatMul node. Name:'/MatMul_3' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 43370127360

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 218, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_fp8_via_fp32.py", line 197, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 372, in quantize
    onnx_model = quantize_func(
                 ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/fp8.py", line 282, in quantize
    quantize_static(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py", line 730, in _quantize_static
    calibrator.collect_data(calibration_data_reader)
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py", line 405, in _collect_data_minmax_calibrator
    calibrator.intermediate_outputs.append(calibrator.infer_session.run(None, inputs))
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running MatMul node. Name:'/MatMul_3' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 43370127360

---
[2026-05-08 18:22:54]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_fp8_via_fp32.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
it failed again. 

Let's abandon this approach and switch to a different one. 

The stem consists of a number of blocks. Some of those are transformers and some of those are audio injections, right? I want to convert every such block into TensorRT engine separately. 

Let's start with trivial conversion from the source weights of bfloat16 to tensorrt bfloat16 (or float16)

1. We need a separate program which will accept the big ONNX which we have and output smaller onnx of given block . Just a single one. 
2. The sh file which would iterate over all the blocks, ignoring those which are already present. 

3. separate program which would convert a smaller ONNX block into TensorRT engine. 
4. the .sh file to which would iterate (3) over all the blocks, ignoring those which are already present. 

5. We need some changes in Wan2.2 to substitute for this new blocks

What do you think, is this a good idea?

---
[2026-05-08 18:26:42]
1 (a)

2. Just ignore how to injector for now. Keep it in pytorch

looks good

---
[2026-05-08 20:32:10]
1-3 finished cleanly

---
[2026-05-08 20:38:04]
[2026-05-08 13:36:40,988] INFO: Creating WanS2V pipeline.
[2026-05-08 13:36:40,988] INFO: Creating WanModel from ./Wan2.2-S2V-14B/
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.44s/it]
[05/08/2026-13:36:54] [TRT] [E] [defaultAllocator.cpp::allocateAsync::69] Error Code 1: Cuda Runtime (out of memory)
[05/08/2026-13:36:54] [TRT] [W] Requested amount of GPU memory (702941056 bytes) could not be allocated. There may not be enough free memory for allocation to succeed.
[05/08/2026-13:36:54] [TRT] [E] [engine.cpp::readEngineFromArchive::1138] Error Code 2: OutOfMemory (Requested size was 702941056 bytes.)
Traceback (most recent call last):
  File "/workspace/Wan2.2/generate_with_trt_blocks.py", line 28, in <module>
    runpy.run_module("generate_with_hook", run_name="__main__")
  File "<frozen runpy>", line 229, in run_module
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/generate_with_hook.py", line 582, in <module>
    generate(args)
  File "/workspace/Wan2.2/generate_with_hook.py", line 497, in generate
    install_export_hook(wan_s2v.noise_model, onnx_path="/workspace/s2v_trt/stem.onnx")
  File "/workspace/Wan2.2/generate_with_trt_blocks.py", line 18, in _install_trt_block_runners
    runners = install_block_runners(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/block_runner.py", line 113, in install_block_runners
    runner = BlockTRTRunner(str(engine_path), device=device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/block_runner.py", line 26, in __init__
    raise RuntimeError(f"failed to deserialize engine at {engine_path}")

---
[2026-05-08 20:43:49]
/workspace/Wan2.2/wan/modules/s2v/model_s2v.py:74: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  @amp.autocast(enabled=False)
[no_blocks] building model structure on meta device
[no_blocks] 4 shard(s); loading non-block tensors
[no_blocks] loaded 180 tensors, skipped 1080 block tensors
[no_blocks] GPU mem after load: 9.5/102.0 GB used
[05/08/2026-13:40:13] [TRT] [E] [defaultAllocator.cpp::allocate::53] Error Code 1: Cuda Runtime (out of memory)
[05/08/2026-13:40:13] [TRT] [W] Requested amount of GPU memory (2882185728 bytes) could not be allocated. There may not be enough free memory for allocation to succeed.
[05/08/2026-13:40:13] [TRT] [E] [executionContext.cpp::ExecutionContext::624] Error Code 2: OutOfMemory (Requested size was 2882185728 bytes.)
[05/08/2026-13:40:14] [TRT] [E] [defaultAllocator.cpp::allocate::53] Error Code 1: Cuda Runtime (out of memory)
[05/08/2026-13:40:14] [TRT] [W] Requested amount of GPU memory (2882185728 bytes) could not be allocated. There may not be enough free memory for allocation to succeed.
[05/08/2026-13:40:14] [TRT] [E] [executionContext.cpp::ExecutionContext::624] Error Code 2: OutOfMemory (Requested size was 2882185728 bytes.)
[05/08/2026-13:40:14] [TRT] [E] [defaultAllocator.cpp::allocate::53] Error Code 1: Cuda Runtime (out of memory)
[05/08/2026-13:40:14] [TRT] [W] Requested amount of GPU memory (2882185728 bytes) could not be allocated. There may not be enough free memory for allocation to succeed.
[05/08/2026-13:40:14] [TRT] [E] [executionContext.cpp::ExecutionContext::624] Error Code 2: OutOfMemory (Requested size was 2882185728 bytes.)
[05/08/2026-13:40:14] [TRT] [E] [defaultAllocator.cpp::allocateAsync::69] Error Code 1: Cuda Runtime (out of memory)
[05/08/2026-13:40:14] [TRT] [W] Requested amount of GPU memory (702941056 bytes) could not be allocated. There may not be enough free memory for allocation to succeed.
[05/08/2026-13:40:14] [TRT] [E] [engine.cpp::readEngineFromArchive::1138] Error Code 2: OutOfMemory (Requested size was 702941056 bytes.)
Traceback (most recent call last):
  File "/workspace/Wan2.2/generate_with_trt_blocks.py", line 32, in <module>
    runpy.run_module("generate_with_hook", run_name="__main__")
  File "<frozen runpy>", line 229, in run_module
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/generate_with_hook.py", line 582, in <module>
    generate(args)
  File "/workspace/Wan2.2/generate_with_hook.py", line 497, in generate
    install_export_hook(wan_s2v.noise_model, onnx_path="/workspace/s2v_trt/stem.onnx")
  File "/workspace/Wan2.2/generate_with_trt_blocks.py", line 19, in _install_trt_block_runners
    runners = install_block_runners(
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/block_runner.py", line 113, in install_block_runners
    runner = BlockTRTRunner(str(engine_path), device=device)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/trt_conv/block_runner.py", line 26, in __init__
    raise RuntimeError(f"failed to deserialize engine at {engine_path}")
RuntimeError: failed to deserialize engine at /workspace/s2v_trt/blocks/block_28.trt

-- please add temporary logging:
nvidia-smi after loading WanModel_S2V
nvidia-smi before and after loading the first trt block
nvidia-smi before and after loading the tenth trt block

---
[2026-05-08 20:48:10]
[smi] after WanModel_S2V load: 9558/97887 MiB (10% util)                                                                                                          [0/1901]
[smi] before loading first TRT block: 9558/97887 MiB (0% util)
[smi] after loading first TRT block: 10232/97887 MiB (14% util)
[smi] after loading tenth TRT block: 16280/97887 MiB (6% util)
[block_runners] max engine scratch = 2.88 GB; allocating one shared buffer
[smi] after shared scratch alloc: 39126/97887 MiB (14% util)
[smi] after all engines loaded: 39214/97887 MiB (2% util)
[block_runners] replaced 40/40 block forwards with TRT engines
[diag] stem type: S2VBlockStem
[diag] stem forward id after install: 140579015892416
[diag] stem forward qualname: S2VBlockStem.forward
Traceback (most recent call last):
  File "/workspace/Wan2.2/generate_with_trt_blocks.py", line 32, in <module>
    runpy.run_module("generate_with_hook", run_name="__main__")
  File "<frozen runpy>", line 229, in run_module
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/generate_with_hook.py", line 582, in <module>
    generate(args)
  File "/workspace/Wan2.2/generate_with_hook.py", line 503, in generate
    video = wan_s2v.generate(
            ^^^^^^^^^^^^^^^^^
  File "/workspace/Wan2.2/wan/speech2video.py", line 734, in generate
    self.noise_model.to(self.device)
  File "/usr/local/lib/python3.12/dist-packages/diffusers/models/modeling_utils.py", line 1528, in to
    return super().to(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1355, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 915, in _apply
    module._apply(fn)
  [Previous line repeated 1 more time]
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 942, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1348, in convert
    raise NotImplementedError(
NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a diff
erent device.

---
[2026-05-08 21:01:02]
good!
it works!
thank you!

now lets move on -- please make trt_blocks_phase3_fp8.sh

---
[2026-05-08 21:09:01]
$ sh trt_blocks_phase3_fp8.sh 
[phase3_fp8] block_00: quantizing to fp8 onnx
[block_fp8] loading /workspace/s2v_trt/blocks/block_00.onnx
[block_fp8] converting bf16 -> fp16 in memory
[block_fp8] converted 44 bf16 entities
[block_fp8] hoisting Constant nodes to initializers
[block_fp8] hoisted 241 Constant nodes
[block_fp8] writing fp16 staging onnx to /workspace/workdir/block_00.fp16.onnx
[block_fp8] building calibration feed from /workspace/s2v_trt/blocks/block_00.pt
[block_fp8] x            (1, 16464, 5120) float16
[block_fp8] e_tensor     (1, 6, 2, 5120) float32
[block_fp8] seq_lens     (1,) int64
[block_fp8] freqs        (1, 16464, 40, 64, 2) float32
[block_fp8] context      (1, 512, 5120) float16
[block_fp8] quantizing -> /workspace/s2v_trt/blocks/block_00.fp8.onnx (FP8)
[05/08/2026-14:07:38] [TRT] [W] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
WARNING:root:No custom ops found. If that's not correct, please make sure that the 'tensorrt' python package is correctly installed and that the paths to 'libcudnn*.so' and TensorRT 'lib/' are in 'LD_LIBRARY_PATH'. If the custom op is not directly available as a plugin in TensorRT, please also make sure that the path to the compiled '.so' TensorRT plugin is also being given via the  '--trt_plugins' flag (requires TRT 10+).
INFO:root:Model /workspace/workdir/block_00.fp16.onnx with opset_version 17 is loaded.
INFO:root:Model is cloned to /workspace/s2v_trt/blocks/block_00.fp16_named.onnx after naming the nodes.
INFO:root:Quantization Mode: fp8
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Successfully enabled 2 EPs for ORT: [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
INFO:root:Quantizable op types in the model: ['MatMul']
INFO:root:Total number of nodes: 383
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
2026-05-08 14:07:41.455872410 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 90 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2026-05-08 14:08:16.752166832 [E:onnxruntime:, sequential_executor.cc:516 ExecuteKernel] Non-zero status code returned while running Where node. Name:'/Where' Status Message: CUDA error cudaErrorIllegalAddress:an illegal memory access was encountered
terminate called after throwing an instance of 'onnxruntime::OnnxRuntimeException'
  what():  /onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:129 std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(ERRTYPE, const char*, const char*, SUCCTYPE, const char*, const char*, int) [with ERRTYPE = cudaError; bool THRW = true; SUCCTYPE = cudaError; std::conditional_t<THRW, void, common::Status> = void] /onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:121 std::conditional_t<THRW, void, onnxruntime::common::Status> onnxruntime::CudaCall(ERRTYPE, const char*, const char*, SUCCTYPE, const char*, const char*, int) [with ERRTYPE = cudaError; bool THRW = true; SUCCTYPE = cudaError; std::conditional_t<THRW, void, common::Status> = void] CUDA failure 700: an illegal memory access was encountered ; GPU=0 ; hostname=f1a1c05469bd ; file=/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_stream_handle.cc ; line=36 ; expr=cudaEventDestroy(event_); 


Aborted

---
[2026-05-08 21:14:17]
$ sh trt_blocks_phase3_fp8.sh 
[phase3_fp8] block_00.fp16_named: quantizing to fp8 onnx
[block_fp8] loading /workspace/s2v_trt/blocks/block_00.fp16_named.onnx
[block_fp8] converting bf16 -> fp32 in memory
[block_fp8] converted 0 bf16 entities
[block_fp8] hoisting Constant nodes to initializers
[block_fp8] hoisted 0 Constant nodes
[block_fp8] writing fp32 staging onnx to /workspace/workdir/block_00.fp16_named.fp32.onnx
[block_fp8] building calibration feed from /workspace/s2v_trt/blocks/block_00.fp16_named.pt
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_block_fp8.py", line 187, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_block_fp8.py", line 155, in main
    snap = torch.load(args.inputs, map_location="cpu", weights_only=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1479, in load
    with _open_file_like(f, "rb") as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 759, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 740, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/workspace/s2v_trt/blocks/block_00.fp16_named.pt'

---
[2026-05-08 21:16:47]
INFO:root:Model /workspace/workdir/block_00.fp32.onnx with opset_version 17 is loaded.
INFO:root:Model is cloned to /workspace/s2v_trt/blocks/block_00.fp32_named.onnx after naming the nodes.
INFO:root:Quantization Mode: fp8
INFO:root:libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Successfully enabled 2 EPs for ORT: [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
INFO:root:Quantizable op types in the model: ['MatMul']
INFO:root:Total number of nodes: 383
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
2026-05-08 14:15:24.646381089 [W:onnxruntime:, transformer_memcpy.cc:74 ApplyImpl] 92 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2026-05-08 14:15:28.885966672 [E:onnxruntime:, sequential_executor.cc:516 ExecuteKernel] Non-zero status code returned while running Cast node. Name:'/Cast_18' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 43370127360

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_block_fp8.py", line 187, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_block_fp8.py", line 174, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 372, in quantize
    onnx_model = quantize_func(
                 ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/fp8.py", line 282, in quantize
    quantize_static(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py", line 730, in _quantize_static
    calibrator.collect_data(calibration_data_reader)
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_patching.py", line 405, in _collect_data_minmax_calibrator
    calibrator.intermediate_outputs.append(calibrator.infer_session.run(None, inputs))
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 266, in run
    return self._sess.run(output_names, input_feed, run_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Cast node. Name:'/Cast_18' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 43370127360

---
[2026-05-08 21:22:27]
do I need to keep files like block_XX.fp32_named.onnx_data  ?? those are quite big
I can see that files block_XX.fp32_named.onnx are being deleted

---
[2026-05-08 22:31:47]
Okay, it has finished. 

How do I run fp8 inference?

---
[2026-05-08 22:56:37]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_block_fp8.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
The FP8 generation is not very good quality. 
What could be the reason? 

Should we make more calibration data? Maybe different calibration method?

---
[2026-05-08 22:59:27]
what is MHA , and why did we exluded it ?

---
[2026-05-08 23:01:54]
Well, I believe right now the scope of this problem is limited to just a single transformer, right? So we can just return it. 

If so, please return it back.

---
[2026-05-09 08:15:49]
okay. It is slightlty better.

lets do 
> Switch calibration_method to entropy
and 
> Capture 3–5 diffusion steps' inputs (e.g., steps 0, 4, 8, 12, 15 of a 16-step run), pass all as calibration data

Do we need to capture that on different prompts? Or a single one is enough?

---
[2026-05-09 12:02:07]
<ide_opened_file>The user opened the file /home/eugene/prj26/videogen1/Wan2.2/trt_conv/quantize_block_fp8.py in the IDE. This may or may not be related to the current task.</ide_opened_file>
$ sh trt_blocks_phase3_fp8.sh 
trt_blocks_phase3_fp8.sh: 29: Syntax error: "(" unexpected (expecting "done")

---
[2026-05-09 12:10:09]
2026-05-09 05:03:37.921512775 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-05-09 05:03:37 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-05-09 05:03:40.117039172 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-05-09 05:03:40 WARNING] ModelImporter.cpp:503: Make sure input seq_lens has Int64 binding.
2026-05-09 05:05:21.805255362 [W:onnxruntime:Default, tensorrt_execution_provider.h:90 log] [2026-05-09 05:05:21 WARNING] UNSUPPORTED_STATE: Skipping tactic 0 due to insufficient memory on requested size of 91155516928 detected for tactic 0x0000000000000000.
2026-05-09 05:05:21.826255565 [E:onnxruntime:Default, tensorrt_execution_provider.h:88 log] [2026-05-09 05:05:21   ERROR] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error (Could not find any implementation for node {ForeignNode[/Cast_10.../Add_31]}.)
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/Wan2.2/trt_conv/quantize_block_fp8.py", line 227, in <module>
    main()
  File "/workspace/Wan2.2/trt_conv/quantize_block_fp8.py", line 214, in main
    quantize(
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/quantize.py", line 357, in quantize
    nodes_to_exclude = find_nodes_from_mha_to_exclude(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 778, in find_nodes_from_mha_to_exclude
    output_map = get_extended_model_outputs(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/graph_utils.py", line 630, in get_extended_model_outputs
    session = create_inference_session(extended_onnx_path, calibration_eps)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/onnx/quantization/ort_utils.py", line 170, in create_inference_session
    return ort.InferenceSession(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 465, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/usr/local/lib/python3.12/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 537, in _create_inference_session
    sess.initialize_session(providers, provider_options, disabled_optimizers)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : TensorRT EP failed to create engine from network for fused node: TensorrtExecutionProvider_TRTKernel_graph_main_graph_8445025524638569544_0_0

---

btw, today I have moved to rtx 6000 ada with 48 gigs.

I believe I have enough memory for converting ~600 MB of onnx to fp8 tensorrt.
why do I run out of memory?

This is the files after the failure:
```
root@C.36379754:/workspace/Wan2.2$ ls -lh /workspace/s2v_trt/blocks/block_00*
-rw-r--r-- 1 root root  51K May  9 05:03 /workspace/s2v_trt/blocks/block_00.fp32_named.extended.onnx
-rw-r--r-- 1 root root 2.7G May  9 05:03 /workspace/s2v_trt/blocks/block_00.fp32_named.extended.onnx_data
-rw-r--r-- 1 root root  50K May  9 05:03 /workspace/s2v_trt/blocks/block_00.fp32_named.onnx
-rw-r--r-- 1 root root 2.7G May  9 05:03 /workspace/s2v_trt/blocks/block_00.fp32_named.onnx_data
-rw-r--r-- 1 root root 671M May  9 04:21 /workspace/s2v_trt/blocks/block_00.onnx
-rw-r--r-- 1 root root 488M May  9 04:18 /workspace/s2v_trt/blocks/block_00.pt
root@C.36379754:/workspace/Wan2.2$ 
```

---
[2026-05-09 12:15:30]
Can I just exclude MHA manually? iThere is just a simple transformer inside, with a residual connection, right?
