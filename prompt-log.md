
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
