
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
