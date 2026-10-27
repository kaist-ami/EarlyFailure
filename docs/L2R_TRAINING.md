# Train and fuse an L2R checkpoint

This guide reproduces the L2R checkpoint lifecycle used by the inference
pipeline:

```text
owned videos
    -> paired RGB/latent manifest
    -> initialized dw basis + expansion weights
    -> factorized rank-basis training
    -> basis-axis contraction
    -> FusedTemporalUpsample replacement
    -> plain 29-key deploy checkpoint
    -> strict=True inference load
```

You do **not** need any step in this guide to run the main pipeline. The two
official fused L2R checkpoints are already included in the package. Start with
the [main README](../README.md#setup) unless you intend to
train a new L2R model.

## Before you begin

The research training dataset and its prepared manifest are **not distributed**.
`early-failure-train-l2r` always requires a real paired manifest supplied with
`--manifest`; there is no hidden or automatically downloaded training set.

The repository contains two procedurally generated videos in
`examples/l2r_training/videos`. They are only smoke-test inputs for the data
preparation command. Two synthetic clips are not sufficient to train a useful
checkpoint.

Install the training and main-pipeline dependencies from the repository root:

```bash
pip install -e ".[pipeline,training]"
```

Use only `.[training]` if you will stop after producing the deploy checkpoint
and do not intend to run step 5 in the same environment.

Data preparation and practical training require a CUDA GPU. Fusion and strict
checkpoint validation can also run on CPU.

## Dataset specification

For the paper-compatible setup, prepare one continuous 49-frame shot per
sample. Frame rate is **not** a training constraint: samples may have different
FPS values because the loader aligns tensors by decoded frame index and never
uses timestamps.

### Source clips accepted by the preparation command

| Property | Requirement |
| --- | --- |
| Frames | At least 49 decoded frames. The centered, contiguous 49 frames are used. |
| FPS | No fixed or uniform FPS is required. Variable-FPS files are also consumed as a decoded frame sequence. |
| Resolution | Any decodable resolution is accepted and center-cropped/resized to `720 x 480` (`W x H`). Use at least `720 x 480` when possible to avoid upsampling; this was also the minimum-resolution filter in the research preprocessing. |
| Content | A single continuous shot is recommended. Pre-cut clips at scene boundaries; this public utility does not run scene detection. |
| Format | RGB-decodable `.mp4`, `.mov`, `.avi`, `.webm`, or `.mkv`. |

The research preprocessing retained each clip's source FPS, so its dataset did
not use one uniform value. The public helper writes the selected frames using
`--fps` (default: 8). This option only sets playback metadata on the prepared
`target.mp4`; it does not resample frames and is not read during training. For
example, the same 49-frame tensor could occupy 6.125 seconds at 8 FPS or about
1.63 seconds at 30 FPS without changing the L2R pair. Consequently, there is no
required clip duration in seconds.

### Prepared pair consumed by training

| Property | CogVideoX-5B | Wan2.1-1.3B |
| --- | --- | --- |
| RGB target | 49 frames, `720 x 480` (`W x H`), H.264 CRF 18 | Same |
| Stored RGB shape | `[49, 480, 720, 3]` (`T,H,W,C`, decoded uint8) | Same |
| Stored latent shape | `[16, 13, 60, 90]` (`C,T,H,W`, float32) | Same |
| Spatial compression | 8x per axis | Same |
| Temporal compression | 49 RGB frames to 13 latent frames | Same |
| RGB supervision indices | `[0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]` | Same |
| Latent convention | VAE latent multiplied by `scaling_factor` | `(latent - latents_mean) / latents_std` |

Thus the model receives latent batches shaped `[B,16,13,60,90]` and RGB
targets shaped `[B,3,13,480,720]`, normalized to `[-1,1]`. The temporal RGB
indices above are the alignment used by the released L2R training code for
**both** backends. Do not substitute `[0,1,5,...,45]`; those are the temporal
group starts before the final `+2` center-frame offset is applied.

The two included synthetic clips already satisfy the input specification, but
they exist only to verify the command chain and are not a training dataset.

## 1. Prepare paired data

Use videos that you own or are permitted to process. The preparation command
selects the centered 49-frame window, center-crops/resizes it to 720 x 480,
writes an H.264 CRF-18 target, decodes that target again, and encodes those
exact decoded frames with the selected Hugging Face VAE. This keeps the RGB
target and latent paired after video compression.

For a directory of real training clips:

```bash
# CogVideoX-5B latents
early-failure-prepare-l2r \
  --backend cogvideox \
  --input-dir /path/to/your/videos \
  --output-dir data/l2r_cogvideox

# Wan2.1-1.3B latents
early-failure-prepare-l2r \
  --backend wan \
  --input-dir /path/to/your/videos \
  --output-dir data/l2r_wan
```

The command downloads only the matching VAE subfolder on first use:

- `zai-org/CogVideoX-5b`
- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`

The built-in IDs use the same pinned revisions documented in the main README.
Use `--model-revision` to intentionally select another remote commit, or use
`--model-id /local/model/path --local-files-only` on an offline machine. Local
model paths and source-video directories are not written to sample metadata;
metadata retains only `local` and the source filename. Pass `--video` more than
once when you do not want to scan a directory.

### Smoke-test the preparation command

This verifies decoding, VAE encoding, and manifest creation. It does not produce
meaningful training data.

```bash
early-failure-prepare-l2r \
  --backend cogvideox \
  --video examples/l2r_training/videos/moving_shapes.mp4 \
  --video examples/l2r_training/videos/color_waves.mp4 \
  --output-dir data/l2r_smoke
```

The output is self-contained:

```text
data/l2r_smoke/
├── manifest.jsonl
├── 0000_moving_shapes/
│   ├── target.mp4
│   ├── latent.safetensors
│   └── metadata.json
└── 0001_color_waves/
    ├── target.mp4
    ├── latent.safetensors
    └── metadata.json
```

Optionally exercise the remaining commands with a tiny rank and one optimizer
step:

```bash
early-failure-init-l2r \
  --backend cogvideox \
  --basis-rank 8 \
  --output runs/l2r_smoke/factorized_init.pt

early-failure-train-l2r \
  --init-checkpoint runs/l2r_smoke/factorized_init.pt \
  --manifest data/l2r_smoke/manifest.jsonl \
  --output-dir runs/l2r_smoke \
  --steps 1 \
  --batch-size 1 \
  --num-workers 0

early-failure-fuse-l2r \
  --checkpoint runs/l2r_smoke/factorized_final.pt \
  --output runs/l2r_smoke/deploy_smoke.pt
```

This is a command-chain test only. `deploy_smoke.pt` is not a usable L2R model
and should not be passed to the main generation pipeline.

CogVideoX latents are multiplied by the VAE `scaling_factor`. Wan latents are
normalized with the VAE `latents_mean` and `latents_std`. Both are stored as
float32 `[C, T, H, W]` tensors with 16 channels.

## Using an existing manifest

If you already have compatible paired data, you may skip the preparation
**utility** and start at initialization. This does not remove the data
requirement: the repository itself does not include such a manifest.

The manifest is JSON Lines with one object per sample. Relative paths are
resolved from the manifest's directory:

```json
{"backend":"cogvideox","latent":"0000_example/latent.safetensors","latent_key":"latent","latent_layout":"cthw","target":"0000_example/target.mp4","target_layout":"thwc","frame_indices":[0,3,7,11,15,19,23,27,31,35,39,43,47]}
```

Required fields:

- `latent`: `.safetensors`, `.npy`, `.npz`, `.pt`, or `.pth` tensor.
- `target`: video, image directory, or one of the tensor formats above.

Optional fields:

- `backend` is `cogvideox` or `wan`; when present, training checks it against
  the initialization checkpoint.
- `latent_key` and `target_key` select a tensor from container files.
- `latent_layout` and `target_layout` accept `cthw`, `tchw`, or `thwc`.
- `frame_indices` maps each latent time index to an RGB frame.

The loader expects 16 latent channels, 3 RGB channels, equal selected temporal
lengths, and an RGB spatial size exactly 8 times the latent size. RGB values may
be uint8, `[0, 1]`, or `[-1, 1]`; they are normalized to `[-1, 1]`. When a
49-frame target is paired with 13 latent frames, the training indices shown in
the dataset table are used automatically if `frame_indices` is omitted.

## 2. Initialize the factorized basis

Initialization is deliberately separate from training so the exact starting
checkpoint can be archived or reused:

```bash
early-failure-init-l2r \
  --backend cogvideox \
  --basis-rank 4096 \
  --seed 1234 \
  --output runs/l2r_cogvideox/factorized_init.pt
```

Use `--backend wan` for a Wan manifest. The paper configuration uses rank 4096,
tanh-parameterized weights, and initialization standard deviation 0.02; these
are the defaults.

Each of the three x2 upsampling stages starts with:

- `dw_weight`: per-channel temporal/spatial basis kernels;
- `expand_weight`: coefficients from the basis to the four subpixel slots.

Together the stages produce the model's x8 spatial upsampling. The factorized
model state has 32 keys. It is a training checkpoint, not an inference
checkpoint.

## 3. Train the factorized model

```bash
early-failure-train-l2r \
  --init-checkpoint runs/l2r_cogvideox/factorized_init.pt \
  --manifest data/l2r_cogvideox/manifest.jsonl \
  --output-dir runs/l2r_cogvideox \
  --batch-size 2 \
  --steps 200000
```

The defaults match the base training setup: pixel L1 loss, AdamW with learning
rate `5e-4` and weight decay `5e-5`, 5% warmup followed by cosine decay, and
gradient clipping at 1.0. `--steps` is the target global step.

Periodic checkpoints contain the model, optimizer, scheduler, backend, and
global step. Resume without resetting optimizer state:

```bash
early-failure-train-l2r \
  --resume runs/l2r_cogvideox/factorized_step_0040000.pt \
  --manifest data/l2r_cogvideox/manifest.jsonl \
  --output-dir runs/l2r_cogvideox \
  --steps 200000
```

The final factorized checkpoint is
`runs/l2r_cogvideox/factorized_final.pt`.

For a separately prepared diffusion-`x0`/RGB manifest, a trained factorized
checkpoint can be used as a new initialization. The paper's fine-tuning loss
adds VGG LPIPS at weight 0.1:

```bash
early-failure-train-l2r \
  --init-checkpoint runs/l2r_cogvideox/factorized_final.pt \
  --manifest /path/to/x0_pairs/manifest.jsonl \
  --output-dir runs/l2r_cogvideox_x0 \
  --steps 20000 \
  --learning-rate 5e-5 \
  --weight-decay 1e-6 \
  --perceptual-weight 0.1
```

The preparation command in step 1 makes VAE latent/RGB pairs; it does not make
diffusion-`x0` pairs.

## 4. Fuse and validate the deploy checkpoint

```bash
early-failure-fuse-l2r \
  --checkpoint runs/l2r_cogvideox/factorized_final.pt \
  --output runs/l2r_cogvideox/deploy_fused.pt
```

Fusion contracts the basis dimension in each `dw_weight`/`expand_weight` pair,
replaces all three factorized modules with `FusedTemporalUpsample`, and writes a
plain float32 state dictionary. Before saving, the command:

1. compares factorized and fused outputs on a deterministic random latent;
2. requires exactly 29 state-dict keys;
3. loads the result into `LatentToRGB` with `strict=True`;
4. reloads the saved file with the inference checkpoint loader;
5. prints the output SHA-256 hash.

Do not pass `--skip-verification` for a release artifact. `--device cuda` makes
the equivalence check faster, while the default CPU path is portable.

## 5. Use the custom L2R checkpoint

Only this final step belongs to the main early-failure pipeline:

```bash
early-failure-video \
  --backend cogvideox \
  --l2r-checkpoint runs/l2r_cogvideox/deploy_fused.pt \
  --prompt "A red panda carefully folds a paper crane on a wooden desk." \
  --output outputs/custom_l2r.mp4
```

Keep the backend aligned across data preparation, training, and inference. A
CogVideoX-trained L2R checkpoint should not be used to inspect Wan latents, or
vice versa.

## Command reference

Every stage is independently inspectable:

```bash
early-failure-prepare-l2r --help
early-failure-init-l2r --help
early-failure-train-l2r --help
early-failure-fuse-l2r --help
```
