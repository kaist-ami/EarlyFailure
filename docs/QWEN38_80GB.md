# Local Qwen3.8-27B refinement on one 80 GB GPU

This is the local, API-free alternative to the default OpenAI prompt refiner.
It uses [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), a
27B multimodal model that can inspect the sampled frames from a failed early
preview.

Qwen is **not** kept resident beside the video model. It is loaded only when
the intervention hierarchy actually reaches prompt refinement:

```text
initial video is predicted to fail
        -> one-frame anchor is not good enough
        -> park Diffusers + L2R + ViCLIP on CPU
        -> launch Qwen3.8-27B worker on the GPU
        -> inspect preview frames and return revised-prompt JSON
        -> worker exits (all Qwen CUDA memory is released)
        -> restore L2R + ViCLIP and continue video diffusion
```

If the initial video succeeds, or the first frame supplies a useful anchor,
Qwen is never started.

## Why Qwen has a separate environment

The released video pipeline pins Transformers 4.57.6 for its Diffusers stack.
Qwen3.8-27B's model configuration targets Transformers 5.8.1. A small
worker environment avoids changing the tested video stack. The worker is a
subprocess rather than a persistent server, so process exit is also a reliable
GPU-memory boundary.

## 1. Install the two environments

First install the main project in the active environment:

```bash
python -m pip install -r requirements.txt
```

Then create the Qwen-only environment from the repository root:

```bash
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install --upgrade pip
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt
```

The requirements file uses the official CUDA 12.4 wheel index and installs the
validated PyTorch 2.6.0, torchvision 0.21.0, and Transformers 5.8.1 versions.
On clusters that use a site-specific CUDA/PyTorch wheel, install compatible
PyTorch 2.6.0 wheels in `.venv-qwen` first, then run the last command.

Sanity-check the isolated environment:

```bash
.venv-qwen/bin/python -c \
  'import torch, transformers; print(torch.cuda.is_available(), transformers.__version__)'
```

The first value must be `True`. The reported versions should be PyTorch 2.6.0
and Transformers 5.8.1.

## 2. Optional but recommended: download Qwen in advance

The model repository is about 55.6 GB. Pre-download it so an early-failure run
does not pause for network transfer:

```bash
.venv-qwen/bin/hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

For a shared cache:

```bash
.venv-qwen/bin/hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --cache-dir /data/hf-cache
```

Pass the same directory to the main command with
`--cache-dir /data/hf-cache`. That directory is used by both the base video
model and the Qwen worker.

For a manually downloaded model, `--qwen-model` may be a local directory:

```bash
--qwen-model /data/models/Qwen3.8-27B --local-files-only
```

`--local-files-only` also applies to the selected Diffusers base model and
ViCLIP, so all three must already be available locally.

## 3. Run the same main pipeline with Qwen

CogVideoX-5B:

```bash
early-failure-video \
  --backend cogvideox \
  --prompt-refiner qwen \
  --qwen-python .venv-qwen/bin/python \
  --prompt "A red panda carefully folds a paper crane on a wooden desk." \
  --output outputs/cogvideox-qwen.mp4
```

Wan2.1-1.3B:

```bash
early-failure-video \
  --backend wan \
  --prompt-refiner qwen \
  --qwen-python .venv-qwen/bin/python \
  --prompt "A red panda carefully folds a paper crane on a wooden desk." \
  --output outputs/wan-qwen.mp4
```

`.venv-qwen/bin/python` is the default, so `--qwen-python` can normally be
omitted. It can also be set once with `QWEN_PYTHON=/path/to/python`.

No OpenAI key is used on this path. Qwen runs in BF16, non-thinking mode, and
receives up to 12 uniformly sampled preview frames. The worker uses the same
`--seed` value as video generation. To reduce vision-token memory or latency:

```bash
--refiner-max-images 8
```

The built-in Qwen repository is pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Override it explicitly with
`--qwen-revision` when testing a different remote revision. Local model paths
do not receive a revision argument. Preview JPEGs exist only in a temporary
worker directory and are removed after the worker exits, including failures.

## Memory and storage expectations

- The intended baseline is one 80 GB CUDA GPU. The Qwen weight repository is
  about 55.6 GB, leaving headroom for its vision inputs and generation cache.
- Selecting Qwen automatically enables Diffusers model CPU offload for both
  backends. `--no-cpu-offload` is rejected because it would break the handoff.
- During Qwen inference, the video model remains in host memory; L2R and ViCLIP
  are also parked on CPU. As a practical baseline, 96--128 GB of system RAM is
  recommended for a comfortable handoff.
- Reserve space for Qwen, the selected video model, ViCLIP, and Hugging Face
  cache metadata. At least 120 GB of free cache storage is a useful starting
  point when both backends may be downloaded.
- Model startup happens only on the refinement branch, but loading 55.6 GB of
  weights still adds noticeable latency. The design favors predictable VRAM
  release over keeping a second model server resident.

## Troubleshooting

`Qwen Python environment not found`
: Create `.venv-qwen` as shown above, pass `--qwen-python`, or set
  `QWEN_PYTHON`.

`Qwen3.8-27B handoff requires an available CUDA GPU`
: The Qwen environment has a CPU-only PyTorch build or cannot see the selected
  GPU. Install the matching CUDA PyTorch build and re-run the sanity check.

An unknown `qwen3_5` model/configuration error
: Upgrade the isolated environment with
  `.venv-qwen/bin/python -m pip install -U "transformers==5.8.1"`.

CUDA out of memory while loading Qwen
: Check for unrelated GPU processes, confirm that CPU offload was not disabled,
  and retry with fewer `--refiner-max-images`. The worker restores the video
  inspector even when Qwen exits with an error.
