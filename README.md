# Early Failure Detection and Intervention in Video Diffusion Models (NeurIPS 2026)

### [Paper](https://arxiv.org/abs/2603.14320) | [Local Qwen guide](docs/QWEN38_80GB.md) | [L2R training guide](docs/L2R_TRAINING.md)

Official implementation of the NeurIPS 2026 paper **Early Failure Detection
and Intervention in Video Diffusion Models**. The pipeline inspects intermediate
video latents, detects likely failures early, and applies a frame anchor or
prompt refinement before regeneration.

Supported backends:

- [`zai-org/CogVideoX-5b`](https://huggingface.co/zai-org/CogVideoX-5b)
- [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)

Both fused L2R checkpoints are included. The selected base model and the
official VBench ViCLIP checkpoint are downloaded automatically on first use.

## Setup

Validated on Linux with Python 3.10, PyTorch 2.5.1, CUDA 12.1, Diffusers
0.38.0, and Transformers 4.57.6. Python 3.10--3.12 is supported.

```bash
git clone https://github.com/kaist-ami/Early-failure-video-diffusion.git
cd Early-failure-video-diffusion

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the validated CUDA 12.1 stack. For another CUDA
wheel, first install matching `torch==2.5.1` and `torchvision==0.20.1`, then
run `python -m pip install -e ".[pipeline]"`.

The default prompt refiner uses the OpenAI Responses API:

```bash
export OPENAI_API_KEY="<your-key>"
```

The API is called only when the initial generation and frame-anchor path are
insufficient. At that point the prompt, predicted score, and sampled preview
frames are sent with `store=False`. Use local Qwen to avoid an external API.

## Inference

```bash
# CogVideoX-5B
early-failure-video \
  --backend cogvideox \
  --prompt "A red panda folds a paper crane on a wooden desk." \
  --output outputs/cogvideox.mp4

# Wan2.1-1.3B
early-failure-video \
  --backend wan \
  --prompt "A red panda folds a paper crane on a wooden desk." \
  --output outputs/wan.mp4
```

Each video is accompanied by a JSON decision trace. The default seed is `42`
and the default length is 49 frames. Run `early-failure-video --help` for
resolution, cache, offline, checkpoint, and model options.

### Local Qwen3.8-27B

Qwen uses a separate environment because it requires Transformers 5.8.1. The
validated configuration targets one 80 GB GPU.

```bash
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install --upgrade pip
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt

early-failure-video \
  --backend cogvideox \
  --prompt-refiner qwen \
  --prompt "A red panda folds a paper crane on a wooden desk." \
  --output outputs/cogvideox-qwen.mp4
```

Qwen is loaded only on the refinement branch and exits before video generation
resumes. See the [Qwen guide](docs/QWEN38_80GB.md) for pre-download, memory,
shared-cache, and offline instructions.

## Released test prompts

Test-set mode writes a normal baseline as `initial.mp4` and our result as
`output.mp4`, using seed `42` for both.

| Backend | Prompt |
| --- | --- |
| CogVideoX-5B | `Iron Man flying in the sky` |
| CogVideoX-5B | `A confused panda in calculus class` |
| Wan2.1-1.3B | `A corgi's head depicted as an explosion of a nebula` |
| Wan2.1-1.3B | `A couple in formal evening wear going home get caught in a heavy downpour with umbrellas by Hokusai, in the style of Ukiyoss` |

```bash
early-failure-video \
  --backend cogvideox \
  --test-set \
  --output-dir outputs/test-set

# Replace cogvideox with wan for the Wan examples.
# Add --prompt-refiner qwen to use the local refiner.
```

## Method

At each inspection step, L2R maps the predicted clean latent (`x0`) to RGB and
ViCLIP scores text-video consistency. A predicted failure follows this order:

1. Probe a one-frame generation.
2. Reuse a strong probe as a temporal semantic anchor.
3. Otherwise refine the prompt with OpenAI or local Qwen.
4. Regenerate from the refined prompt, with an anchor fallback.

Successful generations exit the hierarchy immediately.

## Model assets

- Bundled L2R weights are selected automatically from `--backend` and load as
  strict 29-key state dictionaries.
- ViCLIP is cached under `~/.cache/vbench/ViCLIP`. Pre-download it with
  `early-failure-download-viclip`.
- Base models, Qwen, and ViCLIP weights are not stored in this repository.
- Built-in model downloads use pinned revisions; local paths are redacted from
  generated metadata.

## Optional L2R training

Inference does **not** require L2R training. To train and fuse a custom
checkpoint from compatible paired data:

```bash
python -m pip install -e ".[pipeline,training]"
```

The research training dataset is not distributed; the two synthetic clips in
`examples/l2r_training` are command-chain tests only. See the
[L2R training guide](docs/L2R_TRAINING.md) for the dataset schema and full
workflow.

## Tests and packaging

```bash
python -m pip install -e ".[test]"
pytest

python -m pip install build
python -m build
```

The wheel contains both fused L2R checkpoints, their SHA-256 manifest, and the
vendored ViCLIP license.

## Citation

```bibtex
@article{kwon2026earlyfailure,
  title   = {Early Failure Detection and Intervention in Video Diffusion Models},
  author  = {Kwon, Byung-Ki and Lim, Sohwi and Nam, Hyeon-Woo and Moon, Ye-Bin and Oh, Tae-Hyun},
  journal = {arXiv preprint arXiv:2603.14320},
  year    = {2026}
}
```

## License

Original code is released under the [MIT License](LICENSE). The vendored
ViCLIP inference code remains under Apache License 2.0; its attribution is in
`src/early_failure_video_diffusion/_vendor/viclip`.
