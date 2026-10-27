# Early Failure Detection and Intervention in Video Diffusion Models

This repository is the public implementation of [Early Failure Detection and Intervention in Video Diffusion Models](https://arxiv.org/abs/2603.14320). It includes the inference pipeline, the small fused L2R checkpoints used by that pipeline, and an optional clean L2R training/fusion workflow.

The code adds the paper's early inspection and hierarchical intervention pipeline directly to the official Hugging Face Diffusers implementations of:

- [`zai-org/CogVideoX-5b`](https://huggingface.co/zai-org/CogVideoX-5b)
- [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)

The small L2R inference checkpoints are included in this repository. The base video model and ViCLIP checkpoint are downloaded from Hugging Face on first use, so no source-tree replacement or separate VBench clone is required.

## Choose your path

| Goal | Where to start | Private training data needed? |
| --- | --- | --- |
| Run early failure detection and intervention with the released weights | [Quick start](#quick-start-main-pipeline) | No |
| Use local Qwen3.8-27B instead of the OpenAI API | [80 GB Qwen guide](docs/QWEN38_80GB.md) | No |
| Train and fuse a new L2R checkpoint | [L2R training guide](docs/L2R_TRAINING.md) | Yes |

Most users want the first path. It completely skips L2R data preparation,
initialization, training, and fusion: the correct 29-key fused L2R checkpoint is
selected from the package automatically.

The research training dataset and its prepared manifest are not released. The
two synthetic videos under `examples/l2r_training` only verify the data
preparation command; they are not a usable training dataset.

## Quick start: main pipeline

Python 3.10--3.12, CUDA, and a CUDA-capable PyTorch installation are required.
The released requirements use the validated PyTorch 2.5.1 / CUDA 12.1,
Diffusers 0.38.0, and Transformers 4.57.6 stack. Create a clean environment and
install the tested dependency set directly from `requirements.txt`:

```bash
git clone https://github.com/kaist-ami/Early-failure-video-diffusion.git
cd Early-failure-video-diffusion
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` uses the official PyTorch CUDA 12.1 wheel index and pins
the direct dependencies validated for this release. If the host requires a
different CUDA-specific PyTorch build, install matching `torch==2.5.1` and
`torchvision==0.20.1` wheels first, then install the project with
`python -m pip install -e ".[pipeline]"`; the package metadata prevents an
upgrade to a different PyTorch minor version.

Set an OpenAI API key for the prompt-refinement branch. The API is called only if the initial generation is predicted to fail and the frame-anchor intervention is insufficient.

```bash
export OPENAI_API_KEY="<your-key>"
```

Privacy note: if the hierarchy reaches OpenAI refinement, the original prompt,
predicted score, and up to 12 uniformly sampled preview frames are sent to the
selected OpenAI model. Use `--prompt-refiner qwen` to keep refinement local.
The API key is read only from the process environment and is never written to
the output metadata. The Responses API request uses `store=False`; applicable
provider data-handling policies still govern the request.

The default `gpt-5-mini` model preserves the paper's setup. Select another
vision-capable Responses API model explicitly with `--openai-model` when
needed, for example `--openai-model gpt-5.4-mini`. Check the
[OpenAI model catalog](https://developers.openai.com/api/docs/models) for
current availability.

For an API-free run on one 80 GB GPU, select local
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). It uses a
separate environment because Qwen requires a newer Transformers version than
the video stack:

```bash
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install --upgrade pip
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt

early-failure-video \
  --backend cogvideox \
  --prompt-refiner qwen \
  --prompt "A red panda carefully folds a paper crane on a wooden desk." \
  --output outputs/cogvideox-qwen.mp4
```

The Qwen requirements use the validated PyTorch 2.6.0 / CUDA 12.4,
torchvision 0.21.0, and Transformers 5.8.1 stack. Keep this environment
separate from the main video environment.

On this path the code parks Diffusers, L2R, and ViCLIP on CPU, starts Qwen only
when prompt refinement is reached, and resumes video generation only after the
short-lived Qwen process has exited and released its CUDA memory. Qwen is never
loaded when an earlier branch succeeds. See the
[Qwen3.8-27B 80 GB guide](docs/QWEN38_80GB.md) for pre-download, manual/offline,
memory, and troubleshooting instructions.

Then run either backend:

```bash
# CogVideoX-5B
early-failure-video \
  --backend cogvideox \
  --prompt "A red panda carefully folds a paper crane on a wooden desk." \
  --output outputs/cogvideox.mp4

# Wan2.1-1.3B
early-failure-video \
  --backend wan \
  --prompt "A red panda carefully folds a paper crane on a wooden desk." \
  --output outputs/wan.mp4
```

On the first run, the command automatically:

1. loads the matching bundled L2R checkpoint;
2. downloads the official VBench ViCLIP checkpoint and tokenizer files;
3. downloads the selected Diffusers base model from Hugging Face.

When `--prompt-refiner qwen` is selected, Qwen3.8-27B is additionally
downloaded on the first refinement event unless it was pre-cached. OpenAI is
the default refiner; the Qwen path does not read `OPENAI_API_KEY`.

The output video is accompanied by a `.json` file containing the intervention
decision trace, original/refined prompts, and reproducibility settings. Local
model and checkpoint paths are recorded only as `local` or `custom`, never as
absolute filesystem paths. Treat prompt-bearing metadata as user data when
sharing results. Run `early-failure-video --help` for resolution, seed,
offload, and generation options.

No L2R training command or manifest is involved in this path.

## Paired inference test set

The release includes two fixed examples per backend. Test-set mode generates a
normal full-denoising baseline as `initial.mp4` and the proposed intervention
result as `output.mp4`. Both CogVideoX and Wan test sets use seed `42`.
Within each pair, both videos start from that same seed and use the same model,
resolution, frame count, step count, and guidance scale. Test-set mode rejects
seed overrides so the released comparison stays reproducible.

| Backend | Prompt |
| --- | --- |
| CogVideoX-5B | `Iron Man flying in the sky` |
| CogVideoX-5B | `A confused panda in calculus class` |
| Wan2.1-1.3B | `A corgi's head depicted as an explosion of a nebula` |
| Wan2.1-1.3B | `A couple in formal evening wear going home get caught in a heavy downpour with umbrellas by Hokusai, in the style of Ukiyoss` |

Run either or both backend sets:

```bash
early-failure-video \
  --backend cogvideox \
  --test-set \
  --output-dir outputs/test-set

early-failure-video \
  --backend wan \
  --test-set \
  --output-dir outputs/test-set
```

Add `--prompt-refiner qwen` to either command to use the local Qwen refiner.
The output layout is:

```text
outputs/test-set/
├── cogvideox-5b/
│   ├── openai/
│   │   ├── 01-iron-man-flying-in-the-sky/
│   │   │   ├── initial.mp4
│   │   │   ├── initial.json
│   │   │   ├── output.mp4
│   │   │   └── output.json
│   │   ├── 02-confused-panda-in-calculus-class/
│   │   │   └── ...
│   │   └── manifest.json
│   └── qwen/
│       └── ...
└── wan2.1-1.3b/
    ├── openai/
    │   └── ...
    └── qwen/
        └── ...
```

## Model files

The built-in remote models are pinned to the revisions used for release
validation:

| Component | Repository | Revision |
| --- | --- | --- |
| CogVideoX-5B | `zai-org/CogVideoX-5b` | `8fc5b281006c82b82d34fd2543d2f0ebb4e7e321` |
| Wan2.1-1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `0fad780a534b6463e45facd96134c9f345acfa5b` |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Use `--model-revision` or `--qwen-revision` only when intentionally testing a
different revision. Revisions are not passed for local model directories.

### L2R

Both inference checkpoints are versioned with the package and selected from `--backend`:

```text
src/early_failure_video_diffusion/weights/
├── deploy_fused_crf18_cogvideox.pt
└── deploy_fused_crf18_wan1.5B.pt
```

The historical `wan1.5B` filename is retained for exact checkpoint compatibility; it is the L2R checkpoint used by the Wan2.1-1.3B backend. To use a different compatible checkpoint, pass `--l2r-checkpoint /path/to/checkpoint.pt`.

### ViCLIP

The scorer uses the ViCLIP implementation and checkpoint published for [VBench](https://github.com/Vchitect/VBench). By default, these official files are cached under `~/.cache/vbench/ViCLIP`:

- [`ViClip-InternVid-10M-FLT.pth`](https://huggingface.co/OpenGVLab/VBench_Used_Models/blob/main/ViClip-InternVid-10M-FLT.pth) from `OpenGVLab/VBench_Used_Models`
- [`bpe_simple_vocab_16e6.txt.gz`](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) from OpenAI CLIP

The checkpoint is about 1.7 GB. To download it before a run:

```bash
early-failure-download-viclip
```

The automatic download is pinned to VBench model repository revision
`4a9f73381ee3c9753c0258b50479b388f043e53a` and verified against SHA-256
`7a4d6ad6eac6632db3693f4b97f9f8f6445b65b1e139a6fc6686150522238c56`.
Manually supplied custom ViCLIP checkpoints are not required to match that
official hash.

Use a different cache directory if desired:

```bash
early-failure-download-viclip --cache-dir /data/models/viclip

early-failure-video \
  --backend cogvideox \
  --prompt "A red panda folds a paper crane." \
  --viclip-cache-dir /data/models/viclip \
  --output outputs/example.mp4
```

You may also download the two files manually and provide their paths. This is useful on offline compute nodes:

```bash
early-failure-video \
  --backend cogvideox \
  --prompt "A red panda folds a paper crane." \
  --viclip-checkpoint /data/models/ViClip-InternVid-10M-FLT.pth \
  --viclip-vocab /data/models/bpe_simple_vocab_16e6.txt.gz \
  --local-files-only \
  --output outputs/example.mp4
```

`--local-files-only` also prevents the Diffusers base model from accessing the network, so that model must already be in the Hugging Face cache or supplied through `--model-id /local/model/path`.

## Method

During early denoising, the pipeline predicts the current clean latent (`x0`), maps it to RGB with L2R, and scores text-video consistency with ViCLIP. The adaptive stopper makes an early success/failure decision. A predicted failure follows this hierarchy:

1. Probe a one-frame generation.
2. If that probe is clearly better, reuse its `x0` as a temporal semantic anchor.
3. Otherwise, refine the prompt from sampled failed-preview frames with the selected multimodal refiner: the OpenAI Responses API or local Qwen3.8-27B.
4. Probe and generate with the refined prompt, falling back to its one-frame anchor when needed.

A successful generation exits the hierarchy immediately. Each underlying
generation handles one prompt and one video; CLI test-set mode iterates the
released examples sequentially and writes the paired baseline/intervention
outputs.

## Optional: train a new L2R checkpoint

The optional workflow keeps every transition explicit and independently
checkable:

```text
your videos -> paired manifest -> initialized rank basis -> factorized training
            -> fused 3D kernels -> 29-key deploy checkpoint -> strict=True load
```

Install the extra training dependency and follow the dedicated guide:

```bash
pip install -e ".[training]"
```

The four commands are:

```bash
early-failure-prepare-l2r ...  # own videos -> RGB/latent manifest
early-failure-init-l2r ...     # initialize dw basis + expansion weights
early-failure-train-l2r ...    # train the 32-key factorized representation
early-failure-fuse-l2r ...     # contract basis axes and validate 29 deploy keys
```

`early-failure-train-l2r` requires `--manifest`. You may skip the preparation
utility only if you already have compatible paired data and create that
manifest yourself; the repository does not provide one. Conversely, the main
pipeline above skips this entire workflow by using the bundled fused weights.

The paper-compatible pair uses 49 RGB frames at `720 x 480` (`W x H`) and a
float32 latent of shape `[16,13,60,90]`. FPS does not need to be uniform across
clips because training uses decoded frame indices, not timestamps. CogVideoX
and Wan use the same frame count, resolution, and temporal target alignment;
only their latent normalization differs.

See [Train and fuse an L2R checkpoint](docs/L2R_TRAINING.md) for exact CogVideoX
and Wan dataset specifications and commands, the JSONL schema, resume behavior,
optional diffusion-`x0` fine-tuning, fusion verification, and custom checkpoint
inference.

## Python API

The custom pipelines subclass the official Diffusers classes:

```python
import torch

from early_failure_video_diffusion import LatentToRGB, ViCLIPScorer
from early_failure_video_diffusion.pipelines import EarlyFailureCogVideoXPipeline

l2r = LatentToRGB.from_pretrained("cogvideox").eval().to("cuda")
scorer = ViCLIPScorer(device="cuda")  # downloads ViCLIP when not cached

pipe = EarlyFailureCogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.set_inspector(l2r, scorer)
```

For the complete intervention hierarchy, use `HierarchicalInterventionRunner`; the CLI is the reference entry point. `ViCLIPScorer` also accepts `checkpoint=`, `vocab_path=`, `cache_dir=`, and `local_files_only=` for manual or offline setups.

Stage metadata uses zero-based indices. `stop_step` is where the inspection
policy made its stop decision, while `selected_step` is the preview associated
with the reported score and any returned anchor. They differ when
`stop_reason` is `budget_best`; for example, inspection can stop at step 14
after selecting the strongest preview from step 3 (`steps_executed` is then
15).

## Reproducibility notes

- CogVideoX inspection uses `CogVideoXDDIMScheduler`, matching the released model scheduler.
- Wan uses `UniPCMultistepScheduler` with flow prediction, flow sigmas, and `flow_shift=3.0` at 480p.
- L2R and ViCLIP run in inference mode inside the main pipeline. Optional L2R training is isolated under `early_failure_video_diffusion.training`; ViCLIP training is not included.
- OpenAI refinement reads `OPENAI_API_KEY` through the official client. Local Qwen refinement needs no API credential. Credentials are never accepted as command-line arguments or written to metadata.
- OpenAI refinement sends the prompt, predicted score, and sampled preview frames only when the hierarchy reaches that branch. Qwen processes the same inputs in a short-lived local worker and removes its temporary frame files on exit.
- Built-in Hugging Face model IDs and the automatic ViCLIP download use pinned release revisions. Local paths are redacted from generated JSON metadata.
- Qwen3.8-27B runs as a one-shot BF16 worker only after the initial video and frame-anchor paths both fail. The worker process exits before Diffusers resumes, making the 80 GB single-GPU memory handoff explicit.
- The repository does not include experiment outputs, datasets, base-model weights, Qwen weights, ViCLIP weights, or API credentials.

## Tests

```bash
pip install -e ".[test]"
pytest
```

Unit tests cover the adaptive stopper, bundled L2R checkpoint layout, scheduler math, intervention decision tree, one-shot Qwen handoff, factorized L2R checkpoint round trips, fusion equivalence, and strict 29-key deployment loading without downloading a base model, Qwen, or ViCLIP.

## Build a release artifact

Build distributions from this source tree rather than maintaining a second
editable copy:

```bash
python -m pip install build
python -m build
```

The resulting wheel contains both fused L2R checkpoints, their SHA-256
manifest, and the vendored ViCLIP license. Generated videos, run metadata,
virtual environments, caches, datasets, and API credentials are excluded by
the source manifest and `.gitignore`. Validate the wheel in a clean environment
before publishing it together with the source archive.

## Citation

```bibtex
@article{kwon2026earlyfailure,
  title   = {Early Failure Detection and Intervention in Video Diffusion Models},
  author  = {Kwon, Byung-Ki and Lim, Sohwi and Nam, Hyeon-Woo and Moon, Ye-Bin and Oh, Tae-Hyun},
  journal = {arXiv preprint arXiv:2603.14320},
  year    = {2026}
}
```

## License and third-party code

Except where otherwise noted, the original code in this repository is released
under the [MIT License](LICENSE).

The minimal ViCLIP inference source vendored from VBench remains under Apache License 2.0; its attribution and license are included in `src/early_failure_video_diffusion/_vendor/viclip`. Diffusers, the base models, Qwen3.8-27B, ViCLIP weights, and OpenAI CLIP assets retain their respective licenses and terms.
