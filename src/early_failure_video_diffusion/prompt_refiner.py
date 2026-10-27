from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image

from .types import PromptRevision


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class PromptRefiner(Protocol):
    def refine(
        self,
        *,
        original_prompt: str,
        preview: torch.Tensor,
        score: float,
    ) -> PromptRevision: ...


def tensor_video_to_data_urls(video: torch.Tensor, max_frames: int = 12) -> list[str]:
    """Encode uniformly sampled ``[1, 3, T, H, W]`` previews as JPEG data URLs."""

    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"expected [1, 3, T, H, W], got {tuple(video.shape)}")
    frame_count = min(int(max_frames), int(video.shape[2]))
    indices = np.linspace(0, video.shape[2] - 1, frame_count, dtype=int)
    urls: list[str] = []
    for index in indices:
        frame = video[0, :, int(index)].detach().float().cpu()
        frame = ((frame.clamp(-1.0, 1.0) + 1.0) * 127.5).round().byte()
        array = frame.permute(1, 2, 0).contiguous().numpy()
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{encoded}")
    return urls


def _refinement_instruction(
    original_prompt: str,
    score: float,
) -> str:
    quoted_prompt = json.dumps(original_prompt, ensure_ascii=False)
    return f"""
You are given sampled frames from an early-stage video generation preview.
This is not a final output.
Goal:
Improve overall consistency while preserving the original intent.
Inputs:
- original_prompt: {quoted_prompt}
- predicted_score_at_stop: {float(score):.6f}
- score_range: 0.0 to 1.0 (higher is better)
Task:
- Evaluate the original prompt and preview frames together.
- Refine the prompt to better match the intended subject, action, and scene while fixing visible
issues.
Guidelines:
- Base your revision on both the original prompt and what is visible in the preview.
- Make the prompt clearer, more concrete, and visually stable.
- Clarify subject visibility, action, framing, or scene when needed.
- Do not change the core meaning or add unrelated objects/actions.
- Keep the final prompt concise.
Negative prompt:
- Optional, single comma-separated string
- Short noun phrases only
- No instruction-style wording
- Maximum 5 items
Return ONLY valid JSON:
{{
"refined_prompt": "...",
"negative_prompt": "..."
}}
""".strip()


def _normalize_negative_prompt(value: str, maximum: int = 5) -> str:
    """Normalize the refiner's comma-separated negative-prompt contract."""

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in value.split(","):
        item = " ".join(raw_item.split())
        normalized = item.casefold()
        if not item or normalized in seen:
            continue
        items.append(item)
        seen.add(normalized)
        if len(items) == maximum:
            break
    return ", ".join(items)


def _parse_revision(payload_text: str, original_prompt: str) -> PromptRevision:
    if not isinstance(payload_text, str):
        raise ValueError("prompt-refinement response was not valid JSON")

    # The OpenAI path is schema-constrained, while a local model may still wrap
    # its JSON in a Markdown fence or one short explanatory sentence. Accept the
    # first complete object without weakening validation of the object itself.
    candidate = payload_text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
            candidate = "\n".join(lines).strip()

    candidates = [candidate]
    object_start = candidate.find("{")
    object_end = candidate.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(candidate[object_start : object_end + 1])

    payload = None
    last_error: json.JSONDecodeError | None = None
    for json_text in candidates:
        try:
            payload = json.loads(json_text)
            break
        except json.JSONDecodeError as error:
            last_error = error
    if payload is None:
        raise ValueError(
            "prompt-refinement response was not valid JSON"
        ) from last_error
    if not isinstance(payload, dict):
        raise ValueError("prompt-refinement response must be a JSON object")

    refined_prompt = payload.get("refined_prompt")
    if not isinstance(refined_prompt, str) or not refined_prompt.strip():
        raise ValueError(
            "prompt-refinement response is missing a non-empty refined_prompt"
        )
    negative_prompt = payload.get("negative_prompt", "")
    if negative_prompt is None:
        negative_prompt = ""
    if not isinstance(negative_prompt, str):
        negative_prompt = str(negative_prompt)
    return PromptRevision(
        original_prompt=original_prompt,
        refined_prompt=refined_prompt.strip(),
        negative_prompt=_normalize_negative_prompt(negative_prompt),
    )


class OpenAIPromptRefiner:
    """Multimodal prompt refiner using the OpenAI Responses API.

    No credential is accepted as an argument. The official client reads
    ``OPENAI_API_KEY`` from the process environment.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5-mini",
        max_images: int = 12,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_images = int(max_images)
        self._client = client

    def _client_instance(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "prompt refinement requires the 'pipeline' extra: pip install -e '.[pipeline]'"
                ) from error
            self._client = OpenAI()
        return self._client

    def refine(
        self,
        *,
        original_prompt: str,
        preview: torch.Tensor,
        score: float,
    ) -> PromptRevision:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": _refinement_instruction(original_prompt, score),
            }
        ]
        for data_url in tensor_video_to_data_urls(preview, self.max_images):
            content.append({"type": "input_image", "image_url": data_url})

        response = self._client_instance().responses.create(
            model=self.model,
            store=False,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "prompt_revision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "refined_prompt": {"type": "string"},
                            "negative_prompt": {"type": "string"},
                        },
                        "required": [
                            "refined_prompt",
                            "negative_prompt",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        try:
            output_text = response.output_text
        except AttributeError as error:
            raise ValueError("prompt-refinement response had no output text") from error
        return _parse_revision(output_text, original_prompt)


class QwenPromptRefiner:
    """One-shot local Qwen3.8 refiner with an explicit GPU handoff.

    Qwen runs in a short-lived worker environment so its Transformers version
    cannot conflict with Diffusers. The caller may provide hooks that park the
    video pipeline on CPU before the worker starts and restore its inspector
    after the worker exits. Process exit guarantees that all Qwen GPU memory is
    released before video generation continues.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_QWEN_MODEL,
        revision: str | None = None,
        python_executable: str | Path = ".venv-qwen/bin/python",
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        max_images: int = 12,
        seed: int = 42,
        before_inference: Callable[[], None] | None = None,
        after_inference: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self.revision = (
            revision
            if revision is not None
            else DEFAULT_QWEN_REVISION
            if model == DEFAULT_QWEN_MODEL
            else None
        )
        self.python_executable = str(python_executable)
        self.device = str(device)
        self.cache_dir = (
            str(Path(cache_dir).expanduser().resolve())
            if cache_dir is not None
            else None
        )
        self.local_files_only = bool(local_files_only)
        self.max_images = int(max_images)
        self.seed = int(seed)
        self.before_inference = before_inference
        self.after_inference = after_inference
        if self.max_images < 1:
            raise ValueError("max_images must be positive")

    def _resolve_python(self) -> str:
        expanded = Path(self.python_executable).expanduser()
        if expanded.is_file():
            # Keep virtual-environment launcher symlinks intact. Resolving the
            # symlink would execute the base interpreter and silently bypass
            # the Qwen-specific site-packages installed in the venv.
            return str(expanded.absolute())
        discovered = shutil.which(self.python_executable)
        if discovered is not None:
            return discovered
        raise FileNotFoundError(
            f"Qwen Python environment not found: {self.python_executable}. "
            "Follow docs/QWEN38_80GB.md or pass --qwen-python."
        )

    @staticmethod
    def _write_frames(
        directory: Path, preview: torch.Tensor, maximum: int
    ) -> list[str]:
        urls = tensor_video_to_data_urls(preview, maximum)
        paths: list[str] = []
        for index, data_url in enumerate(urls):
            _, encoded = data_url.split(",", maxsplit=1)
            path = directory / f"frame_{index:02d}.jpg"
            path.write_bytes(base64.b64decode(encoded))
            paths.append(str(path))
        return paths

    def _run_worker(self, command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(command, check=False)

    def refine(
        self,
        *,
        original_prompt: str,
        preview: torch.Tensor,
        score: float,
    ) -> PromptRevision:
        python = self._resolve_python()
        worker = Path(__file__).with_name("_qwen_worker.py")
        handoff_started = False
        with tempfile.TemporaryDirectory(prefix="early-failure-qwen38-") as raw_dir:
            directory = Path(raw_dir)
            request_path = directory / "request.json"
            response_path = directory / "response.json"
            request = {
                "model": self.model,
                "revision": self.revision,
                "device": self.device,
                "cache_dir": self.cache_dir,
                "local_files_only": self.local_files_only,
                "seed": self.seed,
                "frames": self._write_frames(directory, preview, self.max_images),
                "instruction": _refinement_instruction(original_prompt, score),
            }
            request_path.write_text(
                json.dumps(request, ensure_ascii=False), encoding="utf-8"
            )

            try:
                if self.before_inference is not None:
                    # Mark the handoff first so a partially failed offload still
                    # gets the matching restore callback.
                    handoff_started = True
                    self.before_inference()
                result = self._run_worker(
                    [
                        python,
                        "-I",
                        str(worker),
                        "--request",
                        str(request_path),
                        "--response",
                        str(response_path),
                    ]
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        "Qwen3.8 worker failed. The worker output above contains "
                        "the model-loading or inference error."
                    )
                if not response_path.is_file():
                    raise RuntimeError("Qwen3.8 worker did not write a response")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if not isinstance(response, dict):
                    raise RuntimeError("Qwen3.8 worker response was not an object")
                output_text = response.get("output_text")
            finally:
                if handoff_started and self.after_inference is not None:
                    self.after_inference()

        return _parse_revision(output_text, original_prompt)
