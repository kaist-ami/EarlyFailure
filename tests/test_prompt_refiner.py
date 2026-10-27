import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from early_failure_video_diffusion.prompt_refiner import (
    DEFAULT_QWEN_REVISION,
    OpenAIPromptRefiner,
    QwenPromptRefiner,
    _parse_revision,
)


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        payload = {
            "refined_prompt": "clear refined prompt",
            "negative_prompt": "motion blur",
        }
        return type("Response", (), {"output_text": json.dumps(payload)})()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_refiner_uses_injected_client_and_returns_normalized_revision() -> None:
    client = FakeClient()
    refiner = OpenAIPromptRefiner(client=client)
    revision = refiner.refine(
        original_prompt="prompt",
        preview=torch.zeros(1, 3, 2, 8, 8),
        score=0.1,
    )

    assert revision.refined_prompt == "clear refined prompt"
    assert revision.negative_prompt == "motion blur"
    assert client.responses.kwargs["model"] == "gpt-5-mini"
    assert client.responses.kwargs["store"] is False
    content = client.responses.kwargs["input"][0]["content"]
    assert [item["type"] for item in content] == [
        "input_text",
        "input_image",
        "input_image",
    ]
    response_format = client.responses.kwargs["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    assert set(response_format["schema"]["required"]) == {
        "refined_prompt",
        "negative_prompt",
    }
    instruction = content[0]["text"]
    assert '- original_prompt: "prompt"' in instruction
    assert "- predicted_score_at_stop: 0.100000" in instruction
    assert "- score_range: 0.0 to 1.0 (higher is better)" in instruction
    assert "stop_step" not in instruction
    assert "stop_reason" not in instruction


def test_revision_parser_limits_and_deduplicates_negative_prompt_items() -> None:
    revision = _parse_revision(
        json.dumps(
            {
                "refined_prompt": "a stable subject",
                "negative_prompt": (
                    " blur, extra limbs, BLUR, watermark, text, distortion, noise "
                ),
            }
        ),
        "original",
    )

    assert revision.negative_prompt == (
        "blur, extra limbs, watermark, text, distortion"
    )


def test_qwen_refiner_runs_one_shot_worker_with_gpu_handoff(monkeypatch) -> None:
    lifecycle: list[str] = []
    captured: dict = {}
    refiner = QwenPromptRefiner(
        python_executable=sys.executable,
        max_images=2,
        seed=7,
        before_inference=lambda: lifecycle.append("before"),
        after_inference=lambda: lifecycle.append("after"),
    )

    def fake_worker(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        captured.update(request)
        captured["frame_paths"] = list(request["frames"])
        assert all(Path(path).is_file() for path in request["frames"])
        payload = {
            "refined_prompt": "stable subject performing a clear action",
            "negative_prompt": "duplicate subject, motion blur",
        }
        response_path.write_text(
            json.dumps({"output_text": f"```json\n{json.dumps(payload)}\n```"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(refiner, "_run_worker", fake_worker)

    revision = refiner.refine(
        original_prompt="prompt",
        preview=torch.zeros(1, 3, 2, 8, 8),
        score=0.1,
    )

    assert revision.refined_prompt == "stable subject performing a clear action"
    assert revision.negative_prompt == "duplicate subject, motion blur"
    assert lifecycle == ["before", "after"]
    assert captured["model"] == "Qwen/Qwen3.8-27B"
    assert captured["revision"] == DEFAULT_QWEN_REVISION
    assert captured["device"] == "cuda"
    assert captured["seed"] == 7
    assert captured["command"][1] == "-I"
    assert len(captured["frames"]) == 2
    assert '- original_prompt: "prompt"' in captured["instruction"]
    assert "- predicted_score_at_stop: 0.100000" in captured["instruction"]
    assert all(not Path(path).exists() for path in captured["frame_paths"])


def test_qwen_refiner_preserves_virtualenv_python_symlink(tmp_path) -> None:
    launcher = tmp_path / "qwen-venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)

    refiner = QwenPromptRefiner(python_executable=launcher)

    assert refiner._resolve_python() == str(launcher.absolute())


def test_qwen_refiner_restores_after_worker_failure(monkeypatch) -> None:
    lifecycle: list[str] = []
    refiner = QwenPromptRefiner(
        python_executable=sys.executable,
        before_inference=lambda: lifecycle.append("before"),
        after_inference=lambda: lifecycle.append("after"),
    )

    def failed_worker(command: list[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, returncode=1)

    monkeypatch.setattr(refiner, "_run_worker", failed_worker)
    with pytest.raises(RuntimeError, match="Qwen3.8 worker failed"):
        refiner.refine(
            original_prompt="prompt",
            preview=torch.zeros(1, 3, 2, 8, 8),
            score=0.1,
        )

    assert lifecycle == ["before", "after"]
