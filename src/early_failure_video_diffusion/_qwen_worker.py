"""Short-lived Qwen3.8 worker used by :class:`QwenPromptRefiner`.

This file is executed directly by a dedicated Transformers environment. Keep it
independent from the package imports so that the Qwen and Diffusers dependency
sets remain isolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Qwen3.8 refinement")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))

    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as error:
        raise ImportError(
            "The Qwen worker needs torch, accelerate, Pillow, and "
            "transformers>=5.8 in its dedicated environment."
        ) from error

    model_id = str(request["model"])
    device = torch.device(str(request["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen3.8-27B handoff requires an available CUDA GPU")
    torch.manual_seed(int(request.get("seed", 42)))
    torch.cuda.manual_seed_all(int(request.get("seed", 42)))
    common = {"local_files_only": bool(request.get("local_files_only", False))}
    if request.get("cache_dir") is not None:
        common["cache_dir"] = str(request["cache_dir"])
    if request.get("revision") is not None:
        common["revision"] = str(request["revision"])

    display_model = "local model" if Path(model_id).expanduser().exists() else model_id
    print(f"[Qwen handoff] Loading {display_model} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, **common)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        device_map={"": str(device)},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **common,
    ).eval()

    content = [{"type": "image", "path": str(path)} for path in request["frames"]]
    content.append({"type": "text", "text": str(request["instruction"])})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        preserve_thinking=False,
    ).to(model.device)

    print("[Qwen handoff] Refining the failed preview...", flush=True)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    output_text = processor.decode(
        outputs[0, prompt_length:], skip_special_tokens=True
    ).strip()
    args.response.write_text(
        json.dumps({"output_text": output_text}, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[Qwen handoff] Done; releasing Qwen GPU memory.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
