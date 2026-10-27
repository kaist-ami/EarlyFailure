from __future__ import annotations

import math
from typing import Any

import torch
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler

from ..inspection import (
    InspectionHistory,
    InspectionPipelineMixin,
    choose_candidate_scales,
    cogvideox_predicted_x0,
    first_video,
    inspect_step,
    progressive_temporal_mask,
    retrieve_timesteps,
)
from ..stopping import AdaptiveBalancedStopper, StopDecision
from ..types import GenerationResult


class EarlyFailureCogVideoXPipeline(InspectionPipelineMixin, CogVideoXPipeline):
    """CogVideoX pipeline with real-time inspection inside its denoising loop."""

    @torch.inference_mode()
    def generate_with_inspection(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        inspection_prompt: str | None = None,
        stopper: AdaptiveBalancedStopper | None = None,
        failure_threshold: float = 0.22,
        probe_only: bool = False,
        inspect_generation: bool = True,
        anchor_x0: torch.Tensor | None = None,
        anchor_start_step: int = 1,
        anchor_blend_steps: int = 2,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 226,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "CogVideoX intervention currently supports one non-empty prompt"
            )
        if probe_only and num_frames != 1:
            raise ValueError("probe_only=True requires num_frames=1")
        if inspect_generation and stopper is None:
            raise ValueError("inspection requires an AdaptiveBalancedStopper")
        if not isinstance(self.scheduler, CogVideoXDDIMScheduler):
            raise TypeError(
                "CogVideoX inspection currently supports CogVideoXDDIMScheduler only; "
                f"received {type(self.scheduler).__name__}"
            )
        if inspect_generation:
            self._require_inspector()

        inspection_prompt = inspection_prompt or prompt
        height = (
            height
            or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        )
        width = (
            width
            or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        )
        self.check_inputs(
            prompt, height, width, negative_prompt, ["latents"], None, None
        )

        self._guidance_scale = float(guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        positive_embeds, negative_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_cfg,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        prompt_embeds = (
            torch.cat((negative_embeds, positive_embeds), dim=0)
            if do_cfg
            else positive_embeds
        )

        schedule, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        prepared_num_frames = num_frames
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            prepared_num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1,
            latent_channels,
            prepared_num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        pure_noise: torch.Tensor | None = None
        anchor: torch.Tensor | None = None
        skipped_steps = 0
        if anchor_x0 is not None:
            if anchor_start_step < 0 or anchor_start_step >= len(schedule):
                raise ValueError("anchor_start_step is outside the denoising schedule")
            if (
                anchor_x0.ndim != 4
                or anchor_x0.shape[0] != 1
                or anchor_x0.shape[1] != latent_channels
            ):
                raise ValueError(
                    "CogVideoX anchor must have shape [1, latent_channels, H, W] "
                    f"(received {tuple(anchor_x0.shape)})"
                )
            anchor = anchor_x0.unsqueeze(0).to(device=device, dtype=latents.dtype)
            if anchor.shape[-2:] != latents.shape[-2:]:
                raise ValueError(
                    "anchor and generation latent spatial sizes do not match"
                )
            pure_noise = latents.clone()
            skipped_steps = int(anchor_start_step)
            anchor_timestep = schedule[anchor_start_step].reshape(1)
            latents = self.scheduler.add_noise(anchor, pure_noise, anchor_timestep).to(
                prompt_embeds.dtype
            )
            schedule = schedule[anchor_start_step:]

        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height, width, latents.size(1), device
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        self._num_timesteps = len(schedule)
        total_steps = len(schedule) + skipped_steps
        steps_executed = skipped_steps
        history = InspectionHistory()
        inspection_active = bool(inspect_generation)
        accepted_decision: StopDecision | None = None

        with self.progress_bar(total=len(schedule)) as progress_bar:
            for local_step, timestep in enumerate(schedule):
                self._current_timestep = timestep

                if (
                    anchor is not None
                    and pure_noise is not None
                    and local_step < anchor_blend_steps
                ):
                    noised_anchor = self.scheduler.add_noise(
                        anchor, pure_noise, timestep.reshape(1)
                    )
                    temporal_weight = progressive_temporal_mask(
                        latents.shape[1],
                        local_step,
                        anchor_blend_steps,
                        device=latents.device,
                        dtype=latents.dtype,
                    ).view(1, latents.shape[1], 1, 1, 1)
                    latents = (
                        temporal_weight * noised_anchor
                        + (1.0 - temporal_weight) * latents
                    )

                latent_model_input = (
                    torch.cat((latents, latents), dim=0) if do_cfg else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep
                )
                timestep_batch = timestep.expand(latent_model_input.shape[0])
                with self.transformer.cache_context("cond_uncond"):
                    raw_prediction = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep_batch,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].float()

                current_guidance = float(guidance_scale)
                if use_dynamic_cfg:
                    current_guidance = (
                        1.0
                        + guidance_scale
                        * (
                            1.0
                            - math.cos(
                                math.pi
                                * (
                                    (num_inference_steps - timestep.item())
                                    / num_inference_steps
                                )
                                ** 5.0
                            )
                        )
                        / 2.0
                    )

                if do_cfg:
                    prediction_uncond, prediction_text = raw_prediction.chunk(2)
                    scales = choose_candidate_scales(probe_only, current_guidance)
                    candidates = [
                        prediction_uncond
                        + scale * (prediction_text - prediction_uncond)
                        for scale in scales
                    ]
                else:
                    candidates = [raw_prediction]

                selected_index = 0
                decision: StopDecision | None = None
                if inspection_active:
                    assert stopper is not None
                    x0_candidates = torch.cat(
                        [
                            cogvideox_predicted_x0(
                                self.scheduler, candidate, timestep, latents.float()
                            )
                            for candidate in candidates
                        ],
                        dim=0,
                    )
                    _, scores = self._inspect_candidates(
                        x0_candidates, inspection_prompt, layout="btchw"
                    )
                    selected_index, decision = inspect_step(
                        stopper=stopper,
                        history=history,
                        step=local_step + skipped_steps,
                        scores=scores,
                        x0_candidates=x0_candidates,
                    )

                latents = self.scheduler.step(
                    candidates[selected_index],
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0].to(prompt_embeds.dtype)
                steps_executed += 1
                progress_bar.update()

                if decision is not None:
                    snapshot = history.select(decision)
                    if probe_only:
                        result = self._early_result(
                            status="probe",
                            decision=decision,
                            snapshot=snapshot,
                            layout="btchw",
                            steps_executed=steps_executed,
                            total_steps=total_steps,
                        )
                        self._current_timestep = None
                        self.maybe_free_model_hooks()
                        return result
                    if decision.score < failure_threshold:
                        result = self._early_result(
                            status="failed",
                            decision=decision,
                            snapshot=snapshot,
                            layout="btchw",
                            steps_executed=steps_executed,
                            total_steps=total_steps,
                        )
                        self._current_timestep = None
                        self.maybe_free_model_hooks()
                        return result
                    accepted_decision = decision
                    inspection_active = False

        self._current_timestep = None

        if probe_only:
            assert stopper is not None
            decision = stopper.finalize()
            result = self._early_result(
                status="probe",
                decision=decision,
                snapshot=history.select(decision),
                layout="btchw",
                steps_executed=steps_executed,
                total_steps=total_steps,
            )
            self.maybe_free_model_hooks()
            return result

        if inspect_generation and accepted_decision is None:
            assert stopper is not None
            decision = stopper.finalize()
            if decision.score < failure_threshold:
                result = self._early_result(
                    status="failed",
                    decision=decision,
                    snapshot=history.select(decision),
                    layout="btchw",
                    steps_executed=steps_executed,
                    total_steps=total_steps,
                )
                self.maybe_free_model_hooks()
                return result
            accepted_decision = decision

        if output_type == "latent":
            video = latents
        else:
            if additional_frames:
                latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(
                video=video, output_type=output_type
            )

        result = GenerationResult(
            status="completed",
            frames=first_video(video) if output_type != "latent" else video,
            score=accepted_decision.score if accepted_decision is not None else None,
            stop_step=(
                accepted_decision.stop_step if accepted_decision is not None else None
            ),
            selected_step=(
                accepted_decision.selected_step
                if accepted_decision is not None
                else None
            ),
            stop_reason=accepted_decision.reason
            if accepted_decision is not None
            else None,
            steps_executed=steps_executed,
            total_steps=total_steps,
        )
        self.maybe_free_model_hooks()
        return result
