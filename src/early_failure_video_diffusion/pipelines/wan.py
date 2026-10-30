from __future__ import annotations

from typing import Any

import torch
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.schedulers import UniPCMultistepScheduler

from ..inspection import (
    InspectionHistory,
    InspectionPipelineMixin,
    add_flow_noise,
    choose_candidate_scales,
    first_video,
    flow_predicted_x0,
    inspect_step,
    progressive_temporal_mask,
)
from ..stopping import AdaptiveBalancedStopper, StopDecision
from ..types import GenerationResult


class EarlyFailureWanPipeline(InspectionPipelineMixin, WanPipeline):
    """Wan2.1 pipeline with real-time inspection inside its denoising loop."""

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
        continue_on_failure: bool = False,
        anchor_x0: torch.Tensor | None = None,
        anchor_start_step: int = 1,
        anchor_blend_steps: int = 2,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        guidance_scale_2: float | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Wan intervention currently supports one non-empty prompt")
        if probe_only and num_frames != 1:
            raise ValueError("probe_only=True requires num_frames=1")
        if probe_only and continue_on_failure:
            raise ValueError("continue_on_failure is not supported for probes")
        if inspect_generation and stopper is None:
            raise ValueError("inspection requires an AdaptiveBalancedStopper")
        if not isinstance(self.scheduler, UniPCMultistepScheduler):
            raise TypeError(
                "Wan inspection requires UniPCMultistepScheduler; "
                f"received {type(self.scheduler).__name__}"
            )
        if self.scheduler.config.prediction_type != "flow_prediction":
            raise ValueError(
                "Wan inspection requires prediction_type='flow_prediction'"
            )
        if inspect_generation:
            self._require_inspector()

        inspection_prompt = inspection_prompt or prompt
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            None,
            None,
            ["latents"],
            guidance_scale_2,
        )

        if (num_frames - 1) % self.vae_scale_factor_temporal != 0:
            num_frames = (
                num_frames
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        patch_size = (
            self.transformer.config.patch_size
            if self.transformer is not None
            else self.transformer_2.config.patch_size
        )
        height_multiple = self.vae_scale_factor_spatial * patch_size[1]
        width_multiple = self.vae_scale_factor_spatial * patch_size[2]
        height = height // height_multiple * height_multiple
        width = width // width_multiple * width_multiple
        if height < height_multiple or width < width_multiple:
            raise ValueError(
                "requested Wan resolution is smaller than one latent patch"
            )

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale
        self._guidance_scale = float(guidance_scale)
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        prompt_embeds, negative_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        transformer_dtype = (
            self.transformer.dtype
            if self.transformer is not None
            else self.transformer_2.dtype
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_embeds is not None:
            negative_embeds = negative_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        schedule = self.scheduler.timesteps
        latent_channels = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            1,
            latent_channels,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        timestep_mask = torch.ones_like(latents, dtype=torch.float32, device=device)

        pure_noise: torch.Tensor | None = None
        anchor: torch.Tensor | None = None
        skipped_steps = 0
        if anchor_x0 is not None:
            if anchor_start_step < 0 or anchor_start_step >= len(schedule):
                raise ValueError("anchor_start_step is outside the denoising schedule")
            if (
                anchor_x0.ndim != 4
                or anchor_x0.shape[0] != latent_channels
                or anchor_x0.shape[1] != 1
            ):
                raise ValueError(
                    "Wan anchor must have shape [latent_channels, 1, H, W] "
                    f"(received {tuple(anchor_x0.shape)})"
                )
            anchor = anchor_x0.unsqueeze(0).to(device=device, dtype=latents.dtype)
            if anchor.shape[-2:] != latents.shape[-2:]:
                raise ValueError(
                    "anchor and generation latent spatial sizes do not match"
                )
            pure_noise = latents.clone()
            skipped_steps = int(anchor_start_step)
            latents = add_flow_noise(
                self.scheduler, anchor, pure_noise, schedule[anchor_start_step]
            )
            schedule = schedule[anchor_start_step:]

        self.scheduler.set_begin_index(skipped_steps)
        self._num_timesteps = len(schedule)
        total_steps = len(schedule) + skipped_steps
        steps_executed = skipped_steps
        history = InspectionHistory()
        inspection_active = bool(inspect_generation)
        accepted_decision: StopDecision | None = None
        failed_decision: StopDecision | None = None
        failed_snapshot = None
        failed_steps_executed: int | None = None
        boundary_timestep = (
            self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            if self.config.boundary_ratio is not None
            else None
        )

        with self.progress_bar(total=len(schedule)) as progress_bar:
            for local_step, timestep in enumerate(schedule):
                self._current_timestep = timestep
                if boundary_timestep is None or timestep >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance = float(guidance_scale)
                else:
                    current_model = self.transformer_2
                    current_guidance = float(guidance_scale_2)

                if (
                    anchor is not None
                    and pure_noise is not None
                    and local_step < anchor_blend_steps
                ):
                    noised_anchor = add_flow_noise(
                        self.scheduler, anchor, pure_noise, timestep
                    )
                    temporal_weight = progressive_temporal_mask(
                        latents.shape[2],
                        local_step,
                        anchor_blend_steps,
                        device=latents.device,
                        dtype=latents.dtype,
                    ).view(1, 1, latents.shape[2], 1, 1)
                    latents = (
                        temporal_weight * noised_anchor
                        + (1.0 - temporal_weight) * latents
                    )

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    flattened_timestep = (
                        timestep_mask[0, 0, :, ::2, ::2] * timestep
                    ).flatten()
                    timestep_batch = flattened_timestep.unsqueeze(0).expand(
                        latents.shape[0], -1
                    )
                else:
                    timestep_batch = timestep.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    prediction_text = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep_batch,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_cfg:
                    with current_model.cache_context("uncond"):
                        prediction_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep_batch,
                            encoder_hidden_states=negative_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    scales = choose_candidate_scales(probe_only, current_guidance)
                    candidates = [
                        prediction_uncond
                        + scale * (prediction_text - prediction_uncond)
                        for scale in scales
                    ]
                else:
                    candidates = [prediction_text]

                selected_index = 0
                decision: StopDecision | None = None
                if inspection_active:
                    assert stopper is not None
                    x0_candidates = torch.cat(
                        [
                            flow_predicted_x0(
                                self.scheduler,
                                candidate.float(),
                                timestep,
                                latents.float(),
                            )
                            for candidate in candidates
                        ],
                        dim=0,
                    )
                    _, scores = self._inspect_candidates(
                        x0_candidates, inspection_prompt, layout="bcthw"
                    )
                    selected_index, decision = inspect_step(
                        stopper=stopper,
                        history=history,
                        step=local_step + skipped_steps,
                        scores=scores,
                        x0_candidates=x0_candidates,
                    )

                latents = self.scheduler.step(
                    candidates[selected_index], timestep, latents, return_dict=False
                )[0]
                steps_executed += 1
                progress_bar.update()

                if decision is not None:
                    snapshot = history.select(decision)
                    if probe_only:
                        result = self._early_result(
                            status="probe",
                            decision=decision,
                            snapshot=snapshot,
                            layout="bcthw",
                            steps_executed=steps_executed,
                            total_steps=total_steps,
                        )
                        self._current_timestep = None
                        self.maybe_free_model_hooks()
                        return result
                    if decision.score < failure_threshold:
                        if not continue_on_failure:
                            result = self._early_result(
                                status="failed",
                                decision=decision,
                                snapshot=snapshot,
                                layout="bcthw",
                                steps_executed=steps_executed,
                                total_steps=total_steps,
                            )
                            self._current_timestep = None
                            self.maybe_free_model_hooks()
                            return result
                        failed_decision = decision
                        failed_snapshot = snapshot
                        failed_steps_executed = steps_executed
                        inspection_active = False
                        continue
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
                layout="bcthw",
                steps_executed=steps_executed,
                total_steps=total_steps,
            )
            self.maybe_free_model_hooks()
            return result

        if inspect_generation and accepted_decision is None and failed_decision is None:
            assert stopper is not None
            decision = stopper.finalize()
            if decision.score < failure_threshold:
                snapshot = history.select(decision)
                if not continue_on_failure:
                    result = self._early_result(
                        status="failed",
                        decision=decision,
                        snapshot=snapshot,
                        layout="bcthw",
                        steps_executed=steps_executed,
                        total_steps=total_steps,
                    )
                    self.maybe_free_model_hooks()
                    return result
                failed_decision = decision
                failed_snapshot = snapshot
                failed_steps_executed = steps_executed
            else:
                accepted_decision = decision

        if output_type == "latent":
            video = latents
        else:
            latents = latents.to(self.vae.dtype)
            latent_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latent_std_inverse = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latent_std_inverse + latent_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )

        frames = first_video(video) if output_type != "latent" else video
        if failed_decision is not None:
            assert failed_snapshot is not None
            assert failed_steps_executed is not None
            result = self._early_result(
                status="failed",
                decision=failed_decision,
                snapshot=failed_snapshot,
                layout="bcthw",
                steps_executed=failed_steps_executed,
                total_steps=total_steps,
            )
            result.frames = frames
            self.maybe_free_model_hooks()
            return result

        result = GenerationResult(
            status="completed",
            frames=frames,
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
