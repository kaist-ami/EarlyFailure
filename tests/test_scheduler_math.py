import torch
from diffusers.schedulers import CogVideoXDDIMScheduler, UniPCMultistepScheduler

from early_failure_video_diffusion.inspection import (
    add_flow_noise,
    cogvideox_predicted_x0,
    flow_predicted_x0,
)


def test_cogvideox_x0_matches_scheduler_output() -> None:
    scheduler = CogVideoXDDIMScheduler(
        num_train_timesteps=100,
        prediction_type="v_prediction",
        clip_sample=False,
    )
    scheduler.set_timesteps(5)
    timestep = scheduler.timesteps[0]
    sample = torch.randn(1, 2, 16, 3, 4)
    model_output = torch.randn_like(sample)

    predicted = cogvideox_predicted_x0(scheduler, model_output, timestep, sample)
    expected = scheduler.step(model_output, timestep, sample).pred_original_sample
    torch.testing.assert_close(predicted, expected)


def test_wan_flow_prediction_recovers_clean_sample() -> None:
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=100,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )
    scheduler.set_timesteps(5)
    timestep = scheduler.timesteps[1]
    clean = torch.randn(1, 16, 2, 3, 4)
    noise = torch.randn_like(clean)
    noisy = add_flow_noise(scheduler, clean, noise, timestep)

    predicted = flow_predicted_x0(scheduler, noise - clean, timestep, noisy)
    torch.testing.assert_close(predicted, clean, rtol=1e-5, atol=1e-5)
