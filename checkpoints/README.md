# Checkpoints

The official L2R inference checkpoints are already included under
`src/early_failure_video_diffusion/weights` and are loaded automatically for the
selected backend. Nothing needs to be downloaded or linked into this directory.

For a custom compatible checkpoint, use
`--l2r-checkpoint /path/to/checkpoint.pt` or
`LatentToRGB.from_checkpoint(...)`. The loader expects a plain PyTorch state
dictionary (or a dictionary containing `state_dict`) and uses strict loading by
default.

To train a new checkpoint, follow [`docs/L2R_TRAINING.md`](../docs/L2R_TRAINING.md).
The research dataset and prepared training manifest are not distributed.
