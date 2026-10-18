"""Graph-capturable per-slot hard Prefix RTC lock.

The caller stages a boolean mask of shape (B, H, 1) before graph replay.
All tensors retain fixed device addresses. Execute on the capture/replay
stream, as the Thor frontend does for the rest of its PyTorch operations.
"""
import torch


def copy_prefix_masked_(noise, previous, mask):
    """Keep prefix positions bit-exact; leave the other positions untouched."""
    torch.where(mask, previous, noise, out=noise)
