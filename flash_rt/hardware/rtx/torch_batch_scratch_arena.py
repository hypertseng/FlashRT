"""Stable, contiguous workspace views shared by serialized batch graphs."""
import math
import torch

class TorchBatchScratchArena:
    def __init__(self, max_batch_size, *, device="cuda"):
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.device = device
        self._buffers = {}

    def acquire(self, name, shape, *, batch_size, dtype):
        batch_size = int(batch_size)
        shape = tuple(int(x) for x in shape)
        count = math.prod(shape)
        if not shape or any(x < 1 for x in shape) or not 1 <= batch_size <= self.max_batch_size or count % batch_size:
            raise ValueError("scratch shape must scale with supported batch")
        per_sample = count // batch_size
        if name not in self._buffers:
            self._buffers[name] = (torch.empty(per_sample * self.max_batch_size, dtype=dtype, device=self.device), per_sample, dtype)
        base, old_count, old_dtype = self._buffers[name]
        if old_count != per_sample or old_dtype != dtype:
            raise ValueError("scratch layout changed after graph capture")
        return base[:count].view(shape)
