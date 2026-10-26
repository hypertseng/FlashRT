"""Per-slot Prefix RTC control with fixed graph addresses.

Staging changes values, never captured shapes or buffer addresses. The native
pipeline calls apply before and after every denoising update.
"""
import ctypes

import numpy as np
import torch


class DynamicPrefixLock:
    def __init__(self, batch_size, horizon, action_dim=32, *, device="cuda"):
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        shape = (self.batch_size, self.horizon, self.action_dim)
        self.previous = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.scratch = torch.empty_like(self.previous)
        self.lengths = torch.zeros((self.batch_size, 1, 1), dtype=torch.int64, device=device)
        self.index = torch.arange(self.horizon, device=device).view(1, self.horizon, 1)
        self.mask = torch.zeros((self.batch_size, self.horizon, 1), dtype=torch.bool, device=device)
        self._streams = {}

    def stage(self, previous_chunks, prefix_lengths):
        if len(previous_chunks) != self.batch_size or len(prefix_lengths) != self.batch_size:
            raise ValueError("RTC slot count does not match the native batch")
        validated = []
        lengths = []
        for previous, length in zip(previous_chunks, prefix_lengths):
            n = int(length)
            if isinstance(length, bool) or n != length or not 0 <= n <= self.horizon:
                raise ValueError("RTC prefix length must be an integer within the action horizon")
            if n:
                if previous is None:
                    raise ValueError("Positive RTC prefix requires raw previous actions")
                previous = np.asarray(previous, dtype=np.float32)
                if previous.shape != (self.horizon, self.action_dim):
                    raise ValueError("RTC history must contain a complete normalized action chunk")
                if not np.isfinite(previous).all():
                    raise ValueError("RTC history contains non-finite actions")
                if not previous.flags.writeable or not previous.flags.c_contiguous:
                    previous = previous.copy()
            validated.append(previous)
            lengths.append(n)
        for slot, (previous, length) in enumerate(zip(validated, lengths)):
            if length:
                self.previous[slot].copy_(torch.from_numpy(previous))
        self.lengths.copy_(torch.tensor(lengths, dtype=torch.int64).view(self.batch_size, 1, 1))
        torch.lt(self.index, self.lengths, out=self.mask)

    def apply(self, buffer, stream, cudart):
        """Lock a native BF16 noise buffer on the pipeline's actual stream."""
        stream_id = int(getattr(stream, "value", stream) or 0)
        nbytes = self.scratch.numel() * self.scratch.element_size()
        if buffer.nbytes != nbytes:
            raise ValueError("RTC buffer size differs from the native noise buffer")
        ptr = buffer.ptr
        rc = cudart.cudaMemcpyAsync(
            ctypes.c_void_p(self.scratch.data_ptr()), ptr, nbytes, 3,
            ctypes.c_void_p(stream_id))
        if rc:
            raise RuntimeError(f"RTC input copy failed with CUDA status {rc}")
        if stream_id not in self._streams:
            device = self.previous.device
            self._streams[stream_id] = (
                torch.cuda.default_stream(device) if stream_id == 0
                else torch.cuda.ExternalStream(stream_id, device=device))
        with torch.cuda.stream(self._streams[stream_id]):
            torch.where(self.mask, self.previous, self.scratch, out=self.scratch)
        rc = cudart.cudaMemcpyAsync(
            ptr, ctypes.c_void_p(self.scratch.data_ptr()), nbytes, 3,
            ctypes.c_void_p(stream_id))
        if rc:
            raise RuntimeError(f"RTC output copy failed with CUDA status {rc}")
