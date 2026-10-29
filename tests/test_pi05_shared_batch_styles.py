import ctypes
import numpy as np
import pytest

from flash_rt.models.pi05 import pipeline_rtx_batched as batched
from flash_rt.models.pi05.batch_scratch_arena import BatchScratchArena


class HostBuffer:
    next_address = 0x10000000

    def __init__(self, count, dtype):
        self.data = np.zeros(count, dtype=dtype)
        self.nbytes = self.data.nbytes
        self.ptr = ctypes.c_void_p(HostBuffer.next_address)
        HostBuffer.next_address += self.nbytes + 4096

    def upload(self, arr):
        assert arr.nbytes <= self.nbytes
        self.data.view(np.uint8)[:arr.nbytes] = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)

    @classmethod
    def from_numpy(cls, arr):
        result = cls(arr.size, arr.dtype)
        result.upload(arr)
        return result


def make_styles(offset=0):
    # Two denoising steps, two layers, two token rows, six features.
    attn = (np.arange(48, dtype=np.uint16) + offset).reshape(2, 2, 2, 6)
    return {"style_attn": attn, "style_ffn": attn + 100,
            "style_final": attn[:, 0] + 200}


def make_pipeline(width, arena, styles):
    pipe = batched.Pi05BatchedPipeline.__new__(batched.Pi05BatchedPipeline)
    pipe.B = width
    pipe.chunk_size = 2
    pipe.weights = {"precomputed": styles}
    pipe.bufs = {}
    pipe._shared_scratch_arena = arena
    pipe._share_batch_styles = True
    return pipe


def test_shared_styles_use_max_batch_stride_and_survive_reload(monkeypatch):
    monkeypatch.setattr(batched, "DEC_D", 2)
    monkeypatch.setattr(batched, "DEC_L", 2)
    monkeypatch.setattr(batched, "CudaBuffer", HostBuffer)
    arena = BatchScratchArena(4, allocator=HostBuffer)
    styles = make_styles()
    b2 = make_pipeline(2, arena, styles)
    b4 = make_pipeline(4, arena, styles)
    b2._upload_b2_styles()
    b4._upload_b2_styles()

    for name in ("decoder_style_attn", "decoder_style_ffn", "decoder_style_final"):
        buf = b2.bufs[name + "_b2"]
        assert buf is b4.bufs[name + "_b2"]
        for step in range(2):
            for layer in (range(2) if name != "decoder_style_final" else (None,)):
                source = styles[name.removeprefix("decoder_")]
                expected = source[step, layer] if layer is not None else source[step]
                for pipe in (b2, b4):
                    ptr = pipe._style_slice_ptr(name, step, layer)
                    start = (ptr - buf.ptr.value) // 2
                    actual = buf.data[start:start + pipe.B * 2 * 6].reshape(pipe.B, 2, 6)
                    np.testing.assert_array_equal(actual, np.tile(expected[None, ...], (pipe.B, 1, 1)))

    ptr_before = b2.bufs["decoder_style_attn_b2"].ptr.value
    updated = make_styles(400)
    for pipe in (b2, b4):
        pipe.weights["precomputed"] = updated
        pipe._upload_b2_styles()
    assert b2.bufs["decoder_style_attn_b2"].ptr.value == ptr_before
    ptr = b2._style_slice_ptr("decoder_style_attn", 1, 1)
    start = (ptr - ptr_before) // 2
    np.testing.assert_array_equal(
        b2.bufs["decoder_style_attn_b2"].data[start:start + 12].reshape(2, 6),
        updated["style_attn"][1, 1])


def test_unshared_style_keeps_independent_layout(monkeypatch):
    monkeypatch.setattr(batched, "DEC_D", 2)
    monkeypatch.setattr(batched, "DEC_L", 2)
    monkeypatch.setattr(batched, "CudaBuffer", HostBuffer)
    pipe = make_pipeline(2, None, make_styles())
    pipe._share_batch_styles = False
    pipe._upload_b2_styles()
    buf = pipe.bufs["decoder_style_attn_b2"]
    expected_offset = (1 * 2 + 1) * 2 * 2 * 6 * 2
    assert pipe._style_slice_ptr("decoder_style_attn", 1, 1) == buf.ptr.value + expected_offset
