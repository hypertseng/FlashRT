import pytest
import torch
from flash_rt.hardware.rtx.torch_batch_scratch_arena import TorchBatchScratchArena

def test_views_are_contiguous_fixed_address_across_nonleading_batch():
    arena = TorchBatchScratchArena(12, device="cpu")
    a = arena.acquire("splitkv", (3, 2, 4), batch_size=2, dtype=torch.float32)
    b = arena.acquire("splitkv", (3, 9, 4), batch_size=9, dtype=torch.float32)
    assert a.is_contiguous() and b.is_contiguous()
    assert a.data_ptr() == b.data_ptr()
    assert a.shape == (3, 2, 4) and b.shape == (3, 9, 4)
    assert a.untyped_storage().nbytes() == 3*12*4*4
    b.fill_(7)
    assert torch.all(a == 7)

def test_layout_and_boundaries_are_checked():
    arena = TorchBatchScratchArena(12, device="cpu")
    arena.acquire("x", (3, 2, 4), batch_size=2, dtype=torch.float32)
    with pytest.raises(ValueError): arena.acquire("x", (3, 2, 5), batch_size=2, dtype=torch.float32)
    with pytest.raises(ValueError): arena.acquire("x", (3, 2, 4), batch_size=2, dtype=torch.float16)
    with pytest.raises(ValueError): arena.acquire("y", (3, 13, 4), batch_size=13, dtype=torch.float32)
    with pytest.raises(ValueError): arena.acquire("z", (5,), batch_size=2, dtype=torch.float32)
