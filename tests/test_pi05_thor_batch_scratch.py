import pytest
import torch
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
from flash_rt.hardware.rtx.torch_batch_scratch_arena import TorchBatchScratchArena


def test_thor_batch_scratch_uses_stable_storage_across_batch_sizes():
    frontend = object.__new__(Pi05TorchFrontendThor)
    frontend._thor_scratch_arena = TorchBatchScratchArena(12, device="cpu")
    small = frontend._thor_batch_scratch("encoder_x", (2 * 8, 4), 2, torch.float16)
    large = frontend._thor_batch_scratch("encoder_x", (12 * 8, 4), 12, torch.float16)
    assert small.data_ptr() == large.data_ptr()
    assert tuple(small.shape) == (16, 4)
    assert tuple(large.shape) == (96, 4)
    with pytest.raises(ValueError):
        frontend._thor_batch_scratch("encoder_x", (13 * 8, 4), 13, torch.float16)
    with pytest.raises(ValueError, match="layout"):
        frontend._thor_batch_scratch("encoder_x", (2 * 9, 4), 2, torch.float16)


def test_thor_batch_scratch_disabled_allocates_independently():
    frontend = object.__new__(Pi05TorchFrontendThor)
    frontend._thor_scratch_arena = None
    a = frontend._thor_batch_scratch("x", (2, 4), 2, torch.float32, device="cpu")
    b = frontend._thor_batch_scratch("x", (2, 4), 2, torch.float32, device="cpu")
    assert a.data_ptr() != b.data_ptr()


def test_rtx_and_thor_share_the_common_arena_implementation():
    from flash_rt.core.torch_batch_scratch_arena import TorchBatchScratchArena as common
    assert common is TorchBatchScratchArena
