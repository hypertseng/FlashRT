import torch
from flash_rt.hardware.rtx.torch_batch_scratch_arena import TorchBatchScratchArena


def test_compact_kv_reuses_stable_storage_and_preserves_source():
    arena = TorchBatchScratchArena(4, device="cpu")
    original = torch.arange(3 * 4 * 7 * 2, dtype=torch.float32).reshape(3, 4, 7, 1, 2)
    before = original.clone()
    previous = None
    for width, layer, seq in ((2, 0, 5), (4, 2, 6), (2, 1, 5)):
        k = original[layer, :width, :seq]
        out = arena.compact("encoder_k", k, batch_size=width, max_sequence=7)
        assert out.is_contiguous()
        assert torch.equal(out, k)
        if previous is not None:
            assert out.data_ptr() == previous
        previous = out.data_ptr()
    assert torch.equal(original, before)


def test_compact_k_and_v_have_distinct_buffers():
    arena = TorchBatchScratchArena(4, device="cpu")
    value = torch.arange(2 * 3 * 2).reshape(2, 3, 1, 2)
    k = arena.compact("k", value, batch_size=2, max_sequence=3)
    v = arena.compact("v", value + 5, batch_size=2, max_sequence=3)
    assert k.data_ptr() != v.data_ptr()
    assert torch.equal(k, value)
    assert torch.equal(v, value + 5)
