import pytest
from flash_rt.models.pi05.batch_scratch_arena import BatchScratchArena

class Buffer:
    def __init__(self, count, dtype):
        self.count, self.dtype = count, dtype


def test_reuses_one_max_sized_buffer_for_all_widths():
    arena = BatchScratchArena(max_batch_size=12, allocator=Buffer)
    first = arena.acquire("encoder_hidden", 2 * 80, "bf16", batch_size=2)
    later = arena.acquire("encoder_hidden", 11 * 80, "bf16", batch_size=11)
    assert first is later
    assert first.count == 12 * 80
    assert arena.acquire("vision_hidden", 2 * 50, "bf16", batch_size=2) is not first


def test_rejects_unsafe_growth_and_incompatible_layout():
    arena = BatchScratchArena(max_batch_size=12, allocator=Buffer)
    arena.acquire("x", 2 * 10, "bf16", batch_size=2)
    with pytest.raises(ValueError): arena.acquire("x", 13 * 10, "bf16", batch_size=13)
    with pytest.raises(ValueError): arena.acquire("x", 4 * 11, "bf16", batch_size=4)
    with pytest.raises(ValueError): arena.acquire("x", 4 * 10, "fp8", batch_size=4)
    with pytest.raises(ValueError): arena.acquire("y", 21, "bf16", batch_size=2)


def test_padded_zero_scratch_reuses_maximum_allocation():
    allocations = []
    def zeroed(count, dtype):
        allocations.append((count, dtype))
        return Buffer(count, dtype)
    arena = BatchScratchArena(max_batch_size=10, allocator=Buffer, zero_allocator=zeroed)
    first = arena.acquire("fp8", 3 * 128, "fp8", batch_size=2, extra_rows=1, zeroed=True)
    assert first.count == 11 * 128
    assert arena.acquire("fp8", 9 * 128, "fp8", batch_size=8, extra_rows=1, zeroed=True) is first
    assert allocations == [(11 * 128, "fp8")]
    with pytest.raises(ValueError):
        arena.acquire("fp8", 8 * 128, "fp8", batch_size=8)

def test_fixed_parent_scratch_reuses_exact_capacity():
    arena = BatchScratchArena(max_batch_size=12, allocator=Buffer)
    first = arena.acquire_fixed("parent_encoder", 120, "bf16")
    second = arena.acquire_fixed("parent_encoder", 120, "bf16")
    assert first is second
    assert first.count == 120
    with pytest.raises(ValueError):
        arena.acquire_fixed("parent_encoder", 121, "bf16")
    with pytest.raises(ValueError):
        arena.acquire_fixed("parent_encoder", 120, "fp8")
