import torch
import pytest
from flash_rt.frontends.torch.fp8_bf16_offload import offload_candidates

def test_requires_complete_fp8_before_mutation():
    weights={"encoder_ffn_gate_w":torch.arange(6).reshape(2,3), "encoder_ffn_up_w":torch.ones(2,3), "keep":torch.ones(1)}
    with pytest.raises(ValueError):
        offload_candidates(weights, {}, 2)
    assert set(weights)=={"encoder_ffn_gate_w","encoder_ffn_up_w","keep"}

def test_offloads_pair_and_preserves_other_weights():
    weights={"encoder_ffn_gate_w":torch.arange(6).reshape(2,3), "encoder_ffn_up_w":torch.ones(2,3), "keep":torch.ones(1)}
    fp8={f"encoder_ffn_gate_up_w_{i}":(123,456) for i in range(2)}
    keep=weights["keep"]
    result=offload_candidates(weights,fp8,2)
    assert result=={"encoder_ffn_gate_w","encoder_ffn_up_w"}
    assert weights["encoder_ffn_gate_w"].device.type=="cpu"
    assert weights["keep"] is keep

def test_complete_fp8_map_offloads_all_quantized_groups():
    from flash_rt.frontends.torch.fp8_bf16_offload import fp8_candidates
    groups = fp8_candidates(2, 2, 2)
    weights = {k: torch.ones(2, 3) for k in groups}
    fp8 = {name:(123,456) for names in groups.values() for name in names}
    result = offload_candidates(weights, fp8, 2, vision_layers=2, decoder_layers=2)
    assert result == set(groups)
    assert "encoder_ffn_down_w" in result
    assert "vision_attn_qkv_w" in result
    assert "decoder_ffn_down_w" in result

def test_incomplete_one_group_does_not_mutate_weights():
    from flash_rt.frontends.torch.fp8_bf16_offload import fp8_candidates
    groups = fp8_candidates(2, 2, 2)
    weights = {k: torch.ones(2, 3) for k in groups}
    originals = dict(weights)
    fp8 = {name:(123,456) for names in groups.values() for name in names}
    fp8.pop("vision_attn_qkv_w_1")
    with pytest.raises(ValueError):
        offload_candidates(weights, fp8, 2, vision_layers=2, decoder_layers=2)
    assert all(weights[k] is originals[k] for k in weights)

def test_reload_cache_cleanup_is_opt_in(monkeypatch):
    from flash_rt.frontends.torch.fp8_bf16_offload import release_reload_cache
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    release_reload_cache(False)
    assert calls == []
    release_reload_cache(True)
    assert calls == ["sync", "empty"]

def test_discard_retains_metadata_without_storage():
    from flash_rt.frontends.torch.fp8_bf16_offload import discard_candidates
    weights = {"encoder_ffn_gate_w": torch.ones(2,3), "encoder_ffn_up_w": torch.ones(2,3), "keep": torch.ones(1)}
    fp8 = {f"encoder_ffn_gate_up_w_{i}": (123,456) for i in range(2)}
    dropped = discard_candidates(weights, fp8, 2)
    assert dropped == {"encoder_ffn_gate_w", "encoder_ffn_up_w"}
    assert weights["encoder_ffn_gate_w"].device.type == "meta"
    assert weights["encoder_ffn_gate_w"].shape == (2,3)
    assert weights["keep"].device.type == "cpu"


def test_discard_refuses_incomplete_fp8_without_mutation():
    from flash_rt.frontends.torch.fp8_bf16_offload import discard_candidates
    weights = {"encoder_ffn_gate_w": torch.ones(2,3), "encoder_ffn_up_w": torch.ones(2,3)}
    originals = dict(weights)
    with pytest.raises(ValueError):
        discard_candidates(weights, {}, 2)
    assert all(weights[k] is originals[k] for k in weights)
