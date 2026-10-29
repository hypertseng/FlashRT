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
