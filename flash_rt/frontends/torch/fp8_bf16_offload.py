"""Explicit, opt-in offload for FP8 encoder gate/up weights."""

def offload_candidates(weights, fp8, layers):
    keys = ("encoder_ffn_gate_w", "encoder_ffn_up_w")
    if any(k not in weights for k in keys) or any(
            f"encoder_ffn_gate_up_w_{i}" not in fp8 for i in range(layers)):
        raise ValueError("cannot offload without all FP8 gate/up weights")
    moved = {k: weights[k].to("cpu") for k in keys}
    weights.update(moved)
    return set(keys)
