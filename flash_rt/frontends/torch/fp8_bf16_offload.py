"""Opt-in host storage for matrices already represented as FP8."""

def fp8_candidates(vision_layers, encoder_layers, decoder_layers):
    groups = {}
    for key in ("vision_attn_qkv_w", "vision_attn_o_w", "vision_ffn_up_w", "vision_ffn_down_w"):
        groups[key] = [f"{key}_{i}" for i in range(vision_layers)]
    groups["encoder_multi_modal_projector_w"] = ["vision_projector_w"]
    for component, count in (("encoder", encoder_layers), ("decoder", decoder_layers)):
        for suffix in ("attn_qkv_w", "attn_o_w", "ffn_down_w"):
            key = f"{component}_{suffix}"
            groups[key] = [f"{key}_{i}" for i in range(count)]
        fused = [f"{component}_ffn_gate_up_w_{i}" for i in range(count)]
        groups[f"{component}_ffn_gate_w"] = fused
        groups[f"{component}_ffn_up_w"] = fused
    return groups

def offload_candidates(weights, fp8, layers, *, vision_layers=None, decoder_layers=None):
    if vision_layers is None and decoder_layers is None:
        groups = {key: [f"encoder_ffn_gate_up_w_{i}" for i in range(layers)]
                  for key in ("encoder_ffn_gate_w", "encoder_ffn_up_w")}
    elif vision_layers is None or decoder_layers is None:
        raise ValueError("all layer counts are required")
    else:
        groups = fp8_candidates(vision_layers, layers, decoder_layers)
    if any(key not in weights for key in groups) or any(
            name not in fp8 for names in groups.values() for name in names):
        raise ValueError("cannot offload without all corresponding FP8 weights")
    moved = {key: weights[key].to("cpu") for key in groups}
    weights.update(moved)
    return set(groups)


def release_reload_cache(enabled):
    if enabled:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
