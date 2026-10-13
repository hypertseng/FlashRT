"""FlashRT — Action post-processing utilities."""

import numpy as np

LIBERO_ACTION_DIM = 7


def _normalization_mode(mode):
    return str(mode or "quantiles").strip().lower().replace("-", "_")


def normalize_state(state, norm_stats, mode="quantiles"):
    """Normalize proprioception according to the checkpoint contract."""
    values = np.asarray(state, dtype=np.float32)
    stats = norm_stats["state"]
    dim = min(values.shape[-1], len(stats.get("mean", stats.get("q01", []))))
    result = values.copy()
    normalized_mode = _normalization_mode(mode)
    if normalized_mode in ("mean_std", "meanstd"):
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        result[..., :dim] = (values[..., :dim] - mean[:dim]) / (std[:dim] + 1e-8)
    elif normalized_mode in ("quantiles", "quantile", "min_max", "minmax"):
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        result[..., :dim] = np.clip(
            2.0 * (values[..., :dim] - q01[:dim]) / (q99[:dim] - q01[:dim] + 1e-8) - 1.0,
            -1.0,
            1.0,
        )
    else:
        raise ValueError(f"unsupported state normalization mode: {mode!r}")
    return result


def unnormalize_actions(actions, norm_stats, mode="quantiles"):
    """Unnormalize actions according to the checkpoint contract."""
    values = np.asarray(actions, dtype=np.float32)
    stats = norm_stats["actions"]
    normalized_mode = _normalization_mode(mode)
    if normalized_mode in ("mean_std", "meanstd"):
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        dim = min(values.shape[-1], len(mean))
        result = values.copy()
        result[..., :dim] = values[..., :dim] * std[:dim] + mean[:dim]
        return result
    if normalized_mode not in ("quantiles", "quantile", "min_max", "minmax"):
        raise ValueError(f"unsupported action normalization mode: {mode!r}")
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    dim = min(values.shape[-1], len(q01))
    clipped = np.clip(values, -1.0, 1.0)
    unnorm = clipped.copy()
    unnorm[..., :dim] = (
        (clipped[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
