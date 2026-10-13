import numpy as np

from flash_rt.core.utils.actions import normalize_state, unnormalize_actions


def test_mean_std_normalization_round_trip():
    stats = {
        "state": {"mean": [1.0, -2.0], "std": [2.0, 4.0]},
        "actions": {"mean": [1.0, -2.0], "std": [2.0, 4.0]},
    }
    physical = np.asarray([3.0, 2.0], dtype=np.float32)
    normalized = normalize_state(physical, stats, "MEAN_STD")
    np.testing.assert_allclose(normalized, [1.0, 1.0])
    np.testing.assert_allclose(
        unnormalize_actions(normalized, stats, "MEAN_STD"), physical
    )


def test_mean_std_actions_are_not_clipped():
    stats = {
        "actions": {"mean": [0.5], "std": [2.0]},
    }
    result = unnormalize_actions(
        np.asarray([[2.0]], dtype=np.float32), stats, "MEAN_STD"
    )
    np.testing.assert_allclose(result, [[4.5]])


def test_quantile_mode_preserves_legacy_clipping():
    stats = {
        "actions": {"q01": [-2.0], "q99": [4.0]},
    }
    result = unnormalize_actions(
        np.asarray([[-2.0], [2.0]], dtype=np.float32), stats, "QUANTILES"
    )
    np.testing.assert_allclose(result, [[-2.0], [4.0]], atol=1e-5)
