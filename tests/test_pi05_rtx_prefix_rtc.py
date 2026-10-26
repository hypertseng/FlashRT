"""Prefix RTC contracts for the RTX frontend and resident batch graphs."""
import os
import threading

import numpy as np
import pytest


def frontend_type():
    pytest.importorskip("torch")
    pytest.importorskip("flash_rt.flash_rt_kernels")
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
    return Pi05TorchFrontendRtx


@pytest.mark.parametrize("length", [-1, True, 1.5, 11])
def test_invalid_prefix_is_rejected_before_checkpoint_loading(length):
    cls = frontend_type()
    with pytest.raises(ValueError, match="rtc_prefix_len"):
        cls("/nonexistent-checkpoint", rtc_prefix_len=length)


def test_disabled_prefix_fails_before_batch_selection():
    fe = frontend_type().__new__(frontend_type())
    fe._lifecycle_lock = threading.RLock()
    fe.rtc_prefix_len = 0
    with pytest.raises(ValueError, match="not initialized"):
        fe.infer_multi_prompt_batch([
            {"prompt": "test", "observation": {}, "rtc_prefix_len": 2}
        ])


@pytest.mark.skipif(os.getenv("FLASHRT_RUN_RTX_PREFIX_GPU") != "1",
                    reason="requires explicit RTX GPU contract opt-in")
def test_fixed48_prefix_locks_slots_without_replacing_graphs():
    fe = frontend_type()(
        os.environ["PI05_CHECKPOINT"], num_views=2, use_fp8=True,
        num_steps=10, state_prompt_mode="fixed", state_prompt_fixed_max_len=48,
        rtc_prefix_len=2,
    )
    fe.enable_batch_graph_bank(True)
    rng = np.random.default_rng(7)
    obs = [{"image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "state": np.zeros(8, np.float32)} for _ in range(2)]
    prompts = ["pick up the red block", "pick up the blue block"]
    fe.set_prompt(prompts[0], state=obs[0]["state"])
    fe.calibrate(obs[0])
    initial = fe.infer(obs[0], seed=7, rtc_prefix_len=0)
    graph = fe.pipeline._graph
    raw = initial["rtc_raw_actions"]
    assert raw.shape == (10, 32)
    locked = fe.infer(obs[0], seed=11, rtc_prefix_len=2, rtc_prev_action_chunk=raw)
    np.testing.assert_array_equal(locked["rtc_raw_actions"][:2], raw[:2])
    reset = fe.infer(obs[0], seed=7, rtc_prefix_len=0)
    np.testing.assert_array_equal(reset["rtc_raw_actions"], raw)
    assert fe.pipeline._graph is graph
    with pytest.raises(ValueError, match="raw previous"):
        fe.infer(obs[0], rtc_prefix_len=2)

    items = [{"prompt": p, "observation": o, "rtc_prefix_len": 0}
             for p, o in zip(prompts, obs)]
    initial_batch = fe.infer_multi_prompt_batch(items, seed=7)
    batch_graph = fe.pipeline._graph
    unlocked = fe.infer_multi_prompt_batch(items, seed=11)
    for lengths in ((2, 0), (0, 2), (1, 2), (0, 0)):
        batch = [dict(item, rtc_prefix_len=n,
                      rtc_prev_action_chunk=old["rtc_raw_actions"] if n else None)
                 for item, old, n in zip(items, initial_batch, lengths)]
        result = fe.infer_multi_prompt_batch(batch, seed=11)
        assert fe.pipeline._graph is batch_graph
        assert fe.pipeline.B == 2
        for row, old, reference, n in zip(result, initial_batch, unlocked, lengths):
            assert row["actions"].shape == (10, 7)
            assert np.isfinite(row["rtc_raw_actions"]).all()
            if n:
                np.testing.assert_array_equal(row["rtc_raw_actions"][:n], old["rtc_raw_actions"][:n])
            else:
                np.testing.assert_array_equal(row["rtc_raw_actions"], reference["rtc_raw_actions"])
    fe.set_batched_mode(enable=False)
    fe.set_prompt(prompts[0], state=obs[0]["state"])
    restored = fe.infer(obs[0], seed=7, rtc_prefix_len=0)
    assert fe.pipeline._graph is graph
    np.testing.assert_array_equal(restored["rtc_raw_actions"], raw)
