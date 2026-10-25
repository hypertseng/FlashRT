"""CPU contracts for native RTX graph residency; no GPU allocation required."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
Pi05TorchFrontendRtx = pytest.importorskip(
    "flash_rt.frontends.torch.pi05_rtx").Pi05TorchFrontendRtx


def frontend():
    f = Pi05TorchFrontendRtx.__new__(Pi05TorchFrontendRtx)
    f._lifecycle_lock = threading.RLock()
    f._reload_failed = False
    f._batched_active = False
    f._batch_size = 2
    f.pipeline = SimpleNamespace(_graph=object(), max_prompt_len=160)
    f._fixed_pipeline = f.pipeline
    f._prompt_pipeline_cache = {160: f.pipeline}
    f._last_prompt_call = None
    f._batch_ctx = {}
    return f


def test_bank_reports_only_captured_graphs_including_parked_single():
    f = frontend()
    single = f.pipeline
    f._single_ctx = {"pipeline": single}
    f._batch_ctx = {2: {"pipeline": SimpleNamespace(_graph=object(), max_prompt_len=160)},
                    3: {"pipeline": SimpleNamespace(_graph=None, max_prompt_len=160)}}
    f._batched_active = True
    f._batch_size = 8
    f.pipeline = SimpleNamespace(_graph=object(), max_prompt_len=160)
    assert f.batch_graph_bank_keys() == ((1, 160), (2, 160), (8, 160))
    assert single in f._live_pipelines()


def test_select_and_disable_restore_same_single_and_batch_objects():
    f = frontend()
    original = f.pipeline
    batch = SimpleNamespace(_graph=object(), max_prompt_len=160)
    def allocate(*, enable, batch_size):
        f._batched_active = True
        f._batch_size = batch_size
        f.pipeline = batch
        f._fixed_pipeline = None
    f._set_batched_mode = allocate
    f.set_batched_mode(enable=True, batch_size=8)
    assert f.pipeline is batch
    f.set_batched_mode(enable=False)
    assert f.pipeline is original
    assert f._fixed_pipeline is original
    f.set_batched_mode(enable=True, batch_size=8)
    assert f.pipeline is batch


def test_fixed_prompt_without_state_uses_fixed_pipeline_and_main_cache_api():
    f = frontend()
    f._rl_config = None
    f._state_prompt_mode = "fixed"
    f._state_prompt_fixed_max_len = 160
    host = object()
    f._embed_prompt_cached = Mock(return_value=(object(), 7, host))
    f._set_prompt_fixed = Mock()
    f.attn_backend = SimpleNamespace(set_fixed_shape=Mock())
    f.pipeline.set_language_embeds = Mock()
    f.set_prompt("pick up cup")
    f._set_prompt_fixed.assert_called_once_with(7)
    f.pipeline.set_language_embeds.assert_called_once_with(host)
    f._embed_prompt_cached.assert_called_once_with("pick up cup", 161, state=None)


def test_fixed_overflow_rejected_before_graph_replacement():
    f = frontend()
    f._rl_config = None
    f._state_prompt_mode = "fixed"
    f._state_prompt_fixed_max_len = 160
    f._embed_prompt_cached = Mock(return_value=(object(), 161, object()))
    original = f.pipeline
    with pytest.raises(ValueError, match="exceeds fixed capacity"):
        f.set_prompt("long prompt")
    assert f.pipeline is original


def test_request_level_rtc_is_refused_instead_of_ignored():
    f = frontend()
    with pytest.raises(ValueError, match="RTC"):
        f.infer_multi_prompt_batch([{"rtc_prefix_len": 1}])


def test_rl_prompt_keeps_main_batched_cfg_mode():
    f = frontend()
    f._rl_config = {"cfg_beta": 1.5}
    f._batched_active = True
    f._set_prompt_rl = Mock()
    f.attn_backend = SimpleNamespace(set_fixed_shape=Mock())
    f.set_prompt("task")
    assert f._batched_active
    f._set_prompt_rl.assert_called_once_with("task")


def test_single_context_refreshes_prompt_after_weight_reload():
    f = frontend()
    f._weight_version = 2
    f._batched_active = True
    f._last_prompt_call = ("batch", ("task", "task"))
    f._single_ctx = {"pipeline": object(), "_batched_active": False,
                     "_last_prompt_call": ("single", "task", None), "_weight_version": 1}
    f.set_prompt = Mock()
    f.set_batched_mode(enable=False)
    assert f.weight_version == 2
    f.set_prompt.assert_called_once_with("task", state=None)
