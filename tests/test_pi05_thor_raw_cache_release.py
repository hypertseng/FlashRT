import gc
import weakref

import pytest
import torch

from flash_rt.frontends.torch.pi05_thor import _release_raw_checkpoint_cache


def test_release_raw_cache_keeps_converted_and_aliased_weights_alive():
    raw = torch.ones(8, dtype=torch.float32)
    converted = raw.to(torch.float16)
    alias = torch.ones(4, dtype=torch.float16)
    raw_ref = weakref.ref(raw)
    cache = {"raw": {"converted": raw, "alias": alias},
             "fp16": {"converted": converted, "alias": alias}}
    del raw
    _release_raw_checkpoint_cache(cache)
    gc.collect()
    assert raw_ref() is None
    assert not cache["raw"]
    assert torch.equal(cache["fp16"]["converted"], torch.ones(8, dtype=torch.float16))
    assert cache["fp16"]["alias"] is alias


def test_release_raw_cache_rejects_incomplete_conversion():
    value = torch.ones(2)
    cache = {"raw": {"missing": value}, "fp16": {}}
    with pytest.raises(RuntimeError, match="missing"):
        _release_raw_checkpoint_cache(cache)
    assert cache["raw"]["missing"] is value
