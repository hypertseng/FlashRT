# Thor-specific end-to-end tests for Pi05 state-prompt graph reuse.

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

CKPT_PI05 = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT", "<ckpts>/pi05_libero_pytorch"
)
_GPU_AVAILABLE = torch.cuda.is_available()
_IS_THOR = (
    _GPU_AVAILABLE and "thor" in torch.cuda.get_device_name(0).lower()
)
_CKPT_AVAILABLE = os.path.isdir(CKPT_PI05)

_CHILD_COMPARE = r'''
import sys
import numpy as np
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
mode, ckpt, out_path = sys.argv[1:]
rng = np.random.RandomState(0)
obs = {
    "image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    "wrist_image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
}
states = [
    np.zeros(8, dtype=np.float32),
    np.ones(8, dtype=np.float32),
    np.full(8, 0.5, dtype=np.float32),
    np.linspace(-1.0, 1.0, 8).astype(np.float32),
    np.full(8, -0.5, dtype=np.float32),
]
fe = Pi05TorchFrontendThor(
    ckpt, num_views=2, use_fp8=False, state_prompt_mode=mode,
    num_steps=4, autotune=0,
)
capture_count = [0]
original_capture = fe._capture_enc_ae_graph
def counted_capture(*args, **kwargs):
    capture_count[0] += 1
    return original_capture(*args, **kwargs)
fe._capture_enc_ae_graph = counted_capture
actions, prompt_lengths, graph_lengths = [], [], []
for state in states:
    fe.set_prompt("pick up the cup", state=state)
    graph_lengths.append(int(fe.Se))
    if mode == "fixed":
        prompt_lengths.append(int(fe._enc_seqused.item()) - int(fe.sig_S))
    else:
        prompt_lengths.append(int(fe._S_lang))
    result = fe.infer(obs, seed=4321, state=state)
    actions.append(np.asarray(result["actions"], dtype=np.float32))
np.savez(
    out_path,
    actions=np.stack(actions),
    prompt_lengths=np.asarray(prompt_lengths),
    graph_lengths=np.asarray(graph_lengths),
    capture_count=np.asarray(capture_count[0]),
)
'''

_CHILD_SWITCH = r'''
import sys
import numpy as np
from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
ckpt, out_path = sys.argv[1:]
fe = Pi05TorchFrontendThor(
    ckpt, num_views=2, use_fp8=False, state_prompt_mode="fixed",
    num_steps=4, autotune=0,
)
rng = np.random.RandomState(17)
obs = {
    "image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    "wrist_image": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
}
state = np.zeros(8, dtype=np.float32)
flags, actions = [], []
for current_state in (state, None, state):
    fe.set_prompt("pick up the cup", state=current_state)
    flags.append(bool(fe._attn._fixed_shape))
    result = fe.infer(obs, seed=11, state=current_state)
    actions.append(np.asarray(result["actions"], dtype=np.float32))
np.savez(out_path, flags=np.asarray(flags), actions=np.stack(actions))
'''


def _run_child(source: str, *args: str) -> dict[str, np.ndarray]:
    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    handle.close()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", source, *args, handle.name],
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=300,
        )
        if proc.returncode:
            raise RuntimeError(
                f"Thor Pi05 child failed:\n{proc.stdout}\n{proc.stderr}"
            )
        with np.load(handle.name) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    finally:
        os.unlink(handle.name)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


@pytest.mark.skipif(not _IS_THOR, reason="Thor GPU required")
@pytest.mark.skipif(not _CKPT_AVAILABLE, reason=f"Pi0.5 checkpoint missing: {CKPT_PI05}")
def test_thor_fixed_prompt_reuses_one_graph_and_matches_exact():
    fixed = _run_child(_CHILD_COMPARE, "fixed", CKPT_PI05)
    exact = _run_child(_CHILD_COMPARE, "exact", CKPT_PI05)

    assert fixed["prompt_lengths"].tolist() == exact["prompt_lengths"].tolist()
    assert len(set(exact["prompt_lengths"].tolist())) >= 2
    assert len(set(fixed["graph_lengths"].tolist())) == 1
    assert len(set(exact["graph_lengths"].tolist())) >= 2
    assert int(fixed["capture_count"]) == 1
    assert int(exact["capture_count"]) == len(set(exact["graph_lengths"].tolist()))
    assert np.isfinite(fixed["actions"]).all()
    assert np.isfinite(exact["actions"]).all()
    for idx, (fixed_action, exact_action) in enumerate(
        zip(fixed["actions"], exact["actions"])
    ):
        cosine = _cosine(fixed_action, exact_action)
        assert cosine >= 0.9999, (
            f"state {idx}: fixed vs exact action cosine={cosine:.7f}"
        )


@pytest.mark.skipif(not _IS_THOR, reason="Thor GPU required")
@pytest.mark.skipif(not _CKPT_AVAILABLE, reason=f"Pi0.5 checkpoint missing: {CKPT_PI05}")
def test_thor_fixed_prompt_mode_switch_resets_mask_and_replays_deterministically():
    result = _run_child(_CHILD_SWITCH, CKPT_PI05)
    assert result["flags"].tolist() == [True, False, True]
    assert np.isfinite(result["actions"]).all()
    cosine = _cosine(result["actions"][0], result["actions"][2])
    assert cosine >= 0.9999, f"same input after mode switch cosine={cosine:.7f}"
