"""Data pipeline and splits: wake trim, label mapping, exclusion flow, published counts, no subject leakage."""

import json

import numpy as np
import pandas as pd
import pytest

from src import prepare_data as P
from src import splits as S
from src.config import load_config, repo_path

BASE = load_config()


def test_trim_mask_matches_deepsleepnet():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = np.zeros(400, dtype=np.int8)
        y[150:250] = rng.choice([1, 2, 3, 4, -1, -2], 100)
        y[rng.choice(400, 20, replace=False)] = -2
        y[160] = 2
        pos = np.flatnonzero(y >= 0)  # DeepSleepNet prepare_physionet.py: drop Movement / '?' first, then trim
        nw = np.flatnonzero(y[pos] != 0)
        ref = pos[max(nw[0] - 60, 0): min(nw[-1] + 60, len(pos) - 1) + 1]
        assert np.array_equal(np.flatnonzero(P.trim_mask(y, 60) & (y >= 0)), ref)


def test_hypnogram_epochs_maps_labels_and_drops_tail():
    h = P.hypnogram_epochs(np.array([0, 60, 120, 150]), np.array([60, 60, 30, 300]),
                           ["Sleep stage W", "Sleep stage 4", "Movement time", "Sleep stage ?"], 0.0, 200.0,
                           BASE["data"]["labels"]["map"], 30)
    assert h["y"].tolist() == [0, 0, 3, 3, -1, -2] and h["n_outside"] == 9  # stage 4 -> N3; 9 epochs past the signal


def test_artifact_flags():
    x = np.random.default_rng(0).normal(0, 20, (50, 3000)).astype(np.float32)
    x[3], x[7, 1000:1700], x[11, ::10], x[19] = 0.0, 5.0, 192.0, x[19] * 30  # flat, dropout, clipped, huge
    st = P.qc_flags(x, -192.0, 192.0, 100.0, BASE["data"]["qc"], np.ones(50, bool))
    assert st.flat[3] and st.flat[7] and st.clipped[11] and st.high_amp[19]
    assert not st.drop([3, 7, 11, 19])[["flat", "clipped", "high_amp"]].any(axis=None)


def test_exclusion_flow_is_consistent():
    flags = pd.DataFrame(dict(label=[0, 0, 1, 2, -1, -2, 0, 4], in_trim_window=np.array([0, 1, 1, 1, 1, 1, 1, 0], bool),
                              flat=False, clipped=False, high_amp=False))
    ef = P.exclusion_flow(flags, pd.DataFrame(dict(n_hyp_epochs=[10], n_outside_signal=[2])))
    assert ef.remaining.tolist()[:6] == [10, 8, 7, 6, 4, 4]  # past end, '?', Movement, wake trim


@pytest.mark.skipif(len(P.available_recordings(range(20))) != 39, reason="Sleep-EDF-20 not processed")
def test_processed_data_has_the_published_protocol_counts():
    d = P.load_dataset(range(20))
    counts = dict(zip(P.STAGE.values(), np.bincount(d["y"], minlength=5).tolist()))
    assert counts == BASE["data"]["expected_counts"] and len(np.unique(d["rec_id"])) == 39  # AttnSleep Table I


def test_subject_folds_never_share_a_subject():
    for n_folds in (20, 5):
        folds = S.make_subject_folds(list(range(20)), n_folds, 2, 42)
        for f in folds:
            tr, va, te = set(f["train_subjects"]), set(f["val_subjects"]), set(f["test_subjects"])
            assert not tr & te and not va & te and not tr & va
        assert sorted(s for f in folds for s in f["test_subjects"]) == list(range(20))
    leaky = S.make_subject_folds(list(range(20)), 20, 2, 42)
    leaky[3]["train_subjects"].append(3)
    with pytest.raises(AssertionError):
        S.validate_folds(leaky, list(range(20)))


def test_saved_folds_file_matches_config():
    path = repo_path(BASE["paths"]["folds_file"])
    if not path.exists():
        pytest.skip("run `uv run python -m src.splits` first")
    expected = S.make_subject_folds(list(range(20)), BASE["splits"]["n_folds"], BASE["splits"]["n_val_subjects"], BASE["seed"])
    assert [{k: f[k] for k in expected[0]} for f in json.loads(path.read_text())] == expected


def test_epochwise_split_is_a_partition_and_leaky():
    folds = S.make_epochwise_folds(2000, 20, 0.1, seed=0)
    assert np.array_equal(np.sort(np.concatenate([f["test_idx"] for f in folds])), np.arange(2000))
    assert all(not np.intersect1d(f["train_idx"], f["test_idx"]).size for f in folds)
