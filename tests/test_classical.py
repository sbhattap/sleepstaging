"""Features (finite, per-recording, equal to YASA's) and metrics (equal to sklearn; bootstrap over subjects)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src import evaluate as E
from src.classical import features as F
from src.config import load_config

CFG = load_config()["features"]


def _eeg(n, seed):
    t = np.arange(n * 3000) / 100.0
    s = 30 * np.sin(2 * np.pi * 1.5 * t) + 10 * np.sin(2 * np.pi * 10 * t) + 5 * np.random.default_rng(seed).standard_normal(t.size)
    return s.reshape(n, 3000).astype(np.float32)


def test_features_finite_and_equal_to_yasa():
    import mne
    import yasa

    x = _eeg(30, 4)
    x[5] = 0.0  # a flat epoch must still give finite features
    ours = F.signal_features(x, 100.0, CFG)
    assert np.isfinite(ours.to_numpy()).all()
    x[5] = _eeg(1, 9)[0]
    ours = F.signal_features(x, 100.0, CFG)
    raw = mne.io.RawArray(x.reshape(1, -1) * 1e-6, mne.create_info([CFG["channel"]], 100.0, ["eeg"]), verbose=False)
    ref = yasa.SleepStaging(raw, eeg_name=CFG["channel"]).get_features()
    for name in ["std", "iqr", "skew", "kurt", "nzc", "hmob", "hcomp", "sdelta", "theta", "alpha", "sigma", "dt", "abspow", "perm", "higuchi"]:
        np.testing.assert_allclose(ours[f"eeg_{name}"], ref[f"eeg_{name}"], rtol=2e-3, atol=1e-6, err_msg=name)


def test_feature_columns_never_include_labels_or_ids():
    cols = F.META_COLS + list(F.signal_features(_eeg(20, 1), 100.0, CFG).columns)
    noctx, ctx = F.feature_columns(cols, False), F.feature_columns(cols, True)
    assert set(noctx) < set(ctx) and not set(F.META_COLS) & set(ctx) and "time_hour" in ctx and "time_hour" not in noctx


def _preds(seed, noise=0.3, n_subjects=6, n=200):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n_subjects * n)
    return pd.DataFrame(dict(subject=np.repeat(np.arange(n_subjects), n), night=1, epoch_idx=np.tile(np.arange(n), n_subjects),
                             y_true=y, y_pred=np.where(rng.random(len(y)) < noise, rng.integers(0, 5, len(y)), y)))


def test_metrics_match_sklearn():
    d = _preds(0).query("y_true != 3")  # an absent class is left out of macro-F1
    m = E.metrics_from_confusion(E.subject_confusions(d.y_true, d.y_pred, d.subject)[1].sum(0))
    assert m["acc"] == pytest.approx(accuracy_score(d.y_true, d.y_pred)) and m["kappa"] == pytest.approx(cohen_kappa_score(d.y_true, d.y_pred))
    assert m["mf1"] == pytest.approx(f1_score(d.y_true, d.y_pred, average="macro", labels=sorted(set(d.y_true) | set(d.y_pred))))


def test_bootstrap_resamples_subjects_and_averages_seeds():
    one = _preds(1, n_subjects=1)
    same = pd.concat([one.assign(subject=s) for s in range(5)], ignore_index=True)  # identical subjects -> zero-width CI
    r = E.bootstrap_runs([same], 200, 0)["acc"]
    assert r["lo"] == pytest.approx(r["hi"])
    a, b = _preds(3), _preds(3, noise=0.6)
    ra, rb, rab = (E.bootstrap_runs(x, 100, 0)["acc"] for x in ([a], [b], [a, b]))
    assert rab["point"] == pytest.approx((ra["point"] + rb["point"]) / 2) and rab["seed_min"] == pytest.approx(rb["point"])
