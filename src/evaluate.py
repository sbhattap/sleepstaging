"""Pooled metrics with subject-bootstrap 95% CIs, and the shared results table evaluation/summary.csv.

Metrics are pooled over all test folds, as the papers report them. CIs resample SUBJECTS, not epochs: epochs of
one night are strongly correlated, so an epoch bootstrap would give CIs that are far too narrow. Every metric is a
function of the 5x5 confusion matrix, so a replicate is a sum of per-subject matrices (1,000 take milliseconds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import load_config, repo_path

CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")
N_CLASSES = 5
METRICS = ["acc", "mf1", "kappa"] + [f"f1_{c}" for c in CLASS_NAMES]
PRED_KEYS = ["subject", "night", "epoch_idx"]


def metrics_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    """acc, mf1 (over classes present), kappa and per-class f1 from confusion matrices (..., 5, 5), rows = true."""
    cm = np.asarray(cm, dtype=np.float64)
    tp, n_true, n_pred, n = np.diagonal(cm, axis1=-2, axis2=-1), cm.sum(-1), cm.sum(-2), cm.sum((-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        present = (n_true + n_pred) > 0
        f1 = np.where(present, 2 * tp / (n_true + n_pred), np.nan)
        acc, p_e = tp.sum(-1) / n, (n_true * n_pred).sum(-1) / n ** 2
        return dict(acc=acc, mf1=np.nansum(f1, -1) / np.maximum(present.sum(-1), 1), f1=f1,
                    kappa=np.where(np.isclose(p_e, 1.0), np.nan, (acc - p_e) / (1 - p_e)))


def subject_confusions(y_true: np.ndarray, y_pred: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sorted subject IDs, per-subject confusion matrices (n_subjects, 5, 5))."""
    yt, yp = np.asarray(y_true, np.int64), np.asarray(y_pred, np.int64)
    assert 0 <= min(yt.min(), yp.min()) and max(yt.max(), yp.max()) < N_CLASSES, "labels must be 0-4"
    ids, inv = np.unique(np.asarray(subject), return_inverse=True)
    return ids, np.bincount((inv * N_CLASSES + yt) * N_CLASSES + yp, minlength=len(ids) * 25).reshape(-1, 5, 5)


def pick(m: dict[str, np.ndarray], key: str) -> np.ndarray:
    return m[key] if not key.startswith("f1_") else m["f1"][..., CLASS_NAMES.index(key[3:])]


def bootstrap_runs(runs: Sequence[pd.DataFrame], n_boot: int, seed: int, ci_level: float = 0.95) -> dict[str, dict[str, float]]:
    """Metrics averaged over runs (training seeds) with a subject-bootstrap CI and the per-run range.

    Each replicate draws one set of subjects and scores every run on it, so with several seeds the CI is for the
    seed-averaged metric. All runs must cover the same epochs.
    """
    for df in runs:
        assert df[PRED_KEYS].equals(runs[0][PRED_KEYS]) and (df.y_true == runs[0].y_true).all(), "runs scored on different epochs"
    cms = np.stack([subject_confusions(df.y_true, df.y_pred, df.subject)[1] for df in runs])  # (runs, subjects, 5, 5)
    idx = np.random.default_rng(seed).integers(0, cms.shape[1], size=(n_boot, cms.shape[1]))
    boot, point, a = metrics_from_confusion(cms[:, idx].sum(2)), metrics_from_confusion(cms.sum(1)), (1 - ci_level) / 2
    out = {}
    for k in METRICS:
        lo, hi = np.nanquantile(np.nanmean(pick(boot, k), axis=0), [a, 1 - a])
        p = pick(point, k)
        out[k] = dict(point=float(p.mean()), lo=float(lo), hi=float(hi), seed_min=float(p.min()), seed_max=float(p.max()))
    return out


def load_predictions(run: Path | str) -> pd.DataFrame:
    """Pooled test predictions of one run folder, sorted by epoch; asserts every epoch was tested once."""
    df = pd.read_parquet(repo_path(run) / "predictions.parquet").sort_values(PRED_KEYS).reset_index(drop=True)
    assert not df.duplicated(PRED_KEYS).any(), "an epoch was tested twice: folds overlap"
    return df


def summary_row(experiment: str, model: str, label: str, runs: Sequence[str], note: str = "") -> dict[str, Any]:
    """One row of evaluation/summary.csv: point, 95% CI (and seed range) of every metric."""
    mc, preds = load_config()["metrics"], [load_predictions(r) for r in runs]
    row: dict[str, Any] = dict(experiment=experiment, model=model, label=label, note=note, n_seeds=len(runs),
                               runs="+".join(runs), n_epochs=len(preds[0]), n_subjects=int(preds[0].subject.nunique()))
    for k, v in bootstrap_runs(preds, mc["n_boot"], mc["boot_seed"], mc["ci_level"]).items():
        row.update({k: v["point"], f"{k}_lo": v["lo"], f"{k}_hi": v["hi"]})
        if len(runs) > 1:
            row.update({f"{k}_seed_min": v["seed_min"], f"{k}_seed_max": v["seed_max"]})
    return row


def write_summary(rows: Sequence[dict[str, Any]]) -> Path:
    """Replace these (experiment, model) rows in evaluation/summary.csv; other rows are kept."""
    path, new = repo_path(load_config()["paths"]["summary_file"]), pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.set_index(["experiment", "model"]).index.isin(new.set_index(["experiment", "model"]).index)], new])
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return path


def selection_audit(run: Path | str) -> tuple[pd.DataFrame, dict[str, float]]:
    """What choosing the epoch on the TEST fold (AttnSleep's rule) would have given, from the per-epoch test audit.

    Per fold: our epoch (best validation macro-F1) vs the best-test-accuracy epoch; pooled values are n-weighted.
    """
    preds, rows = load_predictions(run), []
    for k, g in preds.groupby("fold"):
        h = pd.read_csv(repo_path(run) / f"history_fold{k}.csv")
        v, t = int(h.val_mf1.idxmax()), int(h.audit_test_acc.idxmax())
        assert np.isclose(h.audit_test_acc[v], (g.y_true == g.y_pred).mean()), f"fold {k}: audit and predictions disagree"
        rows.append(dict(fold=int(k), n_test=len(g), epochs_run=len(h), val_epoch=v, test_epoch=t,
                         **{f"{s}_sel_{m}": h[f"audit_test_{m}"][e] for m in ("acc", "mf1", "kappa") for s, e in (("val", v), ("test", t))}))
    df = pd.DataFrame(rows)
    w = df.n_test / df.n_test.sum()
    pooled = dict(val_selected_acc=float((w * df.val_sel_acc).sum()), test_selected_acc=float((w * df.test_sel_acc).sum()),
                  val_selected_mf1_mean_fold=float(df.val_sel_mf1.mean()), test_selected_mf1_mean_fold=float(df.test_sel_mf1.mean()),
                  folds_val_epoch_before_lr_step=int((df.val_epoch < 10).sum()), max_epochs_run=int(df.epochs_run.max()))
    return df, {**pooled, "optimism_acc": pooled["test_selected_acc"] - pooled["val_selected_acc"]}
