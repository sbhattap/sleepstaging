"""LightGBM under subject-wise CV: train on 17 subjects, early-stop on 2, test on the held-out one (same folds
as every model; validation subjects are used only for early stopping).

CLI:  uv run python -m src.classical.train --experiment exp1_main --model lgbm
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import splits as S
from src.classical import features as F
from src.config import load_config
from src.runs import predictions_frame, run_dir, save_run, set_seeds


def cv_lgbm(cfg: dict[str, Any], folds: Sequence[dict[str, Any]], seed: int, feats: pd.DataFrame | None = None,
            extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """CV of one LightGBM config (YASA hyperparameters); returns (test predictions, run info).

    Folds are subject-wise ({train,val,test}_subjects) or, for the leakage experiment, epoch indices ({...}_idx).
    ``extra`` epochs (e.g. Movement epochs) of each test subject are also predicted, into info["extra_preds"].
    """
    import lightgbm as lgb

    feats = F.load_features() if feats is None else feats
    extra_cols = list(cfg.get("extra_features", []))  # appended once, after the standard features
    cols = [c for c in F.feature_columns(feats.columns, cfg["context"]) if c not in extra_cols] + extra_cols
    X, y = feats[cols].to_numpy(np.float32), feats["label"].to_numpy(np.int64)
    params = dict(cfg["params"])
    es = params.pop("early_stopping_rounds")
    cw = {int(k): float(v) for k, v in cfg["class_weights"].items()}
    set_seeds(seed)
    parts, info, extra_parts = [], [], []
    for fold in folds:
        t0 = time.time()
        m = S.fold_masks(fold, feats["subject"].to_numpy()) if "test_subjects" in fold else \
            {k: np.isin(np.arange(len(y)), fold[f"{k}_idx"]) for k in ("train", "val", "test")}
        clf = lgb.LGBMClassifier(objective="multiclass", class_weight=cw, random_state=seed, n_jobs=cfg["n_jobs"], **params)
        # The validation loss uses the same class weights, so early stopping optimises the training objective.
        clf.fit(X[m["train"]], y[m["train"]], eval_X=(X[m["val"]],), eval_y=(y[m["val"]],), eval_metric="multi_logloss",
                eval_sample_weight=[np.array([cw[int(c)] for c in y[m["val"]]])], callbacks=[lgb.early_stopping(es, verbose=False)])
        parts.append(predictions_frame(feats[m["test"]], fold["fold"], y[m["test"]], clf.predict_proba(X[m["test"]])))
        info.append(dict(fold=fold["fold"], best_iteration=int(clf.best_iteration_ or 0), seconds=round(time.time() - t0, 1)))
        if extra is not None and len(e := extra[extra.subject.isin(fold.get("test_subjects", []))]):
            extra_parts.append(predictions_frame(e, fold["fold"], np.zeros(len(e)), clf.predict_proba(e[cols].to_numpy(np.float32))))
    out = dict(config=cfg, seed=seed, features=cols, folds=info)
    return pd.concat(parts, ignore_index=True), out | ({"extra_preds": pd.concat(extra_parts)} if extra is not None else {})


def run_model(experiment: str, model: str) -> Path:
    """Run one LightGBM model of an experiment config on every fold and save its run folder."""
    preds, info = cv_lgbm(load_config(experiment)["models"][model], S.load_folds(), load_config()["seed"])
    out = run_dir(experiment, model)
    save_run(out, preds, dict(experiment=experiment, model=model, **info))
    print(f"{experiment}/{model}: acc {(preds.y_true == preds.y_pred).mean():.4f} -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp1_main")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    run_model(a.experiment, a.model)
