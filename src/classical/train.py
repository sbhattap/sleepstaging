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


def cv_lgbm(cfg: dict[str, Any], folds: Sequence[dict[str, Any]], seed: int,
            feats: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Subject-wise CV of one LightGBM config (YASA hyperparameters); returns (test predictions, run info)."""
    import lightgbm as lgb

    feats = F.load_features() if feats is None else feats
    cols = F.feature_columns(feats.columns, cfg["context"]) + list(cfg.get("extra_features", []))
    X, y = feats[cols].to_numpy(np.float32), feats["label"].to_numpy(np.int64)
    params = dict(cfg["params"])
    es = params.pop("early_stopping_rounds")
    cw = {int(k): float(v) for k, v in cfg["class_weights"].items()}
    set_seeds(seed)
    parts, info = [], []
    for fold in folds:
        t0, m = time.time(), S.fold_masks(fold, feats["subject"].to_numpy())
        clf = lgb.LGBMClassifier(objective="multiclass", class_weight=cw, random_state=seed, n_jobs=cfg["n_jobs"], **params)
        # The validation loss uses the same class weights, so early stopping optimises the training objective.
        clf.fit(X[m["train"]], y[m["train"]], eval_X=(X[m["val"]],), eval_y=(y[m["val"]],), eval_metric="multi_logloss",
                eval_sample_weight=[np.array([cw[int(c)] for c in y[m["val"]]])], callbacks=[lgb.early_stopping(es, verbose=False)])
        parts.append(predictions_frame(feats[m["test"]], fold["fold"], y[m["test"]], clf.predict_proba(X[m["test"]])))
        info.append(dict(fold=fold["fold"], best_iteration=int(clf.best_iteration_ or 0), seconds=round(time.time() - t0, 1)))
    return pd.concat(parts, ignore_index=True), dict(config=cfg, seed=seed, features=cols, folds=info)


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
