"""Import saved deep-model runs into results/<experiment>/<model>/seed<seed>/ with provenance, instead of retraining.

Checks: the saved run's resolved config equals the experiment config block (key by key); every protocol epoch is
present once, in the fold of its subject. provenance.json records the source and the SHA256 of every imported
file. --verify reloads each checkpoint with the current code and re-predicts every test subject.

CLI:  uv run python -m src.deep.import_runs --experiment exp1_main [--verify]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import splits as S
from src.config import load_config, repo_path
from src.deep.train import cv_deep
from src.evaluate import CLASS_NAMES, PRED_KEYS
from src.runs import run_dir, save_run

PROBS = [f"p_{c}" for c in CLASS_NAMES]


def import_run(experiment: str, model: str, seed: int) -> Path:
    """Copy one saved run (predictions, histories, checkpoints) into this experiment's layout."""
    cfg = load_config(experiment)["models"][model]
    src, ck_src = repo_path(cfg["import_from"][seed]), repo_path(cfg["import_checkpoints_from"][seed])
    saved = json.loads((src / "run_info.json").read_text())["config"]
    diffs = [f"{s}.{k}" for s in ("model", "seq", "train", "augment") for k, v in cfg[s].items() if saved[s].get(k) != v]
    diffs += [k for k in ("normalization",) if saved["data"][k] != cfg["data"][k]] + (["seed"] if saved["seed"] != seed else [])
    assert not diffs and saved["eval"]["track_test_each_epoch"] == cfg["eval"]["track_test_each_epoch"], f"{src}: config differs: {diffs}"
    preds, folds = pd.read_parquet(src / "predictions.parquet")[PRED_KEYS + ["fold", "y_true", "y_pred"] + PROBS], S.load_folds()
    test_of = {s: f["fold"] for f in folds for s in f["test_subjects"]}
    assert (preds.subject.map(test_of) == preds.fold).all(), "an epoch is not in the fold of its subject"
    counts = {c: int((preds.y_true == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    assert counts == load_config()["data"]["expected_counts"], f"{src}: not the protocol epochs: {counts}"
    dst, ck_dst, sha = run_dir(experiment, model, seed), repo_path(f"checkpoints/{experiment}/{model}/seed{seed}"), {}
    infos = [json.loads((src / f"metrics_fold{f['fold']}.json").read_text()) for f in folds]
    assert all(m["seed"] == seed + f["fold"] and m["test_subjects"] == f["test_subjects"] for m, f in zip(infos, folds))
    save_run(dst, preds, dict(experiment=experiment, model=model, config=cfg, seed=seed, imported_from=cfg["import_from"][seed],
                              folds=[{k: m[k] for k in ("fold", "seed", "best_epoch", "best_val_mf1", "epochs_run")} for m in infos]))
    ck_dst.mkdir(parents=True, exist_ok=True)
    for f in folds:
        k = f["fold"]
        for a, b in ((src / f"history_fold{k}.csv", dst / f"history_fold{k}.csv"), (ck_src / f"fold{k}_best.pt", ck_dst / f"fold{k}.pt")):
            shutil.copy2(a, b)
            sha[str(b.relative_to(repo_path(".")))] = hashlib.sha256(b.read_bytes()).hexdigest()
    sha["source predictions.parquet"] = hashlib.sha256((src / "predictions.parquet").read_bytes()).hexdigest()
    (dst / "provenance.json").write_text(json.dumps(dict(source=cfg["import_from"][seed], config_check="identical", sha256=sha), indent=2))
    return dst


def verify_run(experiment: str, model: str, seed: int) -> dict:
    """Re-predict every test fold from the imported checkpoints with the current code; compare with the import."""
    dst = run_dir(experiment, model, seed)
    new, _ = cv_deep(load_config(experiment)["models"][model], S.load_folds(), seed,
                     repo_path(f"checkpoints/{experiment}/{model}/seed{seed}"), dst / "verify", eval_only=True)
    shutil.rmtree(dst / "verify")
    old, new = (d.sort_values(PRED_KEYS).reset_index(drop=True) for d in (pd.read_parquet(dst / "predictions.parquet"), new))
    assert new[PRED_KEYS].equals(old[PRED_KEYS])
    res = dict(model=model, seed=seed, max_abs_prob_diff=float(np.abs(new[PROBS].to_numpy() - old[PROBS].to_numpy()).max()),
               n_label_changes=int((new.y_pred != old.y_pred).sum()))
    (dst / "verify.json").write_text(json.dumps(res, indent=2))
    print(res)
    return res


if __name__ == "__main__":
    exp = sys.argv[sys.argv.index("--experiment") + 1] if "--experiment" in sys.argv else "exp1_main"
    for model, cfg in load_config(exp)["models"].items():
        for seed in cfg.get("import_from", {}):
            import_run(exp, model, seed)
            if "--verify" in sys.argv:
                verify_run(exp, model, seed)
