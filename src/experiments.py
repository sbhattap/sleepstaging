"""One entry point per experiment; each writes results/<experiment>/ and its rows of evaluation/summary.csv.

    exp1  main result: subject-wise 20-fold CV of every model in configs/exp1_main.yaml. Deep runs are imported
          from their saved training runs when present (src/deep/import_runs.py), otherwise trained (GPU).
    exp2  leakage: the same LightGBM under subject-wise vs epoch-wise random folds
    exp3  temporal context: LightGBM without context vs with context, and TinySleepNet
    exp4  missingness: artifact epochs dropped vs kept + flag feature, MNAR bounds, Movement epochs scored

CLI (also `make exp1` ... `make exp4`):  uv run python -m src.experiments exp1
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src import prepare_data as P
from src import splits as S
from src.classical import features as F
from src.classical import train as classical
from src.config import load_config, repo_path
from src.deep import train as deep
from src.deep.import_runs import import_run
from src.evaluate import PRED_KEYS, load_predictions, selection_audit, summary_row, write_summary
from src.runs import run_dir, save_run


def rel(path) -> str:
    return str(path.relative_to(repo_path(".")))


def exp1() -> None:
    rows = []
    for model, m in load_config("exp1_main")["models"].items():
        if m["kind"] == "lgbm":
            runs = [classical.run_model("exp1_main", model)]
        else:  # one run per seed, reported as the seed mean; the saved run was trained with exactly this config block
            runs = [import_run("exp1_main", model, s) if repo_path(m["import_from"][s]).exists() else deep.run_model("exp1_main", model, s)
                    for s in m["seeds"]]
        rows.append(summary_row("exp1_main", model, m["label"], [rel(r) for r in runs],
                                m.get("note", f"mean of {len(runs)} seeds" if len(runs) > 1 else "1 seed")))
    print(f"wrote {write_summary(rows)}")
    # AttnSleep's published numbers pick the epoch with the best TEST accuracy: measure what that rule is worth.
    folds, pooled = selection_audit(run_dir("exp1_main", "attnsleep", 42))
    folds.to_csv(repo_path("evaluation/attnsleep_selection_audit.csv"), index=False)
    repo_path("evaluation/attnsleep_selection_audit.json").write_text(json.dumps(pooled, indent=2))



def lgbm_run(experiment: str, name: str, cfg: dict, folds: list, feats: pd.DataFrame | None = None) -> str:
    """Run one LightGBM config and save it as results/<experiment>/<name>/; returns the repo-relative folder."""
    preds, info = classical.cv_lgbm(cfg, folds, load_config()["seed"], feats)
    save_run(out := run_dir(experiment, name), preds, dict(experiment=experiment, model=name, **info))
    return rel(out)


def exp2() -> None:
    c, seed = load_config("exp2_leakage"), load_config()["seed"]
    cfg, feats = load_config("exp1_main")["models"][c["model"]], F.load_features().sort_values(PRED_KEYS).reset_index(drop=True)
    folds = dict(subject=S.load_folds(), epoch=S.make_epochwise_folds(len(feats), c["epochwise"]["n_folds"], c["epochwise"]["val_frac"], seed))
    write_summary([summary_row("exp2_leakage", f"lgbm_{k}", label, [lgbm_run("exp2_leakage", f"lgbm_{k}", cfg, folds[k], feats)])
                   for k, label in c["splits"].items()])


def exp3() -> None:
    c, done = load_config("exp3_context"), pd.read_csv(repo_path("evaluation/summary.csv")).query("experiment == 'exp1_main'").set_index("model")
    cfg = load_config("exp1_main")["models"][c["model"]] | {"context": False}
    rows = [summary_row("exp3_context", "lgbm_nocontext", c["nocontext_label"], [lgbm_run("exp3_context", "lgbm_nocontext", cfg, S.load_folds())])]
    rows += [summary_row("exp3_context", m, label, done.loc[m, "runs"].split("+"), done.loc[m, "note"]) for m, label in c["reuse_from_exp1"].items()]
    write_summary(rows)


def exp4() -> None:
    c, base = load_config("exp4_missingness"), load_config()
    cfg = load_config("exp1_main")["models"][c["model"]]
    flags = pd.read_parquet(repo_path(base["paths"]["processed_dir"]) / "epoch_flags.parquet")
    full = F.load_features(protocol=False).merge(flags[PRED_KEYS + c["flag_features"]], on=PRED_KEYS)
    full[c["flag_features"]] = full[c["flag_features"]].astype(np.float32)
    feats = full[full.in_trim_window & (full.label >= 0)].reset_index(drop=True)
    main = load_predictions("results/exp1_main/lgbm")
    flagged = main[PRED_KEYS].merge(flags, on=PRED_KEYS)[c["flag_features"]].any(axis=1).to_numpy()
    flag_run = lgbm_run("exp4_missingness", "lgbm_flag", cfg | {"extra_features": c["flag_features"]}, S.load_folds(), feats)
    rows = [summary_row("exp4_missingness", "keep", "Keep flagged epochs, no flag feature (main model)", ["results/exp1_main/lgbm"]),
            summary_row("exp4_missingness", "drop", "(a) Drop flagged epochs from scoring", ["results/exp1_main/lgbm"],
                        f"{int(flagged.sum())} flagged epochs removed", [main[~flagged].reset_index(drop=True)]),
            summary_row("exp4_missingness", "flag_feature", "(b) Keep flagged epochs + flag features", [flag_run])]
    for stage, code in c["mnar_bounds"].items():  # MNAR sensitivity: what if every flagged epoch were truly `stage`?
        rows.append(summary_row("exp4_missingness", f"mnar_all_{stage}", f"MNAR bound: all flagged epochs are truly {stage}",
                                ["results/exp1_main/lgbm"], preds=[main.assign(y_true=np.where(flagged, code, main.y_true))]))
    # Movement epochs inside the scored window, predicted by the same model and scored as the previous / next stage.
    ctx = P.add_context(flags, pd.read_csv(repo_path(base["paths"]["reports_dir"]) / "recording_manifest.csv"))
    move = full.merge(ctx.loc[ctx.in_trim_window & (ctx.label == -1), PRED_KEYS + ["prev_stage", "next_stage"]], on=PRED_KEYS)
    extra = classical.cv_lgbm(cfg, S.load_folds(), base["seed"], feats.drop(columns=c["flag_features"]), extra=move)[1]["extra_preds"]
    codes = {v: k for k, v in P.STAGE.items()}
    for side in ("prev", "next"):
        lab = extra.merge(move[PRED_KEYS + [f"{side}_stage"]], on=PRED_KEYS)[f"{side}_stage"].map(codes)
        both = pd.concat([main, extra.assign(y_true=lab.to_numpy())]).sort_values(PRED_KEYS).reset_index(drop=True)
        rows.append(summary_row("exp4_missingness", f"movement_as_{side}", f"Movement epochs scored as the {side} stage",
                                ["results/exp1_main/lgbm"], f"{len(move)} Movement epochs added", [both]))
    write_summary(rows)


if __name__ == "__main__":
    {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4}[sys.argv[1]]()
