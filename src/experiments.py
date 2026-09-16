"""One entry point per experiment; each writes results/<experiment>/ and its rows of evaluation/summary.csv.

    exp1  main result: subject-wise 20-fold CV of every model in configs/exp1_main.yaml. Deep runs are imported
          from their saved training runs when present (src/deep/import_runs.py), otherwise trained (GPU).

CLI (also `make exp1`):  uv run python -m src.experiments exp1
"""

from __future__ import annotations

import json
import sys

from src.classical import train as classical
from src.config import load_config, repo_path
from src.deep import train as deep
from src.deep.import_runs import import_run
from src.evaluate import selection_audit, summary_row, write_summary
from src.runs import run_dir


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


if __name__ == "__main__":
    {"exp1": exp1}[sys.argv[1]]()
