"""One entry point per experiment; each writes results/<experiment>/ and its rows of evaluation/summary.csv.

    exp1  main result: subject-wise 20-fold CV of every model in configs/exp1_main.yaml

CLI (also `make exp1`):  uv run python -m src.experiments exp1
"""

from __future__ import annotations

import sys

from src.classical import train as classical
from src.config import load_config, repo_path
from src.evaluate import summary_row, write_summary


def rel(path) -> str:
    return str(path.relative_to(repo_path(".")))


def exp1() -> None:
    rows = [summary_row("exp1_main", model, m["label"], [rel(classical.run_model("exp1_main", model))], m.get("note", ""))
            for model, m in load_config("exp1_main")["models"].items()]
    print(f"wrote {write_summary(rows)}")


if __name__ == "__main__":
    {"exp1": exp1}[sys.argv[1]]()
