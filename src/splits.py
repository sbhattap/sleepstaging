"""Subject-wise folds for Sleep-EDF-20, plus the deliberately leaky epoch-wise split used to measure leakage.

Split by subject, not by epoch: neighbouring 30-s epochs of one night look almost identical, so a random epoch
split puts near-copies of every test epoch in training. 20 subjects, 20 folds = leave one subject out; per
fold, 2 training subjects are held out for early stopping only.

CLI:  uv run python -m src.splits      # -> data/splits/sleepedf20_folds.json
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.config import load_config, repo_path


def make_subject_folds(subjects: list[int], n_folds: int, n_val: int, seed: int) -> list[dict[str, Any]]:
    """Folds {fold, test_subjects, val_subjects, train_subjects}; LOSO keeps subject order (fold k tests subject k).

    Validation subjects come from a (seed, fold) generator, so changing one fold never changes another.
    """
    subjects = sorted(subjects)
    groups = [[s] for s in subjects] if n_folds == len(subjects) else \
        [sorted(int(s) for s in g) for g in np.array_split(np.random.default_rng(seed).permutation(subjects), n_folds)]
    folds = []
    for k, test in enumerate(groups):
        rest = [s for s in subjects if s not in test]
        val = sorted(int(s) for s in np.random.default_rng([seed, k]).choice(rest, n_val, replace=False)) if n_val else []
        folds.append(dict(fold=k, test_subjects=test, val_subjects=val, train_subjects=[s for s in rest if s not in val]))
    validate_folds(folds, subjects)
    return folds


def validate_folds(folds: list[dict[str, Any]], subjects: list[int]) -> None:
    """Fail loudly if a subject is in two partitions of a fold, or is not tested exactly once overall."""
    for f in folds:
        tr, va, te = set(f["train_subjects"]), set(f["val_subjects"]), set(f["test_subjects"])
        assert not (tr & te or va & te or tr & va) and tr | va | te == set(subjects) and tr and te, f"fold {f['fold']} leaks"
    assert sorted(s for f in folds for s in f["test_subjects"]) == sorted(subjects), "every subject must be tested once"


def load_folds(path: str | None = None) -> list[dict[str, Any]]:
    """Read the folds file (default: paths.folds_file in base.yaml) and re-validate it."""
    folds = json.loads(repo_path(path or load_config()["paths"]["folds_file"]).read_text())
    validate_folds(folds, sorted(s for f in folds for s in f["test_subjects"]))
    return folds


def fold_masks(fold: dict[str, Any], subject: np.ndarray) -> dict[str, np.ndarray]:
    """Disjoint boolean train / val / test masks over a per-epoch subject array."""
    m = {k: np.isin(subject, fold[f"{k}_subjects"]) for k in ("train", "val", "test")}
    assert not (m["train"] & m["test"]).any() and not (m["val"] & m["test"]).any() and not (m["train"] & m["val"]).any()
    return m


def make_epochwise_folds(n_epochs: int, n_folds: int, val_frac: float, seed: int) -> list[dict[str, Any]]:
    """LEAKY on purpose (leakage experiment only): random k-fold over epochs, so every night is in train and test."""
    chunks = np.array_split(np.random.default_rng(seed).permutation(n_epochs), n_folds)
    folds = []
    for k, test in enumerate(chunks):
        rest = np.random.default_rng([seed, k]).permutation(np.concatenate([c for j, c in enumerate(chunks) if j != k]))
        n_val = int(round(val_frac * len(rest)))
        folds.append(dict(fold=k, train_idx=np.sort(rest[n_val:]), val_idx=np.sort(rest[:n_val]), test_idx=np.sort(test)))
    return folds


if __name__ == "__main__":
    base = load_config()
    folds = make_subject_folds(list(range(*base["data"]["subjects"])), base["splits"]["n_folds"], base["splits"]["n_val_subjects"], base["seed"])
    path = repo_path(base["paths"]["folds_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{**f, "dataset": "sleepedf20", "seed": base["seed"], "protocol": "subject-wise"} for f in folds], indent=2))
    print(f"{len(folds)} subject-wise folds -> {path}")
