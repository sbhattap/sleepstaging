"""Shared by every model: seeds, the prediction format, and run folders.

Run folder results/<experiment>/<model>/ (deep models: .../seed<seed>/), git-ignored:
predictions.parquet (one row per test epoch: subject, night, epoch_idx, fold, y_true, y_pred, p_W ... p_REM)
and run_info.json (config, seed, per-fold details).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import repo_path
from src.evaluate import CLASS_NAMES


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch; deterministic cuDNN makes reruns bit-identical."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, not deterministic


def run_dir(experiment: str, model: str, seed: int | None = None) -> Path:
    return repo_path(f"results/{experiment}/{model}" + ("" if seed is None else f"/seed{seed}"))


def predictions_frame(meta: pd.DataFrame, fold: int, y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Test predictions of one fold in the shared format."""
    df = pd.DataFrame({k: np.asarray(meta[k]) for k in ("subject", "night", "epoch_idx")})
    df["fold"], df["y_true"], df["y_pred"] = fold, np.asarray(y_true, np.int8), proba.argmax(1).astype(np.int8)
    for i, c in enumerate(CLASS_NAMES):
        df[f"p_{c}"] = proba[:, i].astype(np.float32)
    return df


def save_run(out_dir: Path, preds: pd.DataFrame, info: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert not preds.duplicated(["subject", "night", "epoch_idx"]).any(), "an epoch was tested twice"
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
