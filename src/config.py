"""Read the YAML configs in configs/ (base.yaml plus one file per experiment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "base") -> dict[str, Any]:
    """Return configs/<name>.yaml as a dict."""
    with open(REPO_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def repo_path(rel: str | Path) -> Path:
    """Resolve a repo-relative path from a config (absolute paths are returned unchanged)."""
    return REPO_ROOT / rel
