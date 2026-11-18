"""将 monorepo 内 autoapex 包加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIRS = [
    _REPO_ROOT / "common" / "src",
    _REPO_ROOT / "timeseries" / "src",
    _REPO_ROOT / "core" / "src",
    _REPO_ROOT / "features" / "src",
]


def setup_autoapex_path() -> Path:
    for d in _SRC_DIRS:
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    return _REPO_ROOT
