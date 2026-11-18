#!/usr/bin/env python3
"""
FNSPID 快速模拟（已归档）。

请改用:
  python prior_enhance/data/fnspid/prepare_demo.py --symbol AAL --quick
  python prior_enhance/examples/run_fnspid_demo.py --symbol AAL --quick

本脚本保留为兼容入口，转发至 examples/archive/simulate_fnspid.py。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ARCHIVE = Path(__file__).resolve().parent / "archive" / "simulate_fnspid.py"
if not _ARCHIVE.is_file():
    print(f"已移除: 请使用 prepare_demo.py + run_fnspid_demo.py", file=sys.stderr)
    sys.exit(1)

runpy.run_path(str(_ARCHIVE), run_name="__main__")
