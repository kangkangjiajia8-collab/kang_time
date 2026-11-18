"""环境兼容：在 import numpy 之前修补已知问题。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union


def patch_random_randbits() -> None:
    """mutiagent Python 3.10.1 缺少 random.randbits，numpy 依赖它。"""
    import random

    if not hasattr(random, "randbits"):
        random.randbits = random.getrandbits  # type: ignore[attr-defined]


def bootstrap(project_root: Optional[Union[str, Path]] = None) -> None:
    """示例脚本入口：randbits 补丁 + 可选 stdlib patch / sys.path。"""
    patch_random_randbits()
    if project_root is None:
        return
    root = Path(project_root)
    stdlib_patch = root / "prior_enhance" / "_stdlib_patch"
    if stdlib_patch.is_dir():
        patch_path = str(stdlib_patch)
        if patch_path not in sys.path:
            sys.path.insert(0, patch_path)
    for entry in (str(root), str(root / "prior_enhance")):
        if entry not in sys.path:
            sys.path.append(entry)


# 模块 import 时即补丁（供 prior_enhance.__init__ 复用）
patch_random_randbits()
