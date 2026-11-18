"""
stdlib random 补丁：mutiagent 的 Python 3.10.1 缺少 randbits，numpy 导入会失败。

通过 PYTHONPATH 将此目录置于最前，使 `from random import randbits` 可用。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
_stdlib_random_path = Path(sys.base_prefix) / "lib" / _py_ver / "random.py"

_spec = importlib.util.spec_from_file_location("_stdlib_random_impl", _stdlib_random_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载标准库 random: {_stdlib_random_path}")

_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

if not hasattr(_impl, "randbits"):
    _impl.randbits = _impl.getrandbits  # type: ignore[attr-defined]

for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)

randbits = _impl.randbits  # noqa: A001
getrandbits = _impl.getrandbits
