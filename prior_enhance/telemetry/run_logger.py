"""单次运行的结构化透明日志。"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

_RUN_LOGGER: ContextVar[Optional["RunLogger"]] = ContextVar("prior_enhance_run_logger", default=None)

_SENSITIVE_KEYS = frozenset(
    {"api_key", "authorization", "token", "password", "secret", "dashscope_api_key"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any, max_str: int = 2000) -> Any:
    """将对象转为可 JSON 序列化的结构，并截断过长字符串。"""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        if len(obj) > max_str:
            return obj[:max_str] + f"...[+{len(obj) - max_str} chars]"
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x, max_str=max_str) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            if key.lower() in _SENSITIVE_KEYS:
                out[key] = "***REDACTED***"
            else:
                out[key] = _safe_json(v, max_str=max_str)
        return out
    if hasattr(obj, "model_dump"):
        return _safe_json(obj.model_dump(), max_str=max_str)
    return _safe_json(str(obj), max_str=max_str)


class RunLogger:
    """
    单次 pipeline 运行的透明日志器。

    输出:
      - {run_id}.log          人类可读，带阶段标记
      - {run_id}_audit.jsonl  机器可读，逐步审计
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        log_dir: Optional[str] = None,
        console: bool = True,
        level: int = logging.INFO,
    ):
        self.run_id = run_id or self._new_run_id()
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "outputs", "logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, f"{self.run_id}.log")
        self.audit_path = os.path.join(self.log_dir, f"{self.run_id}_audit.jsonl")

        self._logger = logging.getLogger(f"prior_enhance.run.{self.run_id}")
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._logger.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

        self._step_stack: list[str] = []
        self._artifacts: Dict[str, str] = {}

        self.info("RunLogger 初始化", phase="SYSTEM", event="run_start", data={
            "run_id": self.run_id,
            "log_path": self.log_path,
            "audit_path": self.audit_path,
        })

    @staticmethod
    def _new_run_id() -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{ts}_{uuid.uuid4().hex[:8]}"

    def _write_audit(
        self,
        event: str,
        phase: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> None:
        record = {
            "ts": _utc_now_iso(),
            "run_id": self.run_id,
            "phase": phase,
            "step": " > ".join(self._step_stack) if self._step_stack else "",
            "event": event,
            "level": level,
            "message": message,
            "data": _safe_json(data or {}),
        }
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log(
        self,
        level: str,
        message: str,
        *,
        phase: str = "GENERAL",
        event: str = "log",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        prefix = f"[{phase}]"
        if self._step_stack:
            prefix += f"[{'/'.join(self._step_stack)}]"
        full_msg = f"{prefix} {message}"
        getattr(self._logger, level.lower())(full_msg)
        self._write_audit(event, phase, message, data=data, level=level)

    def debug(self, message: str, **kwargs) -> None:
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log("ERROR", message, **kwargs)

    @contextmanager
    def step(self, name: str, phase: str = "GENERAL", **start_data) -> Iterator[None]:
        """标记一个可嵌套的 pipeline 步骤。"""
        self._step_stack.append(name)
        t0 = time.perf_counter()
        self.info(f"▶ 开始: {name}", phase=phase, event="step_start", data=start_data)
        try:
            yield
            elapsed = time.perf_counter() - t0
            self.info(
                f"✓ 完成: {name} ({elapsed:.2f}s)",
                phase=phase,
                event="step_end",
                data={"elapsed_sec": round(elapsed, 3), **start_data},
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self.error(
                f"✗ 失败: {name} ({elapsed:.2f}s) — {exc}",
                phase=phase,
                event="step_error",
                data={
                    "elapsed_sec": round(elapsed, 3),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    **start_data,
                },
            )
            raise
        finally:
            self._step_stack.pop()

    def log_input(self, name: str, payload: Any, phase: str = "GENERAL") -> None:
        self.info(f"输入 | {name}", phase=phase, event="input", data={"name": name, "value": payload})

    def log_output(self, name: str, payload: Any, phase: str = "GENERAL") -> None:
        self.info(f"输出 | {name}", phase=phase, event="output", data={"name": name, "value": payload})

    def log_transform(
        self,
        name: str,
        before: Any,
        after: Any,
        phase: str = "GENERAL",
        notes: Optional[str] = None,
    ) -> None:
        data = {"name": name, "before": before, "after": after}
        if notes:
            data["notes"] = notes
        self.info(f"变换 | {name}", phase=phase, event="transform", data=data)

    def log_artifact(self, name: str, path: str, phase: str = "GENERAL") -> None:
        self._artifacts[name] = path
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        self.info(
            f"产物 | {name} → {path}",
            phase=phase,
            event="artifact",
            data={"name": name, "path": path, "exists": exists, "size_bytes": size},
        )

    def log_config(self, config: Any, phase: str = "SYSTEM") -> None:
        self.info("运行配置", phase=phase, event="config", data={"config": config})

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        self.info(
            "运行结束",
            phase="SYSTEM",
            event="run_end",
            data={"summary": summary or {}, "artifacts": self._artifacts},
        )
        return {
            "run_id": self.run_id,
            "log_path": self.log_path,
            "audit_path": self.audit_path,
            "artifacts": dict(self._artifacts),
        }


def get_run_logger() -> Optional[RunLogger]:
    return _RUN_LOGGER.get()


def bind_run_logger(logger: Optional[RunLogger]):
    """绑定当前上下文 RunLogger，返回 reset token。"""
    return _RUN_LOGGER.set(logger)


def reset_run_logger(token) -> None:
    _RUN_LOGGER.reset(token)


class _LoggerProxy:
    """模块级便捷接口：有 RunLogger 则写入，否则写入 root prior_enhance logger。"""

    _fallback = logging.getLogger("prior_enhance")

    def info(self, msg: str, *, phase: str = "GENERAL", event: str = "log", data: Optional[dict] = None):
        rl = get_run_logger()
        if rl:
            rl.info(msg, phase=phase, event=event, data=data)
        else:
            self._fallback.info(msg)

    def debug(self, msg: str, **kwargs):
        rl = get_run_logger()
        if rl:
            rl.debug(msg, **kwargs)
        else:
            self._fallback.debug(msg)

    def warning(self, msg: str, **kwargs):
        rl = get_run_logger()
        if rl:
            rl.warning(msg, **kwargs)
        else:
            self._fallback.warning(msg)

    def error(self, msg: str, **kwargs):
        rl = get_run_logger()
        if rl:
            rl.error(msg, **kwargs)
        else:
            self._fallback.error(msg)

    def log_input(self, name: str, payload: Any, phase: str = "GENERAL"):
        rl = get_run_logger()
        if rl:
            rl.log_input(name, payload, phase=phase)

    def log_output(self, name: str, payload: Any, phase: str = "GENERAL"):
        rl = get_run_logger()
        if rl:
            rl.log_output(name, payload, phase=phase)

    def log_transform(self, name: str, before: Any, after: Any, phase: str = "GENERAL", notes: str = None):
        rl = get_run_logger()
        if rl:
            rl.log_transform(name, before, after, phase=phase, notes=notes)

    def log_artifact(self, name: str, path: str, phase: str = "GENERAL"):
        rl = get_run_logger()
        if rl:
            rl.log_artifact(name, path, phase=phase)


log = _LoggerProxy()
