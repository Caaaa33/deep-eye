"""全流程执行追踪（Trace）核心。

设计目标：在不影响主流程的前提下，把程序每个阶段的细节原样落盘，便于事后分析。

落盘形式：run 目录下的 ``trace/events.jsonl``，每行一个 JSON 事件::

    {"seq": 12, "ts": "2026-10-30T21:00:00.123+08:00", "elapsed_ms": 1234.5,
     "run_id": "20261030T205900", "stage": "sql_generation", "item_id": "42",
     "event": "llm_response", "data": {...}}

约定：
    * ``stage`` / ``item_id`` 默认取自 contextvars，可由 ``trace_scope`` 或显式参数覆盖；
    * ``data`` 为事件载荷，递归序列化 pydantic / dataclass / Path / numpy 等对象；
    * 无 tracer 时 ``trace_event`` 为 no-op，因此埋点可以无条件写。

开关：环境变量 ``DEEPEYE_TRACE=0`` 关闭；``DEEPEYE_TRACE_DIR`` 覆盖落盘目录。
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import dataclasses
import datetime as _dt
import itertools
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from app.logger import logger


TRACE_ENABLE_ENV = "DEEPEYE_TRACE"
TRACE_DIR_ENV = "DEEPEYE_TRACE_DIR"
DEFAULT_EVENT_FILENAME = "events.jsonl"
DEFAULT_META_FILENAME = "meta.json"

# 单个字符串字段上限。prompt / 模型输出 / SQL 一般远小于该值，超出才截断并标记
_MAX_STR = 40000
_MAX_LIST = 500
_MAX_DEPTH = 8
_MAX_REPR = 2000


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int = _MAX_STR) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _jsonable(value: Any, depth: int = 0) -> Any:
    """把任意对象转换为可 JSON 序列化的结构。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if depth >= _MAX_DEPTH:
        return _truncate(repr(value), _MAX_REPR)

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return _truncate(value.decode("utf-8", errors="replace"))

    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _truncate(str(value), _MAX_REPR),
            "repr": _truncate(repr(value), _MAX_REPR),
        }

    # pydantic v2 / v1
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(), depth + 1)
        except Exception:
            pass
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict) and not isinstance(value, type):
        try:
            dumped = legacy_dict()
            if isinstance(dumped, dict):
                return _jsonable(dumped, depth + 1)
        except Exception:
            pass

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _jsonable(dataclasses.asdict(value), depth + 1)
        except Exception:
            pass

    if isinstance(value, dict):
        result = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _MAX_LIST:
                result["__truncated__"] = f"{len(value) - _MAX_LIST} more items"
                break
            result[str(key)] = _jsonable(item, depth + 1)
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [_jsonable(item, depth + 1) for item in items[:_MAX_LIST]]
        if len(items) > _MAX_LIST:
            result.append(f"<truncated {len(items) - _MAX_LIST} items>")
        return result

    # numpy 标量 / 数组
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist(), depth + 1)
        except Exception:
            pass

    if callable(value):
        return f"<callable {getattr(value, '__name__', type(value).__name__)}>"

    return _truncate(repr(value), _MAX_REPR)


# ---------------------------------------------------------------------------
# 落盘线程
# ---------------------------------------------------------------------------
def _resume_seq(path: Path) -> int:
    """读取已有 events.jsonl 的最后一条 seq，返回下一个可用编号（从文件尾部反向找）。"""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return 1
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            chunk = b""
            block = 64 * 1024
            while pos > 0:
                step = min(block, pos)
                pos -= step
                fh.seek(pos)
                chunk = fh.read(step) + chunk
                lines = [ln for ln in chunk.splitlines() if ln.strip()]
                if len(lines) >= 2 or pos == 0:
                    break
            last_line = lines[-1].decode("utf-8", errors="replace")
        return int(json.loads(last_line).get("seq", 0)) + 1
    except Exception:
        return 1


class TraceWriter:
    """后台线程逐行写 JSONL，避免阻塞主流程。"""

    def __init__(
        self,
        root: Path | str,
        *,
        run_id: Optional[str] = None,
        filename: str = DEFAULT_EVENT_FILENAME,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / filename
        self.run_id = run_id or _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        self._started_at = time.monotonic()
        self._started_wall = _dt.datetime.now().astimezone()
        # 同一 run 目录下各阶段是独立进程，seq 接着已有文件继续，保证整个 run 全局连续
        self._seq = itertools.count(_resume_seq(self.path))
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, name="deepeye-trace-writer", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        self._write_meta()

    # -- public -----------------------------------------------------------
    def emit(self, record: Dict[str, Any]) -> None:
        if self._closed:
            return
        record["seq"] = next(self._seq)
        record["ts"] = _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
        record["elapsed_ms"] = round((time.monotonic() - self._started_at) * 1000, 1)
        record.setdefault("run_id", self.run_id)
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception as exc:  # pragma: no cover - 兜底
            line = json.dumps(
                {
                    "seq": record.get("seq"),
                    "ts": record.get("ts"),
                    "run_id": self.run_id,
                    "event": "trace_serialize_error",
                    "data": {"error": repr(exc), "event": str(record.get("event"))},
                },
                ensure_ascii=False,
            )
        self._queue.put(line)

    def flush(self) -> None:
        """等待队列中的事件全部落盘（用于测试与关键节点）。"""
        try:
            self._queue.join()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        try:
            self._thread.join(timeout=10)
        except Exception:  # pragma: no cover
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    # -- internal ---------------------------------------------------------
    def _drain(self) -> None:
        try:
            with open(self.path, "a", encoding="utf-8", buffering=1) as handle:
                while True:
                    line = self._queue.get()
                    try:
                        if line is None:
                            handle.flush()
                            return
                        handle.write(line + "\n")
                    finally:
                        self._queue.task_done()
        except Exception as exc:  # pragma: no cover
            logger.error(f"[trace] writer thread failed: {exc}")
            traceback.print_exc()

    def _write_meta(self) -> None:
        meta = {
            "run_id": self.run_id,
            "started_at": self._started_wall.isoformat(timespec="seconds"),
            "root": str(self.root),
            "events_file": str(self.path),
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
            "argv": sys.argv,
        }
        try:
            with open(self.root / DEFAULT_META_FILENAME, "w", encoding="utf-8") as handle:
                json.dump(_jsonable(meta), handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[trace] failed to write meta: {exc}")


# ---------------------------------------------------------------------------
# 全局 tracer + 上下文
# ---------------------------------------------------------------------------
_tracer: Optional[TraceWriter] = None
_stage_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("deepeye_trace_stage", default=None)
_item_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("deepeye_trace_item", default=None)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", "disable", "disabled"}


def trace_enabled() -> bool:
    return _tracer is not None


def get_tracer() -> Optional[TraceWriter]:
    return _tracer


def configure_tracing(
    root: Path | str | None = None,
    *,
    stage: Optional[str] = None,
    run_id: Optional[str] = None,
    enabled: Optional[bool] = None,
    force: bool = False,
) -> Optional[TraceWriter]:
    """初始化（或复用）全局 tracer。

    ``root`` 一般为 run 目录下的 ``trace`` 子目录；若已有指向同一目录的 tracer
    则直接复用，便于同一进程内顺序跑多个 stage。
    """
    global _tracer

    if enabled is None:
        enabled = _env_flag(TRACE_ENABLE_ENV, default=True)
    if not enabled:
        reset_tracing()
        return None

    env_root = os.environ.get(TRACE_DIR_ENV)
    resolved_root = Path(env_root) if env_root else (Path(root) if root is not None else Path("trace"))

    if _tracer is not None and not force:
        try:
            same = _tracer.root.resolve() == resolved_root.resolve() and not _tracer.closed
        except Exception:
            same = False
        if same:
            if stage is not None:
                _stage_var.set(stage)
            return _tracer
        _tracer.close()

    _tracer = TraceWriter(resolved_root, run_id=run_id)
    if stage is not None:
        _stage_var.set(stage)
    # 同一 run 目录下多阶段会追加到同一个 events.jsonl，这里记一条会话起点便于切分
    trace_event(
        "trace_session_start",
        stage=stage,
        events_file=str(_tracer.path),
        pid=os.getpid(),
        argv=sys.argv,
    )
    logger.info(f"[trace] enabled: stage={stage}, events -> {_tracer.path}")
    return _tracer


def reset_tracing() -> None:
    global _tracer
    if _tracer is not None:
        _tracer.close()
        _tracer = None


def current_stage() -> Optional[str]:
    return _stage_var.get()


def current_item_id() -> Optional[str]:
    return _item_var.get()


@contextlib.contextmanager
def trace_scope(stage: Optional[str] = None, item_id: Any = None) -> Iterator[None]:
    """在 with 块内设置 stage / item_id 上下文。"""
    stage_token = _stage_var.set(stage) if stage is not None else None
    item_token = _item_var.set(str(item_id)) if item_id is not None else None
    try:
        yield
    finally:
        if item_token is not None:
            _item_var.reset(item_token)
        if stage_token is not None:
            _stage_var.reset(stage_token)


def trace_event(event: str, *, stage: Optional[str] = None, item_id: Any = None, **fields: Any) -> None:
    """记录一个事件。无 tracer 时直接返回。"""
    tracer = _tracer
    if tracer is None:
        return

    resolved_item = str(item_id) if item_id is not None else _item_var.get()
    record: Dict[str, Any] = {
        "event": event,
        "stage": stage if stage is not None else _stage_var.get(),
        "item_id": resolved_item,
    }
    if fields:
        record["data"] = _jsonable(fields)
    tracer.emit(record)


__all__ = [
    "TraceWriter",
    "TRACE_ENABLE_ENV",
    "TRACE_DIR_ENV",
    "configure_tracing",
    "reset_tracing",
    "get_tracer",
    "trace_enabled",
    "trace_event",
    "trace_scope",
    "current_stage",
    "current_item_id",
]