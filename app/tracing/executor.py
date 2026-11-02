"""线程池提交辅助：把 contextvars（stage / item_id）透传到工作线程。

``ThreadPoolExecutor`` 的工作线程不会继承提交线程的 contextvars，因此需要显式
``copy_context()``。注意同一个 Context 对象不能被两个线程同时使用，所以每次提交
都必须复制一份新的。
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

from .tracer import trace_enabled


def submit_with_trace(executor: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any):
    """等价于 ``executor.submit(fn, *args, **kwargs)``，但会携带当前追踪上下文。"""
    if executor is None:
        return fn(*args, **kwargs)
    if not trace_enabled():
        return executor.submit(fn, *args, **kwargs)

    ctx = contextvars.copy_context()

    def _runner() -> Any:
        return ctx.run(fn, *args, **kwargs)

    return executor.submit(_runner)


__all__ = ["submit_with_trace"]
