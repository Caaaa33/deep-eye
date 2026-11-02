"""全流程执行追踪。

用法::

    from app.tracing import configure_tracing, trace_scope, trace_event

    configure_tracing(run_dir / "trace", stage="sql_generation")
    with trace_scope(item_id=item.get_item_id()):
        trace_event("generation_start", budgets={...})
"""

from .executor import submit_with_trace
from .tracer import (
    TRACE_DIR_ENV,
    TRACE_ENABLE_ENV,
    TraceWriter,
    configure_tracing,
    current_item_id,
    current_stage,
    get_tracer,
    reset_tracing,
    trace_enabled,
    trace_event,
    trace_scope,
)

__all__ = [
    "TraceWriter",
    "TRACE_DIR_ENV",
    "TRACE_ENABLE_ENV",
    "configure_tracing",
    "reset_tracing",
    "get_tracer",
    "trace_enabled",
    "trace_event",
    "trace_scope",
    "current_stage",
    "current_item_id",
    "submit_with_trace",
]
