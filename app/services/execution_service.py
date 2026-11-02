import threading
from typing import Any, Optional

from app.db_utils.defaults import DEFAULT_SQL_EXECUTION_TIMEOUT
from app.db_utils import execute_sql_for_data_item, measure_execution_time_for_data_item
from app.pipeline.utils import get_execution_result_hash
from app.tracing import trace_event
from ._bounded_cache import BoundedCache


DEFAULT_EXECUTION_RESULT_CACHE_SIZE = 4096
DEFAULT_EXECUTION_TIME_CACHE_SIZE = 4096

# 结果哈希只对不超过该行数的结果计算，避免超大结果拖慢执行
TRACE_HASH_ROW_LIMIT = 5000


class ExecutionService:
    def __init__(
        self,
        default_timeout: int = DEFAULT_SQL_EXECUTION_TIMEOUT,
        *,
        bigquery_credential_path: Optional[str] = None,
        snowflake_credential_path: Optional[str] = None,
        result_cache_size: int = DEFAULT_EXECUTION_RESULT_CACHE_SIZE,
        time_cache_size: int = DEFAULT_EXECUTION_TIME_CACHE_SIZE,
    ):
        self._result_cache: BoundedCache[tuple[Any, ...], Any] = BoundedCache(result_cache_size)
        self._time_cache: BoundedCache[tuple[Any, ...], float] = BoundedCache(time_cache_size)
        self._result_lock = threading.Lock()
        self._time_lock = threading.Lock()
        self._default_timeout = default_timeout
        self._bigquery_credential_path = bigquery_credential_path
        self._snowflake_credential_path = snowflake_credential_path

    def execute(self, data_item: Any, sql: str, timeout: Optional[int] = None, use_cache: bool = True):
        resolved_timeout = self._resolve_timeout(timeout)
        cache_key = self._build_result_key(data_item, sql, resolved_timeout)
        if use_cache:
            with self._result_lock:
                cached_result = self._result_cache.get(cache_key)
                if cached_result is not None:
                    self._trace_execution(data_item, sql, resolved_timeout, cached_result, cache_hit=True)
                    return cached_result

        result = execute_sql_for_data_item(
            data_item,
            sql,
            timeout=resolved_timeout,
            bigquery_credential_path=self._bigquery_credential_path,
            snowflake_credential_path=self._snowflake_credential_path,
        )
        if use_cache:
            with self._result_lock:
                self._result_cache.set(cache_key, result)
        self._trace_execution(data_item, sql, resolved_timeout, result, cache_hit=False)
        return result

    def measure_time(self, data_item: Any, sql: str, timeout: Optional[int] = None, repeat: int = 10, use_cache: bool = True) -> float:
        resolved_timeout = self._resolve_timeout(timeout)
        cache_key = self._build_time_key(data_item, sql, resolved_timeout, repeat)
        initial_execution_time = None
        if use_cache:
            with self._time_lock:
                cached_time = self._time_cache.get(cache_key)
                if cached_time is not None:
                    trace_event(
                        "sql_measure_time",
                        item_id=_get_item_id(data_item),
                        sql=sql,
                        timeout=resolved_timeout,
                        repeat=repeat,
                        execution_time=cached_time,
                        cache_hit=True,
                    )
                    return cached_time
            with self._result_lock:
                cached_result = self._result_cache.get(self._build_result_key(data_item, sql, resolved_timeout))
            if cached_result is not None and cached_result.result_rows is not None:
                initial_execution_time = cached_result.execution_time

        execution_time = measure_execution_time_for_data_item(
            data_item,
            sql,
            timeout=resolved_timeout,
            repeat=repeat,
            initial_execution_time=initial_execution_time,
        )
        if use_cache:
            with self._time_lock:
                self._time_cache.set(cache_key, execution_time)
        trace_event(
            "sql_measure_time",
            item_id=_get_item_id(data_item),
            sql=sql,
            timeout=resolved_timeout,
            repeat=repeat,
            execution_time=execution_time,
            cache_hit=False,
        )
        return execution_time

    def hash_result(self, data_item: Any, result_rows: Any) -> Any:
        return get_execution_result_hash(data_item, result_rows)

    def _trace_execution(self, data_item: Any, sql: str, timeout: int, result: Any, *, cache_hit: bool) -> None:
        """把一次 SQL 执行的完整结果写入执行痕迹。"""
        rows = getattr(result, "result_rows", None)
        row_count = len(rows) if rows is not None else None
        result_hash = None
        hash_skipped = None
        if rows is not None:
            if row_count is not None and row_count <= TRACE_HASH_ROW_LIMIT:
                try:
                    result_hash = str(self.hash_result(data_item, rows))
                except Exception as exc:
                    result_hash = f"<hash_error: {type(exc).__name__}: {exc}>"
            else:
                hash_skipped = f"row_count={row_count} exceeds limit {TRACE_HASH_ROW_LIMIT}"

        trace_event(
            "sql_execute",
            item_id=_get_item_id(data_item),
            sql=sql,
            timeout=timeout,
            cache_hit=cache_hit,
            result_type=getattr(result, "result_type", None),
            error_message=getattr(result, "error_message", None),
            execution_time=getattr(result, "execution_time", None),
            result_cols=getattr(result, "result_cols", None),
            row_count=row_count,
            rows_preview=rows[:5] if rows is not None else None,
            result_table_str=getattr(result, "result_table_str", None),
            result_hash=result_hash,
            hash_skipped=hash_skipped,
        )

    def reset(self) -> None:
        with self._result_lock:
            self._result_cache.clear()
        with self._time_lock:
            self._time_cache.clear()

    def _resolve_timeout(self, timeout: Optional[int]) -> int:
        return timeout if timeout is not None else self._default_timeout

    @staticmethod
    def _build_result_key(data_item: Any, sql: str, timeout: int) -> tuple[Any, ...]:
        return (
            getattr(data_item, "db_type", "sqlite"),
            data_item.database_path,
            sql,
            timeout,
        )

    @staticmethod
    def _build_time_key(data_item: Any, sql: str, timeout: int, repeat: int) -> tuple[Any, ...]:
        return (
            getattr(data_item, "db_type", "sqlite"),
            data_item.database_path,
            sql,
            timeout,
            repeat,
        )


_execution_service: ExecutionService | None = None


def _get_item_id(data_item: Any) -> Optional[str]:
    getter = getattr(data_item, "get_item_id", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return None
    return None


def configure_execution_service(
    *,
    default_timeout: int = DEFAULT_SQL_EXECUTION_TIMEOUT,
    bigquery_credential_path: Optional[str] = None,
    snowflake_credential_path: Optional[str] = None,
    result_cache_size: int = DEFAULT_EXECUTION_RESULT_CACHE_SIZE,
    time_cache_size: int = DEFAULT_EXECUTION_TIME_CACHE_SIZE,
) -> ExecutionService:
    global _execution_service
    _execution_service = ExecutionService(
        default_timeout=default_timeout,
        bigquery_credential_path=bigquery_credential_path,
        snowflake_credential_path=snowflake_credential_path,
        result_cache_size=result_cache_size,
        time_cache_size=time_cache_size,
    )
    return _execution_service


def get_execution_service() -> ExecutionService:
    global _execution_service
    if _execution_service is None:
        _execution_service = ExecutionService()
    return _execution_service


def reset_execution_service() -> None:
    global _execution_service
    if _execution_service is not None:
        _execution_service.reset()
        _execution_service = None
