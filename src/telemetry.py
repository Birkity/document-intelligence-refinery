"""Best-effort LangSmith tracing helpers for the refinery pipeline.

Tracing stays completely opt-in:

- if ``langsmith`` is not installed, calls become no-ops
- if ``LANGSMITH_TRACING`` is not ``"true"``, calls become no-ops

The repo's provider routing remains unchanged; these helpers only wrap the
existing execution path with observability spans.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

try:
    from langsmith.run_helpers import trace as _langsmith_trace
except ImportError:  # pragma: no cover - optional dependency fallback
    _langsmith_trace = None


def tracing_enabled() -> bool:
    """Return ``True`` when LangSmith tracing is explicitly enabled."""
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


@contextmanager
def traced(
    name: str,
    run_type: str = "chain",
    *,
    inputs: dict[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Yield a LangSmith run context when tracing is enabled, else ``None``."""
    if _langsmith_trace is None or not tracing_enabled():
        yield None
        return

    with _langsmith_trace(
        name,
        run_type=run_type,
        inputs=inputs,
        metadata=metadata,
        tags=tags,
    ) as run:
        yield run
