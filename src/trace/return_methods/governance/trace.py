"""Canonical public interface for TRACE.

The implementation remains in the benchmark-neutral governor and Router
bridge.  This module owns TRACE's public identity and provides the single
method-level entry point used by new experiment code.
"""

from __future__ import annotations

from typing import Any

from ..base import ReturnMethodSpec


TRACE_ARM = "trace"
TRACE_DISPLAY_NAME = "TRACE"
TRACE_METHOD = ReturnMethodSpec(
    method=TRACE_ARM,
    display_name=TRACE_DISPLAY_NAME,
    category="return_governance",
    produces_memory_view=True,
    implementation_kind="proposed_method",
    aliases=("review", "latch"),
)


def compile_trace(*args: Any, **kwargs: Any):
    """Compile TRACE from a frozen Router checkpoint.

    The lazy import prevents the public method catalog from depending on the
    Router implementation during package initialization.
    """

    from ...router_return_governance import compile_router_trace

    return compile_router_trace(*args, **kwargs)


__all__ = ["TRACE_ARM", "TRACE_DISPLAY_NAME", "TRACE_METHOD", "compile_trace"]
