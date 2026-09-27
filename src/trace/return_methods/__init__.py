"""Modular RETURN-method interfaces."""

from .base import ReturnCandidate, ReturnMethodCompilation, ReturnMethodSpec
from .governance import TRACE_ARM, TRACE_METHOD, compile_trace
from .lifecycle import KMU_OP_ARM, TEMPORAL_LWW_ARM
from .semantic_state import (
    CUPMEM_ADAPTER_ARM,
    CUPMEM_ARM,
    CUPMEM_FULL_ARM,
    CUPMEM_SYSTEM_ARM,
)
from .temporal import MEMSTRATA_ARM
from .transactional import MEMTX_ARM
from .prompt_defense import (
    COGNITIVE_ANCHORING_ARM,
    NO_DEFENSE_ARM,
    SOURCE_SCRUTINY_ARM,
)
from .registry import (
    ANSWER_TIME_CONTROL_ARMS,
    DIAGNOSTIC_ADAPTER_ARMS,
    FULL_MEMORY_SYSTEM_ARMS,
    LIFECYCLE_POLICY_ARMS,
    PROMPT_DEFENSE_ARMS,
    PUBLISHED_GOVERNANCE_ARMS,
    RETURN_METHOD_REGISTRY,
    canonical_method_name,
    get_return_method,
    method_display_name,
    prompt_defense_for,
)

__all__ = [
    "ANSWER_TIME_CONTROL_ARMS",
    "COGNITIVE_ANCHORING_ARM",
    "CUPMEM_ADAPTER_ARM",
    "CUPMEM_ARM",
    "CUPMEM_FULL_ARM",
    "CUPMEM_SYSTEM_ARM",
    "DIAGNOSTIC_ADAPTER_ARMS",
    "FULL_MEMORY_SYSTEM_ARMS",
    "KMU_OP_ARM",
    "LIFECYCLE_POLICY_ARMS",
    "MEMSTRATA_ARM",
    "MEMTX_ARM",
    "NO_DEFENSE_ARM",
    "PROMPT_DEFENSE_ARMS",
    "PUBLISHED_GOVERNANCE_ARMS",
    "RETURN_METHOD_REGISTRY",
    "ReturnCandidate",
    "ReturnMethodCompilation",
    "ReturnMethodSpec",
    "SOURCE_SCRUTINY_ARM",
    "TEMPORAL_LWW_ARM",
    "TRACE_ARM",
    "TRACE_METHOD",
    "canonical_method_name",
    "compile_trace",
    "get_return_method",
    "prompt_defense_for",
    "method_display_name",
]
