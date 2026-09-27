"""Lifecycle and memory-update methods for agent re-entry."""

from .kmu_op import (
    KMU_OPERATIONS,
    KMU_OP_ARM,
    KMU_OP_METHOD,
    compile_indexed_kmu_operations,
    compile_kmu_operations,
    compile_router_kmu_operations,
    indexed_kmu_operation_prompt,
    kmu_operation_prompt,
)
from .temporal_lww import (
    TEMPORAL_LWW_ARM,
    TEMPORAL_LWW_METHOD,
    compile_router_temporal_lww,
    compile_temporal_lww,
    router_return_candidates,
)

__all__ = [
    "KMU_OPERATIONS",
    "KMU_OP_ARM",
    "KMU_OP_METHOD",
    "TEMPORAL_LWW_ARM",
    "TEMPORAL_LWW_METHOD",
    "compile_indexed_kmu_operations",
    "compile_kmu_operations",
    "compile_router_kmu_operations",
    "compile_router_temporal_lww",
    "compile_temporal_lww",
    "indexed_kmu_operation_prompt",
    "kmu_operation_prompt",
    "router_return_candidates",
]
