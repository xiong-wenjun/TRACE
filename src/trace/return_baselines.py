"""Lifecycle baselines compiled from one frozen RETURN checkpoint.

The implementations in this module deliberately consume only actor-visible
state.  They never inspect the oracle manifest or TRACE's trusted receipts.
This keeps the baseline comparison paired while preserving the distinction
between temporal reconciliation and evidence-governed admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .router_return_protocol import RouterPostAbsenceCheckpoint
from .return_methods.lifecycle import KMU_OP_ARM, TEMPORAL_LWW_ARM
from .return_methods.semantic_state import (
    CUPMEM_ARM,
    CUPMEM_STATUSES,
    compile_router_cupmem,
    router_cupmem_adjudication_prompt,
)
from .return_methods.temporal import MEMSTRATA_ARM
from .return_methods.transactional import MEMTX_ARM
from .return_methods.policies.checkpoint_replay import (
    CHECKPOINT_REPLAY_ARM,
    CheckpointReplayCompilation,
    compile_checkpoint_replay,
)


LIFECYCLE_BASELINE_ARMS = (
    CHECKPOINT_REPLAY_ARM,
    CUPMEM_ARM,
    MEMSTRATA_ARM,
    MEMTX_ARM,
    TEMPORAL_LWW_ARM,
    KMU_OP_ARM,
)


class LifecycleReturnArm(str, Enum):
    """Memora-only arm identifiers without changing the core Router arms."""

    CHECKPOINT_REPLAY = CHECKPOINT_REPLAY_ARM
    CUPMEM = CUPMEM_ARM
    MEMSTRATA = MEMSTRATA_ARM
    MEMTX = MEMTX_ARM
    TEMPORAL_LWW = TEMPORAL_LWW_ARM
    KMU_OP = KMU_OP_ARM


@dataclass(frozen=True)
class CupMemDecision:
    """One actor-visible CUPMem state-role decision."""

    item_id: str
    status: str
    replacement_item_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("CUPMem decision requires item_id")
        if self.status not in CUPMEM_STATUSES:
            raise ValueError(f"unsupported CUPMem status: {self.status}")
        if self.replacement_item_id == self.item_id:
            raise ValueError("CUPMem item cannot replace itself")

    def record(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "status": self.status,
            "replacement_item_id": self.replacement_item_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LifecycleBaselineCompilation:
    """Auditable selected view for one lifecycle baseline."""

    arm: str
    candidate_item_ids: tuple[str, ...]
    selected_item_ids: tuple[str, ...]
    decisions: tuple[CupMemDecision, ...] = ()
    replay_trace: tuple[Mapping[str, object], ...] = ()
    parse_error: str | None = None

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "return_lifecycle_baseline_v1",
            "arm": self.arm,
            "candidate_item_ids": list(self.candidate_item_ids),
            "selected_item_ids": list(self.selected_item_ids),
            "decisions": [item.record() for item in self.decisions],
            "replay_trace": [dict(item) for item in self.replay_trace],
            "parse_error": self.parse_error,
        }


def cupmem_adjudication_prompt(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> str:
    """Compatibility wrapper for the shared CUPMem adapter."""

    return router_cupmem_adjudication_prompt(checkpoint)


def compile_cupmem(
    checkpoint: RouterPostAbsenceCheckpoint,
    model_output: str,
) -> LifecycleBaselineCompilation:
    """Compatibility wrapper around the benchmark-neutral CUPMem compiler."""

    compilation = compile_router_cupmem(checkpoint, model_output)
    decisions = tuple(
        CupMemDecision(
            item_id=str(row["candidate_id"]),
            status=str(row["status"]),
            replacement_item_id=(
                str(row["replacement_candidate_id"])
                if row.get("replacement_candidate_id") is not None
                else None
            ),
            reason=str(row.get("fallback") or ""),
        )
        for row in compilation.decisions
    )
    return LifecycleBaselineCompilation(
        arm=CUPMEM_ARM,
        candidate_item_ids=compilation.candidate_ids,
        selected_item_ids=compilation.selected_ids,
        decisions=decisions,
        parse_error=compilation.parse_error,
    )


__all__ = [
    "CHECKPOINT_REPLAY_ARM",
    "CheckpointReplayCompilation",
    "CUPMEM_ARM",
    "CUPMEM_STATUSES",
    "CupMemDecision",
    "LIFECYCLE_BASELINE_ARMS",
    "KMU_OP_ARM",
    "MEMSTRATA_ARM",
    "MEMTX_ARM",
    "TEMPORAL_LWW_ARM",
    "LifecycleReturnArm",
    "LifecycleBaselineCompilation",
    "compile_checkpoint_replay",
    "compile_cupmem",
    "cupmem_adjudication_prompt",
]
