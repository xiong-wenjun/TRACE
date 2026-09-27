"""Single-intervention definitions for the five-arm return-governance study."""

from dataclasses import dataclass
from enum import Enum


class CoreArm(str, Enum):
    TRACE = "trace"
    NO_IMPLICIT = "without_implicit_invalidation"
    NO_SELECTION = "without_obligation_selection"
    NO_GATE = "without_complete_view_gate"
    NO_PRIVATE = "without_source_bound_private_readmission"


@dataclass(frozen=True)
class Mechanisms:
    implicit_invalidation: bool = True
    obligation_selection: bool = True
    complete_view_gate: bool = True
    source_bound_private_readmission: bool = True

    @classmethod
    def for_arm(cls, arm: CoreArm | str) -> "Mechanisms":
        arm = CoreArm(arm)
        return cls(
            implicit_invalidation=arm != CoreArm.NO_IMPLICIT,
            obligation_selection=arm != CoreArm.NO_SELECTION,
            complete_view_gate=arm != CoreArm.NO_GATE,
            source_bound_private_readmission=arm != CoreArm.NO_PRIVATE,
        )

