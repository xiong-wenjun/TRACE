"""Checkpoint-Replay lifecycle baseline.

The baseline restores agent1's departure snapshot and replays only
actor-visible absence-ledger events.  It never reads evaluator labels or
TRACE receipt verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...router_return_protocol import (
    RouterLedgerEvent,
    RouterPostAbsenceCheckpoint,
    RouterWorkstateItem,
)
from ..base import ReturnMethodSpec


CHECKPOINT_REPLAY_ARM = "checkpoint_replay"
CHECKPOINT_REPLAY_METHOD = ReturnMethodSpec(
    method=CHECKPOINT_REPLAY_ARM,
    display_name="Checkpoint-Replay",
    category="lifecycle_replay",
    produces_memory_view=True,
)


@dataclass(frozen=True)
class CheckpointReplayCompilation:
    """Auditable state view reconstructed by deterministic event replay."""

    arm: str
    candidate_item_ids: tuple[str, ...]
    selected_item_ids: tuple[str, ...]
    replay_trace: tuple[Mapping[str, object], ...] = ()

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "return_lifecycle_baseline_v1",
            "arm": self.arm,
            "candidate_item_ids": list(self.candidate_item_ids),
            "selected_item_ids": list(self.selected_item_ids),
            "decisions": [],
            "replay_trace": [dict(item) for item in self.replay_trace],
            "parse_error": None,
        }


def _candidate_items(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> tuple[RouterWorkstateItem, ...]:
    rows = {item.item_id: item for item in checkpoint.departure.workstate_items}
    rows.update({item.item_id: item for item in checkpoint.current_workstate_items})
    return tuple(rows[item_id] for item_id in sorted(rows))


def compile_checkpoint_replay(
    checkpoint: RouterPostAbsenceCheckpoint,
) -> CheckpointReplayCompilation:
    """Restore the departure snapshot and replay absence events in order."""

    candidates = _candidate_items(checkpoint)
    candidate_ids = tuple(item.item_id for item in candidates)
    known = set(candidate_ids)
    active = {item.item_id for item in checkpoint.departure.workstate_items}
    trace: list[Mapping[str, object]] = []

    for entry in sorted(
        checkpoint.absence_delta.ledger_entries,
        key=lambda row: row.sequence,
    ):
        payload = entry.payload
        before = set(active)
        action = "ignore"
        affected: list[str] = []
        if entry.event is RouterLedgerEvent.WORKSTATE_CREATED:
            item_id = str(payload.get("item_id") or "")
            if item_id in known:
                active.add(item_id)
                affected.append(item_id)
                action = "create"
        elif entry.event is RouterLedgerEvent.WORKSTATE_SUPERSEDED:
            old_id = str(payload.get("old_item_id") or "")
            replacement_id = str(payload.get("replacement_item_id") or "")
            active.discard(old_id)
            if replacement_id in known:
                active.add(replacement_id)
            affected.extend(item_id for item_id in (old_id, replacement_id) if item_id)
            action = "supersede"
        elif entry.event is RouterLedgerEvent.WORKSTATE_REVOKED:
            item_id = str(payload.get("item_id") or "")
            active.discard(item_id)
            affected.append(item_id)
            action = "revoke"
        elif entry.event is RouterLedgerEvent.WORKSTATE_DISPUTED:
            existing_id = str(payload.get("existing_item_id") or "")
            challenger_id = str(payload.get("challenger_item_id") or "")
            active.discard(existing_id)
            active.discard(challenger_id)
            affected.extend(
                item_id for item_id in (existing_id, challenger_id) if item_id
            )
            action = "quarantine_dispute"
        elif entry.event is RouterLedgerEvent.WORKSTATE_DISPUTE_RESOLVED:
            accepted_id = str(payload.get("accepted_item_id") or "")
            rejected_id = str(payload.get("rejected_item_id") or "")
            active.discard(rejected_id)
            if accepted_id in known:
                active.add(accepted_id)
            affected.extend(
                item_id for item_id in (accepted_id, rejected_id) if item_id
            )
            action = "resolve_dispute"
        if action != "ignore":
            trace.append(
                {
                    "ledger_sequence": entry.sequence,
                    "event": entry.event.value,
                    "action": action,
                    "affected_item_ids": affected,
                    "active_before": sorted(before),
                    "active_after": sorted(active),
                }
            )

    return CheckpointReplayCompilation(
        arm=CHECKPOINT_REPLAY_ARM,
        candidate_item_ids=candidate_ids,
        selected_item_ids=tuple(
            item_id for item_id in candidate_ids if item_id in active
        ),
        replay_trace=tuple(trace),
    )


__all__ = [
    "CHECKPOINT_REPLAY_ARM",
    "CHECKPOINT_REPLAY_METHOD",
    "CheckpointReplayCompilation",
    "compile_checkpoint_replay",
]
