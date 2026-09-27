"""Shared, benchmark-neutral contracts for RETURN method adapters.

The records in this module contain actor-visible lifecycle metadata only.
Benchmark labels, evaluator rubrics, and TRACE receipt verdicts are deliberately
absent so every baseline can be audited for oracle leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping, Sequence


RETURN_PHASES = frozenset(("predeparture", "absence"))
RETURN_OPERATIONS = frozenset(("add", "update", "delete"))


@dataclass(frozen=True)
class ReturnMethodSpec:
    """Publication-facing identity and implementation provenance for a method.

    Runtime artifact identifiers are intentionally separate from display names.
    This lets frozen experiment outputs retain stable keys while paper tables,
    documentation, and new code use one unambiguous method name.
    """

    method: str
    display_name: str
    category: str
    produces_memory_view: bool
    paper_url: str | None = None
    reference_url: str | None = None
    implementation_kind: str = "local_baseline"
    reference_commit: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required_text(self.method, "method")
        required_text(self.display_name, "display_name")
        required_text(self.category, "category")
        if len(self.aliases) != len(set(self.aliases)):
            raise ValueError("method aliases must be unique")
        if self.method in self.aliases:
            raise ValueError("canonical method cannot also be an alias")
        for alias in self.aliases:
            required_text(alias, "method alias")


def required_text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def parse_json_object(text: str) -> Mapping[str, object]:
    """Parse one model JSON object, tolerating a fenced response."""

    candidate = re.sub(r"^```(?:json)?\s*", "", text.strip(), count=1)
    candidate = re.sub(r"\s*```$", "", candidate, count=1)
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, Mapping):
        raise TypeError("method output must be a JSON object")
    return value


@dataclass(frozen=True)
class ReturnCandidate:
    """One normalized candidate offered to a RETURN method."""

    candidate_id: str
    state_key: str
    text: str
    logical_time: int
    version: int
    phase: str
    operation: str = "add"
    source_id: str = "unknown"
    supersedes_candidate_id: str | None = None
    semantic_value: str | None = None
    entity: str | None = None
    attribute: str | None = None
    confidence: float | None = None
    valid_from: int | None = None
    valid_to: int | None = None
    derived_from_candidate_ids: tuple[str, ...] = ()
    provenance_status: str = "legacy"
    provenance_source_ids: tuple[str, ...] = ()
    provenance_receipt_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required_text(self.candidate_id, "candidate_id")
        required_text(self.state_key, "state_key")
        required_text(self.text, "text")
        required_text(self.source_id, "source_id")
        if self.logical_time < 0:
            raise ValueError("logical_time must be non-negative")
        if self.version < 1:
            raise ValueError("version must be positive")
        if self.phase not in RETURN_PHASES:
            raise ValueError(f"unsupported RETURN phase: {self.phase}")
        if self.operation not in RETURN_OPERATIONS:
            raise ValueError(f"unsupported RETURN operation: {self.operation}")
        if self.supersedes_candidate_id == self.candidate_id:
            raise ValueError("a RETURN candidate cannot supersede itself")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.valid_from is not None and self.valid_from < 0:
            raise ValueError("valid_from must be non-negative")
        if self.valid_to is not None and self.valid_to < 0:
            raise ValueError("valid_to must be non-negative")
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            raise ValueError("valid_to cannot precede valid_from")
        if len(self.derived_from_candidate_ids) != len(
            set(self.derived_from_candidate_ids)
        ):
            raise ValueError("derived_from_candidate_ids must be unique")
        if self.candidate_id in self.derived_from_candidate_ids:
            raise ValueError("a RETURN candidate cannot derive from itself")
        if self.provenance_status not in {
            "legacy",
            "valid",
            "replicated",
            "missing",
            "conflicting",
        }:
            raise ValueError(
                "unsupported candidate provenance_status: "
                + str(self.provenance_status)
            )
        for source_id in self.provenance_source_ids:
            required_text(source_id, "candidate provenance source id")
        for receipt in self.provenance_receipt_sha256s:
            if len(str(receipt)) != 64:
                raise ValueError("candidate provenance receipt must be SHA-256")
            bytes.fromhex(str(receipt))

    def record(self, *, max_text_chars: int = 1600) -> dict[str, object]:
        text = self.text
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "…"
        return {
            "candidate_id": self.candidate_id,
            "state_key": self.state_key,
            "text": text,
            "logical_time": self.logical_time,
            "version": self.version,
            "phase": self.phase,
            "operation": self.operation,
            "source_id": self.source_id,
            "supersedes_candidate_id": self.supersedes_candidate_id,
            "semantic_value": self.semantic_value,
            "entity": self.entity,
            "attribute": self.attribute,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "derived_from_candidate_ids": list(
                self.derived_from_candidate_ids
            ),
            # Expose evidence handles, but not the stress-cell label.  The
            # actor may inspect source/receipt material exactly as it could in
            # a deployed system; ``missing``/``conflicting`` are evaluator
            # annotations kept on the protocol item and never rendered here.
            "provenance_source_ids": list(self.provenance_source_ids),
            "provenance_receipt_sha256s": list(
                self.provenance_receipt_sha256s
            ),
            "provenance_source_count": len(set(self.provenance_source_ids)),
            "provenance_receipt_count": len(self.provenance_receipt_sha256s),
        }


@dataclass(frozen=True)
class ReturnMethodCompilation:
    """Auditable memory view produced by one RETURN method."""

    method: str
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    decisions: tuple[Mapping[str, object], ...] = ()
    parse_error: str | None = None

    def __post_init__(self) -> None:
        required_text(self.method, "method")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("selected_ids must be unique")
        if not set(self.selected_ids).issubset(self.candidate_ids):
            raise ValueError("selected_ids must be a subset of candidate_ids")

    def record(self) -> dict[str, object]:
        return {
            "schema_version": "return_method_compilation_v1",
            "method": self.method,
            "candidate_ids": list(self.candidate_ids),
            "selected_ids": list(self.selected_ids),
            "decisions": [dict(item) for item in self.decisions],
            "parse_error": self.parse_error,
            "oracle_fields_visible": False,
        }


def validate_candidates(
    candidates: Sequence[ReturnCandidate],
) -> tuple[ReturnCandidate, ...]:
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.logical_time,
                item.version,
                item.candidate_id,
            ),
        )
    )
    ids = tuple(item.candidate_id for item in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError("RETURN candidate IDs must be unique")
    known = set(ids)
    for item in ordered:
        if (
            item.supersedes_candidate_id is not None
            and item.supersedes_candidate_id not in known
        ):
            raise ValueError(
                "RETURN candidate supersedes an unknown candidate: "
                + item.supersedes_candidate_id
            )
        unknown_dependencies = set(item.derived_from_candidate_ids) - known
        if unknown_dependencies:
            raise ValueError(
                "RETURN candidate derives from unknown candidates: "
                + ",".join(sorted(unknown_dependencies))
            )
    return ordered


__all__ = [
    "RETURN_OPERATIONS",
    "RETURN_PHASES",
    "ReturnCandidate",
    "ReturnMethodCompilation",
    "ReturnMethodSpec",
    "parse_json_object",
    "required_text",
    "validate_candidates",
]
