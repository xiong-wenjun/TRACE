"""Relevance control sharing the production governor's eligibility and gate."""

from collections.abc import Mapping, Sequence

from trace.return_governance import (
    ReturnMemoryGovernor, ReturnMemoryItem, ReturnMemoryKind,
    ReturnObligationGraph, ReturnReadmissionContext, ReturnSelectionStatus,
    ReturnIncompleteAction,
)


class StudyGovernor(ReturnMemoryGovernor):
    """Override selection only; receipts, signing and admission stay inherited."""

    def __init__(self, signer_id: str, key: bytes,
                 relevance_scores: Mapping[str, float] | None = None):
        super().__init__(signer_id, key)
        self.relevance_scores = relevance_scores

    def _selection_report(self, **kwargs):
        if self.relevance_scores is not None:
            kwargs["algorithm"] = "relevance_order_with_coverage_gate_v1"
        return super()._selection_report(**kwargs)

    def _select_items(self, eligible: Sequence[ReturnMemoryItem],
                      context: ReturnReadmissionContext, *, issue_incomplete=False):
        if self.relevance_scores is None:
            return super()._select_items(eligible, context,
                                        issue_incomplete=issue_incomplete)
        graph = ReturnObligationGraph.build(context.obligations, eligible)
        required = {o.obligation_id for o in context.obligations}
        # Mandatory declarations and frontiers are fixed in both selectors.
        mandatory = [m for m in eligible if (
            m.kind is ReturnMemoryKind.OPEN_OBLIGATION and m.obligation_id in required
        ) or (m.kind is ReturnMemoryKind.PROGRESS_FRONTIER and required.intersection(m.depends_on))]
        mandatory_ids = {m.item_id for m in mandatory}
        rest = sorted((m for m in eligible if m.item_id not in mandatory_ids),
                      key=lambda m: (-self.relevance_scores.get(m.item_id, 0.0), m.item_id))
        selected = []
        chars = 0
        failure = self._required_structure_missing(mandatory, context)
        for item in [*mandatory, *rest]:
            if len(selected) >= context.max_items or chars + len(item.text) > context.max_chars:
                if item.item_id in mandatory_ids:
                    failure = failure or "mandatory_workstate_exceeds_budget"
                continue
            selected.append(item)
            chars += len(item.text)
        _, _, coverage, critical = self._selection_metrics(selected, context)
        if not eligible:
            failure = "no_safe_candidate"
        if coverage < context.minimum_obligation_coverage:
            failure = failure or "insufficient_obligation_coverage"
        if critical < context.minimum_critical_dependency_recall:
            failure = failure or "insufficient_critical_dependency_recall"
        if failure:
            if issue_incomplete and selected:
                return tuple(selected), ReturnSelectionStatus.ISSUED, f"incomplete_view_issued:{failure}", graph
            status = (ReturnSelectionStatus.RESET_FALLBACK
                      if context.incomplete_action is ReturnIncompleteAction.RESET
                      else ReturnSelectionStatus.BLOCKED)
            return (), status, failure, graph
        return tuple(selected), ReturnSelectionStatus.ISSUED, None, graph

