"""Cluster-aware paired statistics for the Memora RETURN panel."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from typing import Mapping, Sequence


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _cluster_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    result: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "").strip()
        if not cluster_id:
            raise ValueError("every paired row requires cluster_id")
        result.setdefault(cluster_id, []).append(row)
    return result


def _metric(
    row: Mapping[str, object], arm: str, metric: str
) -> float:
    arms = row.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("paired row requires arms")
    arm_row = arms.get(arm)
    if not isinstance(arm_row, Mapping):
        raise ValueError(f"paired row is missing arm {arm}")
    score = arm_row.get("score")
    if isinstance(score, Mapping):
        value = float(score[metric])
    elif metric == "score" and isinstance(score, (int, float)):
        value = float(score)
    else:
        raise ValueError(f"arm {arm} is missing score")
    if not math.isfinite(value):
        raise ValueError("metric values must be finite")
    return value


def _mean_delta(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment: str,
    comparator: str,
    metric: str,
) -> float:
    if not rows:
        raise ValueError("paired comparison requires rows")
    return sum(
        _metric(row, treatment, metric)
        - _metric(row, comparator, metric)
        for row in rows
    ) / len(rows)


def cluster_sign_flip_test(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment: str,
    comparator: str,
    metric: str,
    alternative: str = "greater",
    monte_carlo_draws: int = 100_000,
) -> dict[str, object]:
    """Paired sign-flip test for the question-weighted mean difference.

    A random sign is assigned to every cluster, while the contribution of a
    cluster remains the sum of its paired question-level differences.  This
    keeps the randomization test aligned with ``_mean_delta`` and the cluster
    bootstrap when persona-period clusters contain different question counts.
    """

    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be greater or less")
    clusters = _cluster_rows(rows)
    cluster_sums = [
        sum(
            _metric(row, treatment, metric)
            - _metric(row, comparator, metric)
            for row in cluster_rows
        )
        for _, cluster_rows in sorted(clusters.items())
    ]
    nonzero = [value for value in cluster_sums if abs(value) > 1e-12]
    if not nonzero:
        return {
            "method": "paired_cluster_sign_flip_question_weighted",
            "clusters": len(clusters),
            "nonzero_clusters": 0,
            "alternative": alternative,
            "observed_mean_delta": 0.0,
            "p_value": 1.0,
        }
    observed = sum(nonzero) / len(rows)

    def is_extreme(value: float) -> bool:
        if alternative == "greater":
            return value >= observed - 1e-12
        return value <= observed + 1e-12

    if len(nonzero) <= 20:
        extreme = 0
        total = 2 ** len(nonzero)
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            statistic = sum(
                sign * value for sign, value in zip(signs, nonzero)
            ) / len(rows)
            extreme += int(is_extreme(statistic))
        p_value = extreme / total
        method = "exact_paired_cluster_sign_flip_question_weighted"
        draws = total
        seed = None
    else:
        payload = "|".join(f"{value:.17g}" for value in nonzero)
        seed = int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)
        generator = random.Random(seed)
        extreme = 0
        for _ in range(monte_carlo_draws):
            statistic = sum(
                (1.0 if generator.getrandbits(1) else -1.0) * value
                for value in nonzero
            ) / len(rows)
            extreme += int(is_extreme(statistic))
        p_value = (extreme + 1) / (monte_carlo_draws + 1)
        method = (
            "monte_carlo_paired_cluster_sign_flip_question_weighted"
        )
        draws = monte_carlo_draws
    return {
        "method": method,
        "clusters": len(clusters),
        "nonzero_clusters": len(nonzero),
        "alternative": alternative,
        "observed_mean_delta": observed,
        "randomizations": draws,
        "seed": seed,
        "p_value": p_value,
    }


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment: str,
    comparator: str,
    metric: str,
    replicates: int = 10_000,
    seed: int = 731,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Bootstrap persona-period clusters while retaining all paired rows."""

    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    clusters = _cluster_rows(rows)
    cluster_ids = tuple(sorted(clusters))
    if len(cluster_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    observed = _mean_delta(
        rows,
        treatment=treatment,
        comparator=comparator,
        metric=metric,
    )
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled: list[Mapping[str, object]] = []
        for _ in cluster_ids:
            sampled.extend(clusters[generator.choice(cluster_ids)])
        draws.append(
            _mean_delta(
                sampled,
                treatment=treatment,
                comparator=comparator,
                metric=metric,
            )
        )
    alpha = 1.0 - confidence
    return {
        "method": "paired_persona_period_cluster_bootstrap",
        "metric": metric,
        "treatment": treatment,
        "comparator": comparator,
        "paired_questions": len(rows),
        "clusters": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "mean_delta": observed,
        "confidence": confidence,
        "confidence_interval": [
            percentile(draws, alpha / 2.0),
            percentile(draws, 1.0 - alpha / 2.0),
        ],
        "one_sided_95_lower_bound": percentile(draws, 0.05),
        "one_sided_greater_probability": sum(
            value > 0.0 for value in draws
        )
        / len(draws),
        "significance": cluster_sign_flip_test(
            rows,
            treatment=treatment,
            comparator=comparator,
            metric=metric,
            alternative="greater",
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values."""

    ordered = sorted(
        ((name, float(value)) for name, value in p_values.items()),
        key=lambda item: item[1],
    )
    count = len(ordered)
    result: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (count - index) * value)
        running = max(running, adjusted)
        result[name] = running
    return result


__all__ = [
    "cluster_sign_flip_test",
    "holm_adjust",
    "paired_cluster_bootstrap",
    "percentile",
]
