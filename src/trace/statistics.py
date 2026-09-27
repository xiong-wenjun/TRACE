from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable, Mapping, Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


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
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_paired_difference(
    rows: Iterable[Mapping[str, object]],
    *,
    method_a: str,
    method_b: str,
    cluster_key: str = "example_id",
    method_key: str = "method",
    outcome_key: str = "correct",
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Paired cluster bootstrap with repeated rows averaged inside a cluster.

    The independent unit is ``cluster_key``. Model seeds or repeated checks must
    appear as multiple rows inside the same cluster, not as new independent N.
    """

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        cluster = str(row[cluster_key])
        method = str(row[method_key])
        if method in {method_a, method_b}:
            grouped[cluster][method].append(float(row[outcome_key]))
    differences: list[float] = []
    missing: list[str] = []
    for cluster, by_method in sorted(grouped.items()):
        if not by_method[method_a] or not by_method[method_b]:
            missing.append(cluster)
            continue
        differences.append(mean(by_method[method_a]) - mean(by_method[method_b]))
    if missing:
        raise ValueError(f"unpaired clusters: {missing[:5]}")
    if not differences:
        raise ValueError("no paired clusters")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    rng = random.Random(seed)
    draws = [
        mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(replicates)
    ]
    return {
        "method_a": method_a,
        "method_b": method_b,
        "independent_clusters": len(differences),
        "difference": mean(differences),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "bootstrap_replicates": replicates,
    }


def paired_cluster_noninferiority(
    rows: Iterable[Mapping[str, object]],
    *,
    method: str,
    reference: str,
    margin: float,
    cluster_key: str = "example_id",
    method_key: str = "method",
    outcome_key: str = "correct",
    replicates: int = 10_000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """One-sided paired cluster-bootstrap non-inferiority analysis.

    The estimand is ``method - reference`` and higher outcomes are better.
    Non-inferiority is established only when the one-sided lower confidence
    bound is strictly above ``-margin``. Repeated model-service seeds remain
    inside the same task/episode cluster and therefore do not inflate N.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if margin < 0.0:
        raise ValueError("non-inferiority margin must be non-negative")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        cluster = str(row[cluster_key])
        name = str(row[method_key])
        if name in {method, reference}:
            grouped[cluster][name].append(float(row[outcome_key]))
    differences: list[float] = []
    missing: list[str] = []
    for cluster, by_method in sorted(grouped.items()):
        if not by_method[method] or not by_method[reference]:
            missing.append(cluster)
            continue
        differences.append(mean(by_method[method]) - mean(by_method[reference]))
    if missing:
        raise ValueError(f"unpaired clusters: {missing[:5]}")
    if not differences:
        raise ValueError("no paired clusters")
    rng = random.Random(seed)
    draws = [
        mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(replicates)
    ]
    lower = percentile(draws, 1.0 - confidence_level)
    estimate = mean(differences)
    return {
        "method": method,
        "reference": reference,
        "independent_clusters": len(differences),
        "estimate_method_minus_reference": estimate,
        "noninferiority_margin": margin,
        "confidence_level": confidence_level,
        "one_sided_lower_confidence_bound": lower,
        "noninferior": lower > -margin,
        "decision_rule": "lower_bound_strictly_greater_than_negative_margin",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _binomial_probability(n: int, k: int) -> float:
    return math.comb(n, k) * (0.5**n)


def exact_mcnemar(
    left: Sequence[bool | int],
    right: Sequence[bool | int],
    *,
    favorable_value: int = 1,
) -> dict[str, object]:
    """Exact paired McNemar test with two-sided and directional p-values.

    ``favorable_value`` defines which binary value is better. For task success
    use 1; for security violations use 0. The directional p-value tests whether
    the left method has more favorable discordant pairs than the right method.
    """

    if len(left) != len(right) or not left:
        raise ValueError("McNemar inputs must be non-empty and equally sized")
    if favorable_value not in {0, 1}:
        raise ValueError("favorable_value must be binary")
    pairs = [(int(bool(a)), int(bool(b))) for a, b in zip(left, right)]
    if any(a not in {0, 1} or b not in {0, 1} for a, b in pairs):
        raise ValueError("McNemar inputs must be binary")
    left_favorable = sum(
        a == favorable_value and b != favorable_value for a, b in pairs
    )
    right_favorable = sum(
        b == favorable_value and a != favorable_value for a, b in pairs
    )
    discordant = left_favorable + right_favorable
    if discordant == 0:
        directional = 1.0
        two_sided = 1.0
    else:
        # Under H0 either method is favorable in a discordant pair with p=.5.
        directional = sum(
            _binomial_probability(discordant, k)
            for k in range(left_favorable, discordant + 1)
        )
        smaller = min(left_favorable, right_favorable)
        two_sided = min(
            1.0,
            2.0
            * sum(
                _binomial_probability(discordant, k)
                for k in range(0, smaller + 1)
            ),
        )
    return {
        "pairs": len(pairs),
        "left_favorable_only": left_favorable,
        "right_favorable_only": right_favorable,
        "discordant_pairs": discordant,
        "favorable_value": favorable_value,
        "directional_p_left_better": directional,
        "two_sided_p": two_sided,
        "test": "exact_mcnemar_binomial",
    }


def transition_regret_recovery(
    rows: Iterable[Mapping[str, object]],
    *,
    method: str,
    control: str,
    oracle: str,
    cluster_key: str = "example_id",
    method_key: str = "method",
    regret_key: str = "transition_regret",
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Estimate event-local transition-regret recovery relative to an oracle.

    Rows may include multiple events and service restarts per independent
    episode/task. Values are first averaged inside ``cluster_key``. Recovery is
    ``(control - method) / (control - oracle)`` where a positive denominator
    exists. Ratios are reported without clipping so overshoot remains visible.
    """

    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    wanted = {method, control, oracle}
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        name = str(row[method_key])
        if name in wanted:
            grouped[str(row[cluster_key])][name].append(float(row[regret_key]))
    savings: list[float] = []
    ratios: list[float] = []
    missing: list[str] = []
    zero_gap = 0
    for cluster, values in sorted(grouped.items()):
        if any(not values[name] for name in wanted):
            missing.append(cluster)
            continue
        method_regret = mean(values[method])
        control_regret = mean(values[control])
        oracle_regret = mean(values[oracle])
        savings.append(control_regret - method_regret)
        denominator = control_regret - oracle_regret
        if denominator > 0:
            ratios.append((control_regret - method_regret) / denominator)
        else:
            zero_gap += 1
    if missing:
        raise ValueError(f"unpaired clusters: {missing[:5]}")
    if not savings:
        raise ValueError("no paired transition-regret clusters")
    rng = random.Random(seed)

    def bootstrap(values: Sequence[float]) -> dict[str, object] | None:
        if not values:
            return None
        draws = [
            mean([values[rng.randrange(len(values))] for _ in values])
            for _ in range(replicates)
        ]
        return {
            "estimate": mean(values),
            "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        }

    return {
        "method": method,
        "control": control,
        "oracle": oracle,
        "independent_clusters": len(savings),
        "control_minus_method_regret": bootstrap(savings),
        "recovery_ratio": bootstrap(ratios),
        "positive_oracle_gap_clusters": len(ratios),
        "zero_or_negative_oracle_gap_clusters": zero_gap,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def accuracy_by_method(
    rows: Iterable[Mapping[str, object]],
    *,
    method_key: str = "method",
    outcome_key: str = "correct",
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[str(row[method_key])].append(float(row[outcome_key]))
    return {method: mean(items) for method, items in sorted(values.items())}
