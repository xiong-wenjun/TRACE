#!/usr/bin/env python3
"""Export a balanced, arm-blinded Memora rubric review packet."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace.trace_method import canonical_return_arm  # noqa: E402
DEFAULT_RETENTION_MANIFEST = (
    ROOT / "results" / "formal" / "RETENTION_MANIFEST.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "human_review" / "memora_rubric_288_v1"
CANONICAL_ARMS = (
    "reset",
    "restore_old",
    "static_no_churn",
    "trace",
)
EVALUATION_TYPES = ("memory_presence", "forgetting_absence")
CONTEXTS_PER_PANEL_TYPE = 4
SEED = "memora-human-review-v1-20260728"
PUBLIC_FIELDS = (
    "item_id",
    "user_question",
    "candidate_response",
    "rubric_question",
    "human_answer_yes_or_no",
    "confidence_1_to_3",
    "notes",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_digest(*parts: object) -> str:
    return _sha256_bytes(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    )


def _canonical_arm(value: str) -> str:
    return canonical_return_arm(value)


def _model_family(actor_model: str) -> str:
    normalized = actor_model.lower()
    if normalized.startswith("qwen"):
        return "Qwen"
    if normalized.startswith("gpt"):
        return "GPT"
    if normalized.startswith("claude"):
        return "Claude"
    raise ValueError(f"unknown model family for {actor_model}")


def _episode_directory(asset_root: Path) -> Path:
    candidates = []
    for path in sorted(asset_root.glob("sol_scored*/episodes")):
        count = sum(1 for _ in path.glob("*.json"))
        if count:
            candidates.append((path, count))
    if len(candidates) != 1:
        raise ValueError(
            f"{asset_root}: expected one non-empty scored episode directory, "
            f"found {candidates}"
        )
    return candidates[0][0]


def _arm_mapping(artifact: Mapping[str, Any]) -> dict[str, str]:
    raw_arms = artifact.get("arms")
    if not isinstance(raw_arms, Mapping):
        raise ValueError("scored artifact has no arms")
    mapping = {
        _canonical_arm(str(raw_arm)): str(raw_arm)
        for raw_arm in raw_arms
    }
    if set(mapping) != set(CANONICAL_ARMS):
        raise ValueError(f"unexpected Memora arms: {sorted(mapping)}")
    return mapping


def _item_index(
    artifact: Mapping[str, Any],
    raw_arm: str,
) -> dict[str, Mapping[str, Any]]:
    arms = artifact["arms"]
    arm = arms[raw_arm]
    items = arm["score"]["item_results"]
    index = {
        str(item["evaluation_question_id"]): item for item in items
    }
    if len(index) != len(items):
        raise ValueError("duplicate evaluation question IDs")
    return index


def _candidate_contexts(
    episode_dir: Path,
    *,
    asset_id: str,
    evaluation_type: str,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for source_path in sorted(episode_dir.glob("*.json")):
        artifact = json.loads(source_path.read_text(encoding="utf-8"))
        arm_mapping = _arm_mapping(artifact)
        indexes = {
            arm: _item_index(artifact, raw_arm)
            for arm, raw_arm in arm_mapping.items()
        }
        common_ids = set.intersection(
            *(set(index) for index in indexes.values())
        )
        for evaluation_id in sorted(common_ids):
            arm_items = {
                arm: indexes[arm][evaluation_id]
                for arm in CANONICAL_ARMS
            }
            reference = arm_items[CANONICAL_ARMS[0]]
            if str(reference["evaluation_type"]) != evaluation_type:
                continue
            invariant_fields = (
                "evaluation_question",
                "evaluation_type",
                "expected_answer",
            )
            for field in invariant_fields:
                values = {
                    str(item[field]) for item in arm_items.values()
                }
                if len(values) != 1:
                    raise ValueError(
                        f"{source_path}:{evaluation_id} differs on {field}"
                    )
            episode = artifact["episode"]
            question = artifact["question_actor_record"]
            contexts.append(
                {
                    "selection_digest": _stable_digest(
                        SEED,
                        asset_id,
                        evaluation_type,
                        episode["episode_id"],
                        evaluation_id,
                    ),
                    "source_path": source_path,
                    "artifact": artifact,
                    "arm_mapping": arm_mapping,
                    "arm_items": arm_items,
                    "episode_id": str(episode["episode_id"]),
                    "cluster_id": str(episode["cluster_id"]),
                    "question_id": str(episode["question_id"]),
                    "evaluation_question_id": evaluation_id,
                    "evaluation_type": evaluation_type,
                    "user_question": str(question["question"]),
                    "rubric_question": str(
                        reference["evaluation_question"]
                    ),
                    "expected_answer": str(reference["expected_answer"]),
                }
            )
    return contexts


def _select_contexts(
    contexts: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    used_episodes: set[str] = set()
    used_clusters: set[str] = set()
    ordered = sorted(contexts, key=lambda row: row["selection_digest"])
    for require_new_cluster in (True, False):
        for row in ordered:
            episode_id = str(row["episode_id"])
            cluster_id = str(row["cluster_id"])
            if episode_id in used_episodes:
                continue
            if require_new_cluster and cluster_id in used_clusters:
                continue
            selected.append(row)
            used_episodes.add(episode_id)
            used_clusters.add(cluster_id)
            if len(selected) == count:
                return selected
    raise ValueError(f"could select only {len(selected)} contexts")


def _answer_for_arm(
    artifact: Mapping[str, Any],
    raw_arm: str,
) -> str:
    answer = str(
        artifact["arms"][raw_arm]["aggregation"]["answer"]
    ).strip()
    if not answer:
        raise ValueError(f"{raw_arm} has an empty candidate answer")
    return answer


def _build_records(
    retention_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assets = [
        asset
        for asset in retention_manifest["assets"]
        if asset["benchmark"] == "Memora"
    ]
    if len(assets) != 9:
        raise ValueError(f"expected nine Memora assets, found {len(assets)}")
    private_rows: list[dict[str, Any]] = []
    source_assets: list[dict[str, Any]] = []
    for asset in sorted(
        assets,
        key=lambda row: (
            float(row["departure_fraction"]),
            str(row["actor_model"]),
        ),
    ):
        asset_root = ROOT / str(asset["path"])
        episode_dir = _episode_directory(asset_root)
        source_assets.append(
            {
                "asset_id": asset["asset_id"],
                "actor_model": asset["actor_model"],
                "model_family": _model_family(
                    str(asset["actor_model"])
                ),
                "departure_fraction": asset["departure_fraction"],
                "complete_paired_units": asset["complete_paired_units"],
                "tree_sha256": asset["tree_sha256"],
                "scored_episode_directory": str(
                    episode_dir.relative_to(ROOT)
                ),
            }
        )
        for evaluation_type in EVALUATION_TYPES:
            contexts = _candidate_contexts(
                episode_dir,
                asset_id=str(asset["asset_id"]),
                evaluation_type=evaluation_type,
            )
            selected = _select_contexts(
                contexts,
                count=CONTEXTS_PER_PANEL_TYPE,
            )
            for context in selected:
                artifact = context["artifact"]
                arm_mapping = context["arm_mapping"]
                arm_items = context["arm_items"]
                for arm in CANONICAL_ARMS:
                    raw_arm = str(arm_mapping[arm])
                    machine_answer = str(
                        arm_items[arm]["predicted_answer"]
                    ).lower()
                    if machine_answer not in {"yes", "no"}:
                        raise ValueError(
                            f"non-binary machine answer: {machine_answer}"
                        )
                    private_key = "|".join(
                        (
                            str(asset["asset_id"]),
                            str(context["episode_id"]),
                            str(context["evaluation_question_id"]),
                            arm,
                        )
                    )
                    private_rows.append(
                        {
                            "private_key": private_key,
                            "asset_id": asset["asset_id"],
                            "actor_model": asset["actor_model"],
                            "model_family": _model_family(
                                str(asset["actor_model"])
                            ),
                            "departure_fraction": asset[
                                "departure_fraction"
                            ],
                            "arm": arm,
                            "source_arm": raw_arm,
                            "evaluation_type": evaluation_type,
                            "episode_id": context["episode_id"],
                            "cluster_id": context["cluster_id"],
                            "question_id": context["question_id"],
                            "evaluation_question_id": context[
                                "evaluation_question_id"
                            ],
                            "source_path": str(
                                context["source_path"].relative_to(ROOT)
                            ),
                            "user_question": context["user_question"],
                            "candidate_response": _answer_for_arm(
                                artifact,
                                raw_arm,
                            ),
                            "rubric_question": context["rubric_question"],
                            "machine_answer": machine_answer,
                            "official_expected_answer": context[
                                "expected_answer"
                            ],
                            "machine_correct": bool(
                                arm_items[arm]["correct"]
                            ),
                        }
                    )
    expected_count = (
        3
        * 3
        * len(EVALUATION_TYPES)
        * CONTEXTS_PER_PANEL_TYPE
        * len(CANONICAL_ARMS)
    )
    if len(private_rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} review items, "
            f"found {len(private_rows)}"
        )
    id_order = sorted(
        private_rows,
        key=lambda row: _stable_digest(
            SEED,
            "opaque-item-id",
            row["private_key"],
        ),
    )
    item_ids = {
        row["private_key"]: f"MHR-{index:04d}"
        for index, row in enumerate(id_order, start=1)
    }
    public_rows = []
    coordinator_rows = []
    for row in private_rows:
        item_id = item_ids[row["private_key"]]
        public_rows.append(
            {
                "item_id": item_id,
                "user_question": row["user_question"],
                "candidate_response": row["candidate_response"],
                "rubric_question": row["rubric_question"],
                "human_answer_yes_or_no": "",
                "confidence_1_to_3": "",
                "notes": "",
            }
        )
        coordinator_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "private_key",
                    "user_question",
                    "candidate_response",
                    "rubric_question",
                }
            }
            | {"item_id": item_id}
        )
    return public_rows, coordinator_rows, source_assets


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PUBLIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_packet(
    rows: Sequence[Mapping[str, Any]],
    annotator: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _stable_digest(
            SEED,
            f"annotator-{annotator}",
            row["item_id"],
        ),
    )


def _counts(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row[field]) for row in rows).items()
        )
    )


def _write_guide(path: Path) -> None:
    path.write_text(
        """# Memora 人工盲评说明

## 任务

每行包含一个用户问题、一个候选回答和一个 yes/no rubric 问题。请只判断
候选回答是否满足 rubric 问题，不要猜测模型、离场比例或方法臂。

## 标注字段

- `human_answer_yes_or_no`：只能填写 `yes` 或 `no`。
- `confidence_1_to_3`：1=不确定，2=一般，3=确定。
- `notes`：可选；记录歧义或需要复核的原因。

## 判定原则

- 只有候选回答明确陈述或可靠蕴含目标事实时才标 `yes`。
- 未提及、模糊暗示、与目标事实冲突时标 `no`。
- 不使用外部知识，不查看 `coordinator_only` 目录。
- Annotator A 与 B 必须独立完成，完成前不要讨论答案。

保存 CSV 时保持 UTF-8、列名和 `item_id` 不变。

## 完成后报告什么

- Cohen's kappa（κ）：两名标注者扣除随机一致后的 agreement。
- Precision：机器判为 `yes` 的样本中，人类共识也为 `yes` 的比例。
- Recall：人类共识为 `yes` 的样本中，机器成功判为 `yes` 的比例。
- F1：Precision 与 Recall 的调和平均，只用于 yes/no 标签。
- Weighted kappa：用于 1–5 分，分差越大惩罚越重。
- Spearman：机器分数与人类分数的排序相关性。
- MAE：机器分数与人类分数的平均绝对差，越低越好。

本 Memora 文件只填写 yes/no，主要报告 κ、Precision、Recall 和 F1。
""",
        encoding="utf-8",
    )


def export_packet(
    retention_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing packet: {output_dir}"
        )
    retention_manifest = json.loads(
        retention_manifest_path.read_text(encoding="utf-8")
    )
    public_rows, coordinator_rows, source_assets = _build_records(
        retention_manifest
    )
    annotator_dir = output_dir / "annotator_packets"
    coordinator_dir = output_dir / "coordinator_only"
    annotator_dir.mkdir(parents=True)
    coordinator_dir.mkdir()
    _write_guide(output_dir / "README.md")
    packet_a = annotator_dir / "annotator_a.csv"
    packet_b = annotator_dir / "annotator_b.csv"
    _write_csv(packet_a, _ordered_packet(public_rows, "a"))
    _write_csv(packet_b, _ordered_packet(public_rows, "b"))
    coordinator_key = coordinator_dir / "key.json"
    coordinator_key.write_text(
        json.dumps(
            {
                "schema_version": "memora_human_review_key_v1",
                "seed": SEED,
                "items": {
                    row["item_id"]: row
                    for row in sorted(
                        coordinator_rows,
                        key=lambda value: value["item_id"],
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exact_cells = Counter(
        (
            str(row["departure_fraction"]),
            str(row["model_family"]),
            str(row["evaluation_type"]),
            str(row["arm"]),
        )
        for row in coordinator_rows
    )
    if set(exact_cells.values()) != {CONTEXTS_PER_PANEL_TYPE}:
        raise ValueError(f"unbalanced exact cells: {exact_cells}")
    manifest = {
        "schema_version": "memora_human_review_manifest_v1",
        "seed": SEED,
        "item_count": len(public_rows),
        "paired_rubric_context_count": (
            len(public_rows) // len(CANONICAL_ARMS)
        ),
        "sampling_design": (
            "3 departure points x 3 model families x 2 rubric types x "
            "4 distinct episode contexts x 4 paired arms"
        ),
        "blinded_fields": [
            "model_family",
            "actor_model",
            "departure_fraction",
            "arm",
            "machine_answer",
            "official_expected_answer",
            "machine_correct",
        ],
        "counts": {
            "departure_fraction": _counts(
                coordinator_rows,
                "departure_fraction",
            ),
            "actor_model": _counts(coordinator_rows, "actor_model"),
            "model_family": _counts(coordinator_rows, "model_family"),
            "arm": _counts(coordinator_rows, "arm"),
            "evaluation_type": _counts(
                coordinator_rows,
                "evaluation_type",
            ),
            "exact_cell_count": len(exact_cells),
            "items_per_exact_cell": CONTEXTS_PER_PANEL_TYPE,
        },
        "source_assets": source_assets,
        "files": {
            "annotator_a": {
                "path": str(packet_a.relative_to(output_dir)),
                "sha256": _sha256_path(packet_a),
            },
            "annotator_b": {
                "path": str(packet_b.relative_to(output_dir)),
                "sha256": _sha256_path(packet_b),
            },
            "coordinator_key": {
                "path": str(coordinator_key.relative_to(output_dir)),
                "sha256": _sha256_path(coordinator_key),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retention-manifest",
        type=Path,
        default=DEFAULT_RETENTION_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    manifest = export_packet(
        args.retention_manifest.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
