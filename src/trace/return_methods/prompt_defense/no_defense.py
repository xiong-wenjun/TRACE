"""Unmodified answer-time control for ManBench-Return.

The control receives the same frozen old/challenger candidate pool as the two
prompt defenses, but adds no defense instruction. Keeping it as an explicit
arm prevents the memory-view baseline ``static`` from being mislabeled as a
prompt-defense control.
"""

from __future__ import annotations

from ..base import ReturnMethodSpec


NO_DEFENSE_ARM = "no_defense"
NO_DEFENSE_DISPLAY_NAME = "No Defense"
NO_DEFENSE_METHOD = ReturnMethodSpec(
    method=NO_DEFENSE_ARM,
    display_name=NO_DEFENSE_DISPLAY_NAME,
    category="answer_time_control",
    produces_memory_view=False,
)


__all__ = ["NO_DEFENSE_ARM", "NO_DEFENSE_DISPLAY_NAME", "NO_DEFENSE_METHOD"]
