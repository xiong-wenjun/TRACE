"""Restore control: expose the returning agent's departure checkpoint."""

from ..base import ReturnMethodSpec


RESTORE_ARM = "restore_old"
RESTORE_METHOD = ReturnMethodSpec(
    method=RESTORE_ARM,
    display_name="Restore",
    category="restore_policy",
    produces_memory_view=True,
)

__all__ = ["RESTORE_ARM", "RESTORE_METHOD"]
