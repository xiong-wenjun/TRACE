"""Static control: keep the benchmark's ordinary shared-memory view."""

from ..base import ReturnMethodSpec


STATIC_ARM = "static"
STATIC_ARTIFACT_ALIASES = ("static_no_churn",)
STATIC_METHOD = ReturnMethodSpec(
    method=STATIC_ARM,
    display_name="Static",
    category="no_lifecycle_governance",
    produces_memory_view=True,
    aliases=STATIC_ARTIFACT_ALIASES,
)

__all__ = ["STATIC_ARM", "STATIC_ARTIFACT_ALIASES", "STATIC_METHOD"]
