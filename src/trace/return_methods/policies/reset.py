"""Reset control: return with no inherited predeparture memory."""

from ..base import ReturnMethodSpec


RESET_ARM = "reset"
RESET_METHOD = ReturnMethodSpec(
    method=RESET_ARM,
    display_name="Reset",
    category="reset_policy",
    produces_memory_view=True,
)

__all__ = ["RESET_ARM", "RESET_METHOD"]
