"""Contract for prompt-only defenses evaluated at agent re-entry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptDefenseSpec:
    method: str
    display_name: str
    instruction: str

    def apply(self, prompt: str) -> str:
        return (
            prompt
            + "\n\n<return_prompt_defense method=\""
            + self.method
            + "\">\n"
            + self.instruction
            + "\n</return_prompt_defense>"
        )


PromptDefenseFactory = Callable[[], PromptDefenseSpec]


__all__ = ["PromptDefenseFactory", "PromptDefenseSpec"]
