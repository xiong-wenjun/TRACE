"""Backward-compatible import path for the canonical :mod:`.cupmem` API.

New code must import ``semantic_state.cupmem``.  This module remains so frozen
launchers and historical artifacts can be reproduced without source edits.
"""

from .cupmem import *  # noqa: F401,F403
from .cupmem import __all__
