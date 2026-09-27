"""Thin repository entry points; benchmark arguments pass through unchanged."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]


def dispatch(description: str, commands: Mapping[str, str], argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('benchmark', choices=commands)
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='Arguments for the selected runner; use BENCHMARK --help.')
    if not args or args[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    selected = parser.parse_args(args)
    script = ROOT / 'scripts' / 'experiments' / commands[selected.benchmark]
    old_argv = sys.argv
    old_path = sys.path[:]
    try:
        sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
        sys.argv = [str(script), *selected.arguments]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    return 0
