#!/usr/bin/env python3
"""Analyze completed TRACE benchmark outputs using their original contracts."""
from scripts.cli import dispatch

COMMANDS = {
    'manbench': 'report_manbench_return_statistics.py',
    'memora': 'report_memora_return_statistics.py',
    'memora-pareto': 'summarize_memora_pareto.py',
    'manbench-provenance': 'report_manbench_provenance_stress.py',
}

if __name__ == '__main__':
    raise SystemExit(dispatch(__doc__, COMMANDS))
