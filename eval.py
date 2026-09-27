#!/usr/bin/env python3
"""Evaluate TRACE and comparison methods on the maintained benchmarks."""
from scripts.cli import dispatch

COMMANDS = {
    'manbench': 'run_manbench_return_batch.py',
    'stale': 'run_stale_type2_mas_return.py',
    'memora': 'generate_memora_mas_return.py',
    'memora-score': 'score_memora_generations.py',
}

if __name__ == '__main__':
    raise SystemExit(dispatch(__doc__, COMMANDS))
