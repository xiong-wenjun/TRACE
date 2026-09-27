#!/usr/bin/env python3
"""Small real-backend integration check, never a benchmark efficacy result."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trace.eight_agent_pipeline import WORKER_IDS
from trace.mas_return_pipeline import EpisodeLongTermMemory
from trace.memory_backends.base import ExternalEpisodeLongTermMemory
from trace.memory_backends.lineage import SourceLedger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backends', nargs='+', choices=['amem','mem0','memobase'], required=True)
    parser.add_argument('--storage-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.storage_dir.mkdir(parents=True, exist_ok=True)
    report = {'schema_version':'native_backend_smoke_v1', 'scientific_efficacy_evidence':False, 'backends':{}}
    for name in args.backends:
        started = time.monotonic()
        try:
            common = dict(storage_dir=args.storage_dir / name,
                          generation_base_url=os.environ['TRACE_SMOKE_GENERATION_BASE_URL'],
                          generation_model=os.environ['TRACE_SMOKE_GENERATION_MODEL'],
                          generation_api_key=os.environ.get('TRACE_SMOKE_GENERATION_API_KEY','local'),
                          embedding_base_url=os.environ['TRACE_SMOKE_EMBEDDING_BASE_URL'],
                          embedding_model=os.environ['TRACE_SMOKE_EMBEDDING_MODEL'],
                          embedding_api_key=os.environ.get('TRACE_SMOKE_EMBEDDING_API_KEY','local'),
                          embedding_dimensions=int(os.environ.get('TRACE_SMOKE_EMBEDDING_DIMENSIONS','4096')))
            if name == 'amem':
                from trace.memory_backends.amem import AMemDriver
                driver = AMemDriver.build(**common, timeout=120, retries=1, max_tokens=4096)
            elif name == 'mem0':
                from trace.memory_backends.mem0 import Mem0Driver
                driver = Mem0Driver.build(**common)
            else:
                from trace.memory_backends.memobase import MemobaseDriver
                driver = MemobaseDriver(base_url=os.environ['TRACE_MEMOBASE_BASE_URL'],
                                        api_key=os.environ['TRACE_MEMOBASE_API_KEY'], timeout=180, retries=0,
                                        ledger=SourceLedger(args.storage_dir / name / 'source_receipts'))
            memory = ExternalEpisodeLongTermMemory(EpisodeLongTermMemory.build('adapter-smoke-'+uuid.uuid4().hex), driver)
            for source, fact in [('w1','My name is Alex. I am vegetarian and never eat meat.'),
                                 ('w2','I am Alex, and I prefer vegetarian Italian restaurants for dinner.')]:
                memory.remember(principal_id=WORKER_IDS[0], task_id='prefix', intent='dining preferences',
                                experience=fact, source_workstate_id=source)
            source_recall = memory.recall(principal_id=WORKER_IDS[0], intent='Alex dining preferences')
            if not source_recall.selected:
                raise AssertionError('native formation/search produced no retrievable memory')
            memory.quarantine_returning_worker()
            keep = memory.fork_return_arm(arm='static_no_churn', admitted_workstate_ids=('w1','w2'))
            drop = memory.fork_return_arm(arm='trace', admitted_workstate_ids=())
            kept = memory.recall(principal_id=WORKER_IDS[0], intent='Alex dining preferences',
                                 bank=keep.returning_worker, branch='static_no_churn')
            dropped = memory.recall(principal_id=WORKER_IDS[0], intent='Alex dining preferences',
                                    bank=drop.returning_worker, branch='trace')
            if not kept.selected or dropped.selected:
                raise AssertionError('branch import/retrieval violates admission')
            report['backends'][name] = dict(status='PASS', source_hits=len(source_recall.selected),
                                            imported_hits=len(kept.selected), excluded_hits=len(dropped.selected),
                                            driver=driver.record())
        except Exception as error:
            # Tracebacks can include native prompts/HTTP bodies. Persist only
            # exception type and the failing stage's status in public receipts.
            report['backends'][name] = dict(status='FAIL', error_type=type(error).__name__)
            print(name, 'FAIL', type(error).__name__, file=sys.stderr)
            if os.environ.get('TRACE_SMOKE_DEBUG') == '1':
                import traceback
                traceback.print_exc()
        report['backends'][name]['elapsed_seconds'] = round(time.monotonic()-started,2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(name, report['backends'][name]['status'], flush=True)
    return int(any(r['status']!='PASS' for r in report['backends'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
