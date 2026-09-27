"""Offline checks for the public CLI and safe, reproducible data preparation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from prepare_data import install, prepare, target_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('entry,args,expected', [
    ('eval.py', ['manbench', '--help'], '--max-episodes'),
    ('eval.py', ['stale', '--help'], '--workers'),
    ('eval.py', ['memora', '--help'], '--state-mode'),
    ('eval.py', ['memora-score', '--help'], '--judge-model'),
    ('analyze.py', ['manbench', '--help'], '--output-dir'),
    ('analyze.py', ['memora', '--help'], '--bootstrap-replicates'),
])
def test_delegates_help_without_model_calls(entry, args, expected):
    result = subprocess.run([sys.executable, str(ROOT / entry), *args],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_invalid_benchmark_is_rejected():
    result = subprocess.run([sys.executable, str(ROOT / 'eval.py'), 'unknown'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2


def test_download_must_match_digest_and_must_not_overwrite(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'benchmark')
    record = {'path': 'file', 'bytes': 9, 'sha256': hashlib.sha256(b'benchmark').hexdigest()}
    dest = tmp_path / 'out' / 'file'
    install(source, dest, record)
    with pytest.raises(FileExistsError):
        install(source, dest, record)
    assert dest.read_bytes() == b'benchmark'
    with pytest.raises(ValueError, match='checksum'):
        install(source, tmp_path / 'bad', {**record, 'sha256': '0' * 64})
    assert not (tmp_path / 'bad').exists()


def test_dataset_paths_cannot_escape_or_follow_symlinks(tmp_path):
    with pytest.raises(ValueError):
        target_path(tmp_path, '../outside')
    with pytest.raises(ValueError):
        target_path(tmp_path, '/outside')
    (tmp_path / 'link').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        target_path(tmp_path, 'link/file')


def test_verify_only_does_not_download_missing_data(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        prepare('stale', tmp_path, verify_only=True)
    assert list(tmp_path.iterdir()) == []


def test_manbench_bundled_files_match_pinned_checksums():
    assert prepare('manbench', ROOT / 'data', verify_only=True)['files'] == 21


def test_configs_use_canonical_data_paths():
    for path in (ROOT / 'configs').rglob('*.json'):
        text = path.read_text()
        json.loads(text)
        assert 'data/vendor/manbench' not in text
        assert 'external/datasets/' not in text
        assert 'external/upstreams/' not in text


def test_frozen_manifest_digests_remain_loadable_after_path_migration():
    from trace.memora_return import load_memora_manifest
    from trace.stale_type2_return import load_sidecar_manifest

    assert len(load_memora_manifest(ROOT / 'configs/shared/memora_return_confirmation_manifest.json')) == 137
    assert len(load_sidecar_manifest(ROOT / 'configs/shared/stale_type2_official_mas_return.json')) == 200
