#!/usr/bin/env python3
"""Download pinned benchmark inputs and verify their recorded SHA-256 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_path(directory: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or not parts.parts:
        raise ValueError(f'Unsafe dataset path: {relative}')
    path = directory.joinpath(*parts.parts)
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError(f'Dataset path contains a symlink: {path}')
        if parent == directory:
            break
    return path


def matches(path: Path, record: dict) -> bool:
    return path.is_file() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': 'TRACE-dataset-preparation/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open('xb') as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def install(source: Path, destination: Path, record: dict) -> None:
    if not matches(source, record):
        raise ValueError(f'Download checksum mismatch: {record["path"]}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing research inputs and works on NFS
    # installations that do not support atomic rename.
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    if not matches(destination, record):
        raise ValueError(f'Installed checksum mismatch: {destination}')


def prepare(name: str, data_root: Path, verify_only: bool = False) -> dict:
    metadata_root = ROOT / 'data' / name
    metadata = json.loads((metadata_root / 'UPSTREAM.json').read_text())
    records = json.loads((metadata_root / 'FILES.sha256.json').read_text())
    directory = data_root / name
    pending = []
    for record in records:
        path = target_path(directory, record['path'])
        if matches(path, record):
            continue
        if path.exists():
            raise ValueError(f'Existing file differs from pinned data; refusing to overwrite: {path}')
        pending.append(record)
    if pending and verify_only:
        raise ValueError(f'{name}: {len(pending)} files missing; run prepare_data.py --dataset {name}')
    if pending:
        print(f'{name}: preparing {len(pending)} pinned files...', file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(prefix='trace-data-') as temp:
            temporary = Path(temp)
            if metadata['download_kind'] == 'github_archive':
                archive = temporary / 'source.tar.gz'
                download(metadata['archive_url'], archive)
                needed = {record['path']: record for record in pending}
                with tarfile.open(archive, 'r:gz') as source:
                    for member in source:
                        relative = member.name.partition('/')[2]
                        if relative not in needed:
                            continue
                        if not member.isfile():
                            raise ValueError(f'Unexpected archive member: {member.name}')
                        record = needed[relative]
                        if member.size != record['bytes']:
                            raise ValueError(f'Unexpected member size: {relative}')
                        stream = source.extractfile(member)
                        if stream is None:
                            raise ValueError(f'Cannot read archive member: {relative}')
                        staged = temporary / 'member'
                        with stream, staged.open('xb') as writer:
                            shutil.copyfileobj(stream, writer)
                        install(staged, target_path(directory, relative), record)
                        staged.unlink()
                        del needed[relative]
                if needed:
                    raise ValueError(f'Pinned archive is missing {len(needed)} files')
            elif metadata['download_kind'] == 'files':
                for index, record in enumerate(pending):
                    staged = temporary / str(index)
                    download(metadata['file_base_url'] + '/' + record['path'], staged)
                    install(staged, target_path(directory, record['path']), record)
            else:
                raise ValueError(f'Bundled {name} files are missing; restore them from Git')
    return {'dataset': name, 'status': 'verified', 'files': len(records),
            'bytes': sum(record['bytes'] for record in records),
            'source_revision': metadata['revision']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'manbench', 'stale', 'Memora'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--verify-only', action='store_true', help='Check local files without downloading.')
    args = parser.parse_args(argv)
    names = ['manbench', 'stale', 'Memora'] if args.dataset == 'all' else [args.dataset]
    try:
        for name in names:
            print(json.dumps(prepare(name, args.data_dir.resolve(), args.verify_only)), flush=True)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
