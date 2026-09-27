#!/usr/bin/env python3
"""Create an audited, metadata-normalized ZIP outside the source directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
if __package__:
    from .audit_anonymous_release import audit
else:
    from audit_anonymous_release import audit


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deny-patterns', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if root == output or root in output.parents:
        parser.error('the output archive must be outside the source directory')
    if output.exists():
        parser.error('refusing to overwrite an existing archive')
    patterns = args.deny_patterns.read_text().splitlines() if args.deny_patterns else []
    issues = audit(root, [p for p in patterns if p.strip()])
    if issues:
        print(json.dumps({'passed': False, 'issues': issues}, indent=2))
        return 1
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SOURCE_MANIFEST.json')
    manifest = {'schema_version': 'anonymous_source_manifest_v1', 'files': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        def add(name: str, data: bytes, executable: bool = False) -> None:
            info = zipfile.ZipInfo('TRACE/' + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if executable else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        for path in files:
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            manifest['files'][rel] = hashlib.sha256(data).hexdigest()
            add(rel, data, path.suffix == '.sh')
        add('SOURCE_MANIFEST.json', (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({'passed': True, 'files': len(files) + 1, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
