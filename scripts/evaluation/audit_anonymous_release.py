#!/usr/bin/env python3
"""Offline, redacted checks for an anonymous source distribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

OFFICIAL_HOSTS = {
    'api.openai.com', 'api.deepseek.com', 'generativelanguage.googleapis.com',
    'dashscope-us.aliyuncs.com', 'api.anthropic.com',
}


def audit(root: Path, deny_patterns: list[str]) -> list[dict[str, object]]:
    issues = []
    patterns = {
        'credential_token': r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,})',
        'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'personal_home': r'/(?:Users|home)/[^/\s]+/',
    }
    for index, expression in enumerate(deny_patterns):
        patterns[f'private_deny_pattern_{index}'] = expression
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_symlink():
            issues.append({'file': str(rel), 'rule': 'symlink'})
            continue
        if path.name in {'.git', '.env', '.DS_Store', '.venv', '__pycache__', '.pytest_cache'} or path.suffix in {'.pyc', '.pyo'}:
            issues.append({'file': str(rel), 'rule': 'private_or_generated_file'})
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            issues.append({'file': str(rel), 'rule': 'opaque_binary_requires_review'})
            continue
        for rule, pattern in patterns.items():
            flags = re.IGNORECASE if rule.startswith('private_deny_pattern_') else 0
            for match in re.finditer(pattern, text, flags):
                issues.append({'file': str(rel), 'line': text[:match.start()].count('\n') + 1, 'rule': rule})
        if rel.parts[0] == 'configs' and path.suffix == '.json':
            def inspect(value):
                if isinstance(value, dict):
                    if 'api_key' in value:
                        issues.append({'file': str(rel), 'rule': 'inline_key_field'})
                    if 'api_base' in value:
                        if urlsplit(value['api_base']).hostname not in OFFICIAL_HOSTS:
                            issues.append({'file': str(rel), 'rule': 'nonofficial_model_endpoint'})
                        if not value.get('api_key_env'):
                            issues.append({'file': str(rel), 'rule': 'missing_credential_variable'})
                    for item in value.values(): inspect(item)
                elif isinstance(value, list):
                    for item in value: inspect(item)
                elif isinstance(value, str) and value.startswith('configs/') and not (root / value).is_file():
                    issues.append({'file': str(rel), 'rule': 'missing_config_dependency'})
            inspect(json.loads(text))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--deny-patterns', type=Path, help='Optional private regex list OUTSIDE the release; one regex per line')
    args = parser.parse_args()
    expressions = args.deny_patterns.read_text().splitlines() if args.deny_patterns else []
    issues = audit(args.root.resolve(), [x for x in expressions if x.strip()])
    print(json.dumps({'passed': not issues, 'issue_count': len(issues), 'issues': issues}, indent=2))
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
