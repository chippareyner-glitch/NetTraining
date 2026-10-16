"""Build a source-only handoff ZIP from an explicit allowlist, without uploading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ['README.md', 'CONTRIBUTING.md', 'pyproject.toml', 'run.py', 'run.ps1',
              'requirements.txt', 'requirements_train.txt', 'requirements_period_noise.txt', '.gitignore']
SOURCE_DIRS = ['src', 'docs', 'tests', 'firmware', 'scripts', '.github']
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.git', '.env'}
ALLOWED_SUFFIXES = {'.py', '.md', '.toml', '.ps1', '.txt', '.html', '.c', '.yml', '.yaml'}
PATTERNS = {
    'local_user_or_project_path': re.compile(r'[A-Za-z]:[\\/](?:Users[\\/]|CH32[\\/])', re.I),
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
    'credential_assignment': re.compile(r'''(?i)\b(?:api_key|access_token|password|secret)\s*=\s*["'][A-Za-z0-9_+./=-]{12,}["']'''),
}


def collect_files() -> list[Path]:
    paths = [ROOT / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file())
    result = []
    for path in paths:
        path.resolve().relative_to(ROOT.resolve())
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.name == '.gitignore' or path.suffix in ALLOWED_SUFFIXES:
            result.append(path)
        else:
            raise ValueError(f'Unexpected source-file type: {relative.as_posix()}')
    return sorted(set(result))


def audit(paths: list[Path]) -> list[dict]:
    findings = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({'path': path.relative_to(ROOT).as_posix(), 'line': number, 'kind': kind})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Audit the selected files without building a ZIP.')
    parser.add_argument('--out', default='dist/plant-signal-source.zip')
    args = parser.parse_args()
    paths = collect_files()
    findings = audit(paths)
    if findings:
        print(json.dumps({'findings': findings}, ensure_ascii=False, indent=2))
        return 1
    if args.check:
        print(f'Source audit passed: {len(paths)} files selected; no configured patterns found.')
        return 0
    output = (ROOT / args.out).resolve()
    output.relative_to(ROOT.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {entry['path'] for entry in manifest}
        for entry in manifest:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
    report = {'files': manifest, 'file_count': len(paths), 'zip_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'configured_pattern_findings': [], 'scope': 'Allowlisted source files only; static checks, no upload.'}
    output.with_suffix('.manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Created {output.relative_to(ROOT).as_posix()}: {len(paths)} files; ZIP hashes verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
