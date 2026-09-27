"""Hash-first replay of the exact mathematical capsule linked in campaign #61.

The original analytic premises are NOT checked by this execution. No network
access or installation; obtain the pinned ZIP using an authorized Drive path.
The CLI only executes the exact pinned artifact, never an arbitrary ZIP.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

CAPSULE_SHA256 = '795115c377b9d37b87f953555f86c7f3f4b514298551e110391f3cba9e03340f'
CAPSULE_BYTES = 86308
DRIVE_ID = '1soVnQdYxLU8LHcQzteabic_Bdn6fD_dE'
HERE = Path(__file__).resolve().parent


def stage_capsule(source: Path, destination: Path, *,
                  expected_sha: str = CAPSULE_SHA256,
                  expected_bytes: int = CAPSULE_BYTES,
                  max_uncompressed: int = 1_048_576) -> int:
    """Validate all entries before creating destination; return file count.

    Keyword identities support synthetic tests. The CLI never overrides them.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('staging destination already exists')
    if source.stat().st_size != expected_bytes:
        raise ValueError('capsule byte-count mismatch')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('capsule SHA-256 mismatch')
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > 256 or sum(i.file_size for i in infos) > max_uncompressed:
            raise ValueError('capsule size/member ceiling exceeded')
        names = set()
        for info in infos:
            name = info.filename
            path = Path(name)
            if (path.is_absolute() or '..' in path.parts or '\\' in name
                    or ':' in name or not name or name in names):
                raise ValueError('unsafe or duplicate member path')
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('only regular file members are permitted')
            names.add(name)
        for name in names:
            if any(parent.as_posix() in names for parent in Path(name).parents
                   if parent != Path('.')):
                raise ValueError('file/directory member collision')
        if archive.testzip() is not None:
            raise ValueError('capsule CRC failure')
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    return len(infos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capsule', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == HERE or HERE in output.parents:
        raise ValueError('output must be outside the replay source directory')
    if output.exists():
        raise FileExistsError('choose a new output directory')
    # staging validates the exact bytes and paths before executing any member.
    source = output / 'source'
    count = stage_capsule(args.capsule, source)
    commands = [
        [sys.executable, '-B', '-S', 'verify_package.py'],
        [sys.executable, '-B', '-S', 'run_validation.py',
         '--output', str(output / 'validation')],
    ]
    runs = []
    for index, command in enumerate(commands):
        proc = subprocess.run(command, cwd=source, capture_output=True, timeout=180)
        (output / f'step{index}.stdout').write_bytes(proc.stdout)
        (output / f'step{index}.stderr').write_bytes(proc.stderr)
        runs.append({'command': command, 'returncode': proc.returncode})
        if proc.returncode:
            break
    passed = len(runs) == 2 and all(x['returncode'] == 0 for x in runs)
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'capsule_sha256': CAPSULE_SHA256, 'capsule_bytes': CAPSULE_BYTES,
              'archive_members': count, 'runs': runs, 'passed': passed,
              'meaning': 'pinned finite replay, not analytic proof or independent review'}
    (output / 'CAPSULE_REPLAY.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
