"""Verify retained diagnostics using their exact historical protocol/runtime.

This dispatcher is only for archival example verification. Live scheduling keeps
using workers.py. No recorded command is executed and no theorem is certified.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile
import hashlib

import workers as current_workers

ROOT = Path(__file__).resolve().parent
LEGACY_PROTOCOL = 'UL-worker-v17.1'
SEALED_ARCHIVE = 'Universal_Law_Mathematical_Conductor_v17.zip'
SEALED_SHA256 = '0deaa4e91cd5cf011236592cc5d91237b52a830e9f03b193227d4bac95ab9814'
SEALED_PREFIX = 'Universal_Law_Mathematical_Conductor_v17/'
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_EXTRACTED_BYTES = 32 * 1024 * 1024
REPLAY_TIMEOUT = 45

# Fixed adapter code only: neither receipt.command nor replay_command is run.
_CHILD = '''import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from workers import verify_receipt
result = verify_receipt(Path(sys.argv[1]), root=Path(sys.argv[2]), replay=sys.argv[3] == '1')
print(json.dumps(result, sort_keys=True))
'''


def _sealed_archive(root):
    return Path(root).resolve().parent / 'history' / 'sealed' / SEALED_ARCHIVE


def _extract_sealed(raw, destination):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) > 1000 or sum(item.file_size for item in entries) > MAX_EXTRACTED_BYTES:
            raise ValueError('sealed v17 extraction exceeds bound')
        for item in entries:
            if not item.filename.startswith(SEALED_PREFIX):
                raise ValueError('sealed v17 member outside expected root')
            relative = PurePosixPath(item.filename[len(SEALED_PREFIX):])
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
                raise ValueError('unsafe sealed v17 archive member')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('sealed v17 archive contains a symlink')
            target = destination.joinpath(*relative.parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(archive.read(item))


def verify_retained_receipt(receipt_path, root=ROOT, replay=False):
    """Dispatch only known protocols, preserving original checker-byte bindings.

    Legacy replay uses the sealed v17 sources in a private temporary directory
    and isolated Python child. A failure is final; protocols are never tried as
    fallbacks, and historical receipts are never rewritten or relabelled.
    """
    receipt_path = Path(receipt_path).resolve()
    if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError('retained receipt exceeds input bound')
    receipt = json.loads(receipt_path.read_bytes())
    if not isinstance(receipt, dict):
        raise ValueError('retained receipt must be an object')
    protocol = receipt.get('protocol_version')
    if protocol == current_workers.PROTOCOL:
        return current_workers.verify_receipt(receipt_path, root=root, replay=replay)
    if protocol != LEGACY_PROTOCOL:
        raise ValueError('unsupported retained worker receipt protocol: ' + str(protocol))
    archive_path = _sealed_archive(root)
    if not archive_path.is_file() or archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('required sealed v17 archive is missing or exceeds bound')
    raw = archive_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SEALED_SHA256:
        raise ValueError('sealed v17 archive hash mismatch')
    with tempfile.TemporaryDirectory(prefix='ul-retained-v17-runtime-') as temp:
        runtime = Path(temp) / 'engine'
        runtime.mkdir()
        _extract_sealed(raw, runtime)
        command = [sys.executable, '-I', '-B', '-c', _CHILD,
                   str(receipt_path), str(runtime), '1' if replay else '0']
        try:
            completed = subprocess.run(command, cwd=runtime,
                env={'PATH': os.defpath, 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'},
                stdin=subprocess.DEVNULL, capture_output=True, timeout=REPLAY_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ValueError('sealed v17 receipt verification exceeded wall bound') from exc
    if len(completed.stdout) > MAX_RECEIPT_BYTES or len(completed.stderr) > MAX_RECEIPT_BYTES:
        raise ValueError('sealed v17 verifier output exceeded bound')
    if completed.returncode:
        raise ValueError('sealed v17 receipt verification failed: ' + completed.stderr.decode('utf8', 'replace')[-6000:])
    result = json.loads(completed.stdout)
    if not isinstance(result, dict) or result.get('valid') is not True or result.get('formal_certification') != 'not_certified':
        raise ValueError('sealed v17 verifier did not return an uncertified diagnostic verdict')
    result['verification_runtime'] = {'protocol': LEGACY_PROTOCOL,
        'sealed_archive_sha256': SEALED_SHA256,
        'scope': 'historical diagnostic custody/replay only; no mathematical certification'}
    return result
