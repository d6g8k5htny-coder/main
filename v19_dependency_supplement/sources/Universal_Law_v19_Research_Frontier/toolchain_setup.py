"""Prepare the bundled byte-pinned Linux x86_64 scoped Lean runtime, offline.

No download, shell command, arbitrary source or profile is accepted. Restoring
ZIP-lost execute permissions happens only after all reviewed file hashes pass.
The host kernel, dynamic loader and C/POSIX libraries remain system trust.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import stat

import formal_lane

BUNDLED_TOOLCHAIN = Path(__file__).resolve().parent / 'toolchains' / 'lean-4.34.1-linux-x86_64'
EXECUTABLES = ('bin/lean', 'bin/leanchecker')
HOST_TRUST = 'Linux kernel, system dynamic loader and system C/POSIX libraries; see OFFLINE_TOOLCHAIN.md'


class _Unsafe(ValueError):
    pass


def _supported():
    return platform.system() == 'Linux' and platform.machine().lower() in ('x86_64', 'amd64')


def default_toolchain() -> Path | None:
    """Choose explicit UL_LEAN_TOOLCHAIN first, then the supported bundled path.

    An explicit empty or nonexistent override never falls back to the bundle.
    This selector neither validates files nor changes permissions.
    """
    if 'UL_LEAN_TOOLCHAIN' in os.environ:
        value = os.environ['UL_LEAN_TOOLCHAIN']
        return Path(value).expanduser() if value else None
    if _supported() and BUNDLED_TOOLCHAIN.is_dir():
        return BUNDLED_TOOLCHAIN
    return None


def _open_root(root):
    # Open each directory through the previous descriptor. O_NOFOLLOW on just
    # the final path would still follow symlinked parent directories.
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_file(root_fd, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise _Unsafe('pinned path escapes toolchain')
    fd = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)


def _check_fd(fd, name, expected):
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected['bytes']:
        raise _Unsafe('pinned file type or size mismatch: ' + name)
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while block := os.read(fd, 1024 * 1024):
        digest.update(block)
    if digest.hexdigest() != expected['sha256']:
        raise _Unsafe('pinned file SHA-256 mismatch: ' + name)
    return metadata


def _inspect(path, restore):
    result = {'state': 'blocked', 'configured': False, 'reason_code': None,
              'reason': None, 'toolchain_root': None, 'restored_executables': [],
              'host_trust': HOST_TRUST, 'full_theorem_certificates': []}
    if not _supported():
        result.update(reason_code='unsupported_platform', reason=(
            'Bundled scoped Lean requires Linux x86_64; detected '
            + platform.system() + ' ' + platform.machine()))
        return result
    selected = path if path is not None else default_toolchain()
    if selected is None or str(selected) == '':
        result.update(reason_code='toolchain_missing', reason='No offline Lean toolchain selected; unset an empty UL_LEAN_TOOLCHAIN to use the bundle')
        return result
    root = Path(os.path.abspath(Path(selected).expanduser()))
    result['toolchain_root'] = str(root)
    if not root.exists() and not root.is_symlink():
        result.update(reason_code='toolchain_missing', reason='Selected offline Lean toolchain does not exist: ' + str(root))
        return result
    root_fd = None
    binaries = {}
    try:
        pins = formal_lane._pins()  # Validates the fixed reviewed manifest hash.
        root_fd = _open_root(root)
        for name, expected in pins['files'].items():
            fd = _open_file(root_fd, name)
            try:
                _check_fd(fd, name, expected)
                if name in EXECUTABLES:
                    binaries[name] = fd
                    fd = None
            finally:
                if fd is not None:
                    os.close(fd)
        result.update(validated_files=len(pins['files']),
                      validated_bytes=sum(p['bytes'] for p in pins['files'].values()),
                      pins_sha256=formal_lane.PINS_SHA256)
        missing_execute = []
        for name in EXECUTABLES:
            metadata = _check_fd(binaries[name], name, pins['files'][name])
            if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
                raise _Unsafe('privileged executable mode is not accepted: ' + name)
            if not metadata.st_mode & stat.S_IXUSR:
                missing_execute.append((name, metadata))
        if missing_execute and not restore:
            result.update(reason_code='execute_permission_missing', reason='ZIP extraction removed user execute bits; run python setup_final.py or python toolchain_setup.py')
            return result
        # Validate every restoration target before changing any target. Never
        # chmod through a path/symlink, nor change another hardlink's mode.
        if any(metadata.st_nlink != 1 for _, metadata in missing_execute):
            result.update(reason_code='unsafe_permission_restore', reason='Refusing execute-bit restoration on a hardlinked binary; extract private writable copies')
            return result
        for name, metadata in missing_execute:
            try:
                os.fchmod(binaries[name], stat.S_IMODE(metadata.st_mode) | stat.S_IXUSR)
            except OSError as exc:
                result.update(reason_code='permission_restore_failed', reason='Cannot restore execute permission; extract into a writable, executable filesystem: ' + str(exc))
                return result
            result['restored_executables'].append(name)
        # This includes native shadow/import-closure checks, independent file
        # validation, and a real sanitized --version subprocess.
        inventory = formal_lane.inventory(root)
        result['inventory'] = inventory
        if inventory['state'] != 'configured':
            result.update(reason_code=inventory['reason_code'], reason=inventory['reason'])
            return result
        result.update(state='configured', configured=True, reason_code=None, reason=None,
                      toolchain_version=inventory['toolchain_version'])
    except (OSError, ValueError) as exc:
        result.update(reason_code='toolchain_integrity_mismatch', reason='Offline pinned runtime rejected: ' + str(exc))
    finally:
        for fd in binaries.values():
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)
    return result


def prepare_toolchain(path=None) -> dict:
    """Hash-check all fixed pins, safely restore only user execute bits, probe Lean.

    Returns a diagnostic dictionary; missing, corrupt, unsupported or unwritable
    toolchains produce state='blocked' with a precise reason_code. Never downloads.
    """
    return _inspect(path, restore=True)


def diagnostics(path=None) -> dict:
    """Read-only hash validation and real version probe; never restore permissions."""
    return _inspect(path, restore=False)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchain')
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    diagnostic = (diagnostics if args.diagnostics else prepare_toolchain)(args.toolchain)
    print(json.dumps(diagnostic, indent=2))
    raise SystemExit(0 if diagnostic['configured'] else 1)
