#!/usr/bin/env python3
"""RF-GATE-01 compatibility entrypoint.

The reviewed compiled-environment gate remains byte-identical in ``gate_helper.py``.
This module re-exports its public API, binds a constrained manifest overlay, and
retains the later exact-runtime/dependency observation contract from current main.
Scientific effect: NONE.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if 'gate_helper' in sys.modules:
    _core = sys.modules['gate_helper']
else:
    _helper_path = Path(__file__).resolve().with_name('gate_helper.py')
    _helper_spec = importlib.util.spec_from_file_location('rf312_gate_helper', _helper_path)
    if _helper_spec is None or _helper_spec.loader is None:
        raise ImportError('cannot load reviewed RF312 helper: ' + str(_helper_path))
    _core = importlib.util.module_from_spec(_helper_spec)
    _helper_spec.loader.exec_module(_core)

# Re-export the reviewed helper API. Tests and callers continue importing `gate`.
for _name in dir(_core):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_core, _name)

ROOT = Path(__file__).resolve().parent
OVERLAY = ROOT / 'manifest.overlay.json'
_HELPER_LOAD_JSON = _core.load_json
_HELPER_SOURCE_CHECK = _core.source_check
_HELPER_EXECUTE = _core.execute
_ALLOWED_OVERLAY_KEYS = frozenset({'schema_version', 'files'})


def _sync_root():
    """Propagate this module's effective ``ROOT`` to the reviewed helper at every call boundary.

    The helper reads its own module global ``ROOT`` (and this module derives the overlay path
    from ``ROOT``) at call time, so a caller that rebinds ``gate.ROOT`` -- the fixtures of the
    runtime tests do -- must be observed by every source, overlay and execution read, not only
    by this wrapper. Import-time constants would silently keep reading the original tree.
    """
    global OVERLAY
    root = Path(ROOT)
    _core.ROOT = root
    OVERLAY = root / 'manifest.overlay.json'
    return root


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_lean_version(text):
    version = text.strip()
    require(re.fullmatch(
        r'Lean \(version 4\.34\.1, [^()\r\n]+, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release\)',
        version) is not None, 'unexpected running Lean version: ' + repr(version))
    return version


def check_worktree_clean(name, status):
    require(status.strip() == '',
            'dependency worktree is not clean: ' + name + ': ' + status.strip()[:200])


def dependency_state(m):
    root = _sync_root()
    actual = {}
    for name in m['dependency_revisions']:
        repo = str(root / '.lake' / 'packages' / name)
        actual[name] = subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
        check_worktree_clean(name, subprocess.check_output(
            ['git', '-C', repo, 'status', '--porcelain', '--untracked-files=normal'], text=True))
    require(actual == m['dependency_revisions'], 'installed dependency HEAD differs from lock')
    return actual


def observation(version, dependencies, digest):
    return {'lean_version': version, 'dependency_revisions': dependencies,
            'dependency_worktrees': 'CLEAN', 'manifest_sha256': digest}


def _overlay_bytes():
    overlay = _sync_root() / 'manifest.overlay.json'
    require(overlay.is_file() and not overlay.is_symlink(), 'missing or symlink manifest overlay')
    return overlay.read_bytes()


def _effective_manifest(text):
    base = _HELPER_LOAD_JSON(text)
    # Only the formal manifest has this shape; lake/alignment JSON passes through unchanged.
    if not (isinstance(base, dict) and 'source_modules' in base and 'targets' in base and 'files' in base):
        return base
    overlay = _HELPER_LOAD_JSON(_overlay_bytes().decode())
    require(isinstance(overlay, dict) and set(overlay) == _ALLOWED_OVERLAY_KEYS,
            'invalid manifest overlay keys')
    require(overlay.get('schema_version') == 1 and isinstance(overlay.get('files'), dict),
            'invalid manifest overlay schema')
    require(all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v)
                for k, v in overlay['files'].items()), 'invalid manifest overlay file binding')
    merged = dict(base)
    merged['files'] = dict(base['files'])
    merged['files'].update(overlay['files'])
    return merged


def source_check():
    root = _sync_root()
    raw = (root / 'manifest.json').read_bytes()
    original = _core.load_json
    try:
        _core.load_json = _effective_manifest
        m, _ = _HELPER_SOURCE_CHECK()
    finally:
        _core.load_json = original
    # The overlay is read after the helper's own contract has passed: a manifest that fails its
    # required-file inventory is refused for that reason, whether or not an overlay exists.
    return m, sha(raw + b'\0RF312-OVERLAY\0' + _overlay_bytes())


def execute(m, digest):
    root = _sync_root()
    out = root / '.lake' / 'formal-evidence'
    out.mkdir(parents=True, exist_ok=True)
    require(not (root / '.lake').is_symlink() and not (root / '.lake/build').is_symlink(),
            'symlink build directory')
    # Keep test monkeypatches visible to the reviewed helper execution.
    _core.run = globals()['run']
    _core.source_check = source_check
    preflight_version = check_lean_version(run(
        ['lake', 'env', 'lean', '--version'], 'version-preflight', out))
    preflight = observation(preflight_version, dependency_state(m), digest)
    receipt = _HELPER_EXECUTE(m, digest)
    try:
        _, after = source_check()
        require(after == digest, 'manifest or overlay changed during execution')
        version = check_lean_version(run(['lake', 'env', 'lean', '--version'], 'version-final', out))
        final = observation(version, dependency_state(m), after)
        require(final == preflight, 'runtime, dependencies or sources changed during execution')
    except BaseException:
        # The reviewed helper has already written its own receipt. A run whose final observation
        # is refused must leave no receipt behind (the logs stay as evidence of the refusal).
        (out / 'receipt.json').unlink(missing_ok=True)
        raise
    receipt = dict(receipt)
    receipt.update(
        lean_version=version,
        dependency_revisions=m['dependency_revisions'],
        dependency_worktrees='CLEAN',
        workflow_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'),
        observations={'preflight': preflight, 'final': final},
        manifest_sha256=digest,
    )
    receipt['logs'] = {p.name: sha(p.read_bytes()) for p in sorted(out.glob('*.log'))}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true',
                   help='run build, compiled inventory, transitive axiom audit and controls')
    p.add_argument('--alignment', type=Path,
                   help='validate a separately authenticated alignment record; never promotes science')
    args = p.parse_args()
    try:
        m, digest = source_check()
        if args.alignment:
            check_alignment(load_json(args.alignment.read_text()), digest,
                            m['targets'], m['files']['SCOPE.md'])
        if args.execute:
            execute(m, digest)
        else:
            print('SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): ' + digest)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as e:
        print('FORMAL_GATE_FAIL: ' + str(e), file=sys.stderr)
        return 1
    return 0


# The helper must see the effective source check even when it calls back internally.
_core.source_check = source_check

if __name__ == '__main__':
    sys.exit(main())
