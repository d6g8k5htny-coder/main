"""Selected synchronous auditor mutation outcomes with live origin evidence.

The existing inventory recorder captures traceback code identities before
unittest cleanup. Only the selected test body and its finite set of standard
unittest assertion helpers authorize a kill. This is a trusted test harness,
not a sandbox for hostile test code or a mathematical certifier.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys

from tests import cbz_inventory_origin

ROOT = Path(__file__).resolve().parents[1]


def retain_full(tool_path, environment_key, process, before_bytes):
    """Preserve full-suite baseline/failed-child logs when evidence is requested."""
    after_bytes = Path(tool_path).read_bytes()
    directory = os.environ.get('CBZ_AUDITOR_EVIDENCE_DIR')
    if directory:
        target = Path(directory) / 'full'
        target.mkdir(exist_ok=True)
        nonce = secrets.token_hex(16)
        stdout, stderr = process.stdout.encode(), process.stderr.encode()
        for suffix, data in (('.before.py', before_bytes), ('.after.py', after_bytes),
                             ('.stdout', stdout), ('.stderr', stderr)):
            with (target / (nonce + suffix)).open('xb') as stream:
                stream.write(data)
        row = {'nonce': nonce, 'environment_key': environment_key,
               'optimize': sys.flags.optimize, 'command': process.args,
               'exit_code': process.returncode,
               'source_before': hashlib.sha256(before_bytes).hexdigest(),
               'source_after': hashlib.sha256(after_bytes).hexdigest(),
               'stdout_sha256': hashlib.sha256(stdout).hexdigest(),
               'stderr_sha256': hashlib.sha256(stderr).hexdigest()}
        with (target / (nonce + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(row, stream, sort_keys=True, indent=2)
            stream.write('\n')
    if before_bytes != after_bytes:
        raise RuntimeError('full-suite subject changed during execution')


def run_selected(tool_path, environment_key, test_id):
    """Run one designated test and bind the report to unchanged subject bytes."""
    try:
        before_bytes = Path(tool_path).read_bytes()
        before = hashlib.sha256(before_bytes).hexdigest()
    except OSError as error:
        return {'classification': 'invalid', 'protocol_error': True, 'error': repr(error)}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env[environment_key] = tool_path
    report = cbz_inventory_origin.run(ROOT, test_id, [test_id], test_id,
                                      env=env, timeout=600)
    try:
        after_bytes = Path(tool_path).read_bytes()
        after = hashlib.sha256(after_bytes).hexdigest()
    except OSError as error:
        after_bytes, after = None, None
        report.update(classification='invalid', protocol_error=True, error=repr(error))
    report['source_before'] = before
    report['source_after'] = after
    if before != after:
        report['classification'] = 'invalid'
    directory = os.environ.get('CBZ_AUDITOR_EVIDENCE_DIR')
    if directory:
        # An explicitly requested evidence write fails the test if it cannot
        # be completed; collision-safe exclusive creation never overwrites a run.
        nonce = secrets.token_hex(16)
        report.update(evidence_nonce=nonce, environment_key=environment_key,
                      optimize=sys.flags.optimize)
        target = Path(directory)
        with (target / (nonce + '.before.py')).open('xb') as stream:
            stream.write(before_bytes)
        if after_bytes is not None:
            with (target / (nonce + '.after.py')).open('xb') as stream:
                stream.write(after_bytes)
        with (target / (nonce + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')
    return report


def outcome(report, expected_test):
    """Reclassify a source-bound report; refuse missing provenance or drift."""
    if type(report) is not dict or type(expected_test) is not str:
        return 'invalid'
    selected = report.get('selected_id')
    before, after = report.get('source_before'), report.get('source_after')
    if (type(selected) is not str or selected.rsplit('.', 1)[-1] != expected_test or
            type(before) is not str or len(before) != 64 or
            any(c not in '0123456789abcdef' for c in before) or before != after):
        return 'invalid'
    actual = cbz_inventory_origin.classify(report, [selected], selected, {})
    return actual if report.get('classification') == actual else 'invalid'


def killed(report, expected_test=None):
    """Refuse anything except an authorized selected-body assertion failure."""
    return outcome(report, expected_test) == 'kill'
