"""Fresh custody controls for the reused authored N6 implementation."""
import json

import pytest

from research.rn import n6_inputs


def test_authenticates_delivered_module_and_warm_cache():
    first, identities = n6_inputs.checked_n6()
    second, again = n6_inputs.checked_n6()
    assert first is second
    assert identities == again
    assert len(identities) == 29
    assert first.__file__.endswith('::rn_n6/side24_taylor.py')


@pytest.mark.parametrize('mutation', ['context', 'access', 'hash', 'path', 'duplicate', 'duplicate_key'])
def test_warm_cache_does_not_bypass_inventory_gate(monkeypatch, mutation):
    n6_inputs.checked_n6()
    original = n6_inputs.runtime.read_bounded

    def changed(path, limit):
        raw = original(path, limit)
        if path != n6_inputs.ROOT / 'drive/inventory.jsonl':
            return raw
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        row = next(x for x in rows if x.get('id') == n6_inputs.RN5_ID)
        if mutation == 'context':
            row['context'] = 'REFERENCE_ONLY'
        elif mutation == 'access':
            row['access_status'] = 'READ_FAILED'
        elif mutation == 'hash':
            row['sha256'] = '0' * 64
        elif mutation == 'path':
            row['path'] = '01_ACTIVE_RESEARCH_PACKAGES/99_DO_NOT_OPEN/source'
        elif mutation == 'duplicate':
            rows.append(dict(row))
        else:
            return raw + b'\n{"id":"x","id":"y"}\n'
        return '\n'.join(json.dumps(x) for x in rows).encode()

    monkeypatch.setattr(n6_inputs.runtime, 'read_bounded', changed)
    with pytest.raises(ValueError, match='ineligible|ambiguous|duplicate JSON key'):
        n6_inputs.checked_n6()


def test_warm_cache_does_not_bypass_coverage_gate(monkeypatch):
    n6_inputs.checked_n6()
    original = n6_inputs.runtime.read_bounded

    def changed(path, limit):
        raw = original(path, limit)
        if path == n6_inputs.ROOT / 'drive/source_map/Payloads.csv':
            lines = raw.decode().splitlines()
            lines = [line.replace('RESEARCH_SOURCE_CHECK_STATUS', 'REFERENCE_ONLY')
                     if n6_inputs.RN5_SHA in line else line for line in lines]
            return '\n'.join(lines).encode()
        return raw

    monkeypatch.setattr(n6_inputs.runtime, 'read_bounded', changed)
    with pytest.raises(ValueError, match='coverage eligibility'):
        n6_inputs.checked_n6()


def test_warm_cache_does_not_bypass_archive_check(monkeypatch):
    n6_inputs.checked_n6()

    def refuse(*_):
        raise ValueError('deliberately corrupted original archive')

    monkeypatch.setattr(n6_inputs.archive_check, 'inspect_archive', refuse)
    with pytest.raises(ValueError, match='corrupted original archive'):
        n6_inputs.checked_n6()
