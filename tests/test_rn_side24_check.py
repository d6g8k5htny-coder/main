"""Source-bound CLI controls; successful algebra is not a spatial theorem."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import rn_side24_check as C


@pytest.fixture(scope='module')
def candidate():
    return C.load_report(C.ROOT / C.CANDIDATE)


def test_frozen_candidate_reconstructs(candidate):
    result = C.check_report(candidate)
    assert len(result['conditioning_pivots']) == 9
    assert all(F(lo) > 0 for lo, _ in result['conditioning_pivots'])
    h = result['conditional_holder']
    assert h['powers'] == {'M': 4, 'S': 4, 'y': 2}
    assert F(h['upper']) ** 4 >= F(h['fourth_power_upper']) > 0


@pytest.mark.parametrize('field,value', [
    ('point', ['1', '0']), ('round_bits', True), ('schema', 'PROMOTED'),
    ('scope', {**C.SCOPE, 'spatial_cover_certified': True}),
])
def test_boundary_mutations_fail_before_reconstruction(candidate, monkeypatch, field, value):
    changed = copy.deepcopy(candidate)
    changed[field] = value
    monkeypatch.setattr(C, 'build_report', lambda *_: pytest.fail('invalid scope entered mathematical replay'))
    with pytest.raises(ValueError):
        C.check_report(changed)


def test_valid_portable_algebra_does_not_replace_field_inputs(candidate, monkeypatch):
    changed = copy.deepcopy(candidate)
    changed['moment_certificates']['y']['claim']['context']['conditioned_law'] += '-different'
    monkeypatch.setattr(C, 'build_report', lambda *_: candidate)
    with pytest.raises(ValueError, match='reconstructed'):
        C.check_report(changed)


def test_caps_and_source_digest_are_part_of_exact_replay(candidate, monkeypatch):
    monkeypatch.setattr(C, 'build_report', lambda *_: candidate)
    for field in ('cap', 'source', 'extra', 'coefficient'):
        changed = copy.deepcopy(candidate)
        if field == 'cap':
            changed['conditional_holder']['upper'] = '0'
        elif field == 'source':
            changed['sources'][C.SOURCE]['sha256'] = '0' * 64
        elif field == 'extra':
            changed['accept_without_replay'] = True
        else:
            changed['moment_certificates']['y']['proof']['coefficients'][0] = ['0', '0']
        with pytest.raises(ValueError, match='reconstructed'):
            C.check_report(changed)


def test_cli_flag_reads_supplied_mutant(candidate, tmp_path):
    changed = copy.deepcopy(candidate)
    changed['scope']['h3_normalizer_included'] = True
    path = tmp_path / 'mutant.json'
    path.write_text(json.dumps(changed))
    for optimization in ([], ['-O']):
        result = subprocess.run([sys.executable, *optimization, str(C.ROOT / 'tools/rn_side24_check.py'),
                                 '--check', str(path)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 1 and 'REJECTED' in result.stdout


def test_cli_rejects_a_numerical_mutation_after_reconstruction(candidate, tmp_path):
    changed = copy.deepcopy(candidate)
    changed['conditional_holder']['upper'] = '0'
    path = tmp_path / 'wrong-bound.json'
    path.write_text(json.dumps(changed))
    result = subprocess.run([sys.executable, '-O', str(C.ROOT / 'tools/rn_side24_check.py'),
                             '--check', str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'reconstructed SIDE24' in result.stdout


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1.2}'])
def test_parser_refuses_ambiguous_json(raw, tmp_path):
    path = tmp_path / 'bad.json'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        C.load_report(path)


def test_missing_or_wrong_source_fails_closed(tmp_path):
    path = tmp_path / C.SOURCE
    path.parent.mkdir(parents=True)
    path.write_text('untrusted replacement')
    with pytest.raises(ValueError, match='source identity'):
        C.checked_sources(tmp_path)


def test_output_refuses_overwrite(tmp_path, monkeypatch):
    output = tmp_path / 'exists.json'
    output.write_text('old bytes')
    monkeypatch.setattr(C, 'build_report', lambda: {})
    assert C.main(['--output', str(output)]) == 1
    assert output.read_text() == 'old bytes'
