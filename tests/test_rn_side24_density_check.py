"""Integrand composition and adversarial source/scope/CLI controls."""
import copy
from fractions import Fraction as F
import json
import subprocess
import sys

import pytest

from tools import rn_side24_density_check as C


@pytest.fixture(scope='module')
def candidate():
    return C.point_check.load_report(C.ROOT / C.CANDIDATE)


def test_candidate_reconstructs_and_composes_exactly(candidate):
    result = C.check_report(candidate)
    density = result['density_window']
    composition = result['composition']
    holder = F(result['moment_reconstruction']['holder_upper'])
    mass = F(density['window_mass_upper'])
    pgrad = F(density['gradient_density'][1])
    assert F(composition['numerator_upper']) == holder * pgrad * mass
    assert F(composition['normalized_upper']) == holder * pgrad * mass / C.H3_FLOOR
    assert 0 < F(composition['normalized_upper']) <= F(composition['simple_rational_upper'])
    assert F(composition['simple_rational_upper']) == F(1888043, 500000000000)
    assert F(density['height_window_length']) == F(1, 48000)
    assert F(result['admission_margins']['minimum_pivot_lower']) > 0


@pytest.mark.parametrize('field,value', [
    ('point', ['1', '0']), ('round_bits', True), ('schema', 'UNCONDITIONAL'),
    ('scope', {**C.SCOPE, 'h3_floor_reproved': True}),
    ('scope', {**C.SCOPE, 'spatial_cover_certified': True}),
])
def test_false_scope_stops_before_replay(candidate, monkeypatch, field, value):
    changed = copy.deepcopy(candidate)
    changed[field] = value
    monkeypatch.setattr(C, 'build_report', lambda *_: pytest.fail('invalid scope entered replay'))
    with pytest.raises(ValueError, match='scope'):
        C.check_report(changed)


@pytest.mark.parametrize('mutation', ['extra_r_squared', 'unconditioned_height',
                                    'drop_density', 'raise_floor', 'wrong_source',
                                    'wrong_powers', 'wrong_margin'])
def test_composition_and_custody_mutations_rejected(candidate, monkeypatch, mutation):
    changed = copy.deepcopy(candidate)
    if mutation == 'extra_r_squared':
        changed['composition']['normalized_upper'] = str(F(changed['composition']['normalized_upper']) / 400)
    elif mutation == 'unconditioned_height':
        changed['density_window']['height_mean'] = changed['density_window']['jet_mean'][0]
    elif mutation == 'drop_density':
        changed['density_window']['gradient_density'] = ['1', '1']
    elif mutation == 'raise_floor':
        changed['composition']['imported_h3_floor'] = '1'
    elif mutation == 'wrong_source':
        changed['sources'][C.PIN_TRANSFORM]['sha256'] = '0' * 64
    elif mutation == 'wrong_powers':
        changed['moment_reconstruction']['powers']['y'] = 4
    else:
        changed['admission_margins']['height_variance_lower'] = '1'
    monkeypatch.setattr(C, 'build_report', lambda *_: candidate)
    with pytest.raises(ValueError, match='reconstructed'):
        C.check_report(changed)


@pytest.mark.parametrize('optimization', [[], ['-O']])
def test_cli_supplied_numerical_mutant_is_replayed(candidate, tmp_path, optimization):
    changed = copy.deepcopy(candidate)
    changed['composition']['normalized_upper'] = '0'
    path = tmp_path / 'bad-integrand.json'
    path.write_text(json.dumps(changed))
    result = subprocess.run([sys.executable, *optimization,
                             str(C.ROOT / 'tools/rn_side24_density_check.py'),
                             '--check', str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and 'reconstructed density/window' in result.stdout


def test_pin_source_identity_is_not_assumed(tmp_path):
    # Reuse the authentic ZIP; replace only the basis bridge source.
    archive = tmp_path / C.point_check.ARCHIVE
    archive.parent.mkdir(parents=True)
    archive.write_bytes((C.ROOT / C.point_check.ARCHIVE).read_bytes())
    pin = tmp_path / C.PIN_TRANSFORM
    pin.parent.mkdir(parents=True)
    pin.write_bytes(b'not the pinned basis transformation')
    with pytest.raises(ValueError, match='pin-transform source'):
        C.checked_imports(tmp_path)


def test_output_refuses_overwrite(tmp_path, monkeypatch):
    output = tmp_path / 'preserved.json'
    output.write_bytes(b'old candidate')
    monkeypatch.setattr(C, 'build_report', lambda: {})
    assert C.main(['--output', str(output)]) == 1
    assert output.read_bytes() == b'old candidate'
