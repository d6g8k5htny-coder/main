"""Exercise the real wedge consumer; stored premises are not a new field proof."""
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from research.cover.ledger import Ledger, REJECTED, RejectKind
from research.interval import Interval as I
from tools import rn_inner_wedge_check as check

CANDIDATE_SHA = 'b685e524cb257ac1c31f1f22f88c5bf4510bcdf7bc575d04b333a2f6613438ba'


def stored_inputs():
    """Decode only the frozen premises consumed by assemble_cover.

    Full numerical/source-law reconstruction remains the actual CLI's job.
    No N6 source body is opened or executed by this helper.
    """
    path = check.ROOT / check.CANDIDATE
    if hashlib.sha256(path.read_bytes()).hexdigest() != CANDIDATE_SHA:
        raise ValueError('original wedge candidate identity changed')
    candidate = check.load_report(path)
    auxiliary = candidate['auxiliary']
    geometry = check.containment()
    if check.encode(geometry) != auxiliary['geometry']:
        raise ValueError('stored geometry differs from actual containment')
    pieces = [dict(box=check.Box(**{k: F(v) for k, v in row['box'].items()}),
                   determinant_lower=F(row['determinant_lower']),
                   q_lower=F(row['q_lower'])) for row in auxiliary['pieces']]
    bounds = dict(pieces=pieces, geometry=geometry,
                  piece_count=auxiliary['piece_count'],
                  pending_pieces=auxiliary['pending_pieces'],
                  conditioning_energy_upper=F(auxiliary['conditioning_energy_upper']))
    majorant = {k: F(candidate['majorant'][k]) for k in
                ('determinant_lower', 'mahalanobis_lower', 'jacobian_upper', 'integrand_upper')}
    return bounds, majorant, candidate['cover']


def test_stored_wedge_assembly_receipt_stays_exact():
    bounds, majorant, expected = stored_inputs()
    actual = check.assemble_cover(bounds, majorant)
    assert check.encode(actual) == expected
    assert actual['total']['certified'] is True
    assert actual['total']['covers_region'] is True
    assert F(actual['total']['enclosure_hi']) > 0


def test_actual_wedge_consumer_cannot_bypass_checked_total(monkeypatch):
    bounds, majorant, _ = stored_inputs()
    def refused(_ledger):
        raise RuntimeError('wedge accounting sentinel')
    # Patch the consumer's lookup, not an unused module binding.
    monkeypatch.setattr(check, 'checked_total', refused)
    with pytest.raises(RuntimeError, match='wedge accounting sentinel'):
        check.assemble_cover(bounds, majorant)


def corrupt_accept(kind):
    original = Ledger.accept
    def accept(self, cid, area, value_range, contribution):
        original(self, cid, area, value_range, contribution)
        rec = self.records[cid]
        if kind == 'unknown':
            rec.disposition = 'UNRECOGNIZED_TERMINAL'
        elif kind == 'outside':
            rec.disposition = REJECTED
            rec.reject_kind = RejectKind.OUTSIDE
            rec.reason = 'synthetic post-accept corruption'
            rec.boundary_area_bound = F(0)
            rec.residual = I.exact(1000000)
        elif kind == 'negative_area':
            rec.area = I.exact(-1)
        else:
            raise ValueError('unknown test mutation')
    return accept


@pytest.mark.parametrize('kind, message', [
    ('unknown', 'unknown disposition'),
    ('outside', 'OUTSIDE residual'),
    ('negative_area', 'area must be nonnegative'),
])
def test_actual_wedge_refuses_post_accept_corruption(kind, message):
    bounds, majorant, _ = stored_inputs()
    with patch.object(Ledger, 'accept', corrupt_accept(kind)):
        with pytest.raises(ValueError, match=message):
            check.assemble_cover(bounds, majorant)


def test_actual_wedge_refuses_wrong_reported_total(monkeypatch):
    bounds, majorant, _ = stored_inputs()
    original = Ledger.total
    def zero_total(self):
        return replace(original(self), enclosure=I.exact(0))
    monkeypatch.setattr(Ledger, 'total', zero_total)
    with pytest.raises(ValueError, match='independently summed records'):
        check.assemble_cover(bounds, majorant)


@pytest.mark.parametrize('optimized', [False, True])
def test_real_consumer_checks_survive_python_optimization(optimized):
    code = """
import runpy
from unittest.mock import patch
ns = runpy.run_path('tests/test_rn_wedge_accounting.py')
check, Ledger = ns['check'], ns['Ledger']
bounds, majorant, _ = ns['stored_inputs']()
with patch.object(Ledger, 'accept', ns['corrupt_accept']('outside')):
    try:
        check.assemble_cover(bounds, majorant)
    except ValueError as error:
        if 'OUTSIDE residual' not in str(error):
            raise
        print('OUTSIDE_CORRUPTION_REFUSED')
    else:
        raise RuntimeError('corrupted wedge was published')
"""
    args = [sys.executable] + (['-O'] if optimized else []) + ['-c', code]
    result = subprocess.run(args, cwd=check.ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'OUTSIDE_CORRUPTION_REFUSED'
