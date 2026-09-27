"""Publication-boundary tests; synthetic accounting is not an RN field proof."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import copy
import os
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.interval import Interval as I
from research.cover import audit
from research.cover.ledger import (
    ACCEPTED, REJECTED, Box, Cell, Ledger, PendingCellsError,
    PartitionError, RejectKind, UncertifiedTotalError,
)


def pair():
    ledger = Ledger('accounting fixture', Box.of(0, 2, 0, 1), 'cartesian')
    ledger.add(Cell('a', Box.of(0, 1, 0, 1), 0))
    ledger.add(Cell('b', Box.of(1, 2, 0, 1), 0))
    ledger.accept('a', I.exact(F(1)), I(F(-2), F(3)), I(F(-2), F(3)))
    return ledger


@pytest.mark.parametrize('residual', [None, I.exact(F(0))])
def test_canonical_outside_preserves_result_and_records(residual):
    ledger = pair()
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1), residual)
    before = copy.deepcopy(ledger.__dict__)
    assert audit.checked_total(ledger) == ledger.total()
    assert audit.checked_total(ledger).certified_enclosure() == I(F(-2), F(3))
    assert ledger.__dict__ == before


@pytest.mark.parametrize('residual', [I.exact(F(1000000)), I(F(-2), F(-1)), I(F(-1), F(1))])
def test_outside_noncanonical_residual_refused_before_publication(residual):
    ledger = pair()
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1))
    # Records are mutable; this bypasses any entry-only guard without code mutation.
    ledger.records['b'].residual = residual
    with pytest.raises(ValueError, match='OUTSIDE residual'):
        audit.checked_total(ledger)


@pytest.mark.parametrize('kind', [RejectKind.EXCLUDED, RejectKind.UNRESOLVED_BOUNDARY])
def test_nonoutside_residual_is_summed_including_signed_values(kind):
    ledger = pair()
    ledger.reject('b', kind, 'bounded omitted part', F(1), I(F(-5), F(7)))
    result = audit.checked_total(ledger)
    assert result.certified_enclosure() == I(F(-7), F(10))
    assert result.area_rejected_bound == F(1)
    assert result.area_rejected_by_kind == {kind: F(1)}


@pytest.mark.parametrize('kind', [RejectKind.EXCLUDED, RejectKind.UNRESOLVED_BOUNDARY])
def test_missing_residual_stays_incomplete(kind):
    ledger = pair()
    ledger.reject('b', kind, 'no bound yet', F(1))
    result = audit.checked_total(ledger)
    assert result.certified is False and result.covers_region is False
    with pytest.raises(UncertifiedTotalError):
        result.certified_enclosure()


def test_noncertifying_stays_noncertifying():
    ledger = pair()
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1))
    ledger.mark_non_certifying('test reference')
    result = audit.checked_total(ledger)
    assert result.certified is False and result.covers_region is True


def test_pending_and_partition_refusals_are_retained():
    ledger = pair()
    with pytest.raises(PendingCellsError):
        audit.checked_total(ledger)
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1))
    ledger.records['b'].cell = Cell('b', Box.of(1, 3, 0, 1), 0)
    with pytest.raises(PartitionError):
        audit.checked_total(ledger)


@pytest.mark.parametrize('field,value', [
    ('disposition', 'UNKNOWN_TERMINAL'), ('reject_kind', 'TYPO'),
    ('reason', ''), ('reason', '  '), ('boundary_area_bound', None),
    ('boundary_area_bound', F(-1)), ('boundary_area_bound', True),
    ('residual', 1),
])
def test_malformed_rejected_record_refuses(field, value):
    ledger = pair()
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1))
    setattr(ledger.records['b'], field, value)
    with pytest.raises(ValueError):
        audit.checked_total(ledger)


@pytest.mark.parametrize('field,value', [
    ('area', None), ('area', I(F(-1), F(1))),
    ('value_range', None), ('contribution', None), ('contribution', '1'),
])
def test_malformed_accepted_record_refuses(field, value):
    ledger = pair()
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1))
    setattr(ledger.records['a'], field, value)
    with pytest.raises(ValueError):
        audit.checked_total(ledger)


@pytest.mark.parametrize('order', [['a'], ['a', 'a', 'b'], ['a', 'b', 'missing']])
def test_order_cannot_omit_or_repeat_records(order):
    ledger = pair()
    ledger._order = order
    with pytest.raises(ValueError, match='record order'):
        audit.checked_total(ledger)


def test_record_identity_and_boolean_marker():
    ledger = pair()
    ledger.certifying = 'False'
    with pytest.raises(ValueError, match='boolean'):
        audit.checked_total(ledger)
    ledger.certifying = True
    ledger.records['a'].cell = Cell('different-id', Box.of(0, 1, 0, 1), 0)
    with pytest.raises(ValueError, match='record identity'):
        audit.checked_total(ledger)


@pytest.mark.parametrize('change', [
    {'enclosure': I.exact(F(0))}, {'area_accounted': I.exact(F(0))},
    {'certified': False}, {'certified': 1}, {'covers_region': False},
    {'area_rejected_bound': F(0)}, {'area_rejected_by_kind': {}},
    {'area_rejected_bound': True}, {'area_rejected_by_kind': {RejectKind.EXCLUDED: True}},
])
def test_a_lying_accumulator_is_rejected(monkeypatch, change):
    ledger = pair()
    ledger.reject('b', RejectKind.EXCLUDED, 'bounded omitted part', F(1), I.exact(F(7)))
    wrong = replace(ledger.total(), **change)
    monkeypatch.setattr(ledger, 'total', lambda: wrong)
    with pytest.raises(ValueError):
        audit.checked_total(ledger)


def test_real_spatial_consumer_calls_checked_boundary(monkeypatch):
    from tools import rn_side24_spatial_check as consumer
    ledger = pair()
    ledger.reject('b', RejectKind.OUTSIDE, 'declared geometry', F(1))
    monkeypatch.setattr(consumer.point_check, 'checked_sources', lambda root: {})
    monkeypatch.setattr(consumer, 'checked_imports', lambda root: {})
    monkeypatch.setattr(consumer, 'checked_target', lambda root: {})
    monkeypatch.setattr(consumer, 'run', lambda *a, **kw: ledger)
    def stop(value):
        assert value is ledger
        raise ValueError('CHECKED_BOUNDARY_REACHED')
    monkeypatch.setattr(consumer, 'checked_total', stop)
    with pytest.raises(ValueError, match='CHECKED_BOUNDARY_REACHED'):
        consumer.build_report()


def test_refusal_is_not_disabled_by_optimized_python():
    root = Path(__file__).resolve().parents[1]
    code = '''
from fractions import Fraction as F
from research.interval import Interval as I
from research.cover.ledger import Box, Cell, Ledger, RejectKind
from research.cover.audit import checked_total
l=Ledger('optimized',Box.of(0,1,0,1),'cartesian')
l.add(Cell('one',Box.of(0,1,0,1),0))
l.reject('one',RejectKind.OUTSIDE,'declared geometry',F(1))
l.records['one'].residual=I.exact(F(7))
try:
    checked_total(l)
except ValueError:
    print('REFUSED')
else:
    raise SystemExit('guard disappeared')
'''
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    outputs = []
    for flags in ([], ['-O']):
        p = subprocess.run([sys.executable, *flags, '-c', code], cwd=root,
                           env=env, capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr
        outputs.append(p.stdout)
    assert outputs == ['REFUSED\n', 'REFUSED\n']


@pytest.mark.parametrize('polar', [False, True])
def test_actual_reference_driver_is_accepted_without_changing_its_total(polar):
    from research.cover import DriverConfig, rn5_annulus_bracket, rn5_annulus_polar, run
    class ConstantReference:
        name = 'audit unit reference'
        label = 'SYNTHETIC constant function; not a Gaussian-field bound'
        certifying = True
        def range_enclosure(self, region, box, prec):
            return I.exact(F(1))
    region = rn5_annulus_polar() if polar else rn5_annulus_bracket()
    ledger = run(region, ConstantReference(), DriverConfig(tol=F(30), max_depth=5))
    before = copy.deepcopy(ledger.__dict__)
    assert audit.checked_total(ledger) == ledger.total()
    assert ledger.__dict__ == before
