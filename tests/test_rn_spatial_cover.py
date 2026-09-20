"""Source-adapter/cover integration at one exact local SIDE24 rectangle.

These exposed same-provider controls retain H3 as an imported premise. They
establish no complete annulus, remote budget, scientific promotion or credit
for organizational independence.
"""
from fractions import Fraction as F

import pytest

from research.cover import (ACCEPTED, PENDING, REFINED, Box, DriverConfig,
                            PendingCellsError, RecoverableEnclosureError,
                            check_exact_partition, run)
from research.interval import Interval as I
from research.rn import side24, side24_cell
from research.rn.conditioning import ConditioningResourceLimit
from research.rn.spatial_cover import RNRectangleRegion, RNSide24Integrand


def pilot_box():
    h = F(1, 2000)
    return Box(1-h, 1+h, 1-h, 1+h)


def config(**changes):
    return DriverConfig(**{'upper_budget': F(6, 10**12), 'max_depth': 1,
                           'max_cells': 5, 'sig_bits': 192, **changes})


@pytest.fixture(scope='module')
def real_pilot():
    region = RNRectangleRegion(pilot_box())
    adapter = RNSide24Integrand(bits=192)
    ledger = run(region, adapter, config())
    return region, adapter, ledger


def test_real_failed_parent_refines_to_four_source_bounded_children(real_pilot):
    region, adapter, ledger = real_pilot
    assert ledger.records['c0'].disposition == REFINED
    assert len(adapter.attempts) == 5
    failed = adapter.attempts[0]
    assert failed['parameter_box'] == failed['cartesian_box'] == pilot_box().as_json()
    assert failed['status'] == 'INCONCLUSIVE'
    assert 'pivot' in failed['reason']
    assert 'proof' not in failed
    assert ledger.records['c0'].contribution is None
    assert ledger.records['c0'].enclosure_failure == failed['reason']

    leaves = ledger.leaves()
    assert len(leaves) == 4 and all(r.disposition == ACCEPTED for r in leaves)
    assert all(r.cell.box.du() == r.cell.box.dv() == F(1, 2000) for r in leaves)
    assert all(r.cell.depth == 1 for r in leaves)
    check_exact_partition(region.domain(), [r.cell.box for r in leaves])
    ledger.check_refinements()
    receipt = ledger.receipt()
    assert receipt['pending_count'] == 0
    assert receipt['cells_by_disposition'] == {'ACCEPTED': 4, 'REFINED': 1,
                                               'REJECTED': 0, 'PENDING': 0}
    assert receipt['enclosure_failures'][0]['disposition'] == REFINED


def test_actual_complete_pilot_upper_counts_geometric_area_exactly_once(real_pilot):
    region, adapter, ledger = real_pilot
    bounded = {tuple(a['parameter_box'].values()): a['proof']
               for a in adapter.attempts if a['status'] == 'BOUNDED'}
    assert len(bounded) == 4
    assert region.domain().param_area() == F(1, 10**6)
    assert sum((r.cell.box.param_area() for r in ledger.leaves()), F(0)) == F(1, 10**6)
    for rec in ledger.leaves():
        proof = bounded[tuple(rec.cell.box.as_json().values())]
        upper = proof['integrand_upper']
        true_area = F(1, 4*10**6)
        assert region.area(rec.cell.box, 48) == I.exact(true_area)
        assert true_area in rec.area  # storage rounding may widen exact area
        assert rec.value_range.lo == rec.contribution.lo == 0
        assert upper in rec.value_range
        # This comparison is independent of the driver's multiplication. It
        # rejects omission or a second multiplication by the geometric area.
        expected = true_area * upper
        assert expected <= rec.contribution.hi <= F(1001, 1000) * expected
        assert rec.contribution.hi <= F(6, 4*10**12)

    total = ledger.total().certified_enclosure()
    assert total.lo == 0
    assert total.hi == sum((rec.contribution.hi for rec in ledger.leaves()), F(0))
    assert F(5589, 10**15) < total.hi < F(5590, 10**15) < F(6, 10**12)
    assert F(1, 10**6) in ledger.total().area_accounted
    assert ledger.total().area_rejected_bound == 0


def test_real_child_proofs_keep_same_law_mark_scope_and_imported_normalizer(real_pilot):
    _, adapter, ledger = real_pilot
    for attempt in adapter.attempts[1:]:
        proof = attempt['proof']
        assert attempt['status'] == 'BOUNDED'
        assert attempt['cartesian_box'] == attempt['parameter_box'] == proof['box'].as_json()
        assert proof['quantifier'] == 'FOR_EVERY_Y_IN_CLOSED_RECTANGLE_AND_EVERY_MARK_IN_FULL_WINDOW'
        assert proof['source_binding']['source_sha256'] == side24.RN5_SHA
        assert proof['imported_h3_floor']['status'] == 'IMPORTED_SOURCE_PREMISE_NOT_REPROVED'
        assert proof['imported_h3_floor']['value'] == F('0.0077592917375327855')
        assert all(proof[key] is False for key in ('h3_reproved', 'scientific_status_changed',
                   'spatial_cover_certified', 'all_small_r_certified', 'original_prize_closed'))
        assert proof['authority'] == 'NONE' and proof['independence_credit'] == 0
        assert set(proof['moment_replays']) == {'M', 'S', 'y'}
        assert all(r['certificate_valid'] and r['requested_checks_passed']
                   for r in proof['moment_replays'].values())
        exact_composition = (proof['holder_upper'] * proof['density']['density_mass_upper']
                             / proof['imported_h3_floor']['value'])
        assert proof['integrand_upper'] >= exact_composition > 0
        assert proof['integrand_range'] == I(0, proof['integrand_upper'])
    assert 'conditional on the pinned imported H3 floor' in ledger.receipt()['note']
    assert 'no full-annulus' in ledger.receipt()['note']


@pytest.mark.parametrize('limits', [{'max_depth': 0}, {'max_cells': 1}])
def test_real_coarse_enclosure_failure_remains_pending_at_resource_limit(limits):
    adapter = RNSide24Integrand()
    ledger = run(RNRectangleRegion(pilot_box()), adapter, config(**limits))
    assert len(adapter.attempts) == 1 and adapter.attempts[0]['status'] == 'INCONCLUSIVE'
    assert ledger.pending()
    assert not ledger.by_disposition(ACCEPTED) and not ledger.rejected()
    assert all(r.contribution is None and r.residual is None for r in ledger.pending())
    assert ledger.receipt()['total'] is None
    assert ledger.receipt()['enclosure_failures']
    ledger.check_partition()
    with pytest.raises(PendingCellsError):
        ledger.total()


def test_changed_real_archive_is_a_hard_failure_not_a_pending_cell(tmp_path, monkeypatch):
    corrupted = tmp_path / 'altered-rn5.zip'
    corrupted.write_bytes(side24.ARCHIVE.read_bytes() + b'changed source')
    monkeypatch.setattr(side24, 'ARCHIVE', corrupted)
    adapter = RNSide24Integrand()
    with pytest.raises(ValueError, match='archive source identity mismatch'):
        run(RNRectangleRegion(pilot_box()), adapter, config())
    assert not any(a['status'] == 'BOUNDED' for a in adapter.attempts)


def test_only_typed_conditioning_resource_failure_requests_refinement(monkeypatch):
    def exhausted(*args, **kwargs):
        raise ConditioningResourceLimit('bounded moment operations exhausted')
    monkeypatch.setattr(side24_cell, 'spatial_cell', exhausted)
    adapter = RNSide24Integrand()
    ledger = run(RNRectangleRegion(pilot_box()), adapter, config(max_depth=0))
    assert ledger.records['c0'].disposition == PENDING
    assert adapter.attempts[0]['reason'] == 'bounded moment operations exhausted'
    with pytest.raises(PendingCellsError):
        ledger.total()


@pytest.mark.parametrize('invalid', [F(-1), True, 0, 1.0, '1'])
def test_adapter_refuses_invalid_upper_without_area_or_completion_fallback(monkeypatch, invalid):
    monkeypatch.setattr(side24_cell, 'spatial_cell', lambda *args, **kwargs: {'integrand_upper': invalid})
    with pytest.raises(ValueError, match='nonnegative exact upper'):
        run(RNRectangleRegion(pilot_box()), RNSide24Integrand(), config())


@pytest.mark.parametrize('rectangle', [Box.of(0, 0, 1, 2), Box.of(0, '1/20', 0, '1/20'),
                                      Box.of(4, 5, 4, 5), (0, 1, 0, 1)])
def test_rectangle_requires_nondegenerate_exact_complete_annulus_containment(rectangle):
    with pytest.raises(ValueError):
        RNRectangleRegion(rectangle)


def test_region_geometry_does_not_accept_an_escaped_cell():
    region = RNRectangleRegion(pilot_box())
    escaped = Box.of(1, 2, 1, 2)
    for operation in (region.classify, region.subdivide, region.area_rational_upper,
                      region.diameter_bound, lambda b: region.cartesian_enclosure(b, 48)):
        with pytest.raises(ValueError, match='escapes'):
            operation(escaped)


@pytest.mark.parametrize('bits', [True, 127, 1025, 192.0])
def test_adapter_precision_validation(bits):
    with pytest.raises(ValueError):
        RNSide24Integrand(bits=bits)
