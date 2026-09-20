"""Upper-target and recoverable-failure controls, using reference laws only.

No RN cell, imported premise or scientific status is certified by these tests.
"""
from fractions import Fraction as F

import pytest

from research.cover import (ACCEPTED, PENDING, REFINED, Box, DriverConfig,
                            PartitionError, PendingCellsError,
                            RecoverableEnclosureError, RejectKind, run)
from research.cover.regions import INSIDE, OUTSIDE, STRADDLE
from research.interval import Interval


class Rectangle:
    name = 'REFERENCE rectangle of exact area six'
    coords = 'cartesian'

    def domain(self):
        return Box.of(0, 2, 0, 3)

    def roots(self):
        return [self.domain()]

    def classify(self, box):
        return INSIDE

    def area(self, box, prec):
        return Interval.exact(box.param_area())

    def area_rational_upper(self, box):
        return box.param_area()

    def diameter_bound(self, box):
        return box.du() + box.dv()

    def subdivide(self, box):
        return box.split_u()


class NonnegativeRange:
    name = 'REFERENCE nonnegative range'
    label = 'A reference range only; no field law, normalizer or RN assertion.'
    certifying = True

    def range_enclosure(self, region, box, prec):
        return Interval(F(0), F(2))


class WideCellFails(NonnegativeRange):
    def range_enclosure(self, region, box, prec):
        if box.du() > 1:
            raise RecoverableEnclosureError('positive covariance pivot not established on this cell')
        return super().range_enclosure(region, box, prec)


class Boundary(Rectangle):
    def classify(self, box):
        return STRADDLE


def test_upper_budget_accepts_an_honest_zero_lower_without_width_acceptance():
    config = DriverConfig(tol=F(0), upper_budget=F(12), max_depth=0)
    led = run(Rectangle(), NonnegativeRange(), config)
    assert led.receipt()['pending_count'] == 0
    assert len(led.by_disposition(ACCEPTED)) == 1
    total = led.total().certified_enclosure()
    assert total == Interval(F(0), F(12))
    # Independent member f=5/4 has integral (2*3)*(5/4)=15/2.
    assert F(15, 2) in total
    rec = led.records['c0']
    assert rec.value_range.lo == rec.contribution.lo == 0
    assert rec.area == Interval.exact(F(6))
    assert rec.contribution == rec.area * rec.value_range
    assert led.receipt()['acceptance_policy']['upper_budget'] == '12'

    default = run(Rectangle(), NonnegativeRange(), DriverConfig(tol=F(0), max_depth=0))
    assert len(default.pending()) == 1
    assert 'acceptance_policy' not in default.receipt()
    with pytest.raises(PendingCellsError):
        default.total()


def test_area_is_neither_omitted_nor_applied_twice():
    # Omitting area would falsely accept upper 2 against budget 6. Applying
    # area twice would reject the exact accepted upper 12 in the prior test.
    led = run(Rectangle(), NonnegativeRange(), DriverConfig(upper_budget=F(6), max_depth=0))
    assert led.pending()[0].contribution == Interval(F(0), F(12))
    with pytest.raises(PendingCellsError):
        led.total()


def test_upper_shares_do_not_assume_geometric_upper_bounds_are_additive():
    class LooseAreas(Rectangle):
        def area_rational_upper(self, box):
            return F(100)  # valid for every cell, deliberately not additive

    good = run(LooseAreas(), WideCellFails(), DriverConfig(upper_budget=F(12), max_depth=1))
    assert good.total().certified_enclosure() == Interval(F(0), F(12))
    bad = run(LooseAreas(), WideCellFails(), DriverConfig(upper_budget=F(6), max_depth=1))
    assert len(bad.pending()) == 2
    # Incorrect shares based on area_upper(child)/area_upper(root) would give
    # both children budget 6 and falsely accept their combined upper 12.
    with pytest.raises(PendingCellsError):
        bad.total()


def test_boundary_residual_spends_budget_and_area_once():
    led = run(Boundary(), NonnegativeRange(), DriverConfig(upper_budget=F(12), max_depth=0))
    rejected = led.rejected()
    assert len(rejected) == 1
    assert rejected[0].reject_kind == RejectKind.UNRESOLVED_BOUNDARY
    assert rejected[0].boundary_area_bound == F(6)
    assert rejected[0].residual == Interval(F(0), F(12))
    assert led.total().certified_enclosure() == Interval(F(0), F(12))
    assert led.receipt()['acceptance_policy']['includes_boundary_residuals'] is True

    over = run(Boundary(), NonnegativeRange(), DriverConfig(upper_budget=F(6), max_depth=1))
    assert not over.rejected()
    assert len(over.pending()) == 2
    assert all(rec.residual == Interval(F(0), F(6)) for rec in over.pending())
    with pytest.raises(PendingCellsError):
        over.total()
    # Existing width mode still retains a bounded rejected boundary at depth.
    default = run(Boundary(), NonnegativeRange(), DriverConfig(tol=F(0), max_depth=0))
    assert not default.pending()
    assert default.total().enclosure == Interval(F(0), F(12))


@pytest.mark.parametrize('region', [Rectangle(), Boundary()])
def test_exact_zero_upper_budget_admits_only_the_supplied_zero_range(region):
    class Zero(NonnegativeRange):
        def range_enclosure(self, region, box, prec):
            return Interval.exact(F(0))
    led = run(region, Zero(), DriverConfig(upper_budget=F(0), max_depth=0))
    assert not led.pending()
    assert led.total().certified_enclosure() == Interval.exact(F(0))


def test_failed_enclosure_refines_and_retains_its_record():
    led = run(Rectangle(), WideCellFails(), DriverConfig(upper_budget=F(12), max_depth=1))
    assert led.records['c0'].disposition == REFINED
    assert led.records['c0'].contribution is None
    assert len(led.by_disposition(ACCEPTED)) == 2
    assert len(led.receipt()['enclosure_failures']) == 1
    assert led.receipt()['enclosure_failures'][0]['disposition'] == REFINED
    assert 'positive covariance pivot' in led.receipt()['enclosure_failures'][0]['reason']
    led.check_partition()
    led.check_refinements()
    assert led.total().certified_enclosure().hi == 12


@pytest.mark.parametrize('region', [Rectangle(), Boundary()])
@pytest.mark.parametrize('config', [DriverConfig(upper_budget=F(12), max_depth=0),
                                  DriverConfig(upper_budget=F(12), max_depth=3, max_cells=1)])
def test_recoverable_failure_resource_limits_leave_pending_without_fabricated_bounds(region, config):
    led = run(region, WideCellFails(), config)
    assert led.pending()
    assert not led.rejected()
    assert all(rec.area is None and rec.contribution is None and rec.residual is None
               for rec in led.pending())
    assert led.receipt()['enclosure_failures']
    assert led.receipt()['total'] is None
    led.check_partition()
    with pytest.raises(PendingCellsError):
        led.total()


@pytest.mark.parametrize('error', [ValueError('source digest mismatch'), TypeError('bad input'),
                                 RuntimeError('unexpected failure')])
def test_only_the_explicit_recoverable_type_is_refined(error):
    class Invalid(NonnegativeRange):
        def range_enclosure(self, region, box, prec):
            raise error
    with pytest.raises(type(error), match=str(error)):
        run(Rectangle(), Invalid(), DriverConfig(upper_budget=F(12)))


def test_partition_and_pending_requirements_cannot_be_bypassed_by_upper_mode():
    led = run(Rectangle(), WideCellFails(), DriverConfig(upper_budget=F(12), max_depth=1))
    del led.records['c0.0']
    led._order.remove('c0.0')
    with pytest.raises(PartitionError, match='GAP'):
        led.total()


def test_uniform_shares_are_sufficient_but_do_not_guarantee_convergence():
    class LeftHalf(NonnegativeRange):
        def range_enclosure(self, region, box, prec):
            if box.u1 <= 1:
                return Interval.exact(F(1))
            if box.u0 >= 1:
                return Interval.exact(F(0))
            return Interval(F(0), F(1))
    # The true integral is exactly 3 < budget 4, but the left half has only
    # budget 2; further refinement cannot change that aggregate allocation.
    for depth in (1, 3):
        led = run(Rectangle(), LeftHalf(), DriverConfig(upper_budget=F(4), max_depth=depth))
        assert led.pending()
        with pytest.raises(PendingCellsError):
            led.total()


@pytest.mark.parametrize('budget', [True, 12, 12.0, '12'])
def test_upper_budget_requires_an_exact_fraction(budget):
    with pytest.raises(TypeError, match='exact Fraction'):
        DriverConfig(upper_budget=budget)


def test_negative_budget_or_unproved_nonnegativity_is_refused():
    with pytest.raises(ValueError, match='non-negative'):
        DriverConfig(upper_budget=F(-1))
    class Signed(NonnegativeRange):
        def range_enclosure(self, region, box, prec):
            return Interval(F(-1), F(2))
    with pytest.raises(ValueError, match='nonnegative range'):
        run(Rectangle(), Signed(), DriverConfig(upper_budget=F(12)))
    # The original width policy continues to support signed ranges.
    led = run(Rectangle(), Signed(), DriverConfig(tol=F(18), max_depth=0))
    assert led.total().certified_enclosure() == Interval(F(-6), F(12))


def test_float_probe_cannot_become_certified_through_upper_acceptance():
    class Probe(NonnegativeRange):
        certifying = False
    led = run(Rectangle(), Probe(), DriverConfig(upper_budget=F(12), max_depth=0))
    assert led.total().certified is False
    assert 'NON-CERTIFYING' in led.receipt()['note']
