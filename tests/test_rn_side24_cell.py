"""Source-cell tests and semantic mutants; samples do not prove uniformity."""
from fractions import Fraction as F

import pytest

from research.cover import Box
from research.interval import Interval as I
from research.rn import side24_cell as c
from research.rn.side24 import MARK_DOMAIN, point_laws
from research.rn.side24_density import density_window
from research.rn.conditioning import ConditioningInconclusive, ConditioningResourceLimit


def square(h):
    return Box(1-h, 1+h, 1-h, 1+h)


def contains(outer, inner):
    return outer.lo <= inner.lo <= inner.hi <= outer.hi


def overlaps(a, b):
    return max(a.lo, b.lo) <= min(a.hi, b.hi)


@pytest.fixture(scope='module')
def accepted_cell():
    return c.spatial_cell(square(F(1, 4000)))


def test_third_derivative_variances_retain_torus_normalization():
    sixth = c.sixth_kernel_derivative_zero(bits=512)
    assert -15 < sixth.lo <= sixth.hi < 0
    assert 0 < c.derivative_variance((3, 0), bits=512).lo < 15
    assert c.derivative_variance((3, 0), bits=512).hi < 15
    assert c.derivative_variance((2, 1), bits=512).lo > 3
    assert c.derivative_variance((3, 0), bits=512) == c.derivative_variance((0, 3), bits=512)


def test_zero_width_transport_has_zero_remainder_and_matches_point_laws():
    result = c.cell_laws(square(F(0)))
    assert result['l2_increment_upper'] == (F(0),)*12
    assert all(x == 0 for row in result['covariance_error_upper'] for x in row)
    direct = point_laws()
    for name, law in result['laws'].items():
        for a, b in zip(law.intercept+law.slope, direct['laws'][name].intercept+direct['laws'][name].slope):
            assert overlaps(a, b)
        for row, direct_row in zip(law.covariance, direct['laws'][name].covariance):
            assert all(overlaps(a, b) for a, b in zip(row, direct_row))
    density = density_window()
    for key in ('gradient_density', 'height_mean', 'height_variance'):
        assert overlaps(result['density'][key], density[key])


def test_nonzero_cell_replays_full_mark_and_has_positive_pivots(accepted_cell):
    result = accepted_cell
    assert result['halfwidths'] == (F(1, 4000),)*2
    assert result['quantifier'] == 'FOR_EVERY_Y_IN_CLOSED_RECTANGLE_AND_EVERY_MARK_IN_FULL_WINDOW'
    assert result['l2_increment_upper'][:6] == (F(0),)*6
    assert all(e > 0 for e in result['l2_increment_upper'][6:])
    assert all(result['covariance_error_upper'][i][j] == 0 for i in range(6) for j in range(6))
    assert all(p.lo > 0 for p in result['six_pin_pivots']+result['jet_pivots'])
    assert all(p.lo > 0 for ps in result['marginal_pivots'].values() for p in ps)
    for name in ('M', 'S', 'y'):
        assert result['laws'][name].context.domain == MARK_DOMAIN
        assert result['moment_replays'][name]['certificate_valid']
        assert result['moment_replays'][name]['requested_checks_passed']
        assert result['moment_certificates'][name]['claim']['context']['domain'] == list(map(str, MARK_DOMAIN))
    assert F('0.000005589') < result['integrand_upper'] < F('0.000005590')
    assert result['integrand_range'].lo == 0
    assert result['integrand_range'].hi == result['integrand_upper']
    raw_upper = result['holder_upper']*result['density']['density_mass_upper']/result['imported_h3_floor']['value']
    assert raw_upper <= result['integrand_upper']
    assert result['holder_upper']**4 >= result['holder_fourth_power_upper']
    assert result['authority'] == 'NONE' and result['independence_credit'] == 0
    assert all(result[key] is False for key in ('field_certified', 'spatial_cover_certified',
        'scientific_status_changed', 'all_small_r_certified', 'h3_reproved', 'original_prize_closed'))


def test_cell_contains_separately_recomputed_point_laws(accepted_cell):
    # Diagnostics at center and corners challenge assembly; the L2 proof,
    # not these finite comparisons, establishes the rectangle quantifier.
    box = accepted_cell['box']
    points = ((F(1), F(1)), (box.u0, box.v0), (box.u0, box.v1),
              (box.u1, box.v0), (box.u1, box.v1))
    for point in points:
        direct = point_laws(point)
        for name, outer in accepted_cell['laws'].items():
            inner = direct['laws'][name]
            assert all(contains(a, b) for a, b in zip(outer.intercept+outer.slope, inner.intercept+inner.slope))
            assert all(contains(a, b) for row, inner_row in zip(outer.covariance, inner.covariance)
                       for a, b in zip(row, inner_row))
        density = density_window(point)
        assert density['density_mass_upper'] <= accepted_cell['density']['density_mass_upper']


def test_covariance_transport_error_formula_and_nested_radius(accepted_cell):
    outer = accepted_cell
    inner = c.transported_law(square(F(1, 8000)))
    assert inner['conditioning_energy'] == outer['conditioning_energy']
    for i in range(12):
        assert inner['l2_increment_upper'][i] <= outer['l2_increment_upper'][i]/2
        assert contains(outer['mean'][i], inner['mean'][i])
        for j in range(12):
            expected = (outer['center_standard_deviation_upper'][i]*outer['l2_increment_upper'][j]
                       +outer['center_standard_deviation_upper'][j]*outer['l2_increment_upper'][i]
                       +outer['l2_increment_upper'][i]*outer['l2_increment_upper'][j])
            assert expected == outer['covariance_error_upper'][i][j]
            assert contains(outer['covariance'][i][j], inner['covariance'][i][j])


def test_zero_spatial_remainder_mutant_excludes_a_real_point_law(monkeypatch):
    box = square(F(1, 4000))
    monkeypatch.setattr(c, 'derivative_variance', lambda *args, **kwargs: I.exact(0))
    weakened = c.transported_law(box)
    _, actual, _ = c._center_law((box.u1, box.v1), 192)
    # Moving height mean differs from its center. A center-only carrier with
    # no spatial remainder demonstrably fails to enclose this source-law value.
    assert not overlaps(weakened['mean'][6], actual.intercept[6])


def test_mean_regression_energy_cannot_be_omitted(monkeypatch):
    original = c._center_law
    def wrong_energy(*args, **kwargs):
        block, central, _ = original(*args, **kwargs)
        return block, central, I.exact(0)
    box = square(F(1, 4000))
    monkeypatch.setattr(c, '_center_law', wrong_energy)
    weakened = c.transported_law(box)
    _, actual, _ = original((box.u1, box.v1), 192)
    assert not overlaps(weakened['mean'][6], actual.intercept[6])


def test_larger_cell_fails_sufficient_pivot_test_without_false_cap():
    with pytest.raises(ConditioningInconclusive, match='pivot'):
        c.spatial_cell(square(F(1, 1000)))


def test_moment_resource_exhaustion_is_recoverable_without_a_cap(monkeypatch):
    from research.rn import certificate
    def exhausted(*args, **kwargs):
        raise certificate.ResourceLimit('test operation limit')
    monkeypatch.setattr(certificate, 'produce', exhausted)
    with pytest.raises(ConditioningResourceLimit, match='moment resource limit'):
        c.spatial_cell(square(F(1, 4000)))


@pytest.mark.parametrize('box', (
    Box.of('-1/20', '1/20', '-1/20', '1/20'),
    Box.of(0, 3, 1, 2), Box.of(17, 18, 1, 2),
    Box(F(1, 2**4097), F(1), F(1), F(2)), (0, 1, 0, 1),
))
def test_unsupported_boxes_refuse(box):
    with pytest.raises((TypeError, ValueError, ConditioningInconclusive)):
        c.transported_law(box)


@pytest.mark.parametrize('alpha', ((True, 0), (1, False), (-1, 0), (4, 0), (2, 2), [1, 0]))
def test_derivative_order_validation_cannot_be_bypassed_by_cache(alpha):
    c.derivative_variance((1, 0))
    with pytest.raises(ValueError):
        c.derivative_variance(alpha)


@pytest.mark.parametrize('bits', (True, 127, 1025, 192.0))
def test_resource_precision_refuses(bits):
    with pytest.raises(ValueError):
        c.cell_laws(square(F(1, 4000)), bits=bits)
