"""Semantic and adversarial controls for the fixed-point density/window bound.

The CDF path below is a separate analytic comparison using shared certified
interval primitives. It is neither numerical sampling as proof nor an
organizationally independent verification.
"""
from fractions import Fraction as F
from functools import lru_cache

import pytest

from research.interval import Interval as I, Phi, exp, pi, sqrt
from research.rn import side24_density as d
from research.rn.conditioning import ConditioningInconclusive


def overlaps(a, b):
    return max(a.lo, b.lo) <= min(a.hi, b.hi)


@lru_cache(maxsize=128)
def cdf_mass(mean, variance, lo, hi):
    # Outward rounding controls series operand sizes. Thirty-five digits
    # resolve the 1e-11 comparison slack comfortably; validity is unconditional.
    sd = sqrt(variance, 35).round_out(128)
    return (Phi(((I.exact(hi)-mean)/sd).round_out(128), 35)
            -Phi(((I.exact(lo)-mean)/sd).round_out(128), 35))


@pytest.mark.parametrize('mean,covariance,q,det', (
    ((0, 0), ((1, 0), (0, 1)), F(0), F(1)),
    ((2, 1), ((4, 2), (2, 2)), F(1), F(4)),
    ((1, -1), ((2, 1), (1, 2)), F(2), F(3)),
))
def test_gradient_density_matches_closed_form(mean, covariance, q, det):
    result = d.gradient_density(mean, covariance)
    assert result['gradient_mahalanobis'].lo <= q <= result['gradient_mahalanobis'].hi
    assert result['gradient_determinant'].lo <= det <= result['gradient_determinant'].hi
    expected = exp(I.exact(-q/2), 100)/(2*pi(100)*sqrt(I.exact(det), 100))
    assert overlaps(result['gradient_density'], expected)


def test_gradient_square_sum_keeps_mahalanobis_nonnegative():
    result = d.gradient_density((I(-1, 1), I(-2, 2)), ((2, 1), (1, 2)))
    assert result['gradient_mahalanobis'].lo == 0
    centered = d.gradient_density((0, 0), ((2, 1), (1, 2)))
    assert result['gradient_density'].hi >= centered['gradient_density'].lo
    shifted = d.gradient_density((3, 4), ((2, 1), (1, 2)))
    assert shifted['gradient_density'].hi < centered['gradient_density'].lo


@pytest.mark.parametrize('mean,variance,window,distance', (
    (0, 1, (F(-1, 2), F(1, 2)), F(0)),
    (2, 1, (F(0), F(1)), F(1)),
    (-2, 1, (F(-1), F(0)), F(1)),
    (I(2, 3), I(F(1, 2), 2), (F(-1, 10), F(1, 10)), F(19, 10)),
    (I(-1, 1), I(F(1, 2), 2), (F(-1, 10), F(1, 10)), F(0)),
))
def test_whole_window_cap_contains_separate_cdf_evaluation(mean, variance, window, distance):
    result = d.height_window_cap(mean, variance, window)
    assert result['height_distance_lower'] == distance
    mean = mean if isinstance(mean, I) else I.exact(mean)
    variance = variance if isinstance(variance, I) else I.exact(variance)
    for m in {mean.lo, (mean.lo+mean.hi)/2, mean.hi}:
        for v in {variance.lo, variance.hi}:
            mass = cdf_mass(I.exact(m), I.exact(v), *window)
            assert mass.hi <= result['window_mass_upper']


def test_window_boundary_and_probability_clamp():
    assert d.height_window_cap(0, 1, (0, 0))['window_mass_upper'] == 0
    assert d.height_window_cap(0, 1, (-100, 100))['window_mass_upper'] == 1
    near = d.height_window_cap(2, 1, (0, 1))
    far = d.height_window_cap(2, 1, (-1, 0))
    assert far['window_mass_upper'] < near['window_mass_upper']
    # Using the farthest rather than nearest endpoint underestimates this mass.
    wrong_sup = exp(I.exact(-F(4, 2)), 90)/sqrt(2*pi(90), 90)
    assert wrong_sup.hi < cdf_mass(I.exact(2), I.exact(1), 0, 1).lo


def test_gradient_exponent_sign_negative_control(monkeypatch):
    original = d.gradient_density((3, 4), ((1, 0), (0, 1)))['gradient_density']
    real_exp = d.exp
    monkeypatch.setattr(d, 'exp', lambda x, prec: real_exp(-x, prec))
    wrong = d.gradient_density((3, 4), ((1, 0), (0, 1)))['gradient_density']
    centered = 1/(2*pi(90))
    assert original.hi < centered.lo < centered.hi < wrong.lo


@pytest.fixture(scope='module')
def point_factor():
    return d.density_window()


def test_source_law_density_window_and_separate_cdf_margin(point_factor):
    result = point_factor
    assert result['point'] == (F(1), F(1))
    assert result['six_pin_order'] == ('M:f', 'M:fx', 'M:fy', 'S:f', 'S:fx', 'S:fy')
    assert result['jet_order'] == ('y:f', 'y:fx', 'y:fy')
    assert result['six_pin_values'] == (F(6, 5), 0, 0, F(57599, 48000), 0, 0)
    assert result['height_window'] == (F(57599, 48000), F(6, 5))
    assert result['height_window_length'] == F(1, 48000)
    assert result['height_mean'].lo > F(6, 5)
    assert result['height_distance_lower'] == result['height_mean'].lo-F(6, 5)
    assert all(p.lo > 0 for key in ('six_pin_pivots', 'jet_pivots', 'gradient_pivots') for p in result[key])
    assert F('0.16280269') < result['gradient_density'].lo <= result['gradient_density'].hi < F('0.16280270')
    assert F('0.11352234') < result['height_variance'].lo <= result['height_variance'].hi < F('0.11352235')
    mass = cdf_mass(result['height_mean'], result['height_variance'], *result['height_window'])
    assert 0 < mass.lo <= mass.hi < result['window_mass_upper']
    # Exact rational comparison: conservative cap is within 3 ppm of the CDF
    # mass enclosure at this one point; this comparison does not prove a cover.
    assert result['window_mass_upper'] < F(1000003, 1000000)*mass.lo
    assert result['density_mass_upper'] == result['gradient_density'].hi*result['window_mass_upper']
    assert F('0.00000400147') < result['density_mass_upper'] < F('0.00000400148')
    assert result['authority'] == 'NONE' and type(result['independence_credit']) is int and result['independence_credit'] == 0
    assert all(result[key] is False for key in ('field_certified', 'spatial_cover_certified',
        'scientific_status_changed', 'all_small_r_certified', 'h3_reproved', 'original_prize_closed'))


def test_h3_floor_is_imported_exact_and_has_pinned_custody(point_factor):
    floor = point_factor['imported_h3_floor']
    assert floor['value'] == F('0.0077592917375327855')
    assert floor['r'] == F(1, 20) and floor['height'] == F(6, 5)
    assert floor['member_sha256'] == d.H3_SHA and floor['member_bytes'] == 7003
    assert floor['status'] == 'IMPORTED_SOURCE_PREMISE_NOT_REPROVED'


@pytest.mark.parametrize('field,value,match', (
    ('H3_SHA', '0'*64, 'source identity'),
    ('H3_BYTES', 7002, 'source size'),
    ('Z_LO', F(1), 'exact imported'),
))
def test_h3_identity_or_floor_corruption_refuses(monkeypatch, field, value, match):
    monkeypatch.setattr(d, field, value)
    with pytest.raises(ValueError, match=match):
        d.imported_h3_floor()


@pytest.mark.parametrize('mean,covariance', (
    ((0, 0), ((1, 2), (2, 1))),
    ((0, 0), ((1, 1), (1, 1))),
    ((0, 0), ((1, 0), (1, 1))),
    ((0.0, 0), ((1, 0), (0, 1))),
    ((True, 0), ((1, 0), (0, 1))),
    ((0,), ((1,),)),
))
def test_invalid_gradient_law_refuses(mean, covariance):
    with pytest.raises((ValueError, TypeError)):
        d.gradient_density(mean, covariance)


@pytest.mark.parametrize('mean,variance,window,options', (
    (0, 0, (0, 1), {}), (0, I(-1, 1), (0, 1), {}),
    (0, 1, (1, 0), {}), (0, 1, (0, 1.0), {}),
    (False, 1, (0, 1), {}), (0, 1, (0, 1), {'bits': 64}),
    (F(1, 2**4097), 1, (0, 1), {}),
))
def test_invalid_height_law_or_resources_refuse(mean, variance, window, options):
    with pytest.raises((ValueError, TypeError, ConditioningInconclusive)):
        d.height_window_cap(mean, variance, window, **options)
