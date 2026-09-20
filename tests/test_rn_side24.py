"""Exact structural and semantic challenges for the source-bound SIDE24 adapter.

Shared interval arithmetic is trusted by these tests. No test supplies external
review independence, a spatial cover, an RN closure or a status change.
"""
from fractions import Fraction as F

import pytest

from research.interval import Interval as I, exp
from research.rn import side24 as s


def overlaps(a, b):
    return max(a.lo, b.lo) <= min(a.hi, b.hi)


def explicit_gaussian_derivative(n, x, prec=100):
    # Deliberately explicit polynomials, separate from production recurrence.
    factors = (F(1), -x, x*x-1, 3*x-x**3, x**4-6*x*x+3)
    return factors[n]*exp(I.exact(-x*x/2), prec)


@pytest.mark.parametrize('order', range(5))
@pytest.mark.parametrize('point', (F(-7, 3), F(1, 7), F(11)))
def test_derivatives_match_explicit_polynomial_oracle(order, point):
    actual = s.kernel_derivative(order, point, prec=110, bits=256)
    finite = sum((explicit_gaussian_derivative(order, point+24*j, 110)
                  for j in (-1, 0, 1)), I.exact(0))
    remainder = s.image_tail(order, abs(point), prec=110)
    independent_polynomial = (finite+I(-remainder, remainder))/s.normalizer(prec=110, bits=256)
    assert overlaps(actual, independent_polynomial)


@pytest.mark.parametrize('order', range(5))
@pytest.mark.parametrize('radius', (F(0), F(1), F(18)))
def test_uniform_tail_majorant_dominates_explicit_omitted_shells(order, radius):
    bound = s.image_tail(order, radius, prec=60)
    for point in (-radius, radius):
        partial = sum((abs(explicit_gaussian_derivative(order, point+24*j, 60))
                       for j in (-3, -2, 2, 3)), I.exact(0))
        assert partial.hi <= bound


def test_exact_normalization_periodicity_evenness_and_sign():
    for point in (F(0), F(24), F(-48)):
        assert s.kernel_derivative(0, point) == I.exact(1)
        assert s.kernel_derivative(1, point) == I.exact(0)
        assert s.kernel_derivative(3, point) == I.exact(0)
    for order in range(5):
        a = s.kernel_derivative(order, F(1, 20))
        assert overlaps(a, s.kernel_derivative(order, F(481, 20)))
        assert overlaps(a, (-1)**order*s.kernel_derivative(order, F(-1, 20)))
    p, q = (F(1, 20), F(0)), (F(0), F(0))
    assert s.covariance(p, (0, 0), q, (1, 0)).lo > 0
    assert s.covariance(p, (1, 0), q, (0, 0)).hi < 0
    assert s.covariance(p, (1, 0), p, (1, 0)).lo > 0
    # Omitting the second-argument sign would make this variance negative.
    assert s.kernel_derivative(2, 0).hi < 0


def test_512_bit_torus_derivative_excludes_planar_and_unnormalized_laws(monkeypatch):
    denominator = s.normalizer(prec=200, bits=512)
    correct = s.kernel_derivative(2, 0, prec=200, bits=512)
    assert denominator.lo > 1
    assert -correct.lo < 1  # Exact torus gradient variance differs from one.
    monkeypatch.setattr(s, 'normalizer', lambda **kwargs: I.exact(1))
    unnormalized = s.kernel_derivative(2, 0, prec=200, bits=512)
    assert not overlaps(correct, unnormalized)


def test_1024_bit_omitted_image_negative_control(monkeypatch):
    # At s=-12 the first omitted positive image is exp(-36^2/2).
    # This precision resolves that correction and distinguishes a zero tail.
    denominator = s.normalizer(prec=370, bits=1024)
    partial_five = sum((exp(I.exact(-(F(-12)+24*j)**2/2), 370)
                        for j in (-2, -1, 0, 1, 2)), I.exact(0))/denominator
    correct = s.kernel_derivative(0, -12, prec=370, bits=1024)
    assert correct.hi >= partial_five.lo
    monkeypatch.setattr(s, 'image_tail', lambda *args, **kwargs: F(0))
    omitted = s.kernel_derivative(0, -12, prec=370, bits=1024)
    assert omitted.hi < partial_five.lo


@pytest.mark.parametrize('order', range(5))
def test_whole_displacement_interval_encloses_members(order):
    domain = I(F(-1, 10), F(1, 5))
    whole = s.kernel_derivative(order, domain)
    for point in (domain.lo, F(0), F(1, 9), domain.hi):
        member = s.kernel_derivative(order, point)
        assert whole.lo <= member.lo <= member.hi <= whole.hi


def test_exact_source_pin_order_and_centered_mark():
    block = s.nine_pin_blocks()
    assert block['source_binding']['archive_sha256'] == s.ARCHIVE_SHA
    assert block['conditioning_order'] == ('M:f', 'M:fx', 'M:fy', 'S:f', 'S:fx', 'S:fy', 'y:f', 'y:fx', 'y:fy')
    assert block['target_order'] == ('M:xx', 'M:yy', 'M:xy', 'S:xx', 'S:yy', 'S:xy', 'y:xx', 'y:yy', 'y:xy')
    assert block['mark_domain'] == (-F(1, 96000), F(1, 96000))
    assert block['conditioning_intercept'] == (F(6, 5), 0, 0, F(6, 5)-F(1, 48000), 0, 0, F(6, 5)-F(1, 96000), 0, 0)
    assert block['conditioning_slope'] == (0, 0, 0, 0, 0, 0, 1, 0, 0)
    assert block['covariance_law'] == block['mean_law'] == block['conditioning_law']
    matrix = block['raw_covariance']
    assert len(matrix) == 18 and all(len(row) == 18 for row in matrix)
    assert all(matrix[i][j] == matrix[j][i] for i in range(18) for j in range(18))
    assert all(matrix[i][i].lo > 0 for i in range(18))


def test_point_laws_conditioning_and_marginal_pivots_are_positive():
    result = s.point_laws()
    assert all(p.lo > 0 for p in result['pivot_intervals'])
    assert len(result['pivot_intervals']) == 9
    assert set(result['laws']) == {'M', 'S', 'y'}
    for name, law in result['laws'].items():
        assert all(p.lo > 0 for p in result['marginal_pivot_intervals'][name])
        assert law.context.domain == s.MARK_DOMAIN
        assert law.mean_law == law.covariance_law == result['conditioning_law']
        assert law.context.order == ('xx', 'yy', 'xy')
        assert len(law.intercept) == len(law.slope) == len(law.covariance) == 3
    # Coarse rational windows from a separate explicit-polynomial diagnostic,
    # not an asserted exact value or an independence claim.
    assert I('-0.050835', '-0.050833').lo < result['laws']['M'].intercept[0].lo
    assert result['laws']['M'].intercept[0].hi < F('-0.050833')
    assert F('-3.325867') < result['laws']['y'].slope[1].lo
    assert result['laws']['y'].slope[1].hi < F('-3.325865')
    assert result['authority'] == 'NONE' and result['independence_credit'] == 0
    assert all(result[key] is False for key in ('field_certified', 'original_prize_closed',
               'spatial_cover_certified', 'scientific_status_changed'))


def test_source_mismatch_refuses_before_returning_a_law(tmp_path, monkeypatch):
    corrupted = tmp_path/'wrong-source.zip'
    corrupted.write_bytes(s.ARCHIVE.read_bytes()+b'corruption')
    monkeypatch.setattr(s, 'ARCHIVE', corrupted)
    with pytest.raises(ValueError, match='source identity mismatch'):
        s.nine_pin_blocks()


@pytest.mark.parametrize('order,point,options', ((5, 0, {}), (-1, 0, {}),
    (True, 0, {}), (0, 0.5, {}), (0, True, {}), (0, I(-7, 7), {}),
    (0, 0, {'bits': 127}), (0, 0, {'prec': 401}), (0, F(1, 2**4097), {})))
def test_kernel_rejects_unsupported_or_unbounded_inputs(order, point, options):
    with pytest.raises((ValueError, TypeError)):
        s.kernel_derivative(order, point, **options)


@pytest.mark.parametrize('point', ((F(-1, 40), 0), (F(1, 40), 0), (18, 0), (1.0, 1)))
def test_pin_collision_or_unbounded_inputs_refuse(point):
    with pytest.raises((ValueError, TypeError)):
        s.nine_pin_blocks(point)
