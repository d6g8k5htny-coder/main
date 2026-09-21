from fractions import Fraction as F
from itertools import permutations, product

import pytest

from research.cover import Box
from research.interval import Interval as I
from research.rn import side24_wedge as wedge
from research.rn.conditioning import ConditioningInconclusive
from research.rn.n6_inputs import checked_n6


def exact_determinant(matrix):
    answer = F(0)
    for perm in permutations(range(3)):
        inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
        term = F((-1)**inversions)
        for i, j in enumerate(perm):
            term *= matrix[i][j]
        answer += term
    return answer


def matrix_coeff(entries):
    return tuple(tuple(I.exact(x) for x in row) for row in entries)


def test_shared_monomials_cancel_before_evaluation():
    # det([[1+s²+t²,s,t],[s,1,0],[t,0,1]]) == 1 everywhere.
    # Separate entry hulls lose this identity on a large rectangle.
    coeff = {
        (0, 0): matrix_coeff(((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        (2, 0): matrix_coeff(((1, 0, 0), (0, 0, 0), (0, 0, 0))),
        (0, 2): matrix_coeff(((1, 0, 0), (0, 0, 0), (0, 0, 0))),
        (1, 0): matrix_coeff(((0, 1, 0), (1, 0, 0), (0, 0, 0))),
        (0, 1): matrix_coeff(((0, 0, 1), (0, 0, 0), (1, 0, 0))),
    }
    polynomial = wedge._determinant_polynomial(coeff, bits=256)
    assert polynomial == {(0, 0): I.exact(1)}
    assert wedge._evaluate(polynomial, (F(100), F(100)), bits=256) == I.exact(1)


def test_determinant_cross_term_and_signs_match_independent_permutations():
    matrix = ((F(4), F(1), F(2)), (F(1), F(5), F(1)), (F(2), F(1), F(6)))
    expected = exact_determinant(matrix)
    assert expected == 94
    polynomial = wedge._determinant_polynomial({(0, 0): matrix_coeff(matrix)}, bits=256)
    assert polynomial == {(0, 0): I.exact(expected)}


def test_even_monomials_keep_sign_and_odd_monomials_change_sign():
    result = wedge._evaluate({(0, 0): I.exact(7), (2, 0): I.exact(3),
                              (0, 4): I.exact(-2), (1, 1): I.exact(5)},
                             (F(2), F(1)), bits=256)
    assert result == I(-5, 29)
    for s, t in product((F(-2), F(0), F(2)), (F(-1), F(0), F(1))):
        value = 7+3*s*s-2*t**4+5*s*t
        assert result.lo <= value <= result.hi


def test_all_symmetric_remainder_corners_fit_multilinear_error():
    matrix = ((F(3), F(-1, 4), F(1, 3)), (F(-1, 4), F(2), F(1, 5)),
              (F(1, 3), F(1, 5), F(4)))
    error = ((F(1, 10), F(1, 20), F(1, 30)),
             (F(1, 20), F(1, 11), F(1, 40)),
             (F(1, 30), F(1, 40), F(1, 12)))
    bound = wedge._determinant_error(matrix_coeff(matrix), error)
    center = exact_determinant(matrix)
    positions = tuple((i, j) for i in range(3) for j in range(i+1))
    for signs in product((-1, 1), repeat=6):
        perturbed = [list(row) for row in matrix]
        for (i, j), sign in zip(positions, signs):
            perturbed[i][j] += sign*error[i][j]
            perturbed[j][i] = perturbed[i][j]
        assert abs(exact_determinant(perturbed)-center) <= bound
    assert wedge._determinant_error(matrix_coeff(matrix), ((F(0),)*3,)*3) == 0


@pytest.mark.parametrize('order', [True, 6.0, 0, 7])
def test_bad_orders_refused_before_adapter_use(order):
    with pytest.raises(ValueError, match='Taylor order'):
        wedge.certify_rectangle(None, wedge.OUTER_BOX, order=order)


@pytest.mark.parametrize('bits', [True, 256.0, 127, 513])
def test_bad_precision_refused_before_adapter_use(bits):
    with pytest.raises(ValueError, match='rounding bits'):
        wedge.certify_rectangle(None, wedge.OUTER_BOX, bits=bits)


@pytest.mark.parametrize('pieces', [True, 12.0, 0, 257])
def test_bad_piece_count_refused_before_adapter_use(pieces):
    with pytest.raises(ValueError, match='piece count'):
        wedge.certify_wedge(None, pieces=pieces)


@pytest.mark.parametrize('box', [
    Box(F(1, 10), F(1, 10), F(-1, 10000), F(1, 10000)),
    Box(F(1, 10), F(11, 100), F(0), F(0)),
    Box(F(99, 1000), F(11, 100), F(-1, 10000), F(1, 10000)),
    Box(F(1, 10), F(11, 100), F(-8, 10000), F(1, 10000)),
])
def test_bad_rectangle_refused_before_adapter_use(box):
    with pytest.raises(ValueError, match='positive-area subset'):
        wedge.certify_rectangle(None, box)


def test_non_box_refused_before_adapter_use():
    with pytest.raises(TypeError, match='exact Box'):
        wedge.certify_rectangle(None, (F(1, 10), F(11, 100), F(0), F(1, 10000)))


def test_wedge_containment_and_area_are_exact():
    result = wedge.containment()
    assert result['x_lower'] > wedge.OUTER_BOX.u0
    assert result['absolute_y_upper'] < wedge.OUTER_BOX.v1
    assert result['area_pi_coefficient'] == (wedge.RADII[1]**2-wedge.RADII[0]**2)*(
        wedge.TURNS[1]-wedge.TURNS[0]) == F(21, 5120000)


def test_bad_geometry_control_refuses(monkeypatch):
    monkeypatch.setattr(wedge, 'PI_UPPER', F(10))
    with pytest.raises(ValueError, match='not contained'):
        wedge.containment()


@pytest.fixture(scope='module')
def actual_wedge():
    n6, before = checked_n6()
    report = wedge.certify_wedge(n6)
    _, after = checked_n6()
    assert before == after
    return n6, report


def test_actual_full_wedge_twelve_piece_certificate(actual_wedge):
    _, report = actual_wedge
    assert report['piece_count'] == len(report['pieces']) == 12
    assert report['pending_pieces'] == 0
    assert report['determinant_lower'] > F(8, 10**25)
    assert report['q_lower'] > 103
    assert report['conditioning_energy_upper'] < 8
    assert report['mark_domain'] == (-F(1, 96000), F(1, 96000))
    assert not report['full_annulus_certified'] and not report['all_radii_certified']
    assert not report['h3_reproved'] and not report['scientific_status_changed']
    records = report['pieces']
    assert records[0]['box'].u0 == wedge.OUTER_BOX.u0
    assert records[-1]['box'].u1 == wedge.OUTER_BOX.u1
    for left, right in zip(records, records[1:]):
        assert left['box'].u1 == right['box'].u0
    assert all(r['box'].v0 == wedge.OUTER_BOX.v0 and r['box'].v1 == wedge.OUTER_BOX.v1 for r in records)


def test_actual_first_piece_center_covariance_determinant_is_contained(actual_wedge):
    n6, report = actual_wedge
    first = report['pieces'][0]
    lift = n6.center_jet_lift(first['center'], order=6, bits=512)
    lookup = {alpha: 6+i for i, alpha in enumerate(lift['jets'])}
    ids = tuple(lookup[alpha] for alpha in ((0, 0), (1, 0), (0, 1)))
    covariance = tuple(tuple(lift['covariance'][i][j] for j in ids) for i in ids)
    determinant = exact_determinant(covariance)
    assert first['determinant'].lo <= determinant.lo <= determinant.hi <= first['determinant'].hi
    fx_mean, fx_variance = lift['mean'][ids[1]], covariance[1][1]
    assert first['fx_mean'].lo <= fx_mean.lo <= fx_mean.hi <= first['fx_mean'].hi
    assert first['fx_variance'].lo <= fx_variance.lo <= fx_variance.hi <= first['fx_variance'].hi
    assert first['q_lower'] <= (fx_mean**2/fx_variance).lo


def test_widened_single_piece_is_a_real_refusal(actual_wedge):
    n6, _ = actual_wedge
    with pytest.raises(ConditioningInconclusive, match='determinant lower bound'):
        wedge.certify_wedge(n6, pieces=1)


def test_zero_determinant_mutant_cannot_be_certified(actual_wedge, monkeypatch):
    n6, report = actual_wedge
    monkeypatch.setattr(wedge, '_determinant_polynomial', lambda *args, **kwargs: {})
    with pytest.raises(ConditioningInconclusive, match='determinant lower bound'):
        wedge.certify_rectangle(n6, report['pieces'][0]['box'])
