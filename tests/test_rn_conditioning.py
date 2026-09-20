"""Exact oracles and negative controls for conditioning, not an RN closure."""
from fractions import Fraction as F
from itertools import product

import pytest

from research.interval import Interval as I
from research.rn import conditioning as C


def transpose(a):
    return tuple(zip(*a))


def multiply(a, b):
    return tuple(tuple(sum((F(x) * F(y) for x, y in zip(row, column)), F(0))
                       for column in transpose(b)) for row in a)


def apply(a, b):
    return tuple(sum((F(x) * F(y) for x, y in zip(row, b)), F(0)) for row in a)


def plus(a, b):
    return tuple(tuple(x + y for x, y in zip(left, right)) for left, right in zip(a, b))


def inverse(a):
    """Independent exact Gauss-Jordan oracle; production uses no inverse."""
    n = len(a)
    rows = [[F(x) for x in row] + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for column in range(n):
        pivot = next(row for row in range(column, n) if rows[row][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        denominator = rows[column][column]
        rows[column] = [x / denominator for x in rows[column]]
        for row in range(n):
            if row != column:
                weight = rows[row][column]
                rows[row] = [x - weight * y for x, y in zip(rows[row], rows[column])]
    return tuple(tuple(row[n:]) for row in rows)


def kwargs(n, m, **extra):
    return dict(conditioning_order=tuple(f"X{i}" for i in range(n)),
                target_order=tuple(f"Y{i}" for i in range(m)),
                covariance_law="one fixed fixture Gaussian law", mean_law="one fixed fixture Gaussian law",
                conditioning_law="one fixed fixture Gaussian law", mark_domain=(F(-2), F(3)), **extra)


def assert_points(actual, expected):
    assert tuple(actual) == tuple(I.exact(value) for value in expected)


def fixture_blocks():
    a = ((F(4), F(1), F(1, 2)), (F(1), F(3), F(-1, 4)), (F(1, 2), F(-1, 4), F(2)))
    transform = ((F(1), F(-2), F(1, 3)), (F(1, 2), F(3), F(-1)), (F(-1), F(1, 4), F(2)))
    residual = ((F(2), F(1, 4), F(0)), (F(1, 4), F(3), F(1, 5)), (F(0), F(1, 5), F(1)))
    cross = multiply(transform, a)
    target = plus(multiply(cross, transpose(transform)), residual)
    return a, cross, target, transform, residual


def test_constructed_joint_gaussian_recovers_exact_residual_and_affine_gain():
    a, cross, target, transform, residual = fixture_blocks()
    c0, c1 = (F(2), F(-1), F(3)), (F(1, 3), F(2), F(-1))
    result = C.condition_gaussian(a, cross, target, c0, c1, **kwargs(3, 3))
    for actual, expected in zip(result.gain, transform):
        assert_points(actual, expected)
    for actual, expected in zip(result.covariance, residual):
        assert_points(actual, expected)
    assert_points(result.intercept, apply(transform, c0))
    assert_points(result.slope, apply(transform, c1))
    assert all(pivot.lo > 0 for pivot in result.pivots)
    assert result.all_conditioning_matrices_spd
    assert not result.joint_psd_established and not result.field_certified
    assert result.independence_credit == 0 and not result.scientific_status_changed


def test_independent_inverse_and_nonzero_affine_prior_means():
    a, cross, target, _, _ = fixture_blocks()
    c0, c1 = (F(2), F(-1), F(3)), (F(1, 3), F(2), F(-1))
    mx0, mx1 = (F(1, 2), F(1, 3), F(-2)), (F(-1), F(2, 5), F(1))
    my0, my1 = (F(7), F(-3), F(1, 4)), (F(1), F(2), F(3))
    result = C.condition_gaussian(a, cross, target, c0, c1, **kwargs(3, 3),
                                 prior_conditioning_intercept=mx0, prior_conditioning_slope=mx1,
                                 prior_target_intercept=my0, prior_target_slope=my1)
    gain = multiply(cross, inverse(a))
    correction = multiply(gain, transpose(cross))
    covariance = tuple(tuple(x - y for x, y in zip(br, cr)) for br, cr in zip(target, correction))
    intercept = tuple(x + y for x, y in zip(my0, apply(gain, tuple(x - y for x, y in zip(c0, mx0)))))
    slope = tuple(x + y for x, y in zip(my1, apply(gain, tuple(x - y for x, y in zip(c1, mx1)))))
    assert_points(result.intercept, intercept)
    assert_points(result.slope, slope)
    for actual, expected in zip(result.covariance, covariance):
        assert_points(actual, expected)
    for mark in (F(-2), F(0), F(1, 3), F(3)):
        assert_points(result.mean(mark), tuple(x + mark * y for x, y in zip(intercept, slope)))
    with pytest.raises(ValueError, match="outside"):
        result.mean(F(4))


def test_wrong_schur_sign_or_cross_transpose_would_fail_exact_oracle():
    a, cross, target, transform, residual = fixture_blocks()
    result = C.condition_gaussian(a, cross, target, (1, 2, 3), (0, 1, 0), **kwargs(3, 3))
    wrong_plus = plus(target, multiply(multiply(cross, inverse(a)), transpose(cross)))
    wrong_gain = multiply(inverse(a), cross)
    assert wrong_plus[0][0] not in result.covariance[0][0]
    assert any(wrong_gain[i][j] not in result.gain[i][j] for i in range(3) for j in range(3))
    assert result.covariance[0][0] == I.exact(residual[0][0])
    assert result.gain[0][0] == I.exact(transform[0][0])


def test_negating_cross_block_negates_mean_but_preserves_covariance():
    a, cross, target, _, _ = fixture_blocks()
    positive = C.condition_gaussian(a, cross, target, (1, 2, 3), (0, 1, 0), **kwargs(3, 3))
    negative = C.condition_gaussian(a, tuple(tuple(-x for x in row) for row in cross), target,
                                    (1, 2, 3), (0, 1, 0), **kwargs(3, 3))
    assert negative.intercept == tuple(-value for value in positive.intercept)
    assert negative.slope == tuple(-value for value in positive.slope)
    assert negative.covariance == positive.covariance


def test_direct_conditioning_matches_two_stage_schur_and_affine_mean():
    a, cross, target, _, _ = fixture_blocks()
    c0, c1 = (F(2), F(-1), F(3)), (F(1, 3), F(2), F(-1))
    direct = C.condition_gaussian(a, cross, target, c0, c1, **kwargs(3, 3))
    joint = tuple(tuple(a[i]) + tuple(cross[j][i] for j in range(3)) for i in range(3))
    joint += tuple(tuple(cross[i]) + tuple(target[i]) for i in range(3))
    first = C.condition_gaussian(((joint[0][0],),), tuple((row[0],) for row in joint[1:]),
                                 tuple(tuple(row[1:]) for row in joint[1:]),
                                 (c0[0],), (c1[0],), **kwargs(1, 5))
    second = C.condition_gaussian(tuple(tuple(row[:2]) for row in first.covariance[:2]),
                                  tuple(tuple(row[:2]) for row in first.covariance[2:]),
                                  tuple(tuple(row[2:]) for row in first.covariance[2:]),
                                  c0[1:], c1[1:], **kwargs(2, 3),
                                  prior_conditioning_intercept=first.intercept[:2],
                                  prior_conditioning_slope=first.slope[:2],
                                  prior_target_intercept=first.intercept[2:],
                                  prior_target_slope=first.slope[2:])
    assert second.intercept == direct.intercept
    assert second.slope == direct.slope
    assert second.covariance == direct.covariance


def test_ldl_solves_match_independent_inverse():
    a, _, _, _, _ = fixture_blocks()
    factor = C.interval_ldlt(a)
    for rhs in ((F(1), F(-2), F(3)), (F(0), F(1), F(0))):
        assert_points(factor.solve(rhs), apply(inverse(a), rhs))
    assert len(factor.pivots) == 3


@pytest.mark.parametrize("matrix", [((0,),), ((-1,),), ((1, 1), (1, 1)), ((1, 2), (2, 1))])
def test_zero_negative_and_singular_pivots_are_rejected(matrix):
    with pytest.raises(C.ConditioningInconclusive, match="pivot"):
        C.interval_ldlt(matrix)


def test_uncertain_ill_conditioned_block_refuses_even_with_spd_midpoint():
    epsilon = F(1, 10**12)
    off = I(1 - epsilon, 1 + epsilon)
    with pytest.raises(C.ConditioningInconclusive):
        C.interval_ldlt(((1, off), (off, 1 + epsilon)))
    # A tiny exact positive pivot can be certified; no numerical threshold is
    # invented. A caller may require an explicit positive admission margin.
    exact = C.interval_ldlt(((1, 1), (1, 1 + epsilon)))
    assert exact.pivots[1] == I.exact(epsilon)
    with pytest.raises(C.ConditioningInconclusive):
        C.interval_ldlt(((1, 1), (1, 1 + epsilon)), pivot_floor=epsilon)


def test_interval_members_enclose_exact_inverse_reference_at_matrix_corners():
    off = I(F(-1, 5), F(1, 5))
    a_box = ((I(2, F(21, 10)), off), (off, I(3, F(31, 10))))
    c_box = ((I(F(-1, 4), F(1, 4)), I(F(1, 3), F(1, 2))),
             (I(F(-1, 2), F(-1, 3)), I(F(-1, 5), F(1, 5))))
    b = ((F(5), F(1, 3)), (F(1, 3), F(6)))
    c0, c1 = (F(2), F(-1)), (F(1, 3), F(2))
    result = C.condition_gaussian(a_box, c_box, b, c0, c1, **kwargs(2, 2))
    for diagonal0, diagonal1, off_value in product((F(2), F(21, 10)), (F(3), F(31, 10)),
                                                 (F(-1, 5), F(0), F(1, 5))):
        a = ((diagonal0, off_value), (off_value, diagonal1))
        for values in product(*((entry.lo, entry.hi) for row in c_box for entry in row)):
            cross = (values[:2], values[2:])
            gain = multiply(cross, inverse(a))
            correction = multiply(gain, transpose(cross))
            for i in range(2):
                assert apply(gain, c0)[i] in result.intercept[i]
                assert apply(gain, c1)[i] in result.slope[i]
                for j in range(2):
                    assert b[i][j] - correction[i][j] in result.covariance[i][j]
    assert result.covariance[0][1] == result.covariance[1][0]


def test_midpoint_substitution_would_lose_an_admitted_member():
    result = C.condition_gaussian(((I(1, 2),),), ((F(1, 2),),), ((2,),), (2,), (1,), **kwargs(1, 1))
    assert F(1) in result.intercept[0]  # A=1, not the midpoint A=3/2
    assert F(1, 2) in result.intercept[0]  # A=2
    assert not result.intercept[0].is_point()
    assert F(1) not in I.exact(F(2, 3))  # explicit weakened midpoint control


def test_negative_schur_diagonal_cannot_describe_a_joint_gaussian():
    with pytest.raises(C.ConditioningInconclusive, match="negative conditional variance"):
        C.condition_gaussian(((1,),), ((2,),), ((1,),), (0,), (0,), **kwargs(1, 1))


def test_nine_conditioning_plus_nine_target_coordinates_fit_boundary():
    identity = tuple(tuple(int(i == j) for j in range(9)) for i in range(9))
    zero = tuple((0,) * 9 for _ in range(9))
    result = C.condition_gaussian(identity, zero, identity, (1,) * 9, (1,) * 9, **kwargs(9, 9))
    assert_points(result.intercept, (0,) * 9)
    assert len(result.covariance) == 9 and len(result.pivots) == 9
    larger = tuple(tuple(int(i == j) for j in range(10)) for i in range(10))
    with pytest.raises(ValueError, match="joint dimension"):
        C.condition_gaussian(larger, zero, identity, (1,) * 10, (1,) * 10, **kwargs(10, 9))


@pytest.mark.parametrize("change", ["mean_law", "conditioning_law", "empty_law", "duplicate_order", "overlap", "bad_order_size"])
def test_law_and_coordinate_mismatches_refused(change):
    options = kwargs(2, 1)
    if change in ("mean_law", "conditioning_law"):
        options[change] = "different Gaussian law"
    elif change == "empty_law":
        options.update(covariance_law="", mean_law="", conditioning_law="")
    elif change == "duplicate_order":
        options["conditioning_order"] = ("X", "X")
    elif change == "overlap":
        options["target_order"] = ("X0",)
    else:
        options["target_order"] = ("Y0", "Y1")
    with pytest.raises(ValueError):
        C.condition_gaussian(((2, 0), (0, 3)), ((0, 0),), ((1,),), (0, 0), (0, 0), **options)


@pytest.mark.parametrize("matrix", [(), ((1, 0),), ((1, 0), (1, 1)), ((1.0,),), ((True,),), (("1e999999999",),)])
def test_invalid_matrix_or_inexact_entries_refused(matrix):
    with pytest.raises(ValueError):
        C.interval_ldlt(matrix)


@pytest.mark.parametrize("bits", [False, 127, 1025])
def test_invalid_precision_refused(bits):
    with pytest.raises(ValueError):
        C.interval_ldlt(((1,),), round_bits=bits)


def test_rounding_widens_exact_values_without_exponential_denominator_growth():
    denominator = (1 << 300) + 151
    tiny = F(1, denominator)
    factor = C.interval_ldlt(((1 + tiny, F(1, 3)), (F(1, 3), 2)), round_bits=128)
    reference = apply(inverse(((1 + tiny, F(1, 3)), (F(1, 3), 2))), (F(1), F(-2)))
    answer = factor.solve((1, -2))
    assert all(value in enclosure for value, enclosure in zip(reference, answer))
    assert all(max(x.numerator.bit_length(), x.denominator.bit_length()) < 512
               for enclosure in answer for x in (enclosure.lo, enclosure.hi))


def test_valid_nondyadic_boundary_mark_is_not_rejected_by_outward_rounding():
    boundary = F(1, (1 << 300) + 151)
    options = kwargs(1, 1)
    options["mark_domain"] = (F(0), boundary)
    result = C.condition_gaussian(((1,),), ((1,),), ((2,),), (0,), (1,), **options)
    assert boundary in result.mean(boundary)[0]


def test_arithmetic_budgets_fail_without_returning_a_bound(monkeypatch):
    with pytest.raises(C.ConditioningResourceLimit):
        C.interval_ldlt(((F(1, 1 << (C.MAX_BITS + 1)),),))
    monkeypatch.setattr(C, "MAX_OPERATIONS", 1)
    with pytest.raises(C.ConditioningResourceLimit):
        C.interval_ldlt(((1, F(1, 2)), (F(1, 2), 2)))
