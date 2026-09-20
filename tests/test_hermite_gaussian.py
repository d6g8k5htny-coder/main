"""Author checks and negative controls; no mathematical promotion or review credit."""
from fractions import Fraction as F
from math import factorial

import pytest
from research.interval import Interval, exp
from research.bands.hermite_gaussian import (
    hermite_coefficients, hermite_gaussian, hermite_majorant, monomial_majorant,
)
from research.bands.lattice import band_enclosure, tail_bound


def closed_coefficients(n):
    coefficients = [0] * (n + 1)
    for k in range(n // 2 + 1):
        coefficients[n - 2*k] = ((-1)**k * factorial(n)
            // (2**k * factorial(k) * factorial(n - 2*k)))
    return tuple(coefficients)


@pytest.mark.parametrize('n', range(21))
def test_recurrence_matches_independent_factorial_formula(n):
    assert hermite_coefficients(n) == closed_coefficients(n)
    if n:
        derivative = tuple(j*c for j,c in enumerate(hermite_coefficients(n)) if j)
        assert derivative == tuple(n*c for c in hermite_coefficients(n-1))


@pytest.mark.parametrize('bad', [-1, True, False, 1.0, F(1), '1'])
def test_order_refuses_non_integer_even_after_cache_hit(bad):
    hermite_coefficients(1)
    with pytest.raises(ValueError): hermite_gaussian(bad, 0)
    with pytest.raises(ValueError): hermite_gaussian(0, bad)
    with pytest.raises(ValueError): monomial_majorant(bad)


def test_exact_majorants_and_separate_axis_product():
    assert [hermite_majorant(n) for n in range(4)] == [1, F(5,2), 5, F(51,2)]
    kernel = hermite_gaussian(1, 1)
    assert kernel.envelope.A == F(25,4)
    assert kernel.envelope.A != hermite_majorant(2)
    assert kernel.envelope.B == F(1,4)
    assert kernel.certified and not kernel.envelope.certified
    assert not kernel.fully_certified


@pytest.mark.parametrize('a,b', [(0,0),(1,0),(0,1),(1,1),(2,1),(4,2),(3,3)])
def test_signed_values_against_independent_exact_polynomial(a,b):
    x,y=F(3,2),F(-2,3)
    def poly(n,t): return sum(c*t**j for j,c in enumerate(closed_coefficients(n)))
    scalar=(-1)**(a+b)*poly(a,x)*poly(b,y)
    expected=scalar*exp(Interval.exact(-(x*x+y*y)/2),35)
    actual=hermite_gaussian(a,b).evaluate(Interval.exact(x),Interval.exact(y),35)
    assert actual == expected


def test_boxes_include_interior_points_and_cross_zero():
    kernel=hermite_gaussian(2,1)
    box=kernel.evaluate(Interval(-1,2),Interval(-2,1),20)
    for x,y in [(0,0),(F(1,2),F(-1,2)),(-1,1),(2,-2)]:
        point=kernel.evaluate(Interval.exact(x),Interval.exact(y),25)
        assert box.lo <= point.lo <= point.hi <= box.hi


def test_tail_is_positive_added_and_grows_for_larger_euclidean_box():
    kernel=hermite_gaussian(2,1)
    enc=band_enclosure(kernel,Interval(F(1,100),F(1,50)),prec=16)
    assert enc.tail > 0
    assert enc.total.lo == enc.truncated.lo-enc.tail
    assert enc.total.hi == enc.truncated.hi+enc.tail
    assert not enc.certified
    axial=tail_bound(kernel.envelope,Interval(-17,17),Interval.exact(0),prec=16)
    square=tail_bound(kernel.envelope,Interval(-17,17),Interval(-17,17),prec=16)
    assert square > axial > 0


def test_negative_control_false_amplitude_loses_global_domination():
    # At (1,1), |k_11|=exp(-1); A=1/4 with B=1/4 is already too small.
    actual=hermite_gaussian(1,1).evaluate(Interval.exact(1),Interval.exact(1),25)
    false_envelope=F(1,4)*exp(Interval.exact(F(-1,2)),25)
    assert actual.lo > false_envelope.hi


def test_negative_control_dropped_hermite_constant_loses_containment():
    # He_2(0)=-1. Replacing He_2 by x² would report zero at the origin.
    actual=hermite_gaussian(2,0).evaluate(Interval.exact(0),Interval.exact(0),25)
    assert actual.lo == actual.hi == -1
    assert not actual.lo <= 0 <= actual.hi


def test_report_cli_detects_review_promotion_and_weakened_tail(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=root/'research/bands/candidates/hermite_gaussian_20260919.json'
    original=json.loads(source.read_text())
    assert len(original['cases'])==66
    assert all(row['tail_lt_1e_60'] for row in original['cases'])
    for mutation in ('promotion','tail'):
        record=json.loads(source.read_text())
        if mutation=='promotion':record['envelope_reviewed']=True
        else:record['cases'][0]['tail_upper']='0'
        target=tmp_path/(mutation+'.json');target.write_text(json.dumps(record))
        run=subprocess.run([sys.executable,str(root/'tools/hermite_envelope_report.py'),'--check',str(target)],capture_output=True,text=True)
        assert run.returncode==1
        assert 'candidate report drift' in run.stderr
