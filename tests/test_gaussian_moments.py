"""Independent exact expansions and refusal/inequality negative controls."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import pytest

from engine.operations.rn_applicability import Context, Law, RN5_ID, RN5_SHA, RETIRED, applicable, apply_moment_bound
from research.interval import Interval
from research.rn.gaussian_moments import (Cost, MomentEngine, bernstein_enclosure,
    coefficient_cap, cauchy_schwarz_squared, evaluate, holder_pow4, moment_cap,
    second_moment_trace)
from research.rn.moment_envelope import det_moment

SOURCE = Path(__file__).resolve().parents[1] / 'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md'


def independent_expansion(mean, lower, degree):
    """Expand in independent standard normals; no Stein recurrence or Q formula."""
    def mul(a, b):
        out = {}
        for i, x in a.items():
            for j, y in b.items():
                k = tuple(i[n]+j[n] for n in range(3))
                out[k] = out.get(k, F(0)) + x*y
        return out
    forms = [{(0, 0, 0): m, **{tuple(int(j==k) for k in range(3)): lower[i][j] for j in range(3)}} for i, m in enumerate(mean)]
    det = mul(forms[0], forms[1])
    for k, v in mul(forms[2], forms[2]).items():
        det[k] = det.get(k, F(0))-v
    polynomial = {(0, 0, 0): F(1)}
    for _ in range(degree):
        polynomial = mul(polynomial, det)
    def central(n):
        if n % 2:
            return 0
        out = 1
        for j in range(1, n, 2):
            out *= j
        return out
    return sum(value*central(a)*central(b)*central(c) for (a,b,c), value in polynomial.items())


@pytest.mark.parametrize('case', range(4))
@pytest.mark.parametrize('degree', (2,4))
def test_correlated_noncentered_against_independent_expansion(case, degree):
    lower = tuple(tuple(F(case+i+j+1, 7) if j <= i else F(0) for j in range(3)) for i in range(3))
    cov = tuple(tuple(sum(lower[i][k]*lower[j][k] for k in range(3)) for j in range(3)) for i in range(3))
    intercept = (F(case-2), F(3-case), F(case+1, 3))
    slope = (F(1, 3), F(-2, 5), F(4, 7))
    engine = MomentEngine(intercept, slope, cov)
    polynomial = engine.determinant(degree)
    assert len(polynomial) <= 2*degree+1
    for mark in (F(-1), F(0), F(2, 3)):
        mean = tuple(a+b*mark for a,b in zip(intercept, slope))
        assert evaluate(polynomial, mark) == independent_expansion(mean, lower, degree)
        if degree == 2:
            assert evaluate(polynomial, mark) == second_moment_trace(mean, cov)


@pytest.mark.parametrize('mu_x,mu_y', product((-1,0,1), repeat=2))
@pytest.mark.parametrize('degree', (2,4))
def test_rn5_old_independent_special_case(mu_x, mu_y, degree):
    v = F(1, 10**6)
    cov = tuple(tuple(v if i==j else F(0) for j in range(3)) for i in range(3))
    assert MomentEngine((mu_x,mu_y,0),(0,0,0),cov).determinant(degree) == (det_moment(F(mu_x),F(mu_y),v,degree),)


@pytest.mark.parametrize('cov', (((0,1,0),(1,1,0),(0,0,1)), ((1,2,0),(2,1,0),(0,0,1)), ((1,0,0),(1,1,0),(0,0,1))))
def test_non_psd_or_asymmetric_rejected(cov):
    with pytest.raises(ValueError):
        MomentEngine((0,0,0),(0,0,0),cov)


def test_singular_law_and_strict_bernstein_improvement():
    cov = ((0,0,0),)*3
    p = MomentEngine((-1,-1,0),(1,-1,0),cov).determinant(4)
    assert p == (1,0,-4,0,6,0,-4,0,1)  # (1-t^2)^4
    domain = Interval(-1,1)
    assert coefficient_cap(p,domain) == 16
    assert moment_cap(p,domain) == 1
    assert bernstein_enclosure(p,domain,depth=2).lo == 0
    # Endpoints alone would miss the interior maximum.
    assert evaluate(p,F(-1)) == evaluate(p,F(1)) == 0
    assert evaluate(p,F(0)) == 1


def test_positive_definite_strict_improvement():
    v=F(1,1000)
    cov=tuple(tuple(v if i==j else F(0) for j in range(3)) for i in range(3))
    p=MomentEngine((-1,-1,0),(1,-1,0),cov).determinant(4)
    assert moment_cap(p,Interval(-1,1)) < coefficient_cap(p,Interval(-1,1))


def test_bernstein_bounds_direct_polynomial_and_subdivision():
    p=(F(1),F(2),F(-3),F(1,7),F(5,9))
    domain=Interval(F(-2,3),F(4,5))
    outer=bernstein_enclosure(p,domain)
    inner=bernstein_enclosure(p,domain,depth=3)
    assert outer.lo <= inner.lo <= inner.hi <= outer.hi
    for j in range(101):
        assert evaluate(p,domain.lo+(domain.hi-domain.lo)*F(j,100)) in inner


def test_missing_probability_sqrt_is_not_a_bound():
    # D=1_A: E|D|1_A=p and ED^2=p. Correct bound is p; wrong one p^(3/2).
    p=F(1,4)
    assert cauchy_schwarz_squared(p,p) == p**2
    assert p*p**2 < p**2
    with pytest.raises(ValueError):
        cauchy_schwarz_squared(1,2)


def test_noncentral_mean_term_overcount_and_conditional_mixing():
    cov=((F(1,4),0,0),(0,F(1,4),0),(0,0,F(1,4)))
    m=(F(2),F(3),F(1))
    correct=second_moment_trace(m,cov)
    assert correct == MomentEngine(m,(0,0,0),cov).determinant(2)[0]
    # Regression mutation inspired by the K3 mean-term defect, not a replay of W12.
    assert correct != correct + 2*(m[0]*m[1]-m[2]**2)**2
    # Conditional means (2,3,1) paired with marginal zeros silently underestimate.
    assert second_moment_trace((0,0,0),cov) < correct


def fixture():
    context=Context(RN5_ID,RN5_SHA,2,('xx','yy','xy'),'exact synthetic Hessian units','synthetic conditional G', (F(-1),F(1)))
    return Law(context,(-1,-1,0),(1,-1,0),((0,0,0),)*3,context.conditioned_law,context.conditioned_law)


@pytest.mark.parametrize('key,value', [('source_id','stale'),('source_sha256','0'*64),('dimension',3),('order',('xx','xy','yy')),('normalization','changed units'),('conditioned_law','marginal'),('domain',(F(0),F(1))),('evidence_tier','FIELD_CERTIFIED')])
def test_all_applicability_axes_reject(key,value):
    law=fixture()
    with pytest.raises(ValueError):
        applicable(replace(law,context=replace(law.context,**{key:value})),law.context)


def test_mixed_laws_and_stale_source_refuse():
    law=fixture()
    for key in ('mean_law','covariance_law'):
        with pytest.raises(ValueError):
            applicable(replace(law,**{key:'marginal'}),law.context)
    with pytest.raises(ValueError):
        apply_moment_bound(law,law.context,degree=4,source_bytes=b'stale')
    for strategy in RETIRED:
        with pytest.raises(ValueError):
            apply_moment_bound(law,law.context,degree=4,source_bytes=SOURCE.read_bytes(),strategy=strategy)


def test_real_dispatch_and_numerical_identity_fingerprint():
    law=fixture()
    result=apply_moment_bound(law,law.context,degree=4,source_bytes=SOURCE.read_bytes())
    assert result['upper']==1
    assert result['field_certified'] is False and result['independence_credit']==0
    assert replace(law,intercept=(2,3,1)).fingerprint() != law.fingerprint()


def test_no_independence_required_holder_and_input_refusals():
    # Three identical deterministic determinants, dependent, exactly |D|^12.
    d=F(1,4)
    assert holder_pow4([{2:d**2,4:d**4}]*3)==d**12
    with pytest.raises(ValueError): holder_pow4([{4:1}]*3)
    with pytest.raises(ValueError): MomentEngine((0,0,0),(0,0,0),((0,0,0),)*3,order=('xx','xy','yy'))
    with pytest.raises(TypeError): MomentEngine((0.0,0,0),(0,0,0),((0,0,0),)*3)
    with pytest.raises(TypeError): evaluate((1,),True)
    with pytest.raises(RuntimeError): MomentEngine((0,0,0),(0,0,0),((0,0,0),)*3,cost=Cost(0))
    for degree in (True,3,8):
        with pytest.raises(ValueError): MomentEngine((0,0,0),(0,0,0),((0,0,0),)*3).determinant(degree)
