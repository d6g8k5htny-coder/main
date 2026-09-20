"""Independent moment expansion, feasibility and enclosure negative controls."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import pytest

from research.interval import Interval as I
from research.rn.gaussian_moments import MomentEngine, covariance_psd, moment_cap
from research.rn.gaussian_families import (FamilyMomentEngine, covariance_family,
    family_bernstein, family_moment_cap)
from engine.operations.rn_applicability import Context, RN5_ID, RN5_SHA
from engine.operations.rn_family_applicability import FamilyLaw, apply_family_bound
from test_gaussian_moments import independent_expansion

SOURCE = Path(__file__).resolve().parents[1]/'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md'


@pytest.mark.parametrize('case',range(4))
@pytest.mark.parametrize('degree',(2,4))
def test_uncertain_families_contain_independent_correlated_expansion(case,degree):
    lower=tuple(tuple(F(case+i+j+1,7) if j<=i else F(0) for j in range(3)) for i in range(3))
    s=tuple(tuple(sum(lower[i][k]*lower[j][k] for k in range(3)) for j in range(3)) for i in range(3))
    a=(F(case-2),F(3-case),F(case+1,3)); b=(F(1,3),F(-2,5),F(4,7)); eps=F(1,50)
    aa=tuple(I(x-eps,x+eps) for x in a); bb=tuple(I(x-eps,x+eps) for x in b)
    ss=tuple(tuple(I(x-eps,x+eps) for x in row) for row in s)
    engine=FamilyMomentEngine(aa,bb,ss); p=engine.determinant(degree)
    bound=family_bernstein(p,I(-1,1),depth=2)
    cap=family_moment_cap(p,I(-1,1),depth=2)
    for sign,mark in product((-1,1),(F(-1),F(0),F(1,3),F(1))):
        m=tuple(x+sign*eps+(y-sign*eps)*mark for x,y in zip(a,b))
        value=independent_expansion(m,lower,degree)
        assert value in bound and value<=cap
    fixed=MomentEngine(a,b,s).determinant(degree)
    assert all(x in y for x,y in zip(fixed,p))


@pytest.mark.parametrize('degree',(2,4))
@pytest.mark.parametrize('depth',(0,1,3))
def test_singleton_recovers_exact_fixed_law(degree,depth):
    s=((1,F(1,2),0),(F(1,2),1,0),(0,0,1));a=(-1,-1,0);b=(1,-1,0)
    fixed=MomentEngine(a,b,s).determinant(degree)
    family=FamilyMomentEngine(a,b,s).determinant(degree)
    assert tuple((x.lo,x.hi) for x in family)==tuple((x,x) for x in fixed)
    assert family_moment_cap(family,I(-1,1),depth=depth)==moment_cap(fixed,I(-1,1),depth=depth)


def test_diagonal_dominance_proves_all_members_psd():
    s=tuple(tuple(I(1,2) if i==j else I(F(-1,3),F(1,3)) for j in range(3)) for i in range(3))
    _,f=covariance_family(s)
    assert f['all_members_psd'] and f['nonempty_established']
    assert f['diagonal_dominance_margins']==(F(1,3),)*3
    for a,b,c in product((F(-1,3),F(1,3)),repeat=3):
        covariance_psd(((1,a,b),(a,1,c),(b,c,1)))


def test_psd_midpoint_does_not_certify_whole_box():
    s=tuple(tuple(I(1) if i==j else I(-1,1) for j in range(3)) for i in range(3))
    _,f=covariance_family(s)
    assert f['quantifier']=='PSD_MEMBERS_ONLY' and not f['all_members_psd']
    assert f['nonempty_established']
    with pytest.raises(ValueError):covariance_psd(((1,-1,-1),(-1,1,-1),(-1,-1,1)))


def test_inconclusive_nonemptiness_is_not_empty_or_all_psd():
    _,f=covariance_family(((I(-1,1),1,0),(1,I(-1,1),0),(0,0,1)))
    assert not f['nonempty_established'] and f['psd_witness'] is None
    assert not f['all_members_psd']
    covariance_psd(((1,1,0),(1,1,0),(0,0,1)))  # feasible endpoint missed by midpoint


def test_midpoint_only_and_dropped_covariance_controls():
    s=((1,I(F(1,2),F(3,4)),0),(I(F(1,2),F(3,4)),1,0),(0,0,0))
    p=FamilyMomentEngine((0,0,0),(0,0,0),s).determinant(2)
    cap=family_moment_cap(p,I(-1,1))
    endpoint=1+2*F(3,4)**2
    midpoint_only=1+2*F(5,8)**2
    dropped_covariance=F(1)
    assert cap>=endpoint>midpoint_only>dropped_covariance


def test_uncertain_mean_endpoint_defeats_midpoint_only():
    p=FamilyMomentEngine((I(1,2),1,0),(0,0,0),((0,0,0),)*3).determinant(4)
    assert family_moment_cap(p,I(0,1))>=16>F(3,2)**4


def test_both_subdivision_children_and_interior_required():
    p=(I(0),I(1,2))
    complete=family_bernstein(p,I(0,1),depth=3)
    left_only=family_bernstein(p,I(0,F(1,2)),depth=2)
    assert 2 in complete and 2 not in left_only
    p=FamilyMomentEngine((-1,-1,0),(1,-1,0),((0,0,0),)*3).determinant(4)
    assert family_bernstein(p,I(-1,1),depth=2).hi==1


@pytest.mark.parametrize('bad',[
    ((-1,0,0),(0,1,0),(0,0,1)),
    ((1,2,0),(2,1,0),(0,0,1)),
    ((1,I(0,1),0),(I(0,2),1,0),(0,0,1)),
    ((1,0),(0,1)),
])
def test_invalid_covariance_families_refused(bad):
    with pytest.raises(ValueError):covariance_family(bad)


@pytest.mark.parametrize('bad',(True,1.0,'1/3'))
def test_scalar_inputs_do_not_silently_change_exactness(bad):
    with pytest.raises(TypeError):FamilyMomentEngine((bad,0,0),(0,0,0),((0,0,0),)*3)


@pytest.mark.parametrize('bad',(-1,13,1.0,True))
def test_invalid_subdivision_refused(bad):
    with pytest.raises(ValueError):family_bernstein((I(1),I(0,1)),I(0,1),depth=bad)


def make_law():
    c=Context(RN5_ID,RN5_SHA,2,('xx','yy','xy'),'SEPARATE_MARGINAL_WHITENING','synthetic-law-A',(F(-1),F(1)))
    law=FamilyLaw(c,(I(-2,-1),1,0),(0,0,0),((1,0,0),(0,1,0),(0,0,1)),c.conditioned_law,c.conditioned_law)
    return c,law


def test_common_law_binding_and_source_bytes():
    c,law=make_law()
    result=apply_family_bound(law,c,degree=4,source_bytes=SOURCE.read_bytes())
    assert result['authority']=='NONE' and result['independence_credit']==0
    assert not result['field_certified'] and not result['spatial_cover_certified'] and not result['original_prize_closed']
    for bad in (replace(law,mean_law='other'),replace(law,covariance_law='other')):
        with pytest.raises(ValueError,match='same conditioned law'):apply_family_bound(bad,c,degree=2,source_bytes=SOURCE.read_bytes())
    with pytest.raises(ValueError,match='source bytes'):apply_family_bound(law,c,degree=2,source_bytes=b'wrong')
    with pytest.raises(ValueError):apply_family_bound(law,replace(c,domain=(F(0),F(1))),degree=2,source_bytes=SOURCE.read_bytes())
    assert law.fingerprint()!=replace(law,intercept=(I(-3,-1),1,0)).fingerprint()
    assert law.fingerprint()==replace(law,slope=(I(0),I(0),I(0))).fingerprint()


def test_variance_only_family_closed_form_controls():
    s=((1,0,0),(0,1,0),(0,0,I(0,1)))
    engine=FamilyMomentEngine((0,0,0),(0,0,0),s)
    for degree,endpoint,midpoint in ((2,F(4),F(7,4)),(4,F(132),F(321,16))):
        assert family_moment_cap(engine.determinant(degree),I(-1,1))==endpoint
        assert endpoint>midpoint


def test_singleton_psd_without_diagonal_dominance():
    s=((1,2,3),(2,4,6),(3,6,9))  # outer product (1,2,3)
    _,f=covariance_family(s)
    assert min(f['diagonal_dominance_margins'])<0 and f['all_members_psd']
    for degree in (2,4):
        p=MomentEngine((1,2,3),(1,0,-1),s).determinant(degree)
        q=FamilyMomentEngine((1,2,3),(1,0,-1),s).determinant(degree)
        assert tuple(x.lo for x in q)==p and all(x.lo==x.hi for x in q)
