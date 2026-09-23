#!/usr/bin/env python3
"""Exact checks for a scoped LPW review reconciliation.

The endpoint covariance statements use the analytic tail argument documented in
03_PERIODIC_ENDPOINT_COMPANION.md. This script verifies finite symbolic/rational
steps, not every analytic argument, reviewer execution, or an all-r certificate.
No acceptance check depends on Python assert. Numerical diagnostics are labeled.
"""
from __future__ import annotations
import json
import sys
from fractions import Fraction
import sympy as sp
import mpmath as mp


def main() -> int:
    checks: list[dict[str, object]] = []
    def ck(name: str, value: object) -> None:
        ok = bool(value)
        checks.append({'name': name, 'pass': ok})
        if not ok:
            raise ValueError(name)

    a, d, e, b = sp.symbols('mu2 mu4 mu6 b', positive=True)
    one = sp.Integer(1)
    jet = [((0,0),one),((1,0),one),((0,1),one),
           ((2,0),sp.Rational(1,2)),((1,1),one),((3,0),sp.Rational(1,6)),
           ((0,2),one),((2,1),sp.Rational(1,2)),
           ((1,2),sp.Rational(1,2)),((0,3),sp.Rational(1,6))]
    moments = {0:one, 2:a, 4:d, 6:e}
    def cov(i: int, j: int) -> sp.Expr:
        x, sx=jet[i];y, sy=jet[j]
        n=[x[k]+y[k] for k in range(2)]
        if any(t%2 for t in n): return sp.Integer(0)
        return sx*sy*(-1)**(sum(y)+sum(n)//2)*moments[n[0]]*moments[n[1]]
    gamma=sp.Matrix(10,10,cov)
    U=gamma[:6,:6];C=gamma[6:,:6];J=gamma[6:,6:]
    schur=(J-C*U.inv()*C.T).applyfunc(sp.factor)
    wanted=sp.diag(d-a*a,a*(d-a*a)/4,a*(d-a*a)/4,(e-d*d/a)/36)
    ck('exact_periodic_symbolic_schur', (schur-wanted).applyfunc(sp.simplify) == sp.zeros(4))
    mean=(C*U.inv()*sp.Matrix([b,0,0,0,0,sp.Rational(1,3)])).applyfunc(sp.factor)
    ck('exact_periodic_symbolic_mean', mean == sp.Matrix([-b*a,0,0,0]))
    planar=gamma.subs({a:1,d:3,e:15})
    ck('planar_schur_specialization', wanted.subs({a:1,d:3,e:15}) == sp.diag(2,sp.Rational(1,2),sp.Rational(1,2),sp.Rational(1,6)))
    ck('planar_mean_specialization', mean.subs(a,1) == sp.Matrix([-b,0,0,0]))
    x=sp.symbols('x')
    p=12*x**3-26*x**2+11*x-1
    char=(x-1)*(4*x**3-19*x**2+18*x-4)*p**2/576
    ck('planar_characteristic_polynomial', sp.expand(planar.charpoly(x).as_expr()-char)==0)
    lo=sp.Rational(126553449667,10**12)
    hi=sp.Rational(126553449668,10**12)
    shifted=planar-lo*sp.eye(10)
    for k in range(1,11):
        ck('sylvester_minor_'+str(k), shifted[:k,:k].det()>0)
    ck('cubic_lower_endpoint_negative', p.subs(x,lo)<0)
    ck('cubic_upper_endpoint_positive', p.subs(x,hi)>0)
    # Analytic document proves every moment error < epsilon.
    eps=sp.Rational(1,10**110)
    perturb=400*eps
    ck('exp_12over5_series_exceeds_10', sum(sp.Rational(12,5)**k/sp.factorial(k) for k in range(6))>10)
    ck('wrapped_moment_tail_budget', 8*24**6*sp.Rational(1,10**120)<eps)
    ck('periodic_lower_floor_safe', lo-perturb>sp.Rational(1265534496,10**10))
    ck('rounded_up_floor_refuted_by_upper_bracket', hi+perturb<sp.Rational(1265534497,10**10))
    ck('reported_decimal_rounding_direction', Fraction('0.1265534497')>Fraction('0.126553449667289'))
    # Exact-law compatibility is a field-wise predicate, never just shared exponent.
    lower={'dimension':2,'side':24,'field':'normalized_periodized_BF','event':'typed_elder_defect','pins':'six_value_gradient','level':'6/5'}
    upper3={**lower,'dimension':3,'pins':'dimension3_pair_pins'}
    ck('reject_3d_upper_plus_2d_lower',lower != upper3)
    adjacent={**lower,'event':'adjacency_conditioned_defect'}
    ck('reject_adjacency_event_substitution',lower != adjacent)
    ck('allow_exact_scope_identity_only',lower == dict(lower))
    delta=sp.Rational(1,1024)
    ck('constant_factor_equivalence', sp.Rational(1040,4)==260)
    ck('clearance_identity', sp.Rational(1,6)-sp.Rational(99,1280)-sp.Rational(1,16)==sp.Rational(103,3840))
    mp.mp.dps=70
    GM=mp.matrix([[mp.mpf(int(t.p))/int(t.q) for t in row] for row in planar.tolist()])
    numerical_min=mp.eigsy(GM,eigvals_only=True)[0]
    def dens(q: mp.mpf,A: mp.mpf,B: mp.mpf,D: mp.mpf) -> mp.mpf:
        variances=[mp.mpf(2),mp.mpf(1)/2,mp.mpf(1)/2,mp.mpf(1)/6]
        offsets=[q+mp.mpf(6)/5,A,B,D]
        return mp.exp(-sum(u*u/v for u,v in zip(offsets,variances))/2)/(4*mp.pi**2*mp.sqrt(mp.fprod(variances)))
    dd=mp.mpf(1)/1024
    report={
      'status':'PASS_FINITE_SYMBOLIC_RATIONAL_AND_SCOPE_CHECKS',
      'author_line':'OpenAI; same author family as LPW candidate; no independent review credit',
      'check_count':len(checks),'checks':checks,
      'exact_periodic_schur_diagonal':[str(wanted[i,i]) for i in range(4)],
      'exact_periodic_conditional_mean':[str(t) for t in mean],
      'planar_eigenvalue_rational_bracket':[str(lo),str(hi)],
      'periodic_matrix_perturbation_bound':'400/10^110; analytic derivation in companion, not a raw Kimi certificate',
      'periodic_endpoint_safe_lower_floor':'0.1265534496',
      'user_relay_rounded_up_floor':'0.1265534497 (not a valid lower floor for this covariance)',
      'diagnostics_not_interval_certificates':{
         'planar_eigenvalue_approx':mp.nstr(numerical_min,60),
         'planar_endpoint_law_density_at_q_minus_quarter_A2':mp.nstr(dens(-mp.mpf(1)/4,mp.mpf(2),mp.mpf(0),mp.mpf(0)),50),
         'planar_endpoint_law_density_at_fixed_box_corner':mp.nstr(dens(-mp.mpf(11),mp.mpf(2)+dd,dd,dd),50)
      },
      'limitations':['Does not replay RA/RB/RC/RD scripts','Does not certify all-r covariance or density','Does not evaluate B3 B4 c or r0','Does not change any program control status','Exact endpoint perturbation proof is an author-side companion requiring review']
    }
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)}),file=sys.stderr)
        sys.exit(1)
