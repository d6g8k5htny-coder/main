#!/usr/bin/env python3
"""
q0_c091_gate_reduction.py

C091 proof-gate reduction. The script records exact algebraic thresholds and
corrects the lower near-diagonal gate to a marked two-scale saddle-pair bound.
It does not manufacture missing interval constants.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

def curvature_limit(a,b,ua,ub,ceiling):
    margin=ceiling-max(ua,ub)
    return None if margin<0 else 8*margin/(b-a)**2

def b2_budget(c_lambda=.8501,rmax=.05,a=2.24,target=.84,residual=2*math.exp(-111)):
    return (c_lambda*(1-a*rmax**3-residual)-target)/rmax**3

def integral_d4_loge(d):
    # integral_0^d x^4 log(e/x) dx
    return d**5*(6/25-math.log(d)/5)

def primitive_d1_loge(d):
    # primitive of x log(e/x)
    return d*d*(3/4-math.log(d)/2)

def two_scale_row(r,delta=.1,area=576,cdec=4.051,m1=6.6):
    ell=r**3/6
    dstar=ell**(1/3)
    I4=integral_d4_loge(dstar)
    I1=primitive_d1_loge(delta)-primitive_d1_loge(dstar)
    coeff=math.pi*area/36*(I4/ell+I1)
    budget=b2_budget(); far=cdec*m1*m1/72
    return {'r':r,'ell':ell,'dstar':dstar,'I4':I4,'I1':I1,'coefficient':coeff,'B2_far_Cdec_4p051':far,'remaining_B2_budget':budget-far,'allowed_Crep_times_Cmark_after_Cdec_4p051':(budget-far)/coeff}

def main():
    nominal={.0125:.9411,.025:.9605,.05:.985}
    banded={r:u+.02 for r,u in nominal.items()}
    intervals={}
    for a,b in [(.0125,.025),(.025,.05)]:
        intervals[f'[{a},{b}]']={
          'secant_slope':(nominal[b]-nominal[a])/(b-a),
          'M_minus_nominal_0p99':curvature_limit(a,b,nominal[a],nominal[b],.99),
          'M_minus_banded_1p01':curvature_limit(a,b,banded[a],banded[b],1.01)}
    x=np.array(sorted(nominal)); y=np.array([nominal[v] for v in x]); a2,a1,a0=np.polyfit(x,y,2)
    budget=b2_budget(); m1=6.6
    far=[]
    for c in [2,4.051,8.2,16,32,64,100,130]:
        value=c*m1*m1/72
        far.append({'C_dec':c,'B2_far':value,'remaining_budget':budget-value})
    report={
      'cycle':'C091',
      'upper_shape':{
        'nominal_rungs':nominal,'banded_rungs':banded,
        'one_sided_theorem':"If U'' >= -M, sup U <= max endpoints + M(b-a)^2/8.",
        'intervals':intervals,
        'monotonicity_shortcut':"U' >= 0 on (0,.05] closes origin and between-rung gates; maxima are .985 nominal and 1.005 banded.",
        'quadratic_diagnostic':{'a2':float(a2),'a1':float(a1),'a0':float(a0),'U_second':float(2*a2),'grade':'diagnostic only'}},
      'uncertainty':{
        'semantic':'CLOSED: an upper theorem consumes +/-0.02 as +0.02 unless inputs are proved already one-sided.',
        'calibration':'OPEN: confidence level, multiplicity, sample size, and source of 0.02 not recovered.',
        'safe_upper_target':1.01},
      'bonferroni':{
        'B2_budget_for_0p84_through_0p05':budget,
        'far_scenarios':far,
        'prior_pointwise_Cnd_d_status':'NOT ACCEPTED AS A COMPLETE MARKED-WINDOW PROOF',
        'replacement':'same-type spatial repulsion plus d^3 critical-value-gap mark law, split at d=ell^(1/3)',
        'two_scale_assumptions':{
          'spatial':'K_ss,r(y,z) <= C_rep d^3 log(e/d)',
          'mark':'under saddle-pair Palm, density of A=(u1+u2)/2 and Z=(u2-u1)/d^3 is <= C_mark, with the same order bound for A marginal',
          'window_probability':'<= C_mark ell min(1,ell/d^3)'},
        'two_scale_rows':[two_scale_row(r) for r in [.05,.025,.0125]],
        'new_numeric_gate':'Certify C_rep*C_mark below about 80 on the full-torus worst-case area, together with the far bound.'},
      'gate_status':{
        'G_TORUS_LOCAL':'closed by separate matrix-transfer certificate',
        'G_U_UNCERTAINTY_SEM':'closed',
        'G_U_SHAPE':'open',
        'G_BONF_MARKED_REPULSION':'open',
        'H4_MARKOV_PRODUCT':'killed',
        'H4_PALM_CHERNOFF_THEOREM':'closed structurally; numerical path certificate open'}}
    Path('/mnt/data/q0_c091_gate_reduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
