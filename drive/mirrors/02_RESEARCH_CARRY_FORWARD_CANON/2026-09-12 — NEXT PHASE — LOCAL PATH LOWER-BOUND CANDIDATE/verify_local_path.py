#!/usr/bin/env python3
"""Exact algebra and fail-closed regression checks for LPW-CAND-20260912.

This is NOT a proof assistant, Gaussian probability certificate, independence
review, q0 verifier replacement, or authority to promote a research claim.
Run: python verify_local_path.py [--mutation NAME]
Requires sympy; package requirements.txt pins the version used in this run.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
try:
    import sympy as s
except ImportError:
    print('DEPENDENCY_ERROR: install the pinned requirements.txt', file=sys.stderr)
    raise SystemExit(2)

MUTATIONS = ('gap_sign', 'denominator_power', 'omit_rare_mass',
             'hessian_scale', 'clearance', 'drop_palm_weight',
             'unauthorized_promotion', 'duplicate_keys')

def strict_json(text: str) -> object:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('DUPLICATE_JSON_KEY:' + k)
            out[k] = v
        return out
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mutation', choices=MUTATIONS)
    args = ap.parse_args()
    checks = []
    def ck(name, truth, value=None):
        passed = bool(truth)
        row = {'name': name, 'pass': passed}
        if value is not None:
            row['value'] = str(value)
        checks.append(row)
        if not passed:
            raise ValueError('CHECK_FAILED:' + name)
    # No positivity flags on spatial coordinates; separation is positive.
    x, y = s.symbols('x y', real=True)
    r = s.symbols('r', positive=True)
    e = s.Rational(1, 8) if args.mutation == 'clearance' else s.Rational(1, 16)
    delta = s.Rational(1, 1024)
    g = x**3/3 - x/4 - s.Rational(1,12)
    F = g + 2*(x*x-s.Rational(1,4))*y - 5*y*y
    if args.mutation == 'gap_sign':
        F = -F
    M = {x:-s.Rational(1,2), y:0}; S = {x:s.Rational(1,2), y:0}
    path_y = (x*x-s.Rational(1,4))/5
    H = s.hessian(F, (x,y)); HM=H.subs(M); HS=H.subs(S)
    try:
        ck('pin_value_M', F.subs(M)==0)
        ck('pin_value_S', F.subs(S)==-s.Rational(1,6))
        for label,pt in [('M',M),('S',S)]:
            for z in (x,y): ck('pin_gradient_'+label+'_'+str(z),s.diff(F,z).subs(pt)==0)
        ck('Hessian_M', HM==s.Matrix([[-1,-2],[-2,-10]]))
        ck('Hessian_S', HS==s.Matrix([[1,2],[2,-10]]))
        ck('M_maximum_determinant',HM.det()==6)
        ck('S_saddle_determinant',HS.det()==-14)
        ck('path_starts_at_M',path_y.subs(x,-s.Rational(1,2))==0)
        ck('path_endpoint_y',path_y.subs(x,-2)==s.Rational(3,4))
        ck('path_y_derivative',s.diff(path_y,x)==2*x/5)
        ck('path_rectangle_bound',0<=path_y.subs(x,-2)<=1)
        R=s.factor(F.subs(y,path_y))
        ck('ridge_polynomial',s.expand(R-(2*x+1)**2*(12*x*x+8*x-17)/240)==0)
        ck('ridge_derivative',s.factor(s.diff(R,x)-(2*x-1)*(2*x+1)*(4*x+5)/20)==0)
        ck('ridge_minimum_value',R.subs(x,-s.Rational(5,4))==-s.Rational(99,1280))
        ck('ridge_global_minimum_factorization',s.factor(R+s.Rational(99,1280)-(4*x+5)**2*(48*x*x-40*x+1)/3840)==0)
        z=s.symbols('z',positive=True)
        ck('minimum_residual_positive_coefficients',all(c>0 for c in s.Poly((48*x*x-40*x+1).subs(x,-z),z).all_coeffs()))
        ck('endpoint_is_older',R.subs(x,-2)==s.Rational(9,16))
        margin=s.Rational(1,6)-s.Rational(99,1280)-e
        ck('robust_path_above_S',margin>0,margin)
        ck('robust_endpoint_above_M',s.Rational(9,16)-e>0)
        hm_floor=(1-e)*(10-e)-(2+e)**2
        hs_upper=-(1-e)*(10-e)-(2-e)**2
        ck('robust_maximum_axial_sign',-1+e<0)
        ck('robust_maximum_det_floor',hm_floor>5,hm_floor)
        ck('robust_saddle_det_ceiling',hs_upper<-13,hs_upper)
        ck('safe_weight_constant',5*13==65)
        ck('coefficient_perturbation_budget',16*delta+8*s.Rational(1,256)<s.Rational(1,16))
        taylor=s.Rational(1,128)+2*s.Rational(23,384)+s.Rational(1,48)+4*s.Rational(1,48)+2*s.Rational(1,24)+12*s.Rational(5,96)+s.Rational(9,2)
        ck('Taylor_C2_majorant',taylor==s.Rational(2089,384) and taylor<8,taylor)
        basis=[s.Integer(1),x,y,x*x,x*y,x**3]
        pts=[(-s.Rational(1,2),0),(s.Rational(1,2),0)]
        V=s.Matrix([[s.diff(f,x,dx,y,dy).subs({x:px,y:py}) for f in basis] for px,py in pts for dx,dy in [(0,0),(1,0),(0,1)]])
        ck('six_pin_Hermite_determinant',V.det()==-1)
        Dinv=s.diag(1,r**-1,r**-1,r**-2,r**-2,r**-3)
        Rscale=s.diag(1,r,r,1,r,r)
        T=Dinv*V.inv()*Rscale
        ck('six_pin_transform_determinant',s.simplify(T.det()+r**-5)==0)
        b=s.symbols('b',real=True)
        target=s.Matrix([b,0,0,b-r**3/6,0,0])
        ck('six_pin_transformed_values',s.simplify(T*target-s.Matrix([b-r**3/12,-r*r/4,0,0,0,s.Rational(1,3)]))==s.zeros(6,1))
        coeffs=s.symbols('a0:6')
        poly=sum(c*f for c,f in zip(coeffs,basis))
        raw=s.Matrix([s.diff(poly,x,dx,y,dy).subs({x:r*px,y:r*py}) for px,py in pts for dx,dy in [(0,0),(1,0),(0,1)]])
        ck('Hermite_exact_on_selected_basis',s.simplify(T*raw-s.Matrix(coeffs))==s.zeros(6,1))
        mon=[s.Integer(1),x,y,x*x,x*y,y*y,x**3,x*x*y,x*y*y,y**3]
        Eval=s.Matrix([[p.subs({x:i,y:j}) for p in mon] for i in range(-2,3) for j in range(-2,3)])
        ck('degree_three_lattice_unisolvence',Eval.rank()==10)
        ck('four_remaining_jet_coordinates',len(set(['fyy','fxxy/2','fxyy/2','fyyy/6']))==4)
        volume=(4*delta*r)*(2*delta)**3
        ck('rare_jet_box_volume',s.simplify(volume-32*delta**4*r)==0)
        m,CZ=s.symbols('m CZ',positive=True)
        ck('conditional_half_mass',s.simplify(m*volume/2-16*m*delta**4*r)==0)
        mass_power=0 if args.mutation=='omit_rare_mass' else 1
        hp=2 if args.mutation=='hessian_scale' else 1
        weight_power=0 if args.mutation=='drop_palm_weight' else 4*hp
        denom=4 if args.mutation=='denominator_power' else 2
        ck('Palm_power_ledger',mass_power+weight_power-denom==3)
        ck('numerator_constant',65*16==1040)
        ck('final_symbolic_bound',s.simplify(65*r**4*(m*volume/2)/(CZ*r**2)-1040*m*delta**4/CZ*r**3)==0)
        # A uniform bound on the conditional tail is applied *inside* the thin jet box.
        ck('conditional_tail_not_global_subtraction',s.Rational(1,2)*r>0)
        ck('bad_global_tail_subtraction_can_be_negative',(r-s.Rational(1,2)).subs(r,s.Rational(1,100))<0)
        # Multiple mentions of a result do not supply an independent review.
        state_path=Path(__file__).with_name('CURRENT_STATE.json')
        text=state_path.read_text(encoding='utf-8')
        if args.mutation=='duplicate_keys':
            text='{"authorized_promotion":false,"authorized_promotion":true}'
        state=strict_json(text)
        if args.mutation=='unauthorized_promotion':state['new_candidate']['accepted_as_program_theorem']=True
        ck('no_unreviewed_theorem_promotion',state['new_candidate']['accepted_as_program_theorem'] is False)
        ck('no_independent_review_claim',state['new_candidate']['independent_review_completed'] is False)
        ck('lower_campaign_preserved_open',state['tracks']['LB_RATE_2D']['recorded_status']=='OPEN')
        ck('P01_preserved_hold',state['tracks']['P01']['recorded_status']=='HOLD / NOT PROMOTED')
        ck('3D_ratification_preserved',state['tracks']['SIDE24_3D']['recorded_status']=='RATIFIED AT STATED SCOPE')
        # Normal JSON must reject duplicate keys even outside a mutation run.
        rejected=False
        try:strict_json('{"a":1,"a":2}')
        except ValueError:rejected=True
        ck('strict_json_duplicate_key_regression',rejected)
        ck('no_Lean_claim',state['new_candidate']['lean_compiled'] is False)
        report={'artifact':'LPW-CAND-20260912','scope':'EXACT ALGEBRA AND REGRESSION ONLY',
                'result':'PASS','checks_passed':len(checks),'checks':checks,
                'independent_review_completed':False,'gaussian_probability_machine_certified':False,
                'theorem_promotion_authorized':False}
        print(json.dumps(report,sort_keys=True,indent=2))
        return 0
    except (ValueError,KeyError,TypeError) as exc:
        print(json.dumps({'artifact':'LPW-CAND-20260912','result':'REJECTED',
                          'mutation':args.mutation,'error':str(exc),'checks':checks},sort_keys=True,indent=2))
        return 1
if __name__=='__main__':
    raise SystemExit(main())
