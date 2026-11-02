#!/usr/bin/env python3
"""Exact-rational author support for the fixed side-24, J=116 tail bound.

No project code, floating point, numerical engine, network or file I/O.
Analytic inequalities are proved separately; finite controls are not that proof.
"""
from fractions import Fraction as Q
from collections import Counter
from itertools import product
import json
import sys

COUNTS = Counter()

def check(ok, group):
    COUNTS[group] += 1
    if not ok:
        raise RuntimeError(group)

def floor(q):
    return q.numerator // q.denominator

def atan_sum(x, terms):
    return sum(((-1)**k*x**(2*k+1)/Q(2*k+1) for k in range(terms)), Q(0))

def tangent_double(x):
    return 2*x/(1-x*x)

def cert_floor(q, k):
    return Q(k) <= q < Q(k+1)

def main():
    # Alternating partial sums + the exact Machin tangent identity.
    a5lo, a5hi = atan_sum(Q(1,5),8), atan_sum(Q(1,5),9)
    a239lo, a239hi = atan_sum(Q(1,239),2), atan_sum(Q(1,239),3)
    pi_lo, pi_hi = 16*a5lo-4*a239hi, 16*a5hi-4*a239lo
    check(Q(314159,100000) < pi_lo < pi_hi < Q(3927,1250), 'pi_bracket')
    t4 = tangent_double(tangent_double(Q(1,5)))
    check(t4 == Q(120,119), 'machin_identity')
    check((t4-Q(1,239))/(1+t4/Q(239)) == 1, 'machin_identity')
    check(4*Q(5,26)-Q(1,239)>0 and 4*Q(1,5)<1, 'machin_quadrant')
    check(16*atan_sum(Q(1,5),10)-4*atan_sum(Q(1,239),5)>pi_lo,
          'pi_nested_control')
    check(16*atan_sum(Q(1,5),11)-4*atan_sum(Q(1,239),4)<pi_hi,
          'pi_nested_control')
    # e > 1+1+1/2+1/6 = 8/3: only positive omitted terms remain.
    check(1+1+Q(1,2)+Q(1,6) == Q(8,3), 'exponential_base')
    check(1+1+Q(1,2)+Q(1,6)+Q(1,24) > Q(8,3), 'exponential_base')

    h_lo, h_hi = Q(314159,1200000), Q(1309,5000)
    J, j0, base, rho = 116, 117, Q(3,8), Q(1,2048)
    check(114*h_hi < 30 < 115*h_lo, 'cutoff_identity')
    first_power = floor((j0*h_lo)**2/2)
    ratio_power = floor((2*j0+1)*h_lo*h_lo/2)
    check(first_power == 469 and cert_floor((j0*h_lo)**2/2,469), 'first_exponent')
    check(ratio_power == 8 and cert_floor((2*j0+1)*h_lo*h_lo/2,8), 'ratio_exponent')
    check(base**8*Q(118,117)**6 < rho < 1, 'uniform_ratio')
    for j,n in product(range(j0,j0+128),range(7)):
        check((2*j+1)*h_lo*h_lo/2 >= 8, 'finite_ratio_controls')
        check(base**8*Q(j+1,j)**n <= base**8*Q(118,117)**6 < rho,
              'finite_ratio_controls')
    for j in range(1,J+1):
        q = (j*h_lo)**2/2
        check(cert_floor(q,floor(q)), 'retained_exponent_controls')

    # Z_J >= 1+2 sum_{j=1}^5 (1-(j*h_hi)^2/2), from exp(-x)>=1-x.
    z_lower = 1+2*sum((1-(j*h_hi)**2/2 for j in range(1,6)), Q(0))
    check(z_lower == Q(36151709,5000000) and z_lower>7, 'normalizer_lower')
    check(all(1-(j*h_hi)**2/2>0 for j in range(1,6)), 'normalizer_lower')
    orders=(0,2,4,6)
    tail={n:2*(j0*h_hi)**n*base**first_power/(1-rho) for n in orders}
    full={n:(Q(1) if n==0 else Q(0))
          +2*sum((base**floor((j*h_lo)**2/2)*(j*h_hi)**n
                  for j in range(1,J+1)), Q(0))+tail[n] for n in orders}
    # A union cover of the two omitted coordinate strips. Intersection is
    # deliberately counted twice for an upper bound, never subtracted twice.
    def outside(u,v):
        return (tail[u]*full[v]+full[u]*tail[v])/z_lower**2
    for outer,cut,u,v in product(range(2,6),range(1,2),orders,orders):
        w={j:Q(1,1+j*j) for j in range(-outer,outer+1)}
        moments=lambda n:sum((a*Q(abs(j))**n for j,a in w.items()),Q(0))
        tails=lambda n:sum((a*Q(abs(j))**n for j,a in w.items() if abs(j)>cut),Q(0))
        direct=sum((w[j]*w[k]*Q(abs(j))**u*Q(abs(k))**v
                    for j,k in product(w,repeat=2) if abs(j)>cut or abs(k)>cut),Q(0))
        union=tails(u)*moments(v)+moments(u)*tails(v)
        check(direct==union-tails(u)*tails(v) and direct<=union, 'two_axis_union')
        check(direct>tails(u)*moments(v), 'negative_single_axis')

    # Trapezoid remainder functional identity on exact polynomial controls:
    # f(r)-f(0)-r*(f'(r)+f'(0))/2 = -1/2 int_0^r x(r-x)f'''(x) dx.
    for r,m in product((Q(1,20),Q(1,7),Q(2)),range(13)):
        f_r, f_0 = r**m, Q(m==0)
        df_r = m*r**(m-1) if m else Q(0)
        df_0 = Q(m==1)
        left=(f_r-f_0-r*(df_r+df_0)/2)/r**3
        integral=(m*(m-1)*(m-2)*r**m/Q(m*(m-1)) if m>=3 else Q(0))
        check(left == -integral/(2*r**3), 'trapezoid_identity')
    check(Q(1,2)*(Q(1,2)-Q(1,3))==Q(1,12), 'trapezoid_mass')
    check(Q(1,2)>6*Q(1,24), 'negative_halved_trapezoid_bound')

    # Sum squared feature ceilings for six literal pins + six literal softs.
    # V5 (last pin): |k_x|^3/12, hence a squared coefficient 1/144.
    r=Q(1,20)
    components=[{(0,0):Q(1)},{(2,0):Q(1)},{(0,2):Q(1)},
                {(4,0):Q(1)},{(2,2):Q(1)},{(6,0):Q(1,144)},
                {(0,4):Q(1)},{(0,4):Q(1)},
                {(4,0):1/r**2},{(4,0):1/r**2},
                {(2,2):1/r**2},{(2,2):1/r**2}]
    coeff={}
    for component in components:
        for powers,c in component.items(): coeff[powers]=coeff.get(powers,Q(0))+c
    expected={(0,0):Q(1),(2,0):Q(1),(0,2):Q(1),(4,0):Q(801),
              (2,2):Q(801),(6,0):Q(1,144),(0,4):Q(2)}
    check(coeff==expected and len(components)==12, 'frame_accounting')
    q0=outside(0,0)+outside(2,0)+outside(0,2)+outside(4,0)+outside(2,2)\
       +2*outside(0,4)+outside(6,0)/144
    q2=2*outside(4,0)+2*outside(2,2)
    tau=q0+q2/r**2
    check(tau==sum((c*outside(*powers) for powers,c in coeff.items()),Q(0)),
          'independent_frame_sum')
    eta=tail[0]/z_lower
    alpha_lower=1/(1+eta)**2
    check(eta<Q(5,10**201), 'retained_mass_ceiling')
    check(1-alpha_lower<=2*eta<Q(1,10**200), 'retained_fraction_loss')
    check(q0<Q(8,10**194), 'general_r_q0_ceiling')
    check(q2<Q(3,2*10**194), 'general_r_q2_ceiling')
    check(tau<Q(7,10**192), 'fixed_r_trace_ceiling')
    check(not cert_floor((j0*h_lo)**2/2,470), 'negative_round_exponent_up')
    check((j0*h_hi)**6>(j0*h_hi)**4, 'negative_drop_derivative_weight')

    def record(q):
        return {'numerator':str(q.numerator),'denominator':str(q.denominator)}
    result={
        'status':'AUTHOR_BOUND_SUPPORT_ONLY','scientific_effect':'NONE',
        'python':sys.version.split()[0],'checks':sum(COUNTS.values()),
        'counts':dict(sorted(COUNTS.items())),
        'source_head':'3279db8bd8cf93905376eb2ff782e8afc3fdb544',
        'frame':'PINS_EXACT+SOFT_EXACT; exact coordinates, no rotated/rounded-basis claim',
        'r':'1/20','J':J,'first_omitted':j0,
        'h_lower':str(h_lo),'h_upper':str(h_hi),'normalizer_lower':str(z_lower),
        'first_exp_floor':first_power,'ratio_exp_floor':ratio_power,'ratio_upper':str(rho),
        'mass_relative_upper':record(eta),'retained_fraction_lower':record(alpha_lower),
        'trace_upper':record(tau),'q0':record(q0),'q2':record(q2),
        'simple_bounds':{'eta':'<5e-201','one_minus_alpha':'<1e-200',
                         'q0':'<8e-194','q2':'<1.5e-194','trace_T_at_r_1_20':'<7e-192'},
        'excluded':'finite-law g/s/beta; consumer matching; mean or expectation conclusion; engine replay',
    }
    print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':
    main()
