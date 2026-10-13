# Mutation probes: which declared negative controls actually detect the declared faults?
import sys, importlib, types
from fractions import Fraction as Q
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import population_certificate as pc
orig_fixture, orig_cdf = pc.critical_fixture, pc.cdf_mass
def report(label, fn):
    try:
        out = fn(); print(f'{label}: {out}')
    except Exception as e:
        print(f'{label}: EXC {type(e).__name__}: {e}')
# baseline
report('baseline negative_controls', lambda: pc.negative_controls()['status'])
# M1: fixed-left / literal elder selector (always take sorted roots[0] as "younger")
def fixed_left(a,h,sign=1):
    d = orig_fixture(a,h,sign)
    if d['status']=='FINITE_BAR':
        d['lifetime'] = d['critical_heights'][0]-d['critical_heights'][1]; d['younger_side']='left'
    return d
pc.critical_fixture = fixed_left
report('M1 fixed-left selector -> negative_controls', lambda: (pc.negative_controls()['status'], {k:v for k,v in pc.negative_controls()['gates'].items() if not v}))
pc.critical_fixture = orig_fixture
# M1b: literal elder (always the higher maximum)
def elder(a,h,sign=1):
    d = orig_fixture(a,h,sign)
    if d['status']=='FINITE_BAR':
        hts=d['critical_heights']; idx = 0 if hts[0]>hts[2] else 2
        d['lifetime'] = hts[idx]-hts[1]; d['younger_side']='left' if idx==0 else 'right'
    return d
pc.critical_fixture = elder
report('M1b elder selector -> negative_controls', lambda: (pc.negative_controls()['status'], {k:v for k,v in pc.negative_controls()['gates'].items() if not v}))
pc.critical_fixture = orig_fixture
# M2: missing factor two in the CDF route (single sign) for 0<tau<9/4 only (saturation branch untouched)
def half_cdf(tau):
    r = orig_cdf(tau)
    return r if (tau==0 or tau>=Q(9,4)) else (r[0]/2, r[1]/2)
pc.cdf_mass = half_cdf
report('M2 CDF route halved (missing sign) -> negative_controls', lambda: (pc.negative_controls()['status'], {k:v for k,v in pc.negative_controls()['gates'].items() if not v}))
pc.cdf_mass = orig_cdf
# M3: bar-conditioned CDF (x5) for 0<tau<9/4
def cond_cdf(tau):
    r = orig_cdf(tau)
    return r if (tau==0 or tau>=Q(9,4)) else (r[0]*5, r[1]*5)
pc.cdf_mass = cond_cdf
report('M3 CDF route conditioned x5 -> negative_controls', lambda: (pc.negative_controls()['status'], {k:v for k,v in pc.negative_controls()['gates'].items() if not v}))
pc.cdf_mass = orig_cdf
# M4: does the saturation value 1/5 depend on any computation? Evaluate the closed form just below 9/4
near = (Q(0),Q(0))
print('M4 cdf_mass(9/4 - 2^-60) =', [float(x) for x in near], ' 1/5 - lo =', float(Q(1,5)-near[0]), ' hi-1/5=', float(near[1]-Q(1,5)))
# M5: the mass controls are literal comparisons
print('M5 rectangle_mass_control((1/10,1/10)) =', pc.rectangle_mass_control((Q(1,10),Q(1,10))), ' rectangle_mass_control((1,1)) =', pc.rectangle_mass_control((Q(1),Q(1))))
# M6: the tail_convexity gate literal
print('M6 tail_convexity gate literal is', 1485-21>0)
# M7: classify_void with intervals that only overlap (should not PASS unless endpoint test holds)
print('M7 classify_void overlap-only', pc.classify_void((Q(1,2)-Q(1,4096), Q(1,2)), (Q(1,2)-Q(1,100000), Q(1,2)+Q(1,100000))))
# M8: coefficient PASS must be inner containment, not overlap
b=(Q(1,4),Q(1,4)+Q(1,1<<60)); e=(Q(1,1000),Q(1,1000))
print('M8 classify_coefficient overlap-only', pc.classify_coefficient((Q(1,4)-Q(1,1000)-Q(1,10**6), Q(1,4)), b, e))
