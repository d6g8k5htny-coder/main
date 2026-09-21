"""Optional NONCERTIFYING high-precision cross-check; never a second certificate.

mpmath point arithmetic is used only to challenge implementation results.
Agreement is not rigorous containment evidence. Missing mpmath is explicit.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from h3_scalar_arithmetic import I, pi, exp_negative_point, cdf_point, pdf
from h3_scalar_certificate import prove


def execute():
    try:import mpmath as mp
    except ImportError:return {'status':'NOT_EXECUTED','reason':'mpmath unavailable','certifying':False}
    mp.mp.dps=150
    def m(x):
        x=F(x);return mp.mpf(x.numerator)/x.denominator
    checks=[]
    def record(label,interval,value):
        checks.append({'case':label,'diagnostic_inside':bool(m(interval.lo)<=value<=m(interval.hi))})
    record('pi',pi(),mp.pi)
    for x in [F(0),F(1,10),F(1,2),F(1),F(2),F(5,2),F(3),F(4),F(6),F(8)]:
        record('cdf:'+str(x),cdf_point(x),(1+mp.erf(m(x)/mp.sqrt(2)))/2)
        record('pdf:'+str(x),pdf(I(x)),mp.exp(-m(x)**2/2)/mp.sqrt(2*mp.pi))
    for x in (F(0),F(1,3),F(1),F(12),F(25),F(50)):
        record('exp:'+str(x),exp_negative_point(x),mp.exp(-m(x)))
    return {'status':'DIAGNOSTIC_PASS' if all(x['diagnostic_inside'] for x in checks) else 'DIAGNOSTIC_DISAGREEMENT',
            'mpmath_version':mp.__version__,'decimal_digits':150,'cases':checks,
            'certifying':False,'independent_scientific_review':False,
            'arb':'NOT_EXECUTED','scope':'Elementary implementation comparison only; not full-law or dual-certified backend validation.'}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True,indent=2))
