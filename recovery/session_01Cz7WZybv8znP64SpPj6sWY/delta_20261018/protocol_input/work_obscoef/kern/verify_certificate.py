"""Standalone proof-witness checker; never imports moment_kernel.

Trusted base: this checker, Python integer/Fraction semantics, alternating-series
and elementary Machin/tail lemmas documented in VERIFY_SPEC_V1.md.
It validates mathematical bound containment, not producer runtime attestations.
"""
from fractions import Fraction as F
import argparse
import json
import os
import re
import time

PINS = {"original": "f38c085ab5b15368e6989a1fd6298490f3b671f5a6c3a01bcdc81b516022bc33",
        "method_v2": "8bdea4b42e9efd2d846cd1aa73691e8194fa26cd937ac02b79991334f88204f6"}
CAPS = {"max_work": 100000, "max_seconds": 3, "max_rational_bits": 8192, "max_squarings": 16}

class Check:
    def __init__(self, bits):
        self.bits = bits
        self.work = 0
        self.start = time.monotonic()
    def tick(self):
        self.work += 1
        if self.work > 100000 or time.monotonic()-self.start > 3:
            raise ValueError("CHECKER_WORK_OR_TIME_LIMIT")
    def q(self, s):
        self.tick()
        if not isinstance(s, str) or len(s)>3000 or not re.fullmatch(r"-?\d+/[1-9]\d*", s):
            raise ValueError("NONCANONICAL_RATIONAL")
        q = F(s)
        if str(q.numerator)+"/"+str(q.denominator) != s:
            raise ValueError("NONCANONICAL_RATIONAL")
        if max(q.numerator.bit_length(),q.denominator.bit_length())>8192:
            raise ValueError("CHECKER_RATIONAL_SIZE_LIMIT")
        return q
    def scalar(self, s):
        q = self.q(s)
        if q<0 or (1<<self.bits)%q.denominator:
            raise ValueError("NON_DYADIC_OR_NEGATIVE_BOUND")
        return q
    def iv(self, v):
        if not isinstance(v,list) or len(v)!=2:
            raise ValueError("INVALID_INTERVAL")
        a,b = self.scalar(v[0]),self.scalar(v[1])
        if a>b: raise ValueError("REVERSED_INTERVAL")
        return a,b
    def contain(self, v, lo, hi, name):
        a,b=self.iv(v)
        if a>lo or b<hi: raise ValueError("INVALID_BOUND_"+name)
        return a,b


def _keys(x, keys):
    if not isinstance(x,dict) or set(x)!=set(keys):
        raise ValueError("UNEXPECTED_SCHEMA_KEYS")

def _integer(x, lo, hi):
    if type(x) is not int or not lo<=x<=hi:
        raise ValueError("INTEGER_GUARD")
    return x

def _atan(x, n):
    # Independent exact witness replay, no producer helper calls.
    t=sum(((-1)**j*x**(2*j+1)/F(2*j+1) for j in range(n)), F(0))
    u=t+(-1)**n*x**(2*n+1)/F(2*n+1)
    return min(t,u),max(t,u)

def _exp(w, expected_x, pairs, ck, max_s=16):
    ck.tick()
    _keys(w, ("x","y","squarings","pairs","levels","result"))
    x,y=ck.q(w["x"]),ck.q(w["y"])
    s=_integer(w["squarings"],0,max_s)
    if w["pairs"]!=pairs or type(w["pairs"]) is not int or x!=expected_x or x<0 or y!=x/F(2**s) or not 0<=y<=1:
        raise ValueError("EXP_ARGUMENT_OR_REDUCTION")
    levels=w["levels"]
    if not isinstance(levels,list) or len(levels)!=s+1:
        raise ValueError("EXP_LEVEL_COUNT")
    # Evaluate the even and odd polynomials separately from their definition.
    term=F(1); total=F(1); even=F(1)
    for j in range(1,2*pairs+2):
        ck.tick();term=term*(-y)/j;total+=term
        if j==2*pairs: even=total
        if max(total.numerator.bit_length(),total.denominator.bit_length())>8192:
            raise ValueError("EXP_PROOF_SIZE_LIMIT")
    a,b=ck.contain(levels[0], max(F(0),total), min(F(1),even), "EXP_SEED")
    if b>1: raise ValueError("EXP_BOUND_ABOVE_ONE")
    for level in levels[1:]:
        a,b=ck.contain(level,a*a,b*b,"EXP_SQUARE")
        if b>1: raise ValueError("EXP_BOUND_ABOVE_ONE")
    if w["result"]!=levels[-1]: raise ValueError("EXP_RESULT_MISMATCH")
    return a,b


def _verify(c):
    _keys(c,("schema","status","outcome","request","sources","semantics","pi","h","alpha","modes","finite","tail_exp","tail_integral_upper","tail_upper","raw","bounds","endpoint_spread","normalizer_input_spread_lower","error_accounting"))
    if c["schema"]!="original-full-spectrum-moments-v1" or c["status"]!="ENCLOSED" or c["sources"]!=PINS or c["semantics"]!="FOR_ALL_L_IN_CLOSED_INPUT_INTERVAL; NO_SCIENTIFIC_TARGET_ASSIGNED":
        raise ValueError("SOURCE_OR_SEMANTIC_IDENTITY")
    r=c["request"]
    _keys(r,("L_lo","L_hi","cutoff","bits","pi_terms","exp_pairs","width_goal","input_role","limits"))
    bits=_integer(r["bits"],16,128);n=_integer(r["pi_terms"],1,32);p=_integer(r["exp_pairs"],1,24);m=_integer(r["cutoff"],1,64)
    ck=Check(bits);lo,hi=ck.q(r["L_lo"]),ck.q(r["L_hi"])
    if not 0<lo<=hi or max(lo.numerator.bit_length(),lo.denominator.bit_length(),hi.numerator.bit_length(),hi.denominator.bit_length())>256:
        raise ValueError("INPUT_INTERVAL")
    if r["input_role"] not in ("TEST_ONLY","UNASSIGNED_PARAMETER_INPUT"):
        raise ValueError("INPUT_ROLE")
    _keys(r["limits"],CAPS)
    for k,v in r["limits"].items(): _integer(v,1,CAPS[k])
    goal=None if r["width_goal"] is None else ck.q(r["width_goal"])
    if goal is not None and goal<=0: raise ValueError("WIDTH_GOAL")
    _keys(c["pi"],("terms","interval"))
    if type(c["pi"]["terms"]) is not int or c["pi"]["terms"]!=n: raise ValueError("PI_TERMS")
    a,b=_atan(F(1,5),n),_atan(F(1,239),n)
    pi=ck.contain(c["pi"]["interval"],16*a[0]-4*b[1],16*a[1]-4*b[0],"PI_MACHIN")
    if pi[0]<=0: raise ValueError("PI_POSITIVITY")
    h=ck.contain(c["h"],2*pi[0]/hi,2*pi[1]/lo,"PHYSICAL_H")
    alpha=ck.contain(c["alpha"],h[0]**2/2,h[1]**2/2,"ALPHA")
    if alpha[0]<=0 or alpha[0]*m*m<3: raise ValueError("INTEGER_M_TAIL_THRESHOLD")
    degrees=("0","2","4","6")
    modes=c["modes"]
    if not isinstance(modes,list) or len(modes)!=m: raise ValueError("MODE_COUNT")
    sums={key:[F(1),F(1)] if key=="0" else [F(0),F(0)] for key in degrees}
    for j,mode in enumerate(modes,1):
        _keys(mode,("m","exp_at_hi","exp_at_lo","weight","contribution"))
        if type(mode["m"]) is not int or mode["m"]!=j: raise ValueError("MODE_ID")
        el=_exp(mode["exp_at_hi"],alpha[1]*j*j,p,ck,r["limits"]["max_squarings"])
        eu=_exp(mode["exp_at_lo"],alpha[0]*j*j,p,ck,r["limits"]["max_squarings"])
        w=ck.contain(mode["weight"],el[0],eu[1],"MODE_WEIGHT")
        _keys(mode["contribution"],degrees)
        for key in degrees:
            d=int(key)
            v=ck.contain(mode["contribution"][key],2*(h[0]*j)**d*w[0],2*(h[1]*j)**d*w[1],"SYMMETRIC_MODE")
            sums[key][0]+=v[0];sums[key][1]+=v[1]
    _keys(c["finite"],degrees);_keys(c["tail_upper"],degrees);_keys(c["raw"],degrees)
    finite={key:ck.contain(c["finite"][key],*sums[key],"FINITE_SUM_WITH_ZERO_MODE") for key in degrees}
    e=_exp(c["tail_exp"],alpha[0]*m*m,p,ck,r["limits"]["max_squarings"])[1]
    ints=c["tail_integral_upper"]
    if not isinstance(ints,list) or len(ints)!=4: raise ValueError("TAIL_RECURRENCE_COUNT")
    integ=[ck.scalar(v) for v in ints]
    if integ[0]<e/(2*alpha[0]*m): raise ValueError("MILLS_BOUND")
    for k in range(1,4):
        if integ[k]<(m**(2*k-1)*e+(2*k-1)*integ[k-1])/(2*alpha[0]):
            raise ValueError("TAIL_RECURRENCE")
    tails={key:ck.scalar(c["tail_upper"][key]) for key in degrees}
    for k,key in enumerate(degrees):
        if tails[key]<2*h[1]**(2*k)*integ[k]: raise ValueError("UNDERSTATED_SPECTRAL_TAIL")
    raw={key:ck.contain(c["raw"][key],finite[key][0],finite[key][1]+tails[key],"FULL_SPECTRUM_SUM") for key in degrees}
    z=raw["0"]
    if z[0]<1: raise ValueError("ZERO_MODE_OR_DENOMINATOR")
    _keys(c["bounds"],("z","mu2","mu4","mu6"))
    bounds={"z":ck.contain(c["bounds"]["z"],*z,"NORMALIZER")}
    for key in ("2","4","6"):
        bounds["mu"+key]=ck.contain(c["bounds"]["mu"+key],raw[key][0]/z[1],raw[key][1]/z[0],"SAME_FULL_NORMALIZATION")
    spread=ck.scalar(c["normalizer_input_spread_lower"])
    if lo==hi:
        if c["endpoint_spread"] is not None or spread!=0: raise ValueError("POINT_INPUT_SPREAD")
    else:
        sp=c["endpoint_spread"]
        _keys(sp,("alpha_Llo","alpha_Lhi","exp_Lhi_at_alpha_hi","exp_Llo_at_alpha_lo"))
        al=ck.contain(sp["alpha_Llo"],2*pi[0]**2/lo**2,2*pi[1]**2/lo**2,"ENDPOINT_ALPHA")
        ah=ck.contain(sp["alpha_Lhi"],2*pi[0]**2/hi**2,2*pi[1]**2/hi**2,"ENDPOINT_ALPHA")
        el=_exp(sp["exp_Lhi_at_alpha_hi"],ah[1],p,ck,r["limits"]["max_squarings"])[0]
        eu=_exp(sp["exp_Llo_at_alpha_lo"],al[0],p,ck,r["limits"]["max_squarings"])[1]
        if spread>max(F(0),2*(el-eu)): raise ValueError("INPUT_SPREAD_OVERCLAIM")
    widths=[v[1]-v[0] for v in bounds.values()]
    outcome="BOUNDS_ONLY" if goal is None else ("TOLERANCE_MET" if all(w<=goal for w in widths) else "HOLD_INPUT_UNCERTAINTY" if spread>goal else "HOLD_TARGET_UNMET")
    if c["outcome"]!=outcome: raise ValueError("FALSE_TOLERANCE_OR_HOLD_DISPOSITION")
    er=c["error_accounting"]
    _keys(er,("producer_rounding_spec","producer_round_quantum","checker_certifies","finite_widths","finite_width_scope"))
    if er["producer_rounding_spec"]!="DYADIC_FLOOR_CEIL_EACH_LOCAL_CALL" or ck.q(er["producer_round_quantum"])!=F(1,2**bits) or er["checker_certifies"]!="FORWARD_CONTAINMENT_ONLY_NOT_MINIMAL_ROUNDING_OR_EXECUTION" or er["finite_width_scope"]!="includes input uncertainty, pi, exp and arithmetic; not a global sum of local errors":
        raise ValueError("ROUNDING_PROVENANCE")
    _keys(er["finite_widths"],degrees)
    if any(ck.q(er["finite_widths"][key])!=finite[key][1]-finite[key][0] for key in degrees):
        raise ValueError("FALSE_FINITE_WIDTH")
    ck.tick()


def verify(c):
    try:
        _verify(c)
    except (KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError,RecursionError) as e:
        raise ValueError("MALFORMED_CERTIFICATE: "+type(e).__name__) from None
    return None


def _strict_json(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("DUPLICATE_JSON_KEY")
            out[k] = v
        return out
    def constant(value):
        raise ValueError("NONFINITE_JSON_CONSTANT")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("certificate");a=p.parse_args()
    try:
        with open(a.certificate,"rb") as f: data=f.read(2000001)
        if len(data)>2000000: raise ValueError("CERTIFICATE_SIZE_LIMIT")
        try:
            c=_strict_json(data)
        except (ValueError,UnicodeError,RecursionError):
            raise ValueError("INVALID_JSON_CERTIFICATE") from None
        verify(c)
        print(json.dumps({"verification":"ACCEPT_BOUNDS_ONLY","outcome":c["outcome"],"input_role":c["request"]["input_role"],"scientific_acceptance":False}))
        return 0
    except (ValueError,OSError) as e:
        print(json.dumps({"verification":"REJECT_OR_HOLD","reason":str(e)}));return 1

if __name__=="__main__": raise SystemExit(main())
