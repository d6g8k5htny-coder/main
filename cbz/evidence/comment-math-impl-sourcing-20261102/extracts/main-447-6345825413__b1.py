#!/usr/bin/env python3
"""Narrow normalization regressions. No H3 replay, network or repository writes.
Usage: python -I -B test_normalization.py MODULE.py OUTPUT.json
Requires mpmath; the actual version is reported, never inferred from a pin.
"""
from __future__ import annotations
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
import mpmath

SOURCE = Path(sys.argv[1]).resolve()
OUTPUT = Path(sys.argv[2]).resolve()
RAW = SOURCE.read_bytes()
MOD = types.ModuleType('normalization_specimen')
MOD.__file__ = str(SOURCE)
exec(compile(RAW, str(SOURCE), 'exec'), MOD.__dict__)
OBS = []


def frac(x):
    s, m, e, _ = x._mpf_
    return F((-1 if s else 1)*int(m)) * (F(2)**e)


def endpoints(v):
    return frac(MOD.mp.mpf(v.a)), frac(MOD.mp.mpf(v.b))


def row(kind, key, actual, passed):
    OBS.append({'kind':kind, 'key':key, 'actual':actual, 'passed':bool(passed)})


class Normalization(unittest.TestCase):
    def contains(self, kind, key, v, q):
        a,b=endpoints(v)
        ok=a <= q <= b
        row(kind,key,{'lo':str(a),'hi':str(b),'target':str(q)},ok)
        self.assertTrue(ok, f'{kind}: {key} does not enclose target')

    def test_01_exact_unity_and_derived_apis(self):
        for dps,kmax in ((100,30),(100,18),(300,30),(60,18)):
            MOD.configure(dps=dps,kmax=kmax,jmax=3)
            for name,call in (
                ('image',lambda:MOD.k1_image(0,0,'iv')),
                ('spectral',lambda:MOD.k1_spectral(0,0,'iv')),
                ('moment-image',lambda:MOD.moment_a(0,'image','iv')),
                ('cov-image',lambda:MOD.cov((0,0),(0,0),0,0,'image','iv')),
            ):
                with self.subTest(dps=dps,kmax=kmax,api=name):
                    self.contains('unity',[dps,kmax,name],call(),F(1))

    def test_02_signed_interval_correction(self):
        helper=getattr(MOD,'_full_normalization_iv',None)
        self.assertTrue(callable(helper), 'normalization correction helper absent')
        MOD.configure(dps=100,kmax=30,jmax=3)
        original=MOD.spectral_tail_1d
        calls=[]
        def fixed_tail(n):
            calls.append(n)
            return MOD.mp.mpf(1)/8
        MOD.spectral_tail_1d=fixed_tail
        try:
            for a,b in ((-5,-2),(-2,3),(0,0),(1,4)):
                v=helper(MOD.iv.mpf([a,b]))
                for q in (F(a),F(a+b,2),F(b)):
                    for delta in (F(0),F(1,32),F(1,8)):
                        with self.subTest(interval=[a,b],q=q,delta=delta):
                            self.contains('signed-correction',[a,b,str(q),str(delta)],v,q/(1+delta))
            self.assertEqual(calls,[0]*4)
        finally:
            MOD.spectral_tail_1d=original

    def test_03_all_three_iv_routes_apply_factor_once(self):
        helper=getattr(MOD,'_full_normalization_iv',None)
        self.assertTrue(callable(helper), 'normalization correction helper absent')
        MOD.configure(dps=100,kmax=30,jmax=3)
        seen=[]
        sentinel=MOD.iv.mpf([17,19])
        def instrument(value):
            seen.append(value)
            return sentinel
        MOD._full_normalization_iv=instrument
        try:
            for name,args in (('e_sum',(2,'0.5')),('s_sum',(1,'0.5')),('k1_image',(1,'0.5'))):
                with self.subTest(api=name):
                    old=len(seen)
                    result=getattr(MOD,name)(*args,mode='iv')
                    ok=len(seen)==old+1 and endpoints(result)==(F(17),F(19))
                    row('routing',name,{'new_calls':len(seen)-old},ok)
                    self.assertTrue(ok)
            old=len(seen)
            for name,args in (('e_sum',(2,'0.5')),('s_sum',(1,'0.5')),('k1_image',(1,'0.5'))):
                getattr(MOD,name)(*args,mode='mpf')
            MOD.moment_a_poisson(1,'iv')
            self.assertEqual(len(seen),old, 'diagnostic/Poisson path incorrectly changed')
        finally:
            MOD._full_normalization_iv=helper

    def test_04_high_precision_image_reference(self):
        # Independent numerical diagnostic, NOT a rigorous reference certificate.
        ref=mpmath.mp.clone(); ref.dps=450
        def he(n,x):
            a,b=ref.mpf(1),x
            if n==0:return a
            for k in range(1,n):a,b=b,x*b-k*a
            return b
        den=sum(ref.exp(-ref.mpf(288)*j*j) for j in range(-10,11))
        for dps,kmax in ((100,18),(300,30)):
            MOD.configure(dps=dps,kmax=kmax,jmax=3)
            for n in range(9):
                for s_text in ('-11','-1.5','-0.5','0','0.5','1.5','11'):
                    s=ref.mpf(s_text)
                    num=sum(((-1)**n)*he(n,s+24*j)*ref.exp(-(s+24*j)**2/2) for j in range(-10,11))
                    q=frac(num/den)
                    for impl in ('spectral','image'):
                        with self.subTest(dps=dps,kmax=kmax,n=n,s=s_text,impl=impl):
                            self.contains('450dps-reference',[dps,kmax,n,s_text,impl],MOD.k1(n,s_text,impl,'iv'),q)

    def test_05_invalid_orders_and_mode_still_refused(self):
        MOD.configure(dps=100,kmax=30,jmax=3)
        for name in ('e_sum','s_sum','k1_image'):
            for n,mode in ((-1,'iv'),(9,'iv'),(0,'bad')):
                with self.subTest(api=name,n=n,mode=mode):
                    with self.assertRaises(MOD.CovInputError):getattr(MOD,name)(n,0,mode)
                    row('input-refusal',[name,n,mode],{},True)

    def test_06_exact_rational_normalization_lemma(self):
        # General theorem is in PROOF.md; this checks finite signed witnesses.
        count=0
        for z in (F(1),F(3,2),F(9)):
            for b in (F(0),F(1,16),F(1,2)):
                for delta in (F(0),b*z/2,b*z):
                    for lo,hi in ((F(-7),F(-2)),(F(-1),F(3)),(F(0),F(0)),(F(2),F(5))):
                        vals=[v/d for v in (lo,hi) for d in (F(1),1+b)]
                        lower,upper=min(vals),max(vals)
                        for r in (lo,(lo+hi)/2,hi):
                            target=(r*z)/(z+delta)
                            self.assertLessEqual(lower,target)
                            self.assertLessEqual(target,upper)
                            count+=1
        row('rational-lemma','finite-grid',{'cases':count},True)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Normalization))
    if SOURCE.read_bytes()!=RAW:raise RuntimeError('specimen mutated during execution')
    report={'python':sys.version.split()[0],'mpmath':mpmath.__version__,'source_sha256':hashlib.sha256(RAW).hexdigest(),
            'methods':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'observations':OBS,
            'checks':len(OBS),'passed':sum(x['passed'] for x in OBS),'pinned_runtime':mpmath.__version__=='1.4.1'}
    OUTPUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='observations'},sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
