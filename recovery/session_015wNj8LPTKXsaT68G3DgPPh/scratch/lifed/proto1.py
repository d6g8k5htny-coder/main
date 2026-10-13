import importlib.util, sys, time
from fractions import Fraction as Fr
def load(name, fn):
    spec=importlib.util.spec_from_file_location(name, fn); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A2=load('ta2','theorem_a.py'); A3=load('ta3','theorem_a3.py')
def f(x): return float(x)
for rs in (Fr(1,4096), Fr(1,512), Fr(1,64)):
    t=time.time(); L2=A2.band_law(Fr(0), rs); t2=time.time()-t
    t=time.time(); L3=A3.band_law(Fr(0), rs); t3=time.time()-t
    print('r_* =', rs, 'times', round(t2,1), round(t3,1))
    for nm,L in (('d2',L2),('d3',L3)):
        print(' ',nm,'e_bb-3/2 [%.3e,%.3e]'%(f(L.e_bb.lo-Fr(3,2)),f(L.e_bb.hi-Fr(3,2))),
              'e_bk [%.3e,%.3e]'%(f(L.e_bk.lo),f(L.e_bk.hi)),
              'e_kk-24 [%.3e,%.3e]'%(f(L.e_kk.lo-24),f(L.e_kk.hi-24)))
    w=L2.alpha['w']; bw=L2.beta['w']; C=L2.C[('w','w')]
    print('  d2 alpha_w+1 [%.3e,%.3e] beta_w [%.3e,%.3e] var_w-2 [%.3e,%.3e]'%(f(w.lo+1),f(w.hi+1),f(bw.lo),f(bw.hi),f(C.lo-2),f(C.hi-2)))
    for n in ('a11','a22','a12'):
        a=L3.alpha[n]; b=L3.beta[n]
        print('  d3',n,'alpha [%.4e,%.4e] beta [%.3e,%.3e]'%(f(a.lo),f(a.hi),f(b.lo),f(b.hi)))
    print('  d3 Cb', [[('%.4f..%.4f'%(f(x.lo),f(x.hi))) for x in row] for row in L3.Cb])
