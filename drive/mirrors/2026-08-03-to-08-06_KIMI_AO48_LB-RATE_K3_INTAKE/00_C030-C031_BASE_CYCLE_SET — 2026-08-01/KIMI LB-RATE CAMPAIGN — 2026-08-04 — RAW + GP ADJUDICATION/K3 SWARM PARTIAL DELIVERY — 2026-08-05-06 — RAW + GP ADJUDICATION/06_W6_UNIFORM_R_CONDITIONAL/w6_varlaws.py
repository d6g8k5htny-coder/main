# w6_varlaws.py -- variance laws v(y;r), v_t(y;r), grad covariance at fixed y, several rungs.
import sys
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
from w6_kernel import *
from w6_analysis import *

CVAL = mp.mpf(-3476069)/6953125
D3VAL = (mp.mpf('1.52') - 2*CVAL)/mp.mpf('0.24')
VSTAR = [BB,0,0,0,0,0,2,mp.mpf('0.5'),D3VAL]

def full_cond(pins, vals, y):
    """Conditional mean vector and covariance of (f, d1f, d2f, d11f, d12f, d22f) at y."""
    targets = [(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)]
    return cond_mean_cov(pins, vals, y, targets)

def vt_vgrad(mean, cov):
    """From (f,grad,H) law: v_val=Var f, S_g=grad cov, v_t = Var(f|grad=0)."""
    v_val = cov[0,0]
    Sg = cov[1:3,1:3]
    c_f = mp.zeros(1,2); c_f[0,0]=cov[0,1]; c_f[0,1]=cov[0,2]
    Sgi = Sg**-1
    vt = v_val - (c_f*Sgi*c_f.transpose())[0]
    return v_val, vt, Sg

def run_point(y, rungs):
    print(f"--- y = ({mp.nstr(y[0],8)}, {mp.nstr(y[1],8)}) ---")
    for r in rungs:
        pins = pin_list(r); vals = pin_values(r, mu_t(r))
        mean, cov, G, Gi = full_cond(pins, vals, y)
        v_val, vt, Sg = vt_vgrad(mean, cov)
        print(f"r={mp.nstr(r,10)}: v={mp.nstr(v_val,12)}  v_t={mp.nstr(vt,12)}  "
              f"Sg=({mp.nstr(Sg[0,0],8)},{mp.nstr(Sg[0,1],8)},{mp.nstr(Sg[1,1],8)})  m={mp.nstr(mean[0],8)}")

if __name__ == '__main__':
    set_dps(100)
    rungs = [mp.mpf(x) for x in ['0.05','0.025','0.0125','0.00625','0.003125','0.0015625']]
    # arch-side strip: x=0, y in {0.4, 0.7, 1.0}; plus C034 station (angle 2.35, d=1)
    pts = [(mp.mpf(0), mp.mpf('0.4')), (mp.mpf(0), mp.mpf('0.7')), (mp.mpf(0), mp.mpf('1.0')),
           (mp.cos(mp.mpf('2.35')), mp.sin(mp.mpf('2.35')))]
    for y in pts:
        run_point(y, rungs)
