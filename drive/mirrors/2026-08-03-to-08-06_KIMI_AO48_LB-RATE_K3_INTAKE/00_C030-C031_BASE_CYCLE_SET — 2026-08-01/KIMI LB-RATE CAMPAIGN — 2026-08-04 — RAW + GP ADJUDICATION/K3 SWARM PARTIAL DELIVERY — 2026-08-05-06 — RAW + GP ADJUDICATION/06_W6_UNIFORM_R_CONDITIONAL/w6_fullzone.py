# w6_fullzone.py -- full I_WP(r): fine wedge (th 55-125, r 0.2-2.6) + coarse tails.
import sys, time
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
import numpy as np
from w6_zone import RungLaw, point_rho
from w6_kernel import set_dps

def wedge(law, nth=36, nr=49):
    ths = np.linspace(np.radians(55), np.radians(125), nth)
    rs = np.linspace(0.2, 2.6, nr)
    A = np.zeros((nth, nr))
    for i,th in enumerate(ths):
        ct, st = mp.cos(mp.mpf(th)), mp.sin(mp.mpf(th))
        for j,rr in enumerate(rs):
            A[i,j] = point_rho(law, (mp.mpf(rr)*ct, mp.mpf(rr)*st))
    return np.trapezoid(np.trapezoid(A*rs[None,:], rs, axis=1), ths, axis=0)

def tails(law, nth=48, nr=46):
    # coarse polar grid over full circle, log radial 0.08..3.0, excluding wedge box
    ths = np.linspace(0, 2*np.pi, nth, endpoint=False)
    rs = np.exp(np.linspace(np.log(0.08), np.log(3.0), nr))
    tot = 0.0
    for i,th in enumerate(ths):
        ct, st = mp.cos(mp.mpf(th)), mp.sin(mp.mpf(th))
        deg = np.degrees(th)
        for j,rr in enumerate(rs):
            in_wedge = (55 <= deg <= 125) and (0.2 <= rr <= 2.6)
            if in_wedge: continue
            rho = point_rho(law, (mp.mpf(rr)*ct, mp.mpf(rr)*st))
            # trapezoid weights (periodic in th; log-r)
            wth = 2*np.pi/nth
            if j == 0: wr = (rs[1]-rs[0])/2
            elif j == nr-1: wr = (rs[-1]-rs[-2])/2
            else: wr = (rs[j+1]-rs[j-1])/2
            tot += rho*rr*wth*wr
    return tot

if __name__ == '__main__':
    set_dps(35)
    for r in ['0.05','0.025','0.0125','0.00625','0.003125']:
        law = RungLaw(r)
        t0=time.time()
        Iw = wedge(law)
        It = tails(law)
        print(f"r={r}: I_wedge={Iw:.10e}  I_tails={It:.3e}  I_total={Iw+It:.10e}  ({time.time()-t0:.0f}s)", flush=True)
