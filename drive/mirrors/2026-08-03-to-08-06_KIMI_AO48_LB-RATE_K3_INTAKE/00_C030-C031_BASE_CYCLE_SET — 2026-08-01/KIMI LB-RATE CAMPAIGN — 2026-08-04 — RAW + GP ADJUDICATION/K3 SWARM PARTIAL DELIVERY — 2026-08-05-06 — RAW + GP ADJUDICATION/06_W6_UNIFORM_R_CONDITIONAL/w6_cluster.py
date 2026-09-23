# w6_cluster.py -- variance laws in the cluster: y = r*yhat, several directions.
import sys
sys.path.insert(0, '/mnt/agents/output/K3_SIDE24_LB/W6_uniform_r')
import mpmath as mp
from w6_kernel import *
from w6_analysis import *

set_dps(120)

def v_vt_at(y, r):
    pins = pin_list(r); vals = pin_values(r, mu_t(r))
    targets = [(0,0),(1,0),(0,1)]
    mean, cov, G, Gi = cond_mean_cov(pins, vals, y, targets)
    v_val = cov[0,0]
    Sg = cov[1:3,1:3]
    c_f = mp.zeros(1,2); c_f[0,0]=cov[0,1]; c_f[0,1]=cov[0,2]
    vt = v_val - (c_f*(Sg**-1)*c_f.transpose())[0]
    return v_val, vt, mean[0]

rungs = [mp.mpf(x) for x in ['0.02','0.01','0.005','0.0025','0.00125']]
print("cluster points y = r*yhat: v/r^p, vt/r^p behavior")
for yhat in [(mp.mpf('0.5'),mp.mpf('0.5')), (mp.mpf(0),mp.mpf(1)), (mp.mpf('-0.76'),mp.mpf('0.24')), (mp.mpf('1'),mp.mpf('0.3'))]:
    print(f" yhat={tuple(mp.nstr(t,6) for t in yhat)}")
    for r in rungs:
        y = (yhat[0]*r, yhat[1]*r)
        v, vt, m = v_vt_at(y, r)
        print(f"  r={mp.nstr(r,9)}: v/r^6={mp.nstr(v/r**6,10)}  vt/r^4={mp.nstr(vt/r**4,10)}  vt/r^6={mp.nstr(vt/r**6,6)}  m-b={mp.nstr(m-BB,6)}")
