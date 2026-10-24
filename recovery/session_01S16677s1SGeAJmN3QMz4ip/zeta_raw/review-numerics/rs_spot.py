import sys, time, numpy as np, mpmath
sys.dont_write_bytecode = True
ST = '/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/stats'
sys.path.insert(0, ST)
import rs_zeta as rs
rng = np.random.default_rng(777)
mpmath.mp.dps = 25
ts = np.r_[rng.uniform(1e5, 1e6, 12), [663318.5083104731 + 0.0015, 999999.3]]
ref = np.array([float(mpmath.siegelz(x)) for x in ts]); v = rs.Z(ts)
print('fresh RS max |err| =', np.max(np.abs(v - ref)), ' max |Z|', np.max(np.abs(ref)))
print('at mid of closest pair: ours %.6e  mpmath %.6e' % (v[-2], ref[-2]))
d = np.load(ST + '/zeta_B.npz'); z = d['zeros']
for lo, hi in [(500000.0, 500300.0), (999700.0, 1000000.0)]:
    t0 = time.time(); n_ex = int(mpmath.nzeros(hi)) - int(mpmath.nzeros(lo))
    n_us = int(np.sum((z >= lo) & (z < hi)))
    print('window [%g,%g]: ours %d  nzeros %d  (%.1fs)' % (lo, hi, n_us, n_ex, time.time() - t0))
# smallest unfolded gaps found, and expected count of pairs hidden inside one grid cell (GUE, h=0.05)
u = d['unfold']; r = np.diff(u)
print('5 smallest unfolded gaps:', np.round(np.sort(r)[:5], 5))
print('expected within-cell hidden pairs (GUE small-gap, h=0.05): %.1f (observed 52)' % (r.size * np.pi ** 2 * 0.05 ** 3 / 36))
print('observed gaps < 0.05: %d ; GUE expectation N*(pi^2/9)*0.05^3 = %.1f' % (np.sum(r < 0.05), r.size * np.pi ** 2 / 9 * 0.05 ** 3))
