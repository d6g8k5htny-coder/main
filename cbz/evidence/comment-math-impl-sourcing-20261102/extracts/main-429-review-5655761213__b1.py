"""Section 3 counterfamily 1: DC-dominant and axial-dominant positive full-square spectra.
L_N = 2 pi N (omega = 1/N), K_N = 2N, fill uniform on {-2N..2N}^2."""
import sys, json
from fractions import Fraction as Q
import numpy as np
sys.path.insert(0, sys.argv[1])
from jetcov import modes, gamma, lam_min_over_frames

out = {}
for N in (4, 8, 16, 32):
    K = 2 * N; om = 1.0 / N
    n1, n2, _ = modes(K)
    size = (4 * N + 1) ** 2
    for eps in (1e-1, 1e-2, 1e-4, 1e-6):
        fill = np.full(n1.shape, 1.0 / size)
        dc = eps * fill.copy(); dc[(n1 == 0) & (n2 == 0)] += 1 - eps
        ax = eps * fill.copy()
        for (a, b) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax[(n1 == a) & (n2 == b)] += (1 - eps) / 4
        rec = {}
        for name, w in (("DC", dc), ("axial", ax)):
            assert np.all(w > 0) and abs(w.sum() - 1) < 1e-12
            assert np.allclose(w, w[::-1])  # q_n = q_-n (grid reversal)
            xi2 = (n1 ** 2 + n2 ** 2) / N ** 2
            m12 = float(np.sum(w * xi2 ** 6)); maxxi = float(np.sqrt(xi2.max()))
            lmin, th, vals = lam_min_over_frames(w, K, om, nth=181)
            G0 = gamma(w, K, om, 0.0)
            varFu = [gamma(w, K, om, t)[1, 1] for t in np.linspace(0, np.pi, 19)]
            pos = min(vals)
            rec[name] = dict(M12=m12, max_abs_xi=maxxi, lam_min_frames=lmin,
                             max_VarFu=max(varFu), bound_8eps=8 * eps,
                             axial_bound=1 / (2 * N ** 2) + 8 * eps,
                             all_positive=bool(pos > 0))
            if name == "DC":
                exact = eps * 2 * (2 * N + 1) / (3 * N)
                assert abs(G0[1, 1] - exact) < 1e-12 * max(1, exact) + 1e-15, (G0[1, 1], exact)
                assert max(varFu) <= 8 * eps and lmin <= 8 * eps
            else:
                assert max(varFu) <= 1 / (2 * N ** 2) + 8 * eps and lmin <= max(varFu)
            assert m12 <= 262144 and maxxi <= 2 * np.sqrt(2) + 1e-12
        out[f"N={N},eps={eps}"] = rec

# exact Fraction M12 for DC family at N=4, eps=1/10 (sanity)
N = 4; eps = Q(1, 10); size = (4 * N + 1) ** 2
m12 = sum((eps / size) * Q(i * i + j * j, N * N) ** 6 for i in range(-2 * N, 2 * N + 1) for j in range(-2 * N, 2 * N + 1))
out["exact_M12_DC_N4_eps0.1"] = str(m12); assert m12 <= 262144
print(json.dumps(out, indent=1))
