"""NON-CERTIFYING Monte Carlo of E[G_r] from the direct point law, to check #447's per-cell band.

Usage: python3 -I mc_probe.py H2_DIR R_NUM R_DEN N SEED
The law is the program's CB7 direct point law (ce.cov in mp, dps 100), copied verbatim except
that mp.inverse replaces mp_inverse6. G = -D_M D_S 1{q_M < 0, D_M > 0, D_S < 0}, as in the
program's own --mc diagnostic. Prints the estimate and its standard error.
"""
import importlib.util
import json
import math
import random
import sys
from fractions import Fraction as F
from pathlib import Path

h2, rn, rd, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
spec = importlib.util.spec_from_file_location('cov_exact', str(Path(h2) / 'cov_exact.py'))
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)
from mpmath import mp, mpf  # noqa: E402

rpt = mpf(rn) / rd
Mx = (-rpt / 2, mpf(0))
Sx = (rpt / 2, mpf(0))
V = [[(Mx, (0, 0), mpf(1))], [(Mx, (1, 0), mpf(1))], [(Mx, (0, 1), mpf(1))],
     [(Sx, (1, 0), 1 / rpt), (Mx, (1, 0), -1 / rpt)], [(Sx, (0, 1), 1 / rpt), (Mx, (0, 1), -1 / rpt)],
     [(Sx, (0, 0), 1 / rpt ** 3), (Mx, (0, 0), -1 / rpt ** 3), (Sx, (1, 0), -1 / (2 * rpt ** 2)), (Mx, (1, 0), -1 / (2 * rpt ** 2))]]
Yv = [[(Mx, (0, 2), mpf(1))], [(Sx, (0, 2), mpf(1))],
      [(Mx, (2, 0), 1 / rpt)], [(Sx, (2, 0), 1 / rpt)],
      [(Mx, (1, 1), 1 / rpt)], [(Sx, (1, 1), 1 / rpt)]]


def lc(FA, FB):
    tot = mpf(0)
    for (xa, aa, ca) in FA:
        for (xb, ab, cb) in FB:
            tot += ca * cb * ce.cov(aa, ab, xa[0] - xb[0], xa[1] - xb[1], mode='mpf')
    return tot


GP = mp.matrix([[lc(V[i], V[j]) for j in range(6)] for i in range(6)])
GY = [[lc(Yv[i], Yv[j]) for j in range(6)] for i in range(6)]
GYV = [[lc(Yv[i], V[j]) for j in range(6)] for i in range(6)]
Gi = mp.inverse(GP)
zVd = [mpf(6) / 5, 0, 0, 0, 0, mpf(-1) / 6]
csol = [sum(Gi[i, k] * zVd[k] for k in range(6)) for i in range(6)]
mu = [float(sum(GYV[i][k] * csol[k] for k in range(6))) for i in range(6)]
Sig = [[float(GY[i][j] - sum(GYV[i][k] * sum(Gi[k, l] * GYV[j][l] for l in range(6)) for k in range(6)))
        for j in range(6)] for i in range(6)]
S = [[(Sig[i][j] + Sig[j][i]) / 2 + (1e-14 if i == j else 0.0) for j in range(6)] for i in range(6)]
L = [[0.0] * 6 for _ in range(6)]
for i in range(6):
    for j in range(i + 1):
        s = S[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
        L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
rng = random.Random(seed)
r = rn / rd
tot = tot2 = 0.0
for _ in range(N):
    z = [rng.gauss(0.0, 1.0) for _ in range(6)]
    y = [mu[i] + sum(L[i][k] * z[k] for k in range(i + 1)) for i in range(6)]
    qM, qS, aM, aS, bM, bS = y
    DM = aM * qM - r * bM * bM
    DS = aS * qS - r * bS * bS
    g = -DM * DS if (qM < 0 and DM > 0 and DS < 0) else 0.0
    tot += g
    tot2 += g * g
mean = tot / N
se = math.sqrt(max(tot2 / N - mean * mean, 0.0) / (N - 1))
print(json.dumps({'r': '%d/%d' % (rn, rd), 'N': N, 'seed': seed, 'mean': mean, 'se': se}))
