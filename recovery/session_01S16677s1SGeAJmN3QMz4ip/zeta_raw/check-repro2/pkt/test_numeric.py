"""Numeric smoke checks for the zeta-fold exploration packet (needs numpy and mpmath; see
requirements.txt). Skipped cleanly when either is missing, e.g. under `python3 -S`.

Exploratory packet; scientific effect NONE. High precision, not interval-certified. Runs in
well under two minutes:
  * H_0(z) = Xi(z/2)/8 at three points (hflow.py, fixed-point trapezoid rule);
  * the backward heat equation by a finite difference in t (not by construction);
  * the exact zero velocity H_zz/H_z at the first zero;
  * re-convergence of the (9/10) collision (t*, x0) from a coarse seed to 15 digits;
  * Riemann-Siegel Z (stats/rs_zeta.py) against mpmath.siegelz at five points;
  * stats/analyze.py:barcode against a brute-force elder-rule computation on random sequences.
"""
import contextlib
import importlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HAVE_DEPS = all(importlib.util.find_spec(m) is not None for m in ("numpy", "mpmath"))
if HAVE_DEPS:
    for sub in ("stats", "dbn"):
        p = str(HERE / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

# published (9/10) collision (results/dbn/fold_pair9_10.json)
T_STAR_910 = "-1.87790907297253526476683971812792711055587932"
X0_910 = "99.3284693055757509480227688859994916526075262"


@unittest.skipUnless(HAVE_DEPS, "numpy and mpmath are required (pip install -r requirements.txt)")
class DeBruijnNewmanFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mpmath as mp
        from hflow import HFlow, xi
        cls.mp, cls.xi = mp, staticmethod(xi)
        cls.hf = HFlow(dps=50)

    def setUp(self):
        self.mp.mp.dps = 50

    def test_H0_matches_Xi_over_8(self):
        mp = self.mp
        for z in (0, 10, 40):
            h = self.hf.H(z, 0)
            ref = self.xi(mp.mpf(1) / 2 + 1j * mp.mpf(z) / 2).real / 8
            self.assertLess(abs(h - ref) / abs(ref), mp.mpf(10) ** -40, z)

    def test_backward_heat_equation_finite_difference(self):
        mp = self.mp
        for z, t in ((50, mp.mpf("-0.3")), (100, mp.mpf(-1)), (30, mp.mpf("0.2"))):
            d = mp.mpf(10) ** -15
            Ht = (self.hf.H(z, t + d) - self.hf.H(z, t - d)) / (2 * d)
            Hzz = self.hf.derivs(z, t, 2)[2]
            self.assertLess(abs(Ht + Hzz), mp.mpf(10) ** -20 * abs(Hzz), (z, t))

    def test_exact_zero_velocity_first_zero(self):
        mp = self.mp
        z1 = mp.findroot(lambda z: self.hf.H(z, 0), mp.mpf("28.27"))
        H, H1, H2 = self.hf.derivs(z1, 0, 2)
        self.assertLess(abs(z1 - 2 * mp.zetazero(1).imag), mp.mpf(10) ** -25)
        self.assertLess(abs(H2 / H1 - mp.mpf("-0.579630577478")), mp.mpf(10) ** -11)

    def test_collision_9_10_reconverges(self):
        mp = self.mp
        x, t = mp.mpf("99.33"), mp.mpf("-1.878")
        for _ in range(30):
            H, H1, H2, H3 = self.hf.derivs(x, t, 3)
            step = mp.lu_solve(mp.matrix([[H1, -H2], [H2, -H3]]), mp.matrix([-H, -H1]))
            x += step[0]
            t += step[1]
            if abs(step[0]) + abs(step[1]) < mp.mpf(10) ** -35:
                break
        self.assertLess(abs(t - mp.mpf(T_STAR_910)), mp.mpf(10) ** -15 * abs(mp.mpf(T_STAR_910)))
        self.assertLess(abs(x - mp.mpf(X0_910)), mp.mpf(10) ** -15 * mp.mpf(X0_910))
        # a genuine double zero: H_zz != 0 and the pair is real just above t*, complex just below
        A = self.hf.derivs(x, t, 2)[2] / 2
        self.assertGreater(abs(A), mp.mpf(10) ** -17)
        self.assertLess(self.hf.H(x, t + mp.mpf("1e-6")) * A, 0)   # minimum below zero: two real zeros
        self.assertGreater(self.hf.H(x, t - mp.mpf("1e-6")) * A, 0)


@unittest.skipUnless(HAVE_DEPS, "numpy and mpmath are required (pip install -r requirements.txt)")
class RiemannSiegel(unittest.TestCase):
    def test_against_siegelz(self):
        import mpmath
        import numpy as np
        import stats_paths
        tmp = tempfile.mkdtemp(prefix="zeta_fold_rs_")
        saved = stats_paths.RAW
        try:
            stats_paths.RAW = Path(tmp)          # keep the coefficient cache out of results/raw
            sys.modules.pop("rs_zeta", None)
            with contextlib.redirect_stdout(io.StringIO()):   # silence the fit-residual report
                rs = importlib.import_module("rs_zeta")
            ts = np.array([1e4 + 0.3, 5e4 + 3.1, 2e5 + 0.9, 5e5 + 1.7, 9e5 + 2.2])
            mpmath.mp.dps = 25
            ref = np.array([float(mpmath.siegelz(float(x))) for x in ts])
            err = np.max(np.abs(rs.Z(ts) - ref))
            self.assertLess(err, 2e-8)
        finally:
            stats_paths.RAW = saved
            sys.modules.pop("rs_zeta", None)
            shutil.rmtree(tmp, ignore_errors=True)


def brute_bars(vals):
    """Superlevel H0 bars of a path graph with distinct vertex values, O(n^2) and no union-find:
    a local maximum dies at the higher of the two path minima on the way to the nearest strictly
    higher vertex on either side. Returns (birth_index, death_value) pairs."""
    import numpy as np
    n = len(vals)
    out = []
    for i in range(n):
        if not ((i == 0 or vals[i - 1] < vals[i]) and (i == n - 1 or vals[i + 1] < vals[i])):
            continue
        deaths = []
        for rng in (range(i - 1, -1, -1), range(i + 1, n)):
            m = np.inf
            for j in rng:
                m = min(m, vals[j])
                if vals[j] > vals[i]:
                    deaths.append(m)
                    break
        if deaths:
            out.append((i, max(deaths)))
    return out


@unittest.skipUnless(HAVE_DEPS, "numpy and mpmath are required (pip install -r requirements.txt)")
class Barcode(unittest.TestCase):
    def test_against_brute_force(self):
        import numpy as np
        from analyze import barcode
        rng = np.random.default_rng(12345)
        for trial in range(150):
            npath = int(rng.integers(1, 4))
            areas, sids = [], []
            for p in range(npath):
                K = int(rng.integers(2, 60))
                sgn = rng.choice([-1, 1]) * (-1) ** np.arange(K)
                areas.append(sgn * rng.lognormal(0, 1.5, K))
                sids.append(np.full(K, p))
            ell, _, _ = barcode(np.concatenate(areas), np.concatenate(sids))
            ref = []
            for a in areas:
                vals = np.r_[0.0, np.cumsum(a)]
                ref += [vals[i] - d for (i, d) in brute_bars(vals) if 0 < i < len(vals) - 1]  # endpoint bars dropped
            self.assertEqual(len(ref), len(ell), trial)
            self.assertTrue(np.allclose(np.sort(ref), np.sort(ell), rtol=1e-12, atol=1e-12), trial)


if __name__ == "__main__":
    unittest.main()
