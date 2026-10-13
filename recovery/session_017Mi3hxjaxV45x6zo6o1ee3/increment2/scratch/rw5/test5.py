import sys, time, importlib.util, pathlib
SP = pathlib.Path("<SCRATCH>")
R = SP / "rw5"
src205 = (SP / "wtw3/frontiers/c8_window_coefficient_d3_torus_20260930/window_coefficient_d3.py").read_text()
src199 = (SP / "wtd4/coefficients/side24_d4_20260930/cone_moment_d4.py").read_text()
imports = src205[src205.index("import argparse"):src205.index("MUTANTS = (")]
toolkit = src205[src205.index("# ---------------- interval arithmetic"):src205.index("def simpson(")]
i3 = src205[src205.index("# ---------------- the d = 3 reference birth integral"):src205.index("def scaled(")]
alg = src199[src199.index("def padd("):src199.index("def ppow(")]
consts = (R / "consts5.py").read_text().replace("PINS_PLACEHOLDER", "{}")
newcode = (R / "newcode5.py").read_text().replace("POLY_ALGEBRA_PLACEHOLDER\n", alg.rstrip() + "\n")
code = imports + consts + "\n\n" + toolkit + i3.rstrip() + "\n" + newcode
(R / "mod5.py").write_text(code)
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("mod5", R / "mod5.py"); M = importlib.util.module_from_spec(spec); sys.modules["mod5"] = M
t0 = time.time(); spec.loader.exec_module(M); print("import", time.time() - t0)
from fractions import Fraction as Fr
t0 = time.time(); pieces, g = M.F_pieces(4); print("pieces", {k: len(M.trim(v)) for k, v in pieces.items()}, "deg", max(len(M.trim(v)) for v in pieces.values()), time.time() - t0)
terms = M.g_terms(4); print("terms", len(terms))
t0 = time.time(); B4 = M.Birth(4); print("model", time.time() - t0, "tail", B4.M.tail, "Rsum", B4.M.Rsum)
D4 = B4.I((-M.INF, M.INF))
S6, S21, S14 = M.Iv.frac(6).sqrt(), M.Iv.frac(21).sqrt(), M.Iv.frac(14).sqrt()
D4cf = M.Iv.frac(Fr(6695, 54)) - M.Iv.frac(Fr(405, 32)) * S6 - (M.Iv.frac(Fr(1375, 72)) * S21 + M.Iv.frac(Fr(6695, 27)) * M.atan_iv(S21 / 7) + M.Iv.frac(Fr(405, 16)) * S6 * M.atan_iv(S14 / 14)) / M.PI
print("D4 quad", D4.pair(), "\nD4 cf  ", D4cf.pair(), D4.intersects(D4cf), D4.width())
for T in (1, 5, 12):
    print("Kfun owen vs sweep", T, M.kfun_owen((Fr(1, 48), Fr(1, 24)), T).pair()[0][:30])
t0 = time.time(); v = B4.I((Fr(0), Fr(1))); print("I[0,1]", v.pair(), time.time() - t0)
