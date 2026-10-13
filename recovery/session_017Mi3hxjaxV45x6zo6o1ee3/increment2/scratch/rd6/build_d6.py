"""Assemble the shared toolkit of cone_moment_d6.py from the verified sources (closed_form_d4.py, cone_moment_d5.py)."""
import sys, inspect, importlib.util, pathlib
sys.argv = ["x"]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
X = load("cf4", "<SCRATCH>/wtd4cf/coefficients/side24_d5_closed_form_20260930/closed_form_d4.py")
D5 = load("cm5", "<SCRATCH>/wtd5/coefficients/side24_d5_20260930/cone_moment_d5.py")
parts = []
head = '''"""HEADER_PLACEHOLDER"""
import argparse
import itertools
import json
import math
import sys
import time
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN, setcontext
from fractions import Fraction as Fr
from functools import lru_cache

MUTANTS = ("shift-variance", "vandermonde", "erfc-branch", "jacobian", "mehta", "recurrence", "layer-init", "two-erf",
           "cumulative-half", "remainder-dropped", "erf-inside", "tail-dropped", "image-shells", "sandwich-exponent")
MUT = None

PREC = 60
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
setcontext(Context(prec=2 * PREC + 20, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999))
ZERO, ONE, HALF = Decimal(0), Decimal(1), Decimal("0.5")
ROOT_DIGITS = 70
K_ORDER, RHO = 40, Fr(6)
CELLS = [(Fr(i, 2), Fr(i + 1, 2)) for i in range(0, 60)] + [(Fr(i), Fr(i + 1)) for i in range(30, 90)]
CONSTANTS_PLACEHOLDER
'''
parts.append(head)
parts.append("\n# ---------------- interval arithmetic over decimal (the Math-#199/#200/#201/#202 toolkit, unchanged)\n")
for name in ["ulp", "Iv", "hull", "ipow", "inroot", "root_frac", "pi_interval"]:
    parts.append(inspect.getsource(getattr(D5, name)) + "\n\n")
parts.append('''PI = pi_interval()
SQRTPI = PI.sqrt()
SQRT2 = Iv.frac(2).sqrt()
SQRT3 = Iv.frac(3).sqrt()
SQRT2PI = (2 * PI).sqrt()
TINY = Decimal(10) ** (-PREC - 8)


''')
for name in ["erf_point", "erf_iv"]:
    parts.append(inspect.getsource(getattr(D5, name)) + "\n\n")
for name in ["gamma_lower_point", "gamma_complete"]:
    parts.append(inspect.getsource(getattr(X, name)) + "\n\n")
parts.append('''GAMMA76 = gamma_complete(Fr(7, 6))
CBRT32 = root_frac(Fr(12), 3) / 2                       # (3/2)^(1/3)


''')
parts.append(inspect.getsource(X.atan_iv) + "\n\n")
parts.append(inspect.getsource(D5.gamma_half_integer) + "\n\n")
parts.append("\n# ---------------- exact arithmetic: surds, and the Q-span of sqrt(r) pi^(k/2) ATOM (Math-#201/#202)\n")
for name in ["sqf", "surd_mul", "surd_sqrt", "surd_inv", "surd_sq"]:
    parts.append(inspect.getsource(getattr(X, name)) + "\n\n")
parts.append("ATOMS = [None]                      # index 0 is the number 1; index i > 0 is arctan(c sqrt r) for ATOMS[i] = (c, r), c sqrt r <= 1\n\n\n")
for name in ["E", "add", "scale", "mul_surd", "mul_pi_half", "mul", "inv_monomial", "atan_elem", "iv_eval", "fl_eval", "show", "M", "gauss_moment_closed"]:
    parts.append(inspect.getsource(getattr(X, name)) + "\n\n")
parts.append("\n# ---------------- exact polynomial algebra and the sector reduction (Math-#199/#200)\n")
for name in ["padd", "pmul", "pscale", "pconst", "pvar", "ppow", "psubst", "univariate", "reduce_sector", "a_integrated_terms", "gamma_half", "mehta_Z", "prefactor", "jk_exact", "jk_identities"]:
    parts.append(inspect.getsource(getattr(X, name)) + "\n\n")
parts.append("\n# ---------------- exact layers: T-functions and the incomplete Gaussian moments (Math-#202)\n")
for name in ["tf_norm", "tf_add", "tf_scale"]:
    parts.append(inspect.getsource(getattr(X, name)) + "\n\n")
parts.append("_G_CACHE = {}\n\n\n")
for name in ["G", "tf_derivative", "tf_at_zero", "g_identities", "outer_moment", "D1_exact", "erfc_moment", "D2_exact", "D3_exact", "organize_tvp", "integrate_branch", "m4_reduction", "D4_exact", "Z4_exact"]:
    parts.append(inspect.getsource(getattr(X, name)) + "\n\n")
parts.append("\n# ---------------- Taylor factors with disc bounds (Math-#199/#200)\n")
for name in ["Factor", "Gauss", "ErfLin", "Poly", "t_mul", "t_add", "t_scale", "t_mul_poly", "poly_taylor_short", "gauss_moment"]:
    parts.append(inspect.getsource(getattr(D5, name)) + "\n\n")
parts.append(inspect.getsource(X.library_exact) + "\n\n")
parts.append(inspect.getsource(X.simpson) + "\n\n" + inspect.getsource(X.float_moment) + "\n\n")
src = "".join(parts)
# mutant hooks: retarget the ones inherited from the D_3/D_4 scripts to this packet's names
src = src.replace('if MUT != "shift-variance" or m != 4 else Fr(1, 24)', 'if MUT != "shift-variance" or m != 5 else Fr(1, 28)')
src = src.replace('m_eff = m + 1 if (MUT == "mehta" and m == 4) else m                          # Z_5 for Z_4 (v1.1: was m == 3, Codex 4149168080)', 'm_eff = m + 1 if (MUT == "mehta" and m == 5) else m                          # Z_6 for Z_5')
src = src.replace('E(-1 / (2 * mu) if MUT != "cumulative-init" else 1 / (2 * mu))', 'E(-1 / (2 * mu) if MUT != "layer-init" else 1 / (2 * mu))')
src = src.replace('Fr(n - 1) if MUT != "cumulative-recurrence" else Fr(n)', 'Fr(n - 1)')
src = src.replace('n - 1 if MUT != "by-parts" else n', 'n - 1')
src = src.replace('if MUT != "one-erf" else (u[0] / lam, u[1])', '')
src = src.replace('E(Fr(1, 2) if MUT != "gauss-half" else Fr(1))', 'E(Fr(1, 2))')
src = src.replace('E(1 / (2 * lam) if MUT != "boundary-term" else 1 / lam)', 'E(1 / (2 * lam))')
src = src.replace('E(Fr(1, 4) if MUT != "atom-pi" else Fr(1, 3), 1, 2)', 'E(Fr(1, 4), 1, 2)')
src = src.replace('-1 if MUT != "erfc-sign" else 1', '-1')
src = src.replace('sign = -1 if MUT != "erfc-sign" else 1', 'sign = -1')
src = src.replace('Fr(1, 2) if MUT != "jacobian" else Fr(1)', 'Fr(1, 2)')                 # m = 3 Jacobian stays exact here
src = src.replace('    if MUT == "jk-init":\n        J[1] = (Fr(0), Fr(2), [Fr(-2)])\n', '')
src = src.replace('jac = Fr(1, 6) if MUT != "jacobian" else Fr(1, 2)', 'jac = Fr(1, 6)')
if MUT_check := [l for l in src.splitlines() if 'MUT' in l and 'MUTANTS' not in l and 'global MUT' not in l]:
    print("remaining MUT hooks:"); [print("   ", l.strip()) for l in MUT_check]
pathlib.Path("cone_moment_d6_base.py").write_text(src)
print("base written", len(src.splitlines()), "lines")
