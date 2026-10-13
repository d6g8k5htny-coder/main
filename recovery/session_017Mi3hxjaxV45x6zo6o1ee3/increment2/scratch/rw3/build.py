import sys, inspect, importlib.util, pathlib
sys.argv = ["x"]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
W = load("w197", "<SCRATCH>/wtwin/frontiers/c8_window_coefficient_planar_20260930/window_coefficient.py")
src = pathlib.Path(W.__file__).read_text()
head = src[src.index("import argparse"):src.index("MUTANTS = (")]
parts = ['"""HEADER_PLACEHOLDER"""\n', head]
parts.append('''MUTANTS = ("image-shells", "lambda-min", "sandwich-power", "window-scale", "d3-cross", "schur-sign", "sphere", "tau-cross",
           "tail-dropped", "no-owen", "gamma-shape")
MUT = None

PREC = 60
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
# Every arithmetic step below names one of the three contexts; operator syntax on Decimals would round to the thread's
# current context, which is widened as a safety net (rule LIBRARY_EXACT checks exp, sqrt and the negation).
setcontext(Context(prec=2 * PREC + 20, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999))
ZERO, ONE, HALF = Decimal(0), Decimal(1), Decimal("0.5")
INF = math.inf
ROOT_DIGITS = 70
SIDE24_D3 = ("0.04177593184059834334", "0.04177593184059834335")   # coefficients/side24_v1/ENCLOSURE.json, dimensions.3
PINS_PLACEHOLDER

''')
parts.append("\n# ---------------- interval arithmetic and special functions (Math-#197 toolkit, unchanged)\n")
for name in ["ulp", "Iv", "hull", "inroot", "root_frac", "root_iv", "pi_interval"]:
    parts.append(inspect.getsource(getattr(W, name)) + "\n\n")
parts.append('''PI = pi_interval()
SQRT2 = Iv.frac(2).sqrt()
SQRT3 = Iv.frac(3).sqrt()
SQRT2PI = (2 * PI).sqrt()
SQRTPI = PI.sqrt()
TINY = Decimal(10) ** (-PREC - 8)


''')
for name in ["erf_point", "erf_iv", "Phi", "phi", "atan_iv", "owen_T", "owen_T_eval", "SHAPE", "gamma_lower_point", "gamma_lower", "gamma_complete"]:
    parts.append(inspect.getsource(getattr(W, name)) + "\n\n")
parts.append('''GAMMA = {Fr(7, 6): gamma_complete(Fr(7, 6)), Fr(4, 3): gamma_complete(Fr(4, 3))}
GAMMA76 = GAMMA[Fr(7, 6)]
CBRT32 = root_frac(Fr(12), 3) / 2                       # (3/2)^(1/3) = 12^(1/3)/2
C72 = 1 / (72 * root_frac(Fr(72), 6))                   # 72^(-7/6)
SIGMA_REF = Iv.frac(Fr(2, 3)).sqrt()                    # sd of f | V=0 for the reference kernel
D2_EXACT = Iv.frac(Fr(29, 6)) - Iv.frac(6).sqrt()        # SIDE24 cone moment m = 2
C3_REF = GAMMA76 * CBRT32 * D2_EXACT / (2 * SQRT3 * PI * PI * SQRTPI)             # SIDE24 (1), d = 3


# ---------------- the Gaussian integrals int_{-inf}^x t^n phi(t) Phi(a t) dt, n <= 4 (Math-#197 section 2, unchanged)
''')
for name in ["G_finite", "G_limit", "G_at", "P_reg", "G_ref"]:
    parts.append(inspect.getsource(getattr(W, name)) + "\n\n")
parts.append(inspect.getsource(W.library_exact) + "\n\n")
for name in ["simpson", "fPhi", "fphi", "m3_float"]:
    parts.append(inspect.getsource(getattr(W, name)) + "\n\n")
out = "".join(parts)
pathlib.Path("toolkit.py").write_text(out)
import re
print("toolkit lines", len(out.splitlines()))
print("MUT hooks:", sorted(set(re.findall(r'MUT [!=]= "([a-z0-9-]+)"', out))))
