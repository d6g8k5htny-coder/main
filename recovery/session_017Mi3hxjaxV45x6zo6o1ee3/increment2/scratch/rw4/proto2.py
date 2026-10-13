import sys, math, time
sys.path.insert(0, "<SCRATCH>/rw4")
import proto as P
from fractions import Fraction as Fr
W, Iv, PI = P.W, P.Iv, P.PI

def gl(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1): p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1); dx = p1 / dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws

def M3_float(b, n=28, R=16.0):
    xs, ws = gl(n)
    nodes = [(R * (x + 1) / 2, w * R / 2) for x, w in zip(xs, ws)]
    Z3 = 48 * math.sqrt(2) * math.pi
    tot = 0.0
    for a, wa in nodes:
        ea = (b - a) ** 2
        for p, wp in nodes:
            ep = ea + (b - a - p) ** 2
            for q, wq in nodes:
                e = ep + (b - a - p - q) ** 2
                tot += wa * wp * wq * a * a * (a + p) ** 2 * (a + p + q) ** 2 * p * q * (p + q) * math.exp(-e / 4)
    return 6 / Z3 * tot

def I_float(B, nb=20, n=28):
    xs, ws = gl(nb); lo, hi = map(float, B); tot = 0.0
    for x, w in zip(xs, ws):
        b = lo + (hi - lo) * (x + 1) / 2
        tot += w * math.sqrt(3 / (4 * math.pi)) * math.exp(-3 * b * b / 4) * M3_float(b, n)
    return tot * (hi - lo) / 2

t0 = time.time()
M3 = P.Model(P.model_for(3))
s3 = W.SQRT3
def lam(b):
    if b == math.inf: return M3.integral()
    if b == -math.inf: return Iv(W.ZERO)
    return M3.integral(s3 * Iv.of(b), 1 / (2 * s3))
def Iref(B): return (lam(B[1]) - lam(B[0])) / (96 * PI)
for B in ((Fr(0), Fr(1)), (Fr(1), Fr(2))):
    t1 = time.time(); f = I_float(B); print(B, f, Iref(B).pair(), time.time() - t1)

# tilted
m = 3
v, mu, sig2 = Fr(17, 25), Fr(33, 34), Fr(2261, 86700)
assert mu * mu * v + sig2 == Fr(2, 3)
D = 1 + m * sig2 / 2
A = 1 / (2 * v) + m * mu * mu / (4 * D)
Bc = mu / (2 * D)
g0 = Bc * Bc / (4 * A) + sig2 / (8 * D)
cprime = Bc / (2 * A)
gout = Fr(1, 4 * m) - g0
print("D A Bc g0 c' gout", D, A, Bc, float(g0), cprime, float(gout))
kappa0 = 1 / Iv.frac(2 * v * A * D).sqrt()
MT = P.Model(P.model_for(3, gamma_out=gout))
sl = Iv.frac(2 * A).sqrt()
def lamT(b):
    if b == math.inf: return MT.integral()
    if b == -math.inf: return Iv(W.ZERO)
    return MT.integral(sl * Iv.of(b), sl * Iv.frac(cprime))
Z3 = 48 * P.SQRT2 * PI
full = kappa0 / Z3 * lamT(math.inf)
D3x = (50 * PI + 200 * (PI / 2 - W.atan_iv(Iv.frac(Fr(1, 2)))) - 228) / (9 * PI)
print("tilted full", full.pair(), full.intersects(D3x))
for B in ((Fr(0), Fr(1)), (Fr(1), Fr(2))):
    print("tilted", B, (kappa0 / Z3 * (lamT(B[1]) - lamT(B[0]))).pair(), "ref", Iref(B).pair())
print("total", time.time() - t0)
