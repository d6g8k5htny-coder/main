# Diagnostic only (not proof): Monte Carlo of Lemma A (A1) and quadrature of the Corollary C width integral.
import random, math
random.seed(20261025)

def gauss_vec(A, mu):
    z = [random.gauss(0, 1) for _ in range(len(A[0]))]
    return [mu[i] + sum(A[i][j] * z[j] for j in range(len(z))) for i in range(len(A))]

# (X1,X2,X3,Y) = mu + A z ; p(X) = (X1*X2 - X3^2)^2, degree M=4
M = 4
cM = max(1.0, (3.0) ** 0.25)          # ||N||_4 = 3^(1/4)
worst = 0.0
for trial in range(6):
    A = [[random.gauss(0, 1) for _ in range(4)] for _ in range(4)]
    mu = [random.gauss(0, 1.5) for _ in range(4)]
    cov = [[sum(A[i][k] * A[j][k] for k in range(4)) for j in range(4)] for i in range(4)]
    sig = [math.sqrt(cov[i][i]) for i in range(3)]
    sy = math.sqrt(cov[3][3]); m = mu[3]
    # p = X1^2 X2^2 - 2 X1 X2 X3^2 + X3^4 ; envelope with |c_alpha|
    a = [abs(mu[i]) + cM * sig[i] for i in range(3)]
    env = a[0]**2 * a[1]**2 + 2 * a[0] * a[1] * a[2]**2 + a[2]**4
    # conditional law of X given Y=h (exact Gaussian regression), sampled
    beta = [cov[i][3] / cov[3][3] for i in range(3)]
    ccov = [[cov[i][j] - cov[i][3] * cov[j][3] / cov[3][3] for j in range(3)] for i in range(3)]
    # Cholesky of 3x3 ccov
    L = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            s = ccov[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 0.0)) if i == j else (s / L[j][j] if L[j][j] > 0 else 0.0)
    for u in (0.0, 1.0, 2.5, 4.0):
        h = m + u * sy
        n = 40000; acc = 0.0
        cm = [mu[i] + beta[i] * (h - m) for i in range(3)]
        for _ in range(n):
            z = [random.gauss(0, 1) for _ in range(3)]
            x = [cm[i] + sum(L[i][k] * z[k] for k in range(3)) for i in range(3)]
            acc += abs((x[0] * x[1] - x[2] ** 2) ** 2)
        lhs = acc / n; rhs = (1 + u) ** M * env
        worst = max(worst, lhs / rhs)
        print(f"trial {trial} u={u}: E[|p| | Y=h]={lhs:.4g}  bound={rhs:.4g}  ratio={lhs/rhs:.3f}")
print("(A1) max ratio over samples (must be <= 1):", round(worst, 4))

# Corollary C width: I(a,b,L) = int_R exp(-((a(r+|v|)|v| - L)_+)^2 / s8) dv  vs  C*min(...)
def width(k, s, r, Lc=1.0, c=1.0):
    L = k * r**3 + Lc * s**4; s8 = s**8
    g = lambda v: c * k * s * (r + abs(v)) * abs(v)
    # integrate on [0, V] (symmetric), trapezoid with fine grid
    V = 50 * max(L / (k * s * r), math.sqrt(L / (k * s)), s**3 / (k * r), s**1.5 / math.sqrt(k))
    N = 200000; dv = V / N; tot = 0.0
    for i in range(N + 1):
        v = i * dv; w = 0.5 if i in (0, N) else 1.0
        tot += w * math.exp(-max(g(v) - L, 0.0) ** 2 / s8)
    return 2 * tot * dv, min((k * r**3 + s**4) / (k * s * r), math.sqrt((k * r**3 + s**4) / (k * s)))
for (k, s, r) in [(1, 1e-2, 1e-3), (1, 3e-2, 1e-4), (1, 1e-1, 1e-5), (0.5, 5e-2, 1e-3), (2, 2e-3, 1e-3)]:
    I, W = width(k, s, r)
    print(f"k={k} s={s} r={r}: width integral={I:.4g}  W={W:.4g}  ratio={I/W:.3f}")
