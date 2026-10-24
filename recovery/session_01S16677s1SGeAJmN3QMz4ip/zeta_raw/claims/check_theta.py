"""Theta / Mellin / Poisson / Epstein checks for section 4 of the note, and the
project's torus covariance normalizer."""
from mpmath import mp

mp.dps = 30
out = []
def rep(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

def Theta_direct(u):  # sum_{n in Z} exp(-pi u n^2), direct series (jtheta)
    return mp.jtheta(3, 0, mp.exp(-mp.pi * u))

def Theta(u):
    # direct series for u >= 0.05; for smaller u the transformation formula, which is
    # itself checked against the direct series in section 1 below.
    if u >= mp.mpf("0.05"):
        return Theta_direct(u)
    return Theta_direct(1 / u) / mp.sqrt(u)

# 1. Theta transformation
for u in ["0.37", "2", "0.05"]:
    u = mp.mpf(u)
    rep(f"Theta({u}) [direct series]={mp.nstr(Theta_direct(u),20)}  u^(-1/2) Theta(1/u) [direct series]={mp.nstr(Theta_direct(1/u)/mp.sqrt(u),20)}")

# 2. Mellin representation for Re s > 1
def Lam(s):
    return mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)
for s in [mp.mpf(2), mp.mpf(3), mp.mpf("4.5"), mp.mpc(2, 3), mp.mpf("1.3")]:
    I = mp.quad(lambda u: (Theta(u) - 1) * u ** (s / 2 - 1), [0, mp.mpf("0.01"), mp.mpf("0.1"), 1, 10, mp.inf])
    rep(f"s={mp.nstr(s,5)}: (1/2) int (Theta-1) u^(s/2-1) du = {mp.nstr(I/2,18)};  pi^(-s/2)Gamma(s/2)zeta(s) = {mp.nstr(Lam(s),18)}")
# divergence at s<=1: integrand ~ u^{(s-3)/2} near 0
s = mp.mpf("0.9")
part = [mp.quad(lambda u: (Theta(u) - 1) * u ** (s / 2 - 1), [eps, 1]) for eps in [mp.mpf("1e-4"), mp.mpf("1e-8")]]
rep(f"s=0.9: int_eps^1 grows as eps->0: eps=1e-4 -> {mp.nstr(part[0],8)}, eps=1e-8 -> {mp.nstr(part[1],8)} (diverges; the representation needs Re s > 1)")
# Riemann's continued form, valid for all s != 0, 1: Lam(s) = int_1^inf psi(u)(u^{s/2}+u^{(1-s)/2}) du/u - 1/s - 1/(1-s)
def Lam_cont(s):
    psi = lambda u: (Theta(u) - 1) / 2
    return mp.quad(lambda u: psi(u) * (u ** (s / 2) + u ** ((1 - s) / 2)) / u, [1, 10, mp.inf]) - 1 / s - 1 / (1 - s)
for s in [mp.mpc("0.3", "5"), mp.mpc("0.5", "14.134725141734693790")]:
    rep(f"s={mp.nstr(s,8)}: continued theta formula {mp.nstr(Lam_cont(s),15)};  direct {mp.nstr(Lam(s),15)};  value at 1-s: {mp.nstr(Lam_cont(1-s),15)}")

# 3. The project's normalizer and Poisson summation (image sum <-> Fourier series)
def NF(L, d):  # sum_{j in Z^d} exp(-2 pi^2 |j|^2 / L^2) = Theta(2 pi / L^2)^d
    return Theta(2 * mp.pi / L**2) ** d
def NC(L, d):  # sum_{n in Z^d} exp(-|L n|^2 / 2) = Theta(L^2 / (2 pi))^d
    return Theta(L**2 / (2 * mp.pi)) ** d
for (L, d) in [(24, 2), (24, 3), (2, 1), (mp.mpf("3.5"), 3)]:
    L = mp.mpf(L)
    direct = mp.nsum(lambda j: mp.exp(-2 * mp.pi**2 * j * j / L**2), [-mp.inf, mp.inf]) ** d
    rep(f"L={mp.nstr(L,4)}, d={d}: Fourier normalizer sum_j exp(-omega^2|j|^2/2) = {mp.nstr(direct,20)} = Theta(2pi/L^2)^d = {mp.nstr(NF(L,d),20)}; "
        f"Poisson: (2pi)^(d/2) L^-d N_F = {mp.nstr((2*mp.pi)**(mp.mpf(d)/2)*L**(-d)*NF(L,d),20)} vs image normalizer N_C = {mp.nstr(NC(L,d),20)}")
# covariance: image sum vs Fourier series vs Jacobi theta_3 product, 1D factor
def K1_image(x, L):
    return mp.nsum(lambda n: mp.exp(-(x + L * n) ** 2 / 2), [-mp.inf, mp.inf]) / mp.nsum(lambda n: mp.exp(-(L * n) ** 2 / 2), [-mp.inf, mp.inf])
def K1_fourier(x, L):
    a = lambda j: mp.exp(-2 * mp.pi**2 * j * j / L**2)
    return mp.nsum(lambda j: a(j) * mp.cos(2 * mp.pi * j * x / L), [-mp.inf, mp.inf]) / mp.nsum(a, [-mp.inf, mp.inf])
def K1_theta(x, L):
    q = mp.exp(-2 * mp.pi**2 / L**2)
    return mp.jtheta(3, mp.pi * x / L, q) / mp.jtheta(3, 0, q)
for (x, L) in [(mp.mpf("0.7"), mp.mpf(3)), (mp.mpf("1.3"), mp.mpf(24)), (mp.mpf(12), mp.mpf(24))]:
    rep(f"x={x}, L={L}: K image={mp.nstr(K1_image(x,L),18)}, Fourier={mp.nstr(K1_fourier(x,L),18)}, theta_3 ratio={mp.nstr(K1_theta(x,L),18)}, plain Gaussian e^(-x^2/2)={mp.nstr(mp.exp(-x*x/2),18)}")
# size of the periodization correction to local jets at L=24: K_L^{(2k)}(0) vs Gaussian
L = mp.mpf(24)
for k in [1, 2, 3]:
    der_img = mp.diff(lambda x: K1_image(x, L), 0, 2 * k)
    der_gau = mp.diff(lambda x: mp.exp(-x * x / 2), 0, 2 * k)
    rep(f"L=24: d^{2*k}K/dx^{2*k}(0) = {mp.nstr(der_img,25)}; Gaussian: {mp.nstr(der_gau,25)}; diff ~ {mp.nstr(abs(der_img-der_gau),3)} (cf. e^(-L^2/2)={mp.nstr(mp.exp(-L*L/2),3)})")

# 4. Mellin transform in the side length L of the project's normalizer:
#    int_0^inf (N_F(L,d) - 1) L^{-s-1} dL = (1/2)(2 pi^2)^{-s/2} Gamma(s/2) Z_d(s/2),  Z_d(w) = sum' |n|^{-2w}
def Zd(w, d):
    if d == 1:
        return 2 * mp.zeta(2 * w)
    if d == 2:
        return 4 * mp.zeta(w) * mp.dirichlet(w, [0, 1, 0, -1])
    if d == 4:
        return 8 * (1 - 4 ** (1 - w)) * mp.zeta(w) * mp.zeta(w - 1)
for (d, s) in [(1, mp.mpf(3)), (1, mp.mpc(2, 1)), (2, mp.mpf(3)), (4, mp.mpf(6))]:
    I = mp.quad(lambda L: (NF(L, d) - 1) * L ** (-s - 1), [0, 1, 5, 20, mp.inf])
    rhs = (2 * mp.pi**2) ** (-s / 2) * mp.gamma(s / 2) * Zd(s / 2, d) / 2
    rep(f"d={d}, s={mp.nstr(s,4)}: int_0^inf (N_F(L)-1) L^(-s-1) dL = {mp.nstr(I,16)};  (1/2)(2pi^2)^(-s/2)Gamma(s/2)Z_d(s/2) = {mp.nstr(rhs,16)}")

# 5. Epstein zeta of Z^d via Mellin of Theta^d - 1 and its functional equation
def LamE(w, d):  # pi^{-w} Gamma(w) Z_d(w), continued via theta transformation
    f = lambda u: Theta(u) ** d - 1
    return mp.quad(lambda u: f(u) * (u ** w + u ** (mp.mpf(d) / 2 - w)) / u, [1, 10, mp.inf]) - 1 / w - 1 / (mp.mpf(d) / 2 - w)
for d in [2, 4]:
    w = mp.mpc("0.8", "3.1")
    direct = mp.pi ** (-w) * mp.gamma(w) * Zd(w, d)
    rep(f"Epstein d={d}: Lambda_d(w) theta-continued={mp.nstr(LamE(w,d),14)}, factorized={mp.nstr(direct,14)}, Lambda_d(d/2-w)={mp.nstr(LamE(mp.mpf(d)/2-w,d),14)}")
# zeros of Z_4 off its symmetry line Re w = 1: zeta zeros at Re w = 1/2 and zeta(w-1) zeros at Re w = 3/2
rho = mp.zetazero(1)
for w in [rho, rho + 1]:
    rep(f"Z_4 at w={mp.nstr(w,12)}: theta-continued Lambda_4 = {mp.nstr(LamE(w,4),5)}  (zero; symmetry line of Z_4 is Re w = d/4 = 1)")

open("check_theta.out", "w").write("\n".join(out) + "\n")
