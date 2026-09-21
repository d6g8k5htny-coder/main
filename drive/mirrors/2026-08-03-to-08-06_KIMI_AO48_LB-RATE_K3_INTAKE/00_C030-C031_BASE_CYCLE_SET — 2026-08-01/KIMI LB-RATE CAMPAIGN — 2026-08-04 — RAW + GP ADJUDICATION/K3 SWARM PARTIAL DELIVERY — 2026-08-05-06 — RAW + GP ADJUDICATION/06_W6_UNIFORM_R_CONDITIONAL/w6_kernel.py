# w6_kernel.py -- W6 uniform-in-r kernel + Gaussian conditioning machinery.
# Model (K-B convention, per W2_DERIVATION C3): normalized periodized Bargmann-Fock field
# on T^2_24, Var f = 1, K(x) = K1(x1) K1(x2),
#   K1(s) = (1/Z) sum_j exp(-k_j^2/2) exp(i k_j s),  k_j = pi j / 12,  Z = sum_j exp(-k_j^2/2).
# Wrapped dual (Poisson):  K1(s) = sum_n exp(-(s+24n)^2/2) / (1 + 2 e^{-288} + ...).
# Derivatives: K1^{(a)}(s) = sum_n (-1)^a H_a(s+24n) e^{-(s+24n)^2/2}  (probabilists' Hermite),
# with wrap error < 4 e^{-288} for all s (relative), and image-truncation error controlled.
# All arithmetic: mpmath, dps set by caller. No randomness. Deterministic.
import mpmath as mp

PI = mp.pi

def set_dps(dps):
    mp.mp.dps = dps

def Zspec(J=64):
    """Spectral normalizer Z = sum_{j=-J}^{J} e^{-(pi j/12)^2/2}; tail < 1e-300 at J=64."""
    return mp.nsum(lambda j: mp.e**(-(PI*j/12)**2/2), [-J, J])

# probabilists' Hermite polynomials He_n(x) by recurrence
def hermite_he(n, x):
    if n == 0:
        return mp.mpf(1)
    if n == 1:
        return x
    h0, h1 = mp.mpf(1), x
    for k in range(1, n):
        h0, h1 = h1, x*h1 - k*h0
    return h1

def K1_wrapped(s, a=0, nim=3):
    """K1^{(a)}(s) via wrapped images: sum_n (-1)^a He_a(s+24n) e^{-(s+24n)^2/2}.
    Images |n|<=nim; error < 2 e^{-(24*(nim+1)-|s|)^2/2}. For |s|<=24*nim+12 tiny."""
    s = mp.mpf(s)
    tot = mp.mpf(0)
    sgn = (-1)**a
    for n in range(-nim, nim+1):
        u = s + 24*n
        tot += hermite_he(a, u) * mp.e**(-u*u/2)
    return sgn*tot

def K1_spec(s, a=0, J=None):
    """K1^{(a)}(s) via spectral sum (cross-check)."""
    if J is None:
        J = 64
    Z = Zspec(J)
    tot = mp.mpf(0)
    for j in range(-J, J+1):
        kj = PI*j/12
        tot += mp.e**(-kj*kj/2) * (1j*kj)**a * mp.e**(1j*kj*s)
    return mp.re(tot/Z)

def K1d(s, a):
    """1D kernel derivative order a at s (wrapped dual is primary)."""
    return K1_wrapped(s, a)

def K2d(s1, s2, a1, a2):
    """d^alpha K(s) = K1^{(a1)}(s1) K1^{(a2)}(s2)."""
    return K1d(s1, a1)*K1d(s2, a2)

def cov_der(s1, s2, al, be):
    """Cov( d^al f(x), d^be f(y) ) = (-1)^|be| d^{al+be} K(s),  s = x - y."""
    return (-1)**(be[0]+be[1]) * K2d(s1, s2, al[0]+be[0], al[1]+be[1])

# ---------------- pins ----------------
# Stations at rung r:  M = (-r/2, 0), S = (r/2, 0), Y = M + r*(-0.76, 0.24) = r*(-1.26, 0.24)
D_YOFF = (mp.mpf('-0.76'), mp.mpf('0.24'))  # (Y - M)/r
D_ETA = (mp.mpf('-1.26'), mp.mpf('0.24'))   # Y/r
BB = mp.mpf(6)/5                            # b = 6/5

def stations(r):
    r = mp.mpf(r)
    M = (-r/2, mp.mpf(0)); S = (r/2, mp.mpf(0))
    Y = (M[0] + r*D_YOFF[0], M[1] + r*D_YOFF[1])
    return M, S, Y

def pin_list(r):
    """9 pins: (f, d1f, d2f) at M, S, Y. Returns list of (station, multiindex)."""
    M, S, Y = stations(r)
    pins = []
    for st in (M, S, Y):
        pins.append((st, (0,0)))
        pins.append((st, (1,0)))
        pins.append((st, (0,1)))
    return pins

def pin_values(r, mu_t):
    r = mp.mpf(r); ell = r**3/6
    return [BB, mp.mpf(0), mp.mpf(0),
            BB - ell, mp.mpf(0), mp.mpf(0),
            mp.mpf(mu_t), mp.mpf(0), mp.mpf(0)]

def gram(pins):
    n = len(pins)
    G = mp.zeros(n, n)
    for i in range(n):
        (si, ai) = pins[i]
        for j in range(i, n):
            (sj, aj) = pins[j]
            G[i,j] = cov_der(si[0]-sj[0], si[1]-sj[1], ai, aj)
            G[j,i] = G[i,j]
    return G

def cross_cov(pins, y, target_al):
    """c_j = Cov(pin_j, d^{target_al} f(y))."""
    c = mp.zeros(len(pins), 1)
    for j, (sj, aj) in enumerate(pins):
        c[j] = cov_der(sj[0]-y[0], sj[1]-y[1], aj, target_al)
    return c

def cond_mean_cov(pins, vals, y, targets):
    """Conditional mean vector and covariance of the derivatives `targets` at y given pins.
    targets: list of multiindices. Returns (mean vector, cov matrix)."""
    G = gram(pins)
    Gi = G**-1
    v = mp.zeros(len(pins), 1)
    for i, vi in enumerate(vals):
        v[i] = vi
    nt = len(targets)
    Szz = mp.zeros(nt, nt)
    C = mp.zeros(len(pins), nt)
    for a, ta in enumerate(targets):
        for b, tb in enumerate(targets):
            Szz[a,b] = cov_der(0, 0, ta, tb) if a <= b else Szz[b,a]
        C[:, a] = cross_cov(pins, y, ta)
    for a in range(nt):
        for b in range(a+1, nt):
            Szz[b,a] = Szz[a,b]
    # Cov(d^ta f(y), pin_j): note cross_cov uses s = station - y; symmetric anyway.
    mean = (C.transpose()*Gi)*v
    cov = Szz - (C.transpose()*Gi)*C
    return mean, cov, G, Gi

def mu_t(r):
    """mu_t(r) = E[f(Y) | 6 pins at M,S and grad f(Y) = 0]  (8 functionals)."""
    M, S, Y = stations(r)
    pins = []
    for st in (M, S):
        pins.append((st, (0,0))); pins.append((st, (1,0))); pins.append((st, (0,1)))
    pins.append((Y, (1,0))); pins.append((Y, (0,1)))
    r = mp.mpf(r); ell = r**3/6
    vals = [BB, 0, 0, BB-ell, 0, 0, 0, 0]
    G = gram(pins); Gi = G**-1
    v = mp.zeros(8,1)
    for i,vi in enumerate(vals): v[i]=vi
    c = cross_cov(pins, Y, (0,0))
    return (c.transpose()*Gi*v)[0]
