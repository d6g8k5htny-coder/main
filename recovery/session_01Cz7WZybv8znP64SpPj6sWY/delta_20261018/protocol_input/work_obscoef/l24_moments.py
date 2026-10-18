"""Independent high-precision evaluation of the ORIGINAL 1-D spectral law at L=24
(and comparison L values): z, a=mu2, b=mu4, c=mu6, kappa4, kappa6, beta, nu3,
componentwise condition numbers, and C_alg(2,L) by angular quadrature vs the
closed-form c_{2,inf} of ORIGINAL (32). Pure analysis of the stated formulas;
no field, sampler, driver or experiment."""
import sys
import mpmath as mp

mp.mp.dps = 400

def moments(L, mmax=None):
    h = 2*mp.pi/L
    al = h*h/2
    if mmax is None:
        mmax = int(mp.sqrt(mp.mpf(mp.mp.dps)*mp.log(10)*1.3/al)) + 10
    S = [mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    for m in range(1, mmax+1):
        w = 2*mp.e**(-al*m*m)
        q2 = (h*m)**2
        S[0] += w; S[1] += w*q2; S[2] += w*q2**2; S[3] += w*q2**3
    z = S[0]
    return z, S[1]/z, S[2]/z, S[3]/z, mmax

def poisson_cumulants(L, K=6):
    # phi(s) = exp(-s^2/2) * N(s)/N(0), N(s)=1+2 sum_k eps_k cosh(s k L), eps_k=exp(-k^2L^2/2)
    # log-moment generating: g(s)=log E exp(sQ)... use real s: E exp(sQ) = exp(s^2/2) N(-is)/N(0)?
    # Use mpmath Taylor of log of cumulant generating function in imaginary direction:
    # E e^{isQ} = e^{-s^2/2} sum_k exp(-k^2L^2/2) cos(s k L)/sum_k exp(-k^2L^2/2)   (Poisson)
    # Poisson: sum_m exp(-(hm)^2/2+ishm) = (sqrt(2pi)/h) exp(-s^2/2) sum_k exp(-k^2L^2/2) cosh(kLs)
    def logphi(s):
        num = 1 + 2*mp.fsum(mp.e**(-(k*L)**2/2)*mp.cosh(s*k*L) for k in range(1, K+1))
        den = 1 + 2*mp.fsum(mp.e**(-(k*L)**2/2) for k in range(1, K+1))
        return -s*s/2 + mp.log(num/den)
    t = mp.taylor(logphi, 0, 6)
    # log phi(s) = sum kappa_n (i s)^n / n!
    k2 = -t[2]*2
    k4 = t[4]*24
    k6 = -t[6]*720
    return k2, k4, k6

def calg2(a, b, c, nodes=None):
    k4 = b - 3*a*a
    k6 = c - 15*a*b + 30*a**3
    def integrand(th):
        e1, e2 = mp.cos(th), mp.sin(th)
        v1, v2 = -e2, e1
        B11 = a*a*(1+2*e1*e1) + k4*e1*e1
        B22 = a*a*(1+2*e2*e2) + k4*e2*e2
        B12 = a*a*2*e1*e2
        detB = B11*B22 - B12*B12
        D1 = e1*(a*a + k4*v1*v1)
        D2 = e2*(a*a + k4*v2*v2)
        # D^T B^{-1} D
        q = (B22*D1*D1 - 2*B12*D1*D2 + B11*D2*D2)/detB
        Gam = 3*a*a + k4*(v1**4 + v2**4) - q
        F = Gam/2
        s4 = e1**4 + e2**4
        s6 = e1**6 + e2**6
        V = 6*a**3 + 9*a*k4*s4 + (k6 - k4*k4/a)*s6
        return F*V**(mp.mpf(2)/3)/mp.sqrt(detB)
    I = mp.quad(integrand, mp.linspace(0, 2*mp.pi, 9))
    pref = mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi)*(2*mp.pi)**2*a)
    return pref*I

def c2inf():
    return 2/(3*mp.pi*mp.sqrt(3))*mp.cbrt(mp.mpf(3)/2)*mp.gamma(mp.mpf(7)/6)/mp.sqrt(mp.pi)

def report(L):
    z, a, b, c, mmax = moments(L)
    beta = b - a*a
    nu3 = c - b*b/a
    k4 = b - 3*a*a
    k6 = c - 15*a*b + 30*a**3
    cond_beta = (b + 2*a*a)/abs(beta)
    cond_nu3 = (c + 3*b*b/a)/abs(nu3)
    cond_k4 = (b + 6*a*a)/abs(k4)
    cond_k6 = (c + 15*a*b + 15*a*b + 90*a**3)/abs(k6)  # |dk6/dc|c+|dk6/db|b+|dk6/da|a with da-term 15b+90a^2 times a
    out = dict(L=L, mmax=mmax, z=z, a=a, b=b, c=c, beta=beta, nu3=nu3, k4=k4, k6=k6,
               log10_cond_beta=mp.log10(cond_beta), log10_cond_nu3=mp.log10(cond_nu3),
               log10_cond_k4=mp.log10(cond_k4), log10_cond_k6=mp.log10(cond_k6))
    return out

if __name__ == "__main__":
    Ls = [mp.mpf(x) for x in sys.argv[1:]] or [mp.mpf(24)]
    for L in Ls:
        r = report(L)
        print("L =", mp.nstr(L, 6), "mmax", r["mmax"])
        for k in ("z", "a", "b", "c", "beta", "nu3", "k4", "k6"):
            print("  %-5s %s" % (k, mp.nstr(r[k], 25)))
        print("  1-a           ", mp.nstr(1 - r["a"], 12))
        for k in ("log10_cond_beta", "log10_cond_nu3", "log10_cond_k4", "log10_cond_k6"):
            print("  %-16s %s" % (k, mp.nstr(r[k], 6)))
        k2p, k4p, k6p = poisson_cumulants(L)
        print("  Poisson-dual kappa2,kappa4,kappa6:", mp.nstr(k2p, 25), mp.nstr(k4p, 12), mp.nstr(k6p, 12))
        print("  rel diff a vs dual  ", mp.nstr(abs(k2p - r["a"])/r["a"], 5))
        if r["k4"] != 0:
            print("  rel diff k4 vs dual ", mp.nstr(abs(k4p - r["k4"])/abs(k4p), 5))
            print("  rel diff k6 vs dual ", mp.nstr(abs(k6p - r["k6"])/abs(k6p), 5))
        eps1 = mp.e**(-L*L/2)
        print("  eps1=exp(-L^2/2)", mp.nstr(eps1, 8), " 2 eps1 L^4", mp.nstr(2*eps1*L**4, 8), " -2 eps1 L^6", mp.nstr(-2*eps1*L**6, 8))
