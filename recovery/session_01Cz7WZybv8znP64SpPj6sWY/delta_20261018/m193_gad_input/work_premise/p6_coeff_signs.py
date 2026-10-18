"""Signs of the individual coefficients in (10) for actual lattice weights: positivity of V(e)
is NOT termwise in (10) (p6-coefficient kappa6-kappa4^2/a goes negative at small L) but is
termwise in the decomposition nu3*S6 + 9a*beta*sum_{i!=j} e_i^4 e_j^2 + 36a^3*e3(e^2)."""
import mpmath as mp
for L, dps in [(0.3, 1600), (0.7, 400), (1.0, 120), (1.5, 80), (2.0, 80), (3.0, 80), (6.283185307179586, 80), (10.0, 80)]:
    mp.mp.dps = dps
    h = 2*mp.pi/mp.mpf(L)
    K = int(mp.sqrt(2*(dps*mp.log(10)+60))/h) + 3
    w = [mp.e**(-(h*k)**2/2) for k in range(-K, K+1)]
    z = mp.fsum(w)
    mu = [mp.fsum(((h*k)**(2*j))*w[k+K] for k in range(-K, K+1))/z for j in range(4)]
    a, b, c = mu[1], mu[2], mu[3]
    k4 = b-3*a**2; k6 = c-15*a*b+30*a**3
    print('L=%-6s 6a^3=%s  9a*k4=%s  k6-k4^2/a=%s | nu3=%s 9a*beta=%s 36a^3=%s' % (
        mp.nstr(mp.mpf(L), 4), mp.nstr(6*a**3, 3), mp.nstr(9*a*k4, 3), mp.nstr(k6-k4**2/a, 3),
        mp.nstr(c-b**2/a, 3), mp.nstr(9*a*(b-a**2), 3), mp.nstr(36*a**3, 3)))
