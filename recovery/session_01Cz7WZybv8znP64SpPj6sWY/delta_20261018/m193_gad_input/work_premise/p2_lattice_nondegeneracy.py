"""Premise audit, part 2: nondegeneracy for the ACTUAL full-lattice weights (3)-(4).

For several L (very coarse to fine lattice) and d=2..6, at high precision:
  * 1D moments a,b,c of the discrete Gaussian on hZ, h=2pi/L (full lattice, tail < 10^-dps);
  * beta=b-a^2, nu3=c-b^2/a, kappa4, kappa6, and the witness lower bounds
      a >= 2h^2 w1/z,  beta >= 2h^4 w1/z^2,  nu3 >= 72 h^6 w1 w2/(z(w1+4w2));
  * V(e) by (10) vs an INDEPENDENT Schur complement from multinomial moments,
    over axis / diagonal / mixed / random unit directions, and V(e)>=nu3/d^2;
  * lambda_min B(e) >= min(beta,2a^2)/2  and  lambda_min Gamma_e >= min(beta,2a^2)/2,
    Gamma_e built from (12)-(13) with a Householder transverse frame.
"""
import itertools, random
import mpmath as mp

random.seed(7)

def moments(L, dps):
    mp.mp.dps = dps
    h = 2*mp.pi/L
    lim = mp.sqrt(2*(dps*mp.log(10) + 60))/h
    K = int(lim) + 3
    ks = range(-K, K+1)
    w = {k: mp.e**(-(h*k)**2/2) for k in ks}
    z = mp.fsum(w.values())
    mu = lambda j: mp.fsum(((h*k)**(2*j))*w[k] for k in ks)/z
    return h, z, w, [mu(j) for j in range(5)]  # mu0..mu8

def unit(v):
    n = mp.sqrt(mp.fsum(x*x for x in v))
    return [x/n for x in v]

def householder_U(e):
    d = len(e)
    j = max(range(d), key=lambda i: abs(e[i]))
    sg = 1 if e[j] >= 0 else -1
    aj = [mp.mpf(sg*(i == j)) for i in range(d)]
    v = [aj[i] + e[i] for i in range(d)]
    vv = mp.fsum(x*x for x in v)
    R = mp.matrix(d, d)
    for i in range(d):
        for k in range(d):
            R[i, k] = 2*v[i]*v[k]/vv - (1 if i == k else 0)
    cols = [[R[i, k] for i in range(d)] for k in range(d) if k != j]
    return cols

def mineig(M):
    ev = mp.eigsy(M, eigvals_only=True)
    return min(ev)

results = []
for L, dps in [(0.3, 1600), (0.5, 700), (1.0, 120), (2.0, 80), (mp.pi, 80), (2*mp.pi, 80), (10.0, 80), (30.0, 80)]:
    h, z, w, mu = moments(mp.mpf(L), dps)
    a, b, c = mu[1], mu[2], mu[3]
    k4 = b - 3*a**2; k6 = c - 15*a*b + 30*a**3
    beta = b - a**2; nu3 = c - b**2/a
    w1 = w[1]; w2 = w[2]
    wit_a = 2*h**2*w1/z; wit_b = 2*h**4*w1/z**2; wit_n = 72*h**6*w1*w2/(z*(w1+4*w2))
    lamH = min(beta, 2*a**2)
    print('L=%s dps=%d: a=%s beta=%s nu3=%s k4=%s' % (mp.nstr(mp.mpf(L), 6), dps, mp.nstr(a, 6),
          mp.nstr(beta, 6), mp.nstr(nu3, 6), mp.nstr(k4, 6)))
    print('   witnesses hold: a>=%s %s, beta>=%s %s, nu3>=%s %s' % (
        mp.nstr(wit_a, 4), a >= wit_a, mp.nstr(wit_b, 4), beta >= wit_b, mp.nstr(wit_n, 4), nu3 >= wit_n))
    for d in range(2, 7):
        dirs = [[1] + [0]*(d-1), [1]*d, [1, 1] + [0]*(d-2), [3, 1] + [0]*(d-2), [1, 2, 3, 4, 5, 6][:d]]
        dirs += [[random.gauss(0, 1) for _ in range(d)] for _ in range(6)]
        worst = dict(V_over_floor=mp.inf, B_over=mp.inf, G_over=mp.inf, relerr=mp.mpf(0))
        for raw in dirs:
            e = unit([mp.mpf(x) for x in raw])
            p4 = mp.fsum(x**4 for x in e); p6 = mp.fsum(x**6 for x in e)
            V10 = 6*a**3 + 9*a*k4*p4 + (k6 - k4**2/a)*p6
            # independent: multinomial moments of (Q.e) via 1D moments
            def Emon(cnt):
                r = mp.mpf(1)
                for n in cnt:
                    if n % 2:
                        return mp.mpf(0)
                    r *= mu[n//2]
                return r
            E6 = mp.mpf(0)
            for cnt in itertools.product(range(7), repeat=d):
                if sum(cnt) != 6 or any(n % 2 for n in cnt):
                    continue
                coef = mp.factorial(6)
                for n, x in zip(cnt, e):
                    coef = coef/mp.factorial(n)*x**n
                E6 += coef*Emon(cnt)
            v = []
            for i in range(d):
                s = mp.mpf(0)
                for cnt in itertools.product(range(4), repeat=d):
                    if sum(cnt) != 3:
                        continue
                    coef = mp.factorial(3)
                    for n, x in zip(cnt, e):
                        coef = coef/mp.factorial(n)*x**n
                    cc = list(cnt); cc[i] += 1
                    s += coef*Emon(cc)
                v.append(s)
            Vs = E6 - mp.fsum(x*x for x in v)/a
            worst['relerr'] = max(worst['relerr'], abs(Vs - V10)/abs(Vs))
            worst['V_over_floor'] = min(worst['V_over_floor'], Vs/(nu3/d**2))
            # B(e)
            B = mp.matrix(d, d)
            for i in range(d):
                for j in range(d):
                    B[i, j] = a**2*((i == j) + 2*e[i]*e[j]) + (k4*e[i]**2 if i == j else 0)
            worst['B_over'] = min(worst['B_over'], mineig(B)/(lamH/2))
            # Gamma_e via (12)-(13)
            U = householder_U(e)
            n = d-1
            pairs = [(x, y) for x in range(n) for y in range(x, n)]
            R = mp.matrix(len(pairs), d); C = mp.matrix(len(pairs), len(pairs))
            for P, (x, y) in enumerate(pairs):
                for i in range(d):
                    R[P, i] = a**2*(x == y)*e[i] + k4*U[x][i]*U[y][i]*e[i]
                for Qi, (s_, t_) in enumerate(pairs):
                    iso = a**2*((x == y)*(s_ == t_) + (x == s_)*(y == t_) + (x == t_)*(y == s_))
                    C[P, Qi] = iso + k4*mp.fsum(U[x][i]*U[y][i]*U[s_][i]*U[t_][i] for i in range(d))
            G = C - R*mp.inverse(B)*R.T
            G = (G + G.T)/2
            worst['G_over'] = min(worst['G_over'], mineig(G)/(lamH/2))
        ok = worst['V_over_floor'] >= 1 and worst['B_over'] >= 1 - mp.mpf(10)**(-20) and worst['G_over'] >= 1 - mp.mpf(10)**(-20)
        print('   d=%d: min V/(nu3/d^2)=%s  max rel|V10-Vschur|=%s  min lamB/(lamH/2)=%s  min lamGamma/(lamH/2)=%s  OK=%s'
              % (d, mp.nstr(worst['V_over_floor'], 6), mp.nstr(worst['relerr'], 3),
                 mp.nstr(worst['B_over'], 6), mp.nstr(worst['G_over'], 6), ok))
        results.append(ok)
print('ALL OK:', all(results))
