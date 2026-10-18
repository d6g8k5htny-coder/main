# Illustrative only: inflation of n/ln(2J/alpha) from common (Q,r) vs per-bin (Q_j,r_j).
# Best case: T_j=0, H=beta=grid=0, EN_j=m_j. Tails: exact Poisson E(N-Q)_+ (oracle) or P13 with S2=m^2+m.
import math
eta = 0.05; J = 8
def pois_tail_table(m, Qmax):
    # sf[k] = P(N > k), k=0..Qmax ; tail(Q) = m*P(N>=Q) - Q*P(N>=Q+1) = m*sf[Q-1] - Q*sf[Q]
    pmf = [0.0]*(Qmax+2)
    # log-space pmf
    for k in range(Qmax+2):
        pmf[k] = math.exp(-m + k*math.log(m) - math.lgamma(k+1)) if m > 0 else (1.0 if k == 0 else 0.0)
    sf = [0.0]*(Qmax+2)
    acc = 0.0
    # compute sf from the top for accuracy: P(N>k) = 1 - cdf ; use complement sum of upper pmf tail
    upper = 0.0
    # approximate P(N > Qmax+1) by remaining mass
    total = sum(pmf)
    rem = max(0.0, 1.0 - total)
    upper = rem
    for k in range(Qmax+1, -1, -1):
        sf[k] = upper
        upper += pmf[k]
    def tail(Q):
        return max(m*sf[Q-1] - Q*sf[Q], 0.0) if Q >= 1 else m
    return tail
def solve(m0, route):
    ms = [m0*2.0**(2*j/3) for j in range(J)]
    if route == "p13":
        Qhi = int(max(80, 60*ms[-1] + 80))
        tails = [lambda Q, m=m: (m*m + m)/(4*Q) for m in ms]
    else:
        Qhi = int(max(80, 3*ms[-1] + 60*math.sqrt(ms[-1]) + 80))
        tails = [pois_tail_table(m, Qhi+1) for m in ms]
    per = []
    for m, tf in zip(ms, tails):
        best = math.inf
        for Q in range(1, Qhi+1):
            r = (eta*m - 2*tf(Q))/2
            if r > 0:
                v = (Q/r)**2/2
                if v < best: best = v
        per.append(best)
    bestc, Qc = math.inf, None
    for Q in range(1, Qhi+1):
        r = min((eta*m - 2*tf(Q))/2 for m, tf in zip(ms, tails))
        if r > 0:
            v = (Q/r)**2/2
            if v < bestc: bestc, Qc = v, Q
    return max(per), bestc, Qc
for route in ["poisson", "p13"]:
    for m0 in [0.5, 1, 3, 10, 100, 1000]:
        a, b, Qc = solve(m0, route)
        print(f"{route:8s} m0={m0:<6} per-bin n/ln={a:.4g}  common n/ln={b:.4g} (Q={Qc})  ratio={b/a:.4g}")
print("(m7/m0)^2 =", 2.0**(28/3))
