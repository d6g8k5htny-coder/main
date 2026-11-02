"""Compare the binding's d=2 closed forms with the Math-#481 packet's reduction of ORIGINAL (PROOF.md lines 22-47)."""
import math, sys
sys.path.insert(0, '.')
from check_binding import route_b_parts, det

def moments(L, K):
    q = [2*math.pi*k/L for k in range(-K, K+1)]; w = [math.exp(-x*x/2) for x in q]; z = sum(w)
    return [sum(wi*x**p for x, wi in zip(q, w))/z for p in (2, 4, 6)]

for L, K in ((3.0, 8), (6.0, 14), (24.0, 60)):
    a, b, c = moments(L, K)
    worst = {'D': 0.0, 'V': 0.0, 'Gamma': 0.0}
    integ_binding = integ_481 = 0.0; n = 2880
    for i in range(n):
        th = 2*math.pi*i/n; e = (math.cos(th), math.sin(th)); U = [[-e[1]], [e[0]]]
        V, B, G, _ = route_b_parts(a, b, c, 2, e, U)
        t = e[0]**2 * e[1]**2
        D481 = a*a*b + (b - 3*a*a)*(b + a*a)*t
        V481 = c - b*b/a + 3*(b*b/a + 3*a*b - 3*a**3 - c)*t
        G481 = a*a*(b*b - a**4)/D481
        worst['D'] = max(worst['D'], abs(det(B)/D481 - 1)); worst['V'] = max(worst['V'], abs(V/V481 - 1)); worst['Gamma'] = max(worst['Gamma'], abs(G[0][0]/G481 - 1))
        integ_binding += (G[0][0]/2) * V**(2/3) / math.sqrt(det(B))
        integ_481 += V481**(2/3) / D481**1.5
    h = 2*math.pi/n
    C_binding = math.gamma(7/6)/(24**(1/3)*math.sqrt(math.pi)*(2*math.pi)**2*a) * integ_binding*h
    C_481 = math.gamma(7/6)*a*(b*b - a**4)/(8*24**(1/3)*math.pi**2.5) * integ_481*h
    print(f"L={L:>4}: kappa4={b-3*a*a:+.3e}  max rel diff D={worst['D']:.1e} V={worst['V']:.1e} Gamma={worst['Gamma']:.1e}  C_binding={C_binding:.15f}  C_481_reduction={C_481:.15f}  rel={C_binding/C_481-1:.1e}")
print("Math-#481 certified enclosure for C_alg(2,24): [0.07332163930293274928728086, 0.07349219930919442993700213]")
