from fractions import Fraction as Fr

def bars(vals, sup=True):
    """Elder-rule H0 barcode of a piecewise-monotone path given by the values at its
    critical points and endpoints (in order). Returns finite bars (birth, death) and essential."""
    v = [x if sup else -x for x in vals]
    n = len(v)
    # sort indices by decreasing value (superlevel filtration)
    order = sorted(range(n), key=lambda i: -v[i])
    comp = {}
    parent = {}
    birth = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    out = []
    for i in order:
        parent[i] = i; birth[i] = v[i]
        for j in (i - 1, i + 1):
            if j in parent:
                a, b = find(i), find(j)
                if a == b: continue
                # elder survives
                young, old = (a, b) if birth[a] < birth[b] else (b, a)
                if birth[young] != v[i]:
                    out.append((birth[young], v[i]))
                parent[young] = old
    if not sup:
        out = [(-b, -d) for b, d in out]
    return out

F = lambda z: z**3 / 3 - z
L = Fr(3, 2)
pts = [-L, Fr(-1), Fr(1), L]
vals = [F(z) for z in pts]
print("values", vals, "gap", F(-1) - F(1))
print("superlevel finite bars", [(b, d, abs(b - d)) for b, d in bars(vals, True)])
print("sublevel finite bars", [(b, d, abs(b - d)) for b, d in bars(vals, False)])
L3 = Fr(3)
vals3 = [F(z) for z in [-L3, Fr(-1), Fr(1), L3]]
print("L=3 super", [(b, d, abs(b - d)) for b, d in bars(vals3, True)], "sub", [(b, d, abs(b - d)) for b, d in bars(vals3, False)])
# triple collision v3 = x^3 - 6 x tau, F = x^4/4 - 3 tau x^2
tau = Fr(1, 7)
F3 = lambda x: x**4 / 4 - 3 * tau * x**2
# critical points 0, +-sqrt(6 tau); use r^2 = 6 tau
r2 = 6 * tau
print("triple: F(0)-F(r) =", F3(0) - (r2**2 / 4 - 3 * tau * r2), "9tau^2 =", 9 * tau**2, "r^4/4 =", r2**2 / 4)
# information loss
for Lx in (Fr(1), Fr(2), Fr(5, 3)):
    G1 = lambda z: (1 + Lx**2 / 3) * z
    G2 = lambda z: z**3 / 3 + z
    print("L", Lx, "endpoints", G1(-Lx), G1(Lx), G2(-Lx), G2(Lx))
