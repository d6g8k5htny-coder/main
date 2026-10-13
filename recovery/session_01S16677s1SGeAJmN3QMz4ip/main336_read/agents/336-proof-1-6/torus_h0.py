# Whole-torus superlevel H0 persistence of F_lambda (d=2) by union-find with the elder rule on a nonuniform rectilinear grid.
import numpy as np, instance as I, sys
lam = float(sys.argv[1]) if len(sys.argv) > 1 else I.lam0
def axis_grid(n_coarse=600, fine_lo=1e-8, per_decade=48):
    g = set(np.round(np.linspace(-I.L/2, I.L/2, n_coarse, endpoint=False), 12).tolist())
    fine = np.geomspace(fine_lo, I.L/4, int(per_decade*np.log10(I.L/4/fine_lo)))
    g |= set(np.round(fine, 15).tolist()) | set(np.round(-fine, 15).tolist()) | {0.0}
    return np.array(sorted(g))
y1 = axis_grid(); y2 = axis_grid()
n1, n2 = len(y1), len(y2)
Y1, Y2 = np.meshgrid(y1, y2, indexing='ij')
z1 = np.sqrt(2*I.w[0])*np.sin(I.om*Y1/2); z2 = np.sqrt(2*I.w[1])*np.sin(I.om*Y2/2)
chart = (np.abs(Y1) < I.L/4) & (np.abs(Y2) < I.L/4)
h = I.w[0]*np.cos(I.om*Y1) + I.w[1]*np.cos(I.om*Y2)
F = h.copy()
m = chart & (z1**2 + z2**2 < I.R**2)
F[m] = I.Flam(z1[m], z2[m], lam)
F = F.ravel(); N = F.size
print(f"lambda={lam:.4e} grid {n1}x{n2}={N} vertices; F range [{F.min():.6g},{F.max():.17g}] b={I.b:.17g}")
# edges: 8-connectivity with periodic wrap
idx = np.arange(N).reshape(n1, n2)
def edges(shift1, shift2):
    a = idx; bidx = np.roll(np.roll(idx, -shift1, axis=0), -shift2, axis=1)
    return a.ravel(), bidx.ravel()
E = [edges(1,0), edges(0,1), edges(1,1), edges(1,-1)]
eu = np.concatenate([e[0] for e in E]); ev = np.concatenate([e[1] for e in E])
emin = np.minimum(F[eu], F[ev])
order = np.argsort(-emin, kind='stable')
eu, ev, emin = eu[order], ev[order], emin[order]
parent = np.arange(N); rootval = F.copy()  # value of the maximum (elder) of each component
def find(x):
    r = x
    while parent[r] != r: r = parent[r]
    while parent[x] != r: parent[x], x = r, parent[x]
    return r
bars = []
for u, v, lev in zip(eu, ev, emin):
    ru, rv = find(u), find(v)
    if ru == rv: continue
    # elder rule: the component with the larger max survives; the younger dies at this level
    if rootval[ru] < rootval[rv]: ru, rv = rv, ru
    death = lev; birth = rootval[rv]
    parent[rv] = ru
    if birth - death > 0: bars.append((birth, death, birth - death))
bars.sort(key=lambda t: -t[2])
print(f"finite bars found: {len(bars)}; essential birth = {F.max():.17g}")
for bb in bars[:8]:
    print(f"   birth={bb[0]:.17g} death={bb[1]:.17g} lifetime={bb[2]:.6e}")
print(f"predicted: birth F(M-)={I.b+lam**2*I.Phi(I.zm):.17g} death F(S)={I.b+lam**2*I.Phi(I.z0):.17g} lifetime={I.kappa*lam**2:.6e}  F(M+)={I.b+lam**2*I.Phi(I.zp):.17g}")
