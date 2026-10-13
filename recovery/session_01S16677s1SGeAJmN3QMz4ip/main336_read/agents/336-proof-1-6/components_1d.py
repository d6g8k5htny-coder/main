# Section 6 axial-interval structure at levels between F(S) and F(M-), and just below F(S), via exact core formula; plus 2D union-find inside the core.
import numpy as np, instance as I
lam = I.lam0
FS = I.b + lam**2*I.Phi(I.z0); FM = I.b + lam**2*I.Phi(I.zm); FP = I.b + lam**2*I.Phi(I.zp)
Psi = lambda s: -s**4/4 + lam*s**2/2 + I.eta*lam**1.5*s
s = np.linspace(-I.r1/2, I.r1/2, 2_000_001)
def intervals(q):
    ok = Psi(s) >= q - I.b
    edges = np.diff(ok.astype(int)); starts = np.where(edges == 1)[0]+1; ends = np.where(edges == -1)[0]
    if ok[0]: starts = np.r_[0, starts]
    if ok[-1]: ends = np.r_[ends, len(s)-1]
    return [(s[a], s[b]) for a, b in zip(starts, ends)]
for frac in [0.01, 0.5, 0.99]:
    q = FS + frac*(FM - FS)
    iv = intervals(q); print(f"q=FS+{frac}(FM-FS): {len(iv)} axial intervals: {[(f'{a:+.3e}', f'{b:+.3e}') for a, b in iv]}  all within |s|<r1/4={I.r1/4:.3e}: {all(max(abs(a),abs(b)) < I.r1/4 for a,b in iv)}")
q = FS - 0.01*(FM - FS); iv = intervals(q); print(f"q just below FS: {len(iv)} interval(s) {[(f'{a:+.3e}', f'{b:+.3e}') for a, b in iv]}")
q = FM + 0.5*(FP - FM); iv = intervals(q); print(f"q between FM and FP: {len(iv)} interval(s) {[(f'{a:+.3e}', f'{b:+.3e}') for a, b in iv]}")
print(f"Psi(+-r1/2) = {Psi(I.r1/2):.4e}, {Psi(-I.r1/2):.4e}  vs q-b at FS = {FS-I.b:.4e} (must be below): {Psi(I.r1/2) < FS-I.b and Psi(-I.r1/2) < FS-I.b}")
