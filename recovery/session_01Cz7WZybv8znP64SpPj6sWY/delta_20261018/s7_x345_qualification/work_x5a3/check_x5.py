import sympy as sp

r = sp.symbols('r', positive=True)
k, b = sp.symbols('k b', real=True)
out = []

# --- Congruence: X5 reading D^{-1} H D^{-1}, D = diag(sqrt r, I) vs E1's D_r = diag(r^-1/2, I) vs literal D H D
for m in (1, 2, 3):
    al = sp.symbols('al', real=True)
    be = sp.Matrix(sp.symbols(f'be1:{m+1}', real=True))
    As = sp.symbols(f'a1:{m*(m+1)//2+1}', real=True)
    A = sp.zeros(m, m); idx = 0
    for i in range(m):
        for j in range(i, m):
            A[i, j] = A[j, i] = As[idx]; idx += 1
    H = sp.BlockMatrix([[sp.Matrix([[r*al]]), r*be.T], [r*be, A]]).as_explicit()
    D = sp.diag(sp.sqrt(r), *([1]*m))
    Dr = sp.diag(1/sp.sqrt(r), *([1]*m))
    X5mat = sp.simplify(D.inv()*H*D.inv())
    E1mat = sp.simplify(Dr*H*Dr)
    lit = sp.simplify(D*H*D)
    target = sp.BlockMatrix([[sp.Matrix([[al]]), sp.sqrt(r)*be.T], [sp.sqrt(r)*be, A]]).as_explicit()
    out.append((f'm={m} X5 D^-1 H D^-1 == E1 D_r H D_r', sp.simplify(X5mat-E1mat) == sp.zeros(m+1, m+1)))
    out.append((f'm={m} X5 matrix == [[al, sqrt r be^T],[sqrt r be, A]]', sp.simplify(X5mat-target) == sp.zeros(m+1, m+1)))
    out.append((f'm={m} det(X5 matrix) == det H / r', sp.simplify(X5mat.det() - H.det()/r) == 0))
    out.append((f'm={m} literal D H D top-left, off-diag', (sp.simplify(lit[0, 0]), sp.simplify(lit[0, 1]))))
    out.append((f'm={m} det(D H D) / det H', sp.simplify(lit.det()/H.det())))
    # (5.3) Schur/adjugate identity, polynomial (no inverse)
    lhs = sp.expand(H.det()/r)
    adjA = A.adjugate() if m > 1 else sp.Matrix([[1]])
    rhs = sp.expand(al*A.det() - r*(be.T*adjA*be)[0, 0])
    out.append((f'm={m} (5.3) det H/r == al detA - r be^T adj(A) be', sp.simplify(lhs-rhs) == 0))

# --- (5.1)/(5.2) integral identities for a generic smooth g(u)=f(a+u) on [0,r]
u, s = sp.symbols('u s', real=True)
g = sp.Function('g')
# generic quintic satisfying pins g'(0)=g'(r)=0, g(r)-g(0)=-k r^3
c = sp.symbols('c0:6', real=True)
G = sum(c[i]*u**i for i in range(6))
sol = sp.solve([sp.diff(G, u).subs(u, 0), sp.diff(G, u).subs(u, r), (G.subs(u, r)-G.subs(u, 0)) + k*r**3], [c[1], c[2], c[3]], dict=True)[0]
Gp = G.subs(sol)
g1, g2, g3 = [sp.diff(Gp, u, n) for n in (1, 2, 3)]
out.append(('int_0^r g2 == 0 (axial pins)', sp.simplify(sp.integrate(g2, (u, 0, r))) == 0))
dbl = sp.integrate(sp.integrate(g3, (u, 0, s)), (s, 0, r))
out.append(('X5 N1: int_0^r int_0^s g3 == -r g2(0)', sp.simplify(dbl + r*g2.subs(u, 0)) == 0))
out.append(('X5 N2: int u(r-u) g3 == 2 k r^3', sp.simplify(sp.integrate(u*(r-u)*g3, (u, 0, r)) - 2*k*r**3) == 0))
out.append(('weight int u(r-u) == r^3/6', sp.simplify(sp.integrate(u*(r-u), (u, 0, r)) - r**3/6) == 0))
out.append(('X5 N2: g2(0) == -(1/r) int (r-u) g3', sp.simplify(g2.subs(u, 0) + sp.integrate((r-u)*g3, (u, 0, r))/r) == 0))
out.append(('right endpoint: g2(r) == (1/r) int u g3', sp.simplify(g2.subs(u, r) - sp.integrate(u*g3, (u, 0, r))/r) == 0))
# with g3 == 12k + eta: main terms
out.append(('main term left: -(1/r) int (r-u) 12k == -6kr', sp.simplify(-sp.integrate((r-u)*12*k, (u, 0, r))/r + 6*k*r) == 0))
out.append(('main term right: (1/r) int u 12k == 6kr', sp.simplify(sp.integrate(u*12*k, (u, 0, r))/r - 6*k*r) == 0))
out.append(('error weights: (1/r) int (r-u) * r == r^2/2', sp.simplify(sp.integrate((r-u)*r, (u, 0, r))/r - r**2/2) == 0))
# U3 = 12k at the pins (3.1)
fa, fc = b, b - k*r**3
U3 = sp.Rational(6)/r**2*(0 + 0 - 2*(fc-fa)/r)
out.append(('(3.1) U3 at pins == 12k', sp.simplify(U3 - 12*k) == 0))

# --- limit type: diag(-6k, A0) and diag(6k, A0); product of |det| = 36 k^2 det(A0)^2
for name, val in out:
    print(f'{name}: {val}')
