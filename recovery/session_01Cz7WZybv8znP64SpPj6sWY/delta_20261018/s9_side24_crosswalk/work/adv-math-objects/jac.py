# Pin-frame Jacobians and targets, both sources, re-derived from their literal definitions (1D axial block + transverse pairs)
import sympy as sp
r, b, k, kap, x = sp.symbols('r b k kappa x', positive=True)
fa, fxa, fc, fxc = sp.symbols('fa fxa fc fxc')   # f(M), f_x(M), f(S), f_x(S)
# P (3.1): O=(f(a),f_x(a),f(c),f_x(c)) -> (U0,U1,U2,U3)
U = sp.Matrix([(fa+fc)/2, (fc-fa)/r, (fxc-fxa)/r, 6/r**2*(fxa+fxc-2*(fc-fa)/r)])
TP = U.jacobian([fa, fxa, fc, fxc])
print('P axial |det| =', sp.simplify(abs(TP.det())), ' (P 3.2 says 12/r^4)')
# chain LS-DER-046 (3): X=(f(M),f(S),f_t(M),f_t(S)) -> (V+, V-corr, G_t+, G_t-)
Y = sp.Matrix([(fc+fa)/2, (fc-fa-r*(fxc+fxa)/2)/r**3, (fxc+fxa)/2, (fxc-fxa)/r])
TC = Y.jacobian([fa, fxa, fc, fxc])
print('chain axial |det| =', sp.simplify(abs(TC.det())), ' (046 (4): r^-3 * r^-1 for t-gradient pair = r^-4)')
print('transverse pair |det| both: (1/2,1/2;-1/r,1/r) ->', sp.simplify(sp.Matrix([[sp.Rational(1,2),sp.Rational(1,2)],[-1/r,1/r]]).det()))
# relation V-corr = -U3/12 identically
print('V-corr + U3/12 =', sp.simplify(Y[1] + U[3]/12))
# targets
subsP = {fa: b, fc: b - k*r**3, fxa: 0, fxc: 0}
subsC = {fa: b, fc: b - kap*r**3/6, fxa: 0, fxc: 0}
print('P target:', [sp.simplify(e.subs(subsP)) for e in U], ' (P 3.3: b-kr^3/2, -kr^2, 0, 12k)')
print('chain target:', [sp.simplify(e.subs(subsC)) for e in Y], ' (046 (5): b-kappa r^3/12, -kappa/6, 0, 0)')
# contact limits via Taylor at midpoint 0 with generic jet
c = sp.symbols('c0:6')
f = sum(c[j]*x**j/sp.factorial(j) for j in range(6))
fx = sp.diff(f, x)
sub = {fa: f.subs(x, -r/2), fc: f.subs(x, r/2), fxa: fx.subs(x, -r/2), fxc: fx.subs(x, r/2)}
print('U3 -> ', sp.limit(sp.simplify(U[3].subs(sub)), r, 0), ' ; V-corr ->', sp.limit(sp.simplify(Y[1].subs(sub)), r, 0), ' (c3 = f_ttt)')
# density relation at contact: p_X = |detT_C| p_Y = |detT_P| pi ; with p_Y(u=-kappa/6)=12*pi(U3=2kappa) since U3=-12u
print('P/chain full Jacobian ratio:', sp.simplify(abs(TP.det())/abs(TC.det())), '=> 12*pi_U3 = p_u, consistent with |dU3/du|=12')
