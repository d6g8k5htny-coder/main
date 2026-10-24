"""Adversarial math re-derivation: FlatSaddle Tail Lemma vs P2 (d8acf0bc).
Independent of the lens script. Read-only; prints to stdout."""
import sympy as sp

out = []
def rec(k, v):
    out.append((k, v))

r, kap, mu, nu, z_t, z_s, eps, m = sp.symbols('r kappa mu nu z_t z_s epsilon m', real=True)
Tttt, Ttts, Ttss, Tsss = sp.symbols('T_ttt T_tts T_tss T_sss', real=True)

# ---------- A. det H_S with the (F1) relative error kept ----------
# f_tt(S) = kappa r (1 + eps r), f_ts(S) = r nu, f_ss(S) = -mu
a = kap*r*(1 + eps*r); b = r*nu; c = -mu
detHS = a*c - b**2
mut = mu + r*nu**2/kap                     # carrier's mu-tilde
muhat = -(c - b**2/a)                      # exact Schur variable
rec('A1 detHS == -f_tt(S)*muhat exactly', sp.simplify(detHS - (-a*muhat)) == 0)
# carrier claim: detHS = -kappa r mut (1+O(r)). On the tail set mut = m r^2:
err = sp.simplify(detHS - (-kap*r*mut*(1 + eps*r)))
err_tail = sp.series(sp.simplify(err.subs(mu, m*r**2 - r*nu**2/kap)), r, 0, 4).removeO()
rec('A2 detHS - [-kap r mut (1+eps r)] on tail set (leading)', sp.simplify(err_tail))
rec('A3 compare: kap r mut on tail set', sp.simplify(kap*r*m*r**2))
rec('A4 muhat - mut on tail set (leading)',
    sp.simplify(sp.series((muhat - mut).subs(mu, m*r**2 - r*nu**2/kap), r, 0, 3).removeO()))

# ---------- B. Third-point model with cubic Taylor terms ----------
# local field at S: f(S+y) = 1/2 y^T H y + 1/6 T[y,y,y], H = [[A,B],[B,Cc]]
A_, B_, C_ = sp.symbols('A B C', real=True)
yt, ys = sp.symbols('y_t y_s', real=True)
f = sp.Rational(1, 2)*(A_*yt**2 + 2*B_*yt*ys + C_*ys**2) + sp.Rational(1, 6)*(
    Tttt*yt**3 + 3*Ttts*yt**2*ys + 3*Ttss*yt*ys**2 + Tsss*ys**3)
ft, fs = sp.diff(f, yt), sp.diff(f, ys)
Hy = sp.Matrix([[sp.diff(f, yt, 2), sp.diff(f, yt, ys)], [sp.diff(f, yt, ys), sp.diff(f, ys, 2)]])
subsH = {A_: kap*r, B_: r*nu, C_: -(m*r**2) + (r*nu)**2/(kap*r)}   # muhat = m r^2 exactly
# B1: f_s mean on the linear null line, quadratic terms included
line = {yt: -B_*ys/A_}
fs_line = sp.expand(fs.subs(line).subs(subsH).subs(ys, r**2*z_s))
rec('B1 f_s on linear null line, y_s=r^2 z_s (collect r)', sp.collect(sp.expand(fs_line), r))
rec('B1b retained -r^2 z_s muhat term order', sp.expand(-r**2*z_s*m*r**2))
# B2: solve grad f(y)=0 with y=r^2 z: f_t=0 for z_t (perturbatively), f_s=0 for T_sss
zt1 = sp.symbols('zt1', real=True)
ztry = -nu*z_s/kap + r*zt1
ft_e = sp.expand(ft.subs({yt: r**2*ztry, ys: r**2*z_s}).subs(subsH)/r**3)
zt1_sol = sp.solve(sp.series(ft_e, r, 0, 2).removeO().coeff(r, 1), zt1)[0]
rec('B2 null-line shift z_t = -nu z_s/kap + r*zt1, zt1', sp.simplify(zt1_sol))
ysub = {yt: r**2*(-nu*z_s/kap + r*zt1_sol), ys: r**2*z_s}
fs_e = sp.expand(fs.subs(ysub).subs(subsH))
Tsss_sol = sp.solve(sp.expand(sp.series(fs_e/r**4, r, 0, 1).removeO()), Tsss)[0]
rec('B2 T_sss pinned by f_s(y)=0 (leading)', sp.simplify(Tsss_sol))
detHy = sp.expand(Hy.det().subs(ysub).subs(subsH).subs(Tsss, Tsss_sol))
detHy_lead = sp.simplify(sp.series(detHy, r, 0, 4).removeO())
rec('B3 det H(y) leading after Kac-Rice pinning', sp.factor(detHy_lead))
rec('B3b det H_S (exact, model)', sp.factor(sp.simplify(Hy.det().subs({yt: 0, ys: 0}).subs(subsH))))
# carrier's xi with T_sss treated as free:
xi_car = -(z_t*Ttss + z_s*Tsss)
rec('B4 carrier mut + r^2 xi with pinned T_sss, on line (leading /r^2)',
    sp.simplify(sp.series(((m*r**2) + r**2*xi_car.subs(z_t, -nu*z_s/kap).subs(Tsss, Tsss_sol))/r**2, r, 0, 1).removeO()))
# B5: value offset at the pinned third point (1D fold check with nu=0, other thirds 0)
off = f.subs(ysub).subs(subsH).subs(Tsss, Tsss_sol)
off0 = sp.simplify(sp.series(off.subs({nu: 0, Tttt: 0, Ttts: 0, Ttss: 0}), r, 0, 7).removeO())
rec('B5 offset f(y)-f(S), nu=0 & other thirds 0', sp.factor(off0))
off_full = sp.simplify(sp.series(off, r, 0, 7).removeO())
rec('B5b offset general (leading) / r^6', sp.factor(sp.simplify(off_full/r**6)))
# quadratic part only (lens claim -1/2 r^4 z_s^2 muhat)
q2 = sp.Rational(1, 2)*(A_*yt**2 + 2*B_*yt*ys + C_*ys**2)
rec('B5c quadratic part on linear line', sp.factor(sp.simplify(q2.subs(line).subs(subsH).subs(ys, r**2*z_s))))
# generic value offset (mu ~ 1, not tail): order r^4
q2gen = q2.subs({A_: kap*r, B_: r*nu, C_: -mu, yt: r**2*z_t, ys: r**2*z_s})
rec('B6 generic value offset (mu~1) lowest r power', sp.Poly(sp.expand(q2gen), r).monoms()[-1])

# ---------- C. Var(f_sss | f_s, f_tts) and others, unit squared exponential ----------
t, s = sp.symbols('t s', real=True)
K = sp.exp(-(t**2 + s**2)/2)
def cov(al, be):
    return sp.simplify(((-1)**(be[0]+be[1])*sp.diff(K, t, al[0]+be[0], s, al[1]+be[1])).subs({t: 0, s: 0}))
def condvar(target, given):
    S = sp.Matrix([[cov(i, j) for j in given] for i in given])
    cv = sp.Matrix([cov(target, j) for j in given])
    return sp.simplify(cov(target, target) - (cv.T*S.inv()*cv)[0])
rec('C1 Var(f_sss|f_s,f_tts)', condvar((0, 3), [(0, 1), (2, 1)]))
rec('C2 Var(f_sss|f_s,f_ts,f_tts,f_ttts)', condvar((0, 3), [(0, 1), (1, 1), (2, 1), (3, 1)]))
rec('C3 Var(f_sss|f_s,f_tts,f_tttts)', condvar((0, 3), [(0, 1), (2, 1), (4, 1)]))

# ---------- D. Ledgers ----------
# carrier: area r^4, T-integral r^-3, S-block r^4, M-factor r^2, denominator r^-2
rec('D1 carrier ledger', 4 - 3 + 4 + 2 - 2)
# carrier S-block via correct factors: dmu (r^2 z_s), |det H_S| (kap r * r^2 z_s), |det H(y)| (kap r * r^2 z_s), sd_s^-1 (r^-4 z_s^-2)
cs = sp.symbols('c_s', positive=True)
zsp = sp.symbols('zeta_s', positive=True)
Sblk = (cs*zsp*r**2)*(kap*r*r**2*zsp)*(kap*r*r**2*zsp)/(cs*zsp**2*r**4)
rec('D2 S-block scaling', sp.simplify(Sblk))
# P2 region I in nested disk: X = M + r(p,q), p = r Zt, q = r Zs
Zt, Zs = sp.symbols('Z_t Z_s', positive=True)
p_, q_ = r*Zt, r*Zs
Dl2 = q_**2 + r**2*p_**2
rhoI = r**-2 * r**-3/Dl2 * r**6*q_**2*(1 + (p_/q_)**2)**3
rec('D3 P2 rho_I nested (no Gaussian)', sp.simplify(rhoI))
rec('D4 P2 nested count power', sp.simplify(sp.powsimp(r**-2*r**-3*r**-2*r**8*r**4)))
# P2 region II
rec('D5 P2 region II ledger', -2 - 3 + 3 - 2 - 6)
cpos = sp.symbols('c', positive=True)
rec('D6 6!(2 r^2/c)^6', sp.expand(sp.factorial(6)*(2*r**2/cpos)**6))
# gamma family: tau = r^g
g = sp.symbols('gamma', positive=True)
tau = r**g
led = tau**2 * (kap*r*tau)**-1 * (kap**2*r**2*tau) * (kap*r**2) / (kap**2*r**2)
rec('D7 carrier ledger at tau=r^gamma', sp.powsimp(sp.simplify(led), force=True))
# P2 per-factor at d = tau
L, sig = sp.symbols('L sigma', positive=True)
rec('D8 P2 product at d=tau', sp.simplify((L**2*r*tau/(4*sig))*(4*L**2*r**2/sig**2)*(L**2*r*tau/(2*sig))))
# literal l.140
rec('D9 literal l.140 product', sp.simplify(kap**2*r**2*(cs*zsp*r**2)*(r**2)*(cs*zsp)/(cs*zsp**2*r**4)*(cs*zsp*r**2)))
rec('D10 l.159-163 literal', 4 - 3 + 2 + 2 + 1 - 2)

for k_, v_ in out:
    print(f'{k_}: {v_}')
