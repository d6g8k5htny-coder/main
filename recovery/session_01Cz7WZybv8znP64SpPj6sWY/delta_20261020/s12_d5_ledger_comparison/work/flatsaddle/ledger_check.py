"""FlatSaddle Tail Lemma vs P2 (d8acf0bc): identity and power-ledger re-derivation.
Read-only support computation; prints results to stdout."""
import sympy as sp

r, kap, mu, nuS, nuM, eta, zt, zs = sp.symbols('r kappa mu nu_S nu_M eta z_t z_s', real=True)
fttt, ftts, ftss, fsss = sp.symbols('f_ttt f_tts f_tss f_sss', real=True)
out = {}

# --- 1. det H_S and corrected curvature (carrier l.79-80) ---
HS = sp.Matrix([[kap*r, r*nuS], [r*nuS, -mu]])
mutS = mu + r*nuS**2/kap
out['1_detHS_eq_-kap_r_mutS'] = sp.simplify(HS.det() - (-kap*r*mutS)) == 0

# --- 2. f_s mean on the t-null line equals -r^2 z_s mutS (carrier l.131-132) ---
zt_star = -nuS*zs/kap
m_fs = r**2*zt_star*(r*nuS) + r**2*zs*(-mu)
out['2_fs_mean_identity'] = sp.simplify(m_fs - (-r**2*zs*mutS)) == 0
# f_t mean vanishes on the line (carrier l.119, l.125)
m_ft = r**3*(kap*zt + nuS*zs)
out['2b_ft_mean_zero_on_line'] = sp.simplify(m_ft.subs(zt, zt_star)) == 0

# --- 3. det H_M and eta-tilde sign (carrier l.81-83) ---
HM = sp.Matrix([[-kap*r, r*nuM], [r*nuM, -mu - r*eta]])
mutM = sp.expand(HM.det()/(kap*r))
diff = sp.simplify((mutM - mutS)/r)
out['3_mutM'] = str(sp.expand(mutM))
out['3_(mutM-mutS)/r_correct'] = str(sp.expand(diff))
carrier_etatilde = eta - nuM**2/kap + nuS**2/kap
out['3_carrier_etatilde_matches'] = sp.simplify(diff - carrier_etatilde) == 0
out['3_discrepancy_(correct-carrier)'] = str(sp.simplify(diff - carrier_etatilde))

# --- 4. value offset on the line (carrier l.148-150) ---
y_t, y_s = r**2*zt_star, r**2*zs
quad = sp.Rational(1, 2)*(kap*r*y_t**2 + 2*r*nuS*y_t*y_s - mu*y_s**2)
out['4_quadratic_offset_on_line'] = str(sp.factor(sp.simplify(quad)))
out['4_quad_equals_-half_r4_zs2_mutS'] = sp.simplify(quad - (-sp.Rational(1, 2)*r**4*zs**2*mutS)) == 0
out['4_carrier_kept_term_only'] = str(sp.simplify(sp.Rational(1, 2)*kap*r*y_t**2))
# on the tail set mutS = c r^2: offset is O(r^6), sign of -mutS
c = sp.symbols('c', real=True)
off_tail = sp.simplify((-sp.Rational(1, 2)*r**4*zs**2*mutS).subs(mu, c*r**2 - r*nuS**2/kap))
out['4_offset_on_tail_set'] = str(off_tail)

# --- 5. det H(y) to order r^3 (carrier l.136-137) ---
a = zt*fttt + zs*ftts      # z . grad f_tt
cc = zt*ftts + zs*ftss     # z . grad f_ts
xi = -(zt*ftss + zs*fsss)  # carrier's xi
Hy = sp.Matrix([[kap*r + r**2*a, r*nuS + r**2*cc], [r*nuS + r**2*cc, -mu - r**2*xi]])
dety = sp.expand(Hy.det())
# put mu = mutS - r nu^2/kap with mutS = m r^2 (tail set), expand in r
m = sp.symbols('m', real=True)
dety_tail = sp.expand(dety.subs(mu, m*r**2 - r*nuS**2/kap))
lead = sp.expand(sp.series(dety_tail/(-kap*r), r, 0, 3).removeO())
out['5_detHy/(-kap r)_to_r^2'] = str(sp.collect(lead, r))
xi_eff = sp.simplify(lead.coeff(r, 2) - m)
out['5_xi_eff_minus_carrier_xi'] = str(sp.simplify(xi_eff - xi))
out['5_detHy_order_r'] = 'r^3 on tail set (coeff of r^1 vanishes): ' + str(sp.simplify(lead.coeff(r, 1)) == 0)

# --- 6. S-block substitution (carrier l.138-141) ---
cs, v, p0, X = sp.symbols('c_s v p0 Xi', positive=True)
dmu = cs*zs*r**2          # d mu-tilde = cs zs r^2 dv
absmu = cs*zs*r**2        # |mu-tilde| = cs zs r^2 |v|
absdety = r**2            # |mu-tilde + r^2 xi| = r^2 |cs zs v + xi|
sd_inv = 1/(cs*zs**2*r**4)
Sblock = sp.simplify(kap**2*r**2*p0*dmu*absmu*absdety*sd_inv)
out['6_Sblock_prefactor_correct'] = str(Sblock)   # times C1 = int |v| E|cs zs v + xi| phi(v) dv
carrier_line140 = sp.simplify(kap**2*r**2*p0*(cs*zs*r**2)*(r**2)*(cs*zs)*(1/(cs*zs**2*r**4))*(cs*zs*r**2))
out['6_carrier_line140_literal'] = str(carrier_line140)
out['6_carrier_line141_final'] = str(kap**2*r**4*cs*p0)
# xi is linear in z: on the line z_t = -nu z_s/kap, C1 = O(|z_s|)
out['6_xi_on_line'] = str(sp.factor(xi.subs(zt, zt_star)))

# --- 7. Assembly and exponent ledger (carrier l.152-163) ---
area, T, S, Mf, den = 4, -3, 4, 2, -2
out['7_ledger_area+T+S+M-den'] = area + T + S + Mf + den
kpow = -1 + 2 + 1 - 2
out['7_kappa_power'] = kpow
out['7_literal_text_decomposition_l159-163'] = 4 - 3 + 2 + 2 + 1 - 2
out['7_relative_form_4-3+3+1'] = 4 - 3 + 3 + 1

# --- 8. P2 Region I inside the nested disk (P11),(P14),(P16),(P18),(P19) ---
Zt, Zs, L, sig = sp.symbols('Z_t Z_s L sigma', positive=True)
p_, q_ = r*Zt, r*Zs                         # X = M + r(p,q), |X-M| = r^2 |Z|
Delta = sp.sqrt(q_**2 + r**2*p_**2)
dens = r**-3*Delta**-2                       # (P11) without Gaussian factor
dets = r**6*q_**2*(1 + (p_/q_)**2)**3        # (P16)
rho = sp.simplify(r**-2*dens*dets)           # Z^-1 (P14) * density * det product
out['8_rhoI_nested'] = str(sp.simplify(rho))
out['8_rhoI_r_power_at_fixed_Z'] = sp.limit(sp.log(rho.subs({Zt: 1, Zs: 1}))/sp.log(r), r, 0)
out['8_q2/Delta2'] = str(sp.simplify(q_**2/Delta**2))
# count in nested disk: dX = r^2 dp dq = r^4 dZ
out['8_count_power'] = 'r^4 (area) * r (intensity) = r^5'
# per-factor (P2 l.269-276) at d = r^2 |Z|
d = r**2*sp.sqrt(Zt**2 + Zs**2)
out['8_detA_bound_power'] = sp.degree(sp.expand((L**2*r*d/(4*sig)).subs({Zt: 1, Zs: 1})/sp.sqrt(2)), r)
out['8_detB_bound'] = '4 L^2 r^2 / sigma^2  -> r^2'
out['8_detC_bound_power'] = sp.degree(sp.expand((L**2*r*d/(2*sig)).subs({Zt: 1, Zs: 1})/sp.sqrt(2)), r)
out['8_product_bound'] = str(sp.simplify(L**6*r**4*d**2/sig**4))

# --- 9. P2 Region II ledger (P17),(P20), l.360-365 ---
out['9_regionII_ledger'] = -2 - 3 + 3 - 2 - 6
xx = sp.symbols('x', positive=True)
# e^{-x} <= 6!/x^6 with x = c/(2 r^2)
cpos = sp.symbols('c', positive=True)
bound = sp.simplify(sp.factorial(6)/(cpos/(2*r**2))**6)
out['9_absorption_bound'] = str(bound)
out['9_net_power'] = -10 + 12

# --- 10. gamma family (carrier S6): tau = r^gamma, same ledger ---
g = sp.symbols('gamma', positive=True)
tau = r**g
carrier_gamma = sp.simplify(tau**2 * (kap*r*tau)**-1 * (kap**2*r**2*tau) * (kap*r**2) / (kap**2*r**2))
out['10_carrier_ledger_gamma'] = str(sp.powsimp(carrier_gamma, force=True))
out['10_P2_bound_gamma'] = 'C r * (pi tau^2) = C r^(1+2 gamma)'
out['10_window_free_iff'] = 'offset ~ tau^2 * mutS ~ tau^3 << ell ~ r^3  <=>  gamma > 1'

# --- 11. constants for unit squared-exponential covariance ---
t_, s_ = sp.symbols('t s', real=True)
K = sp.exp(-(t_**2 + s_**2)/2)
def cov(al, be):
    e = (-1)**(be[0] + be[1])*sp.diff(K, t_, al[0]+be[0], s_, al[1]+be[1])
    return sp.simplify(e.subs({t_: 0, s_: 0}))
jets = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2)]
target = (0, 3)
Sig = sp.Matrix([[cov(i, j) for j in jets] for i in jets])
cvec = sp.Matrix([cov(target, j) for j in jets])
cond = sp.simplify(cov(target, target) - (cvec.T*Sig.inv()*cvec)[0])
out['11_Var(f_sss | 2-jet + f_tts,f_tss)'] = cond
out['11_c_s = sqrt(Var)/2'] = str(sp.sqrt(cond)/2) + ' = sqrt(3/2)? ' + str(sp.simplify(sp.sqrt(cond)/2 - sp.sqrt(sp.Rational(3, 2))) == 0)
out['11_sd_f coeff sqrt(Var)/6'] = str(sp.simplify(sp.sqrt(cond)/6)) + ' = 1/sqrt(6)? ' + str(sp.simplify(sp.sqrt(cond)/6 - 1/sp.sqrt(6)) == 0)

# --- 12. competing regime mutS ~ r (carrier l.165-166) ---
vv = (r)/(cs*zs*r**2)
out['12_gaussian_exponent_mutS~r'] = str(sp.simplify(-vv**2/2))

# --- 13. kappa <-> 6k identification (carrier l.55, F1 vs P2 (P4),(P13)) ---
k, x = sp.symbols('k x', real=True)
Hc = sp.Symbol('b') + k*(2*x**3 - 3*r*x**2)
out['13_H\'\'(0), H\'\'(r)'] = (str(sp.diff(Hc, x, 2).subs(x, 0)), str(sp.diff(Hc, x, 2).subs(x, r)))
ell = k*r**3
out['13_kappa=6 ell/r^3'] = str(sp.simplify(6*ell/r**3))

for kk, vv_ in out.items():
    print(f'{kk}: {vv_}')
