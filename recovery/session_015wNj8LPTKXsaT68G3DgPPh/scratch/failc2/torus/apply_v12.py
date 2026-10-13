"""Apply the v1.2 (Lemma T, torus transfer) changes to the packet's certificate.py and copy torus.py."""
import shutil
PK = '/home/user/wt_fail/frontiers/c8_elder_failure_coefficient_20261001'
shutil.copy('/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/failc2/torus/torus.py', PK + '/torus.py')
s = open(PK + '/certificate.py').read()
rep = [
("""import boxes as BX
""", """import boxes as BX
import torus as TOR
"""),
("""        out['Ik'] = {'Ik': Ik, 'boxes': Ik_boxes, 'tail': t4, 'const': const, 'Ib': Ib, 'C_fail': C, 'Wtot': Ktot}""",
 """        out['Ik'] = {'Ik': Ik, 'boxes': Ik_boxes, 'tail': t4, 'const': const, 'Ib': Ib, 'C_fail': C, 'Wtot': Ktot,
                     'K_m4': Km.get(BX.K_LO, BX.K_HI)[-4]}
        out['torus'] = TOR.transfer(TOR.E_I2_upper())"""),
("""    if 'Ik' in raw:
        I = raw['Ik']""",
 """    if 'torus' in raw and 'Ik' in raw:
        T = raw['torus']
        dJ = (-T['dJ'], T['dJ'])
        sec = {'jet covariance deviation delta (order <= 6)': (0.0, T['delta']), 'chi^2 bound': (0.0, T['chi2']),
               '|J^L - J| bound (all k, J_fail and J_2)': (0.0, T['dJ']),
               '(pi_0 p_b(0))^L / (pi_0 p_b(0))^ref': T['pi0p_ratio']}
        for k in KS:
            J = raw['J'][k]
            JfL, J2L = add(J['J_fail'], dJ), add(J['J_2'], dJ)
            row = {'J_fail': JfL, 'J_2': J2L}
            for b in (0.0, 1.0):
                mu = mul(T['mean_coeff_A'], (b, b))
                s = isqrt(T['var_A'])
                t = div(mu, s)
                ph = div(exp_neg(divc(sqr(t), 2.0)), SQRT2PI)
                p = div(ph, s)
                m2L = sub(mul(add(sqr(mu), T['var_A']), sub((1.0, 1.0), Phi(t))), mul(mul(mu, s), ph))
                P = div(p, mul((36.0 * k * k, 36.0 * k * k), m2L))
                row['b=%g' % b] = {'p_b(0)/z_0': P, 'alpha_1': mul(P, sub(JfL, J2L)), 'alpha_2': mul(P, J2L),
                                   'a_fail': mul(P, JfL)}
            sec['k=%g' % k] = row
        I = raw['Ik']
        dC = mul(mul(mul(I['const'], I['Ib']), divc(I['K_m4'], 3.0)), (T['dJ'], T['dJ']))
        CL = mul(T['pi0p_ratio'], add(I['C_fail'], (-dC[1], dC[1])))
        sec['C_fail^{B,K}'] = CL
        res['torus model, every L >= 10, every frame (Lemma T)'] = sec
    if 'Ik' in raw:
        I = raw['Ik']"""),
]
for a, b in rep:
    assert a in s, a[:60]
    s = s.replace(a, b)
open(PK + '/certificate.py', 'w').write(s)
print('applied')
