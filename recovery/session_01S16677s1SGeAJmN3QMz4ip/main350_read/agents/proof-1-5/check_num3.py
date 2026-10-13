exec(open('check_num2.py').read().split('for t in [mpf')[0])
def direct_cum(T):
    return 2*c0*quad(lambda e: (mpf(2)/5)*min(u0, sqrt(T/max(kappa(e), mpf(10)**-30)))**(mpf(5)/2), [0, eta_c - mpf(10)**-12])
for T in [mpf('0.001'), mpf('0.01')]:
    print("T=%s  direct 2D cumulative = %s   (3/2)C_edge T^(2/3) = %s" % (T, direct_cum(T), mpf(3)/2*C_edge*T**(mpf(2)/3)))
print("(26) mass direct = %s ; eta_c u0^(3/2)/(5 v0) = %s" % (direct_cum(u0**2), eta_c*u0**(mpf(3)/2)/(5*v0)))
eps = mpf('0.1'); kmin = kappa(eta_c - eps)
C_tip = c0*quad(lambda e: kappa(e)**(-mpf(5)/4), [0, eta_c - eps])
def tip_cum(T):
    return 2*c0*quad(lambda e: (mpf(2)/5)*min(u0, sqrt(T/max(kappa(e), mpf(10)**-30)))**(mpf(5)/2), [0, eta_c - eps])
for T in [u0**2*kmin/2, u0**2*kmin/10]:
    print("(9) T=%s tip cumulative direct = %s ; (4/5) C_tip T^(5/4) = %s" % (T, tip_cum(T), mpf(4)/5*C_tip*T**(mpf(5)/4)))
r = mpf('0.05'); t = mpf(10)**-6
small = 2*c0*quad(lambda uu: t**(mpf(1)/3)*uu**(-mpf(1)/2)*q(t/uu**2), [2*sqrt(t), r], maxdegree=4)
print("(28) contribution u<=r=%s at t=1e-6: %s ; limit (12 c0 Q_c/7) r^(7/6) = %s ; bound with A_q=Q_c same" % (r, small, 12*c0*Q_c/7*r**(mpf(7)/6)))
