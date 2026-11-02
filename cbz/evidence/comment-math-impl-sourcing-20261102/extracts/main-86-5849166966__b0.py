#!/usr/bin/env python3
"""Small mathematical audit probes of a user-pasted LB1 source, 2026-09-26.

This is an independently written extraction/reproduction, not the raw historical
certificate. It does not authenticate the pasted file's advertised SHA256 and
does not execute its full gridded certificate or change any research status.
Source snippets consumed: wp_rho probability subtraction and final factor;
log_phi_bar asymptotic polynomial; rho_bar's noncentral Wick construction.
"""
from decimal import Decimal as D, getcontext
from fractions import Fraction

getcontext().prec = 90
PI = D('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534')

class MP:
    mpf = staticmethod(D)
    sqrt = staticmethod(lambda x: D(x).sqrt())
    exp = staticmethod(lambda x: D(x).exp())
    pi = PI
    nstr = staticmethod(lambda x, n: format(x, '.29g'))

mp = MP()

def emit(name, value):
    print(name + ': ' + mp.nstr(value, 30))

def Phi_bar(t):
    # Entire erf power series, only used here for |t|<=2.1.
    # This numerical probe is not directed interval arithmetic.
    x = t/mp.sqrt(2)
    term, total, k = x, x, 0
    while True:
        k += 1
        term *= -x*x/k
        add = term/(2*k+1)
        total += add
        if abs(add) < D('1e-85'):
            break
    return (1-2*total/mp.sqrt(PI))/2
phi = lambda t: mp.exp(-t*t/2) / mp.sqrt(2*mp.pi)

# The exact line order of the pasted wp_rho uses t1 > t2 and Q(t1)-Q(t2).
b = mp.mpf('1.2')
ell = mp.mpf('0.025')**3 / 6
mu = mp.mpf('1.187')
sigma = mp.sqrt(mp.mpf('5.94e-5'))
t1, t2 = (b-mu)/sigma, (b-ell-mu)/sigma
source_pw = Phi_bar(t1) - Phi_bar(t2)
true_pw = Phi_bar(t2) - Phi_bar(t1)
emit('wp_rho source subtraction (negative)', source_pw)
emit('correct window probability (positive)', true_pw)
if not (source_pw < 0 < true_pw):
    raise SystemExit('unexpected probability ordering')

# Missing sqrt counterexample to the proposed generic determinant/event bound.
# Z ~ N(0,1), Hessian diag(Z,-Z): always saddle off Z=0, |det H|=Z^2.
# E = {2 <= Z <= 2.1}; E[D^2]=E[Z^4]=3, P(type)=1.
lo, hi = mp.mpf(2), mp.mpf('2.1')
p_event = Phi_bar(lo)-Phi_bar(hi)
actual = lo*phi(lo)-hi*phi(hi)+p_event
source_bound = mp.sqrt(3)*p_event
cs_bound = mp.sqrt(3*p_event)
emit('window probability in saddle counterexample', p_event)
emit('true E[|det H| 1_E]', actual)
emit('invalid sqrt(E[D^2])*P(E)', source_bound)
emit('valid Cauchy-Schwarz upper bound', cs_bound)
if not (source_bound < actual <= cs_bound):
    raise SystemExit('counterexample not established')

# The pasted log_phi_bar uses -1035 u^5; fifth Mills coefficient is -945.
# Integration by parts gives Q(t)>phi(t)/t*S5(t) for t>0, where S5
# terminates in -945u^5. Exact Fraction arithmetic proves source<S5,
# so source<Q. This does not rely on evaluating the tiny exact tail.
for t in (30, 40):
    u = Fraction(1,t*t)
    source_S = 1-u+3*u**2-15*u**3+105*u**4-1035*u**5
    lower_S = 1-u+3*u**2-15*u**3+105*u**4-945*u**5
    deficit = lower_S-source_S
    print('tail t='+str(t)+' exact lower-polynomial deficit: '+str(deficit))
    emit('tail t='+str(t)+' lower-polynomial relative deficit',
         D(deficit.numerator)/D(deficit.denominator)/(D(lower_S.numerator)/D(lower_S.denominator)))
    if not (source_S < lower_S and deficit == 90*u**5):
        raise SystemExit('tail underestimate not established')

# Independent exact noncentral determinant moment: Hxx=1+X, Hyy=1+Y,
# Hxy=Z with X,Y,Z independent standard normal.
# E[(Hxx*Hyy-Hxy^2)^2] = 2*2 - 2*1*1*1 + 3 = 5.
# Wick on raw second moments C=Sigma+mu*mu^T gives 7 instead.
mu = [mp.mpf(1), mp.mpf(0), mp.mpf(1)]
S = [[mp.mpf(int(i == j)) for j in range(3)] for i in range(3)]
C = [[S[i][j]+mu[i]*mu[j] for j in range(3)] for i in range(3)]
def source_em(a,b,c,d):
    return C[a][b]*C[c][d]+C[a][c]*C[b][d]+C[a][d]*C[b][c]
code_d2 = source_em(0,0,2,2)-2*source_em(0,1,1,2)+source_em(1,1,1,1)
exact_d2 = mp.mpf(5)
det_mean_hessian = mu[0]*mu[2]-mu[1]**2
emit('raw-second-moment Wick D2', code_d2)
emit('true noncentral D2', exact_d2)
emit('conservative excess 2*(det mean H)^2', 2*det_mean_hessian**2)
if code_d2-exact_d2 != 2*det_mean_hessian**2:
    raise SystemExit('noncentral moment relation failed')

print('PROBES_PASS: explicit defects and conservative-moment distinction reproduced')
