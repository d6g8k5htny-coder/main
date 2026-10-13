"""Exact Fraction re-derivation of every rational bound in ANALYTIC_GUARDS.md,
plus comparison with surgery_field.py constants (imported read-only, no bytecode)."""
import sys, os
sys.dont_write_bytecode = True
from fractions import Fraction as Q
WT = '/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/wt350/experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, WT)
import surgery_field as sf

ok = True
def chk(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(('PASS ' if cond else 'FAIL ') + name + ((' : ' + detail) if detail else ''))

# Section 1 parameters (doc lines 36-40)
R, r1, A = Q(1,8), Q(1,32), Q(1, 2**16)
u0, v0 = 3*A**2, 2*A**3
a_star, delta = r1/4, r1**4/64
b = Q(4,9)
chk('u0=3/2^32', u0 == Q(3, 2**32))
chk('v0=1/2^47', v0 == Q(1, 2**47))
chk('a*=1/128', a_star == Q(1,128))
chk('delta=2^-26', delta == Q(1, 2**26))
chk('r1=R/4', r1 == R/4)
chk('R^2=1/64<1/9=w2', R**2 == Q(1,64) and R**2 < Q(1,9))
chk('b = w1+w2 = 1/3+1/9', b == Q(1,3)+Q(1,9))
# code constants (surgery_field.py lines 28-30)
chk('code R,R1', (sf.R, sf.R1) == (R, r1))
chk('code A,U0,V0', (sf.A, sf.U0, sf.V0) == (A, u0, v0))
chk('code A_STAR,DELTA,B', (sf.A_STAR, sf.DELTA, sf.B) == (a_star, delta, b))
chk('code D,L', (sf.D, sf.L) == (2, Q(1)))

# Section 2: M = 4 e^-2 handled transcendental elsewhere. (G3) bound constant 4.
# Section 3: (G5) 32/(3 r1) < 64/r1
chi_grad = 32/(3*r1)
chk('(G5) 32/(3r1) = 1024/3', chi_grad == Q(1024,3))
chk('(G5) 32/(3r1) < 64/r1', chi_grad < 64/r1, f'{chi_grad} < {64/r1}')
# (G6)
B0, D0 = r1**2/2, r1
B1, D1 = (64/r1)*(r1**2/2) + r1, (64/r1)*r1 + 1
chk('(G6) B1 = 33 r1', B1 == 33*r1, str(B1))
chk('(G6) D1 = 65', D1 == 65)
B1_tight = chi_grad*(r1**2/2) + r1
chk('(G6) tighter B1 with (G5) = 19 r1/3 <= 33 r1', B1_tight == 19*r1/3 and B1_tight <= B1, str(B1_tight))
chk('(G6) tighter D1 with (G5) = 35/3 <= 65', chi_grad*r1 + 1 == Q(35,3))

# Section 4: transition width
w = R**2/2 - r1**2
chk('(G7) width w = 7/1024 > 0', w == Q(7,1024) and w > 0)
chk('R^2/2 < 1', R**2/2 < 1)
# (G8)
ann = r1**3/16
g_star = r1**3/64
chk('(G8) r1^3/16 = 2^-19 > r1^3/64 = 2^-21', ann == Q(1,2**19) and g_star == Q(1,2**21) and ann > g_star)
chk('(G8) (r1/2)^3/2 = r1^3/16', (r1/2)**3/2 == ann)
chk('(G8) rho^2/2 <= 2 used for z-bound (rho<=r1)', r1**2/2 <= 2)

# Section 5: (G9)
gp = B1*u0 + D1*v0
hp = B0*u0 + D0*v0
chk('(G9) B1 u0 + D1 v0 == 101441/2^47', gp == Q(101441, 2**47), str(gp))
chk('(G9) 101441 < 2^25', 101441 < 2**25)
chk('(G9) gp < 2^-22', gp < Q(1,2**22))
chk('(G9) 2^-22 <= g*/2 (equality)', Q(1,2**22) <= g_star/2 and Q(1,2**22) == g_star/2)
chk('(G9) B0 u0 + D0 v0 == 1537/2^52', hp == Q(1537, 2**52), str(hp))
chk('(G9) 1537 < 2^24', 1537 < 2**24)
chk('(G9) hp < 2^-28 = delta/4', hp < Q(1,2**28) and Q(1,2**28) == delta/4)
# root containment
chk('u0 < a*^2/4 = 2^-16', u0 < a_star**2/4 and a_star**2/4 == Q(1,2**16))
chk('v0 < a*^3/4 = 2^-23', v0 < a_star**3/4 and a_star**3/4 == Q(1,2**23))
# wedge coverage exact: (2/(3 sqrt3)) u0^(3/2) = v0  <=> v0^2 = 4 u0^3/27
chk('wedge exact v0^2 == 4 u0^3/27', v0**2 == 4*u0**3/27)
# PROOF (16): |P(s)| <= a*^4/4 + u0 a*^2/2 + v0 a* < delta/4 ; a*^4 = delta/4
chk('a*^4 == delta/4', a_star**4 == delta/4)
Pmax = a_star**4/4 + u0*a_star**2/2 + v0*a_star
chk('|P| bound < delta/4', Pmax < delta/4, f'{Pmax} < {delta/4}')
chk('|P| bound < 5 delta/32 (PROOF 16)', Pmax < 5*delta/32)
# barrier
chk('c(r1^2/4) = r1^2/16', (r1**2/4)/4 == r1**2/16)
chk('(r1^2/16)(r1/2)^2 == delta', (r1**2/16)*(r1/2)**2 == delta)
chk('r1^2/16 < 1', r1**2/16 < 1)
chk('R^2 > delta', R**2 > delta)
chk('b - delta + delta/4 = b - 3delta/4', b - delta + delta/4 == b - 3*delta/4)
chk('separation: b-3delta/4 < b-delta/2 < b-delta/4', b-3*delta/4 < b-delta/2 < b-delta/4)
# old critical heights of h = (1/3)cos(2pi y1) + (1/9)cos(2pi y2)
w1, w2 = Q(1,3), Q(1,9)
vals = sorted({w1*c1 + w2*c2 for c1 in (1,-1) for c2 in (1,-1)})
chk('old heights {-4/9,-2/9,2/9,4/9}', vals == [Q(-4,9), Q(-2,9), Q(2,9), Q(4,9)], str(vals))
chk('old non-max heights <= b - Delta, Delta = 2 w2', max(vals[:-1]) <= b - 2*w2)
chk('delta < Delta = 2/9', delta < 2*w2)
chk('delta == R^4/16384 (PROOF 217)', delta == R**4/16384)
# tied-barcode remark line 276-279: at v=0,u>0 maxima at s=+-sqrt(u) height b+u^2/4, saddle at s=0 height b
import sympy as sp
s_, u_ = sp.symbols('s u', positive=True)
P = -s_**4/4 + u_*s_**2/2
chk('P(sqrt u) = u^2/4', sp.simplify(P.subs(s_, sp.sqrt(u_)) - u_**2/4) == 0)
chk('P(0)=0', P.subs(s_, 0) == 0)

# Code guard_certificate replay and its bounds dict vs doc constants
g = sf.guard_certificate()
bd = g['bounds']
chk('code psi_derivative == 4', bd['psi_derivative'] == 4)
chk('code chi_gradient == 32/(3 r1)', bd['chi_gradient'] == chi_grad)
chk('code B0,D0,B1,D1', (bd['B0'], bd['D0'], bd['B1'], bd['D1']) == (B0, D0, B1, D1))
chk('code annular_gradient, g_star', (bd['annular_gradient'], bd['g_star']) == (ann, g_star))
chk('code gradient_perturbation', bd['gradient_perturbation'] == gp)
chk('code height_perturbation', bd['height_perturbation'] == hp)
chk('code critical_centered_height', bd['critical_centered_height'] == Pmax)
chk('code all checks True', all(g['checks'].values()), str({k: v for k, v in g['checks'].items() if not v}))
chk('code status', g['status'] == 'ACCEPTED_ANALYTIC_PREMISE')
print('ALL OK' if ok else 'SOME FAILED')
