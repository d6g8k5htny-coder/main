# Independent mpmath check: do the certified intervals enclose the true values?
import json, sys
from fractions import Fraction as Q
from mpmath import mp, mpf, quad, findroot, cbrt, exp, log, sqrt
mp.dps = 60
cert = json.load(open(sys.argv[1]))
def F(x): return Q(x)
def tomp(q): q=Q(q); return mpf(q.numerator)/mpf(q.denominator)
B = mpf(9)/(28*cbrt(2))
blo,bhi = map(tomp, cert['coefficient_interval'])
print('B true', B, 'cert B in?', blo<=B<=bhi, 'width', bhi-blo)
tlo,thi = map(tomp, cert['poisson_target_interval'])
print('exp(-B) in target?', tlo<=exp(-B)<=thi, 'width', thi-tlo)
def integrand(th, tau):
    return th*(9-th**2)*min((mpf(3)/(3+th**2))**(mpf(5)/2), (tau/(4*th**3))**(mpf(5)/4))
def m_direct(tau):
    s = findroot(lambda r: 36*r**3 - tau*(3+r**2)**2, mpf('0.1'))
    return (mpf(1)/15)*(quad(lambda th: th*(9-th**2)*(mpf(3)/(3+th**2))**(mpf(5)/2),[0,s]) + quad(lambda th: th*(9-th**2)*(tau/(4*th**3))**(mpf(5)/4),[s,1])), s
def m_cdf(tau, s):
    return (mpf(1)/15)*(3 - 3*(1-s**2)/(1+s**2/3)**(mpf(3)/2) + (tau/4)**(mpf(5)/4)*((mpf(36)/7)*(s**(-mpf(7)/4)-1) - 4*(1-s**(mpf(1)/4))))
# total mass check
tot = (mpf(1)/15)*quad(lambda th: th*(9-th**2)*(mpf(3)/(3+th**2))**(mpf(5)/2),[0,1])
print('total mass (tau>=9/4) =', tot, ' expected 1/5; single-sign half =', tot/2)
# closed form at s=1, tau=9/4
print('closed form at s=1,tau=9/4 =', m_cdf(mpf(9)/4, mpf(1)))
ok_all = True
for row in cert['rows']:
    j=row['j']; tau=mpf(1)/2**(3*j); n=4**j
    md, s = m_direct(tau); mc = m_cdf(tau, s)
    dlo,dhi = map(tomp,row['direct']['interval']); clo,chi = map(tomp,row['cdf_mass'])
    ilo,ihi = map(tomp,row['intersection_mass'])
    rlo,rhi = map(tomp,row['direct']['root_bracket'])
    lam = n*md
    ok = dlo<=md<=dhi and clo<=md<=chi and ilo<=md<=ihi and rlo<=s<=rhi
    ok_all &= ok
    E = mpf(9)/8*tau**(mpf(2)/3) + tau**(mpf(7)/12)
    print(f'j={j} m_direct={md} m_cdf={mc} |diff|={float(abs(md-mc)):.3e} s={s} in_direct={dlo<=md<=dhi} in_cdf={clo<=md<=chi} in_inter={ilo<=md<=ihi} root_in={rlo<=s<=rhi} n*m={lam} |n*m-B|={float(abs(lam-B)):.3e} E={float(E):.3e} ratio={float(abs(lam-B)/E):.3f}')
    if j==8:
        V = (1-md)**n
        vlo,vhi = map(tomp,row['void_interval'])
        print('  V8 true', V, 'in void interval?', vlo<=V<=vhi, 'void width', vhi-vlo, '|V-exp(-B)|', abs(V-exp(-B)), ' exp(-n m)', exp(-n*md), '|V-exp(-nm)|', abs(V-exp(-n*md)))
print('ALL_ENCLOSED', ok_all)
