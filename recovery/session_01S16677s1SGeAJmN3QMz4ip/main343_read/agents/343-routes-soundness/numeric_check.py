import json, sys
from fractions import Fraction as Q
from mpmath import mp, mpf, quad, findroot, exp, log, cbrt, nstr
mp.dps = 50
c = json.load(open(sys.argv[1]))
def q(x): return Q(x)
def mpq(x): f = Q(x); return mpf(f.numerator)/mpf(f.denominator)
B = mpf(9)/(28*cbrt(2))
print('B =', nstr(B, 40))
Blo, Bhi = (mpq(x) for x in c['coefficient_interval'])
print('cert B interval encloses B:', Blo <= B <= Bhi, ' width', nstr(Bhi-Blo, 5))
def split(tau):
    lo, hi = mpf(0), mpf(1)
    for _ in range(170):
        mid = (lo+hi)/2
        if 36*mid**3 <= tau*(3+mid**2)**2: lo = mid
        else: hi = mid
    return (lo+hi)/2
def m_closed(tau):
    s = split(tau)
    first = 3 - 3*(1-s**2)/(1+s**2/3)**mpf('1.5')
    second = mpf(36)/7*(s**mpf('-1.75')-1) - 4*(1-s**mpf('0.25'))
    return (first + (tau/4)**mpf('1.25')*second)/15, s
def m_direct(tau):
    s = split(tau)
    core = lambda th: th*(9-th**2)*(3/(3+th**2))**mpf('2.5')
    tail = lambda th: th*(9-th**2)*(tau/(4*th**3))**mpf('1.25')
    return (quad(core,[0,s]) + quad(tail,[s,1]))/15
def m_min(tau):
    f = lambda th: th*(9-th**2)*min((3/(3+th**2))**mpf('2.5'), (tau/(4*th**3))**mpf('1.25'))
    s = split(tau)
    return quad(f,[0,s,1])/15
print('total mass closed-form limit s->1:', nstr(m_closed(mpf(9)/4 - mpf(10)**-30)[0], 20))
ok_all = True
for row in c['rows']:
    j = row['j']; tau = mpq(row['tau']); n = row['n']
    mc, s = m_closed(tau); md = m_direct(tau); mm = m_min(tau)
    lam = n*mc
    E = mpf(9)/8*tau**(mpf(2)/3) + tau**(mpf(7)/12)
    dlo, dhi = (mpq(x) for x in row['direct']['normalized_interval'])
    clo, chi = (mpq(x) for x in row['cdf_normalized'])
    ilo, ihi = (mpq(x) for x in row['intersection_normalized'])
    rlo, rhi = (mpq(x) for x in row['direct']['root_bracket'])
    inner = [mpq(x) for x in row['predicted_envelopes']['inner']]
    outer = [mpq(x) for x in row['predicted_envelopes']['outer']]
    enc_d = dlo <= lam <= dhi; enc_c = clo <= lam <= chi; enc_i = ilo <= lam <= ihi
    ok_all &= enc_d and enc_c and enc_i and rlo <= s <= rhi
    print(f'j={j}: n*m closed={nstr(lam,25)} |closed-direct|={nstr(abs(mc-md)*n,3)} |closed-min|={nstr(abs(mc-mm)*n,3)}')
    print(f'      direct encloses {enc_d} (w={nstr(dhi-dlo,3)}), cdf encloses {enc_c} (w={nstr(chi-clo,3)}), intersection encloses {enc_i}, root encl {rlo<=s<=rhi}')
    print(f'      dev n*m-B={nstr(lam-B,8)}  E(tau)={nstr(E,8)}  |dev|<=E {abs(lam-B)<=E}  inner=[{nstr(inner[0],8)},{nstr(inner[1],8)}] outer=[{nstr(outer[0],8)},{nstr(outer[1],8)}]  admissible n*m range [0,{n/5}]  outer contains whole admissible range: {outer[0]<=0 and outer[1]>=mpf(n)/5}')
    print(f'      status {row["status"]} coeff {row["coefficient_status"]}')
print('ALL enclosures OK:', ok_all)
r8 = c['rows'][7]; n = 65536; tau = mpf(2)**-24
m8, _ = m_closed(tau)
V = (1-m8)**n; T = exp(-B)
vlo, vhi = (mpq(x) for x in r8['void_interval']); tlo, thi = (mpq(x) for x in r8['target_interval'])
print('V8 =', nstr(V,30), ' encl', vlo<=V<=vhi, ' width', nstr(vhi-vlo,3))
print('exp(-B) =', nstr(T,30), ' encl', tlo<=T<=thi, ' width', nstr(thi-tlo,3))
print('|V8-exp(-n m)| =', nstr(abs(V-exp(-n*m8)),5), ' bound 1/(16n)=', nstr(mpf(1)/(16*n),5))
print('|V8-exp(-B)| =', nstr(abs(V-T),5), ' endpoint distance cert =', nstr(mpq(r8['finite_void']['endpoint_distance']),5), ' threshold 2^-13 =', nstr(mpf(2)**-13,5))
# second-order expansion check: n m - B behaviour
for j in (6,7,8):
    tau = mpf(2)**(-3*j); n = 4**j
    lam = n*m_closed(tau)[0]
    print(f'j={j}: (n m - B)/tau^(7/12) = {nstr((lam-B)/tau**(mpf(7)/12),8)}   (n m - B)/tau^(2/3) = {nstr((lam-B)/tau**(mpf(2)/3),8)}')
