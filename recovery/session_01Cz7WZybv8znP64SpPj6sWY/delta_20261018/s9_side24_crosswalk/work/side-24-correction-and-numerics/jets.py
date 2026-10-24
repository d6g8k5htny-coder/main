import mpmath as mp
mp.mp.dps=300
L=24; q=mp.exp(-mp.mpf(L)**2/2)
He={2:lambda x:x**2-1,4:lambda x:x**4-6*x**2+3,6:lambda x:x**6-15*x**4+45*x**2-15}
He0={2:-1,4:3,6:-15}
# exact 1-D normalized jets (images |n|<=3) vs first-order
def m_exact(n):
    num=sum(He[n](mp.mpf(L*k))*mp.exp(-mp.mpf(L*k)**2/2) for k in range(-3,4))
    den=sum(mp.exp(-mp.mpf(L*k)**2/2) for k in range(-3,4))
    return num/den
for n in (2,4,6):
    lin=2*(He[n](mp.mpf(L))-He0[n])*q
    print(f'm{n}-m{n}^0 exact = {mp.nstr(m_exact(n)-He0[n],12)}   first-order 2[He_n(L)-He_n(0)]q = {mp.nstr(lin,12)}   diff = {mp.nstr(m_exact(n)-He0[n]-lin,5)}')
print('PROOF E =',mp.nstr(mp.mpf(21175738586478)*mp.mpf(10)**-125,6))
# max over a grid of directions of |delta d_u^6 K(0)| and |delta d_u^4 K(0)| (first order)
import itertools
d2=2*L**2*q; d4=2*(L**4-6*L**2)*q; d6=2*(L**6-15*L**4+45*L**2)*q
m0={0:1,2:-1,4:3,6:-15}
def dir_moment(u,n,dm):
    # d_u^n K(0) = sum over multinomial (a,b,c) even of n!/(a!b!c!) u^a.. m_a m_b m_c ; first-order variation
    tot=mp.mpf(0)
    for a in range(0,n+1,2):
        for b in range(0,n-a+1,2):
            c=n-a-b
            if c%2: continue
            coef=mp.factorial(n)/(mp.factorial(a)*mp.factorial(b)*mp.factorial(c))
            mon=u[0]**a*u[1]**b*u[2]**c
            ms=[a,b,c]
            var=sum(dm.get(ms[i],0)*mp.fprod(m0[ms[j]] for j in range(3) if j!=i) for i in range(3))
            tot+=coef*mon*var
    return tot
dm={2:d2,4:d4,6:d6}
best={4:0,6:0}
for i in range(0,41):
    for j in range(0,81):
        th=mp.pi*i/40; ph=2*mp.pi*j/80
        u=[mp.sin(th)*mp.cos(ph),mp.sin(th)*mp.sin(ph),mp.cos(th)]
        for n in (4,6):
            best[n]=max(best[n],abs(dir_moment(u,n,dm)))
print('max_u |delta d_u^6 K(0)| ~',mp.nstr(best[6],6),'  max_u |delta d_u^4 K(0)| ~',mp.nstr(best[4],6))
print('CL-AUD-LSDER071 A3 ||J24-J0||_max = 3.1195e-117 ; delta m6 q =',mp.nstr(d6,6))
