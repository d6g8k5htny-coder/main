import mpmath as mp
mp.mp.dps = 400
def mom(L):
    L = mp.mpf(L)
    h = 2*mp.pi/L; al = h**2/2
    M = int(mp.ceil(mp.sqrt(3000/al)))+10
    z = mp.mpf(0); s2=s4=s6=mp.mpf(0)
    for m in range(-M, M+1):
        w = mp.e**(-al*m*m); x2 = (h*m)**2
        z += w; s2 += w*x2; s4 += w*x2**2; s6 += w*x2**3
    return s2/z, s4/z, s6/z
for L in ['0.5','1','2','3','6','12','24']:
    a,b,c = mom(L)
    nu3 = c-b*b/a; beta = b-a*a
    print(L, mp.nstr(a,8), mp.nstr(b,8), mp.nstr(c,8), 'nu3=',mp.nstr(nu3,8),
          'amp_nu3=10^%s' % mp.nstr(mp.log10((c+3*b*b/a)/abs(nu3)),4),
          'amp_beta=10^%s' % mp.nstr(mp.log10((b+2*a*a)/abs(beta)),4))
