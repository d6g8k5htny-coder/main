import mpmath as mp
mp.mp.dps=120
def mom(L):
    h=2*mp.pi/L; al=h**2/2
    M=int(mp.ceil(mp.sqrt(400/al)))+5
    z=mp.mpf(0);s2=s4=s6=mp.mpf(0)
    for m in range(-M,M+1):
        w=mp.e**(-al*m*m); x=(h*m)**2
        z+=w;s2+=x*w;s4+=x*x*w;s6+=x**3*w
    return s2/z,s4/z,s6/z
for L in [0.5,1,2,3,6,12,24]:
    a,b,c=mom(mp.mpf(L))
    nu=c-b*b/a; beta=b-a*a
    print(L, mp.nstr(a,8), mp.nstr(nu,8), 'amp_nu3=10^%s'%mp.nstr(mp.log10((c+3*b*b/a)/abs(nu)),4), 'amp_beta=10^%s'%mp.nstr(mp.log10((b+2*a*a)/abs(beta)),4))
