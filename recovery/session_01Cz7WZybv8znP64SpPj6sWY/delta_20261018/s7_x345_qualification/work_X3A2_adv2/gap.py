# Independent illustrative probe (d=2, m=1): joint covariance of (U_r, A_r=f_yy(M)) via Fourier modes.
import mpmath as mp
mp.mp.dps=30
L=mp.mpf(3); N=7
modes=[(i,j) for i in range(-N,N+1) for j in range(-N,N+1)]
w={n: mp.e**(-2*mp.pi**2*(n[0]**2+n[1]**2)/L**2) for n in modes}
Z=sum(w.values()); a={n:w[n]/Z for n in modes}
def funcs(n,th,r):
    # mode e(z)=exp(i k.z), k=2pi n/L ; frame u=(cos,sin), e1=(-sin,cos)
    kx=2*mp.pi/L*(n[0]*mp.cos(th)+n[1]*mp.sin(th)); ky=2*mp.pi/L*(-n[0]*mp.sin(th)+n[1]*mp.cos(th))
    I=mp.mpc(0,1)
    e=lambda s: mp.e**(I*kx*s)      # value of mode at s*u
    A=-r/2; C=r/2
    f_a,f_c=e(A),e(C); fx_a,fx_c=I*kx*e(A),I*kx*e(C)
    fy_a,fy_c=I*ky*e(A),I*ky*e(C)
    if r==0:
        return [1,I*kx,-kx**2,-I*kx**3,I*ky,-kx*ky,-ky**2]
    U0=(f_a+f_c)/2; U1=(f_c-f_a)/r; U2=(fx_c-fx_a)/r
    U3=6/r**2*(fx_a+fx_c-2*(f_c-f_a)/r)
    V0=(fy_a+fy_c)/2; V1=(fy_c-fy_a)/r
    Ayy=-ky**2*e(A)
    return [U0,U1,U2,U3,V0,V1,Ayy]
def cov(th,r):
    M=mp.matrix(7,7)
    for n in modes:
        v=funcs(n,th,r)
        for i in range(7):
            for j in range(7):
                M[i,j]+=a[n]*v[i]*mp.conj(v[j])
    return mp.matrix([[mp.re(M[i,j]) for j in range(7)] for i in range(7)])
for r in [0, mp.mpf('0.1'), mp.mpf('0.01')]:
    mins=[]
    for q in range(0,13):
        th=mp.pi/2*q/12
        E=mp.eigsy(cov(th,r))[0]
        mins.append(min(E))
    print('r',r,'min over frames',mp.nstr(min(mins),6),'max of per-frame min',mp.nstr(max(mins),6))
