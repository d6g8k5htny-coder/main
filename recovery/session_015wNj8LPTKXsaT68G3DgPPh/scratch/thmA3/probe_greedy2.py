from probe_m3 import *
r0=R3R(0)
def boxval(blo,bhi):
    return exp(-blo*blo/4)*I(3,bhi,N=6000)/(4*sqrt(2*pi))/m3(blo)/r0
T=1.003
for khi in (1, 1.5, 1.75, 1.875, 31/16, 63/32, 2):
    f=(khi/2)**3
    b=0.0; out=[0.0]
    while b<1:
        if boxval(b,1.0)*f<=T: out.append(1.0); break
        lo, hi = b, 1.0
        for _ in range(40):
            mid=(lo+hi)/2
            if boxval(b,mid)*f<=T: lo=mid
            else: hi=mid
        nb=max(b+1/1024, int(lo*1024)/1024)
        out.append(nb); b=nb
    print(khi, len(out)-1, [round(x*1024) for x in out])
