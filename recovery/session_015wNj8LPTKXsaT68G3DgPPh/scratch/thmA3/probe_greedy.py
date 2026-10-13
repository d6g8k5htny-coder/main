from probe_m3 import *
r0=R3R(0)
def boxval(blo,bhi):
    return exp(-blo*blo/4)*I(3,bhi,N=6000)/(4*sqrt(2*pi))/m3(blo)/r0
for T in (1.003, 1.005):
    b=0.0; out=[0.0]
    while b<1:
        # largest dyadic step 2^-j * m such that boxval <= T
        lo, hi = b, 1.0
        if boxval(b,1.0)<=T: out.append(1.0); break
        for _ in range(40):
            mid=(lo+hi)/2
            if boxval(b,mid)<=T: lo=mid
            else: hi=mid
        # round down to a multiple of 1/1024
        nb=max(b+1/1024, int(lo*1024)/1024)
        out.append(nb); b=nb
    print(T, len(out)-1, [round(x*1024) for x in out])
