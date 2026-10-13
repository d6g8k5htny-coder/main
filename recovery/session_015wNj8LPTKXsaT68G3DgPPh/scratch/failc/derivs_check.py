import math, random
def xroot(B,T):
    # root of 2x^3+3Bx^2-B^3=T on x>=max(-B,B/2)
    lo=max(-B,B/2); f=lambda x:2*x**3+3*B*x*x-B**3-T
    hi=lo+1.0
    while f(hi)<0: hi=lo+2*(hi-lo)
    for _ in range(200):
        m=(lo+hi)/2
        if f(m)<0: lo=m
        else: hi=m
    return (lo+hi)/2
def V(B,T):
    x=xroot(B,T); return -3*B*(2*x-B)*(x+B)
def VT(B,T):
    x=xroot(B,T); return -B*(4*x+B)/(2*x*(x+B))
def VB(B,T):
    x=xroot(B,T); return -6*x*x+4.5*B*B-1.5*B**3/x
def VTT(B,T):
    x=xroot(B,T); return B*(4*x*x+2*B*x+B*B)/(12*x**3*(x+B)**3)
def VBT(B,T):
    x=xroot(B,T); return -(4*x*x+2*B*x+B*B)*(2*x-B)/(4*x**3*(x+B))
def VBB(B,T):
    x=xroot(B,T); return 3*B+6*x-9*B*B/(2*x)-3*B**3/(4*x*x)+3*B**4/(4*x**3)
random.seed(1); worst=[0]*5
for _ in range(2000):
    B=random.uniform(-4,4); T=random.uniform(0.05,30)
    h=1e-4
    fd_T=(V(B,T+h)-V(B,T-h))/(2*h); fd_B=(V(B+h,T)-V(B-h,T))/(2*h)
    fd_TT=(VT(B,T+h)-VT(B,T-h))/(2*h); fd_BT=(VT(B+h,T)-VT(B-h,T))/(2*h); fd_BB=(VB(B+h,T)-VB(B-h,T))/(2*h)
    for i,(a,b) in enumerate(((fd_T,VT(B,T)),(fd_B,VB(B,T)),(fd_TT,VTT(B,T)),(fd_BT,VBT(B,T)),(fd_BB,VBB(B,T)))):
        worst[i]=max(worst[i],abs(a-b)/(1+abs(b)))
print('max rel errors VT,VB,VTT,VBT,VBB:',['%.1e'%w for w in worst])
# monotonicity checks of x+B in B and T, and H formula consistency
