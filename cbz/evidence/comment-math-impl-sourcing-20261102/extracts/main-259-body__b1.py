"""Independent falsification probes. Floating-point checks are not proofs."""
import json, math, random
from decimal import Decimal, localcontext
from fractions import Fraction

def moments(L,N):
    z=sum(math.exp(-.5*(2*math.pi*j/L)**2) for j in range(-N,N+1))
    return {p:sum((2*math.pi*j/L)**p*math.exp(-.5*(2*math.pi*j/L)**2) for j in range(-N,N+1))/z for p in (0,2,4,6)}

def mul(a,b):
    out={}
    for (i,j),x in a.items():
        for (k,l),y in b.items():out[i+k,j+l]=out.get((i+k,j+l),0)+x*y
    return out

def power(a,n):
    out={(0,0):1}
    for _ in range(n):out=mul(out,a)
    return out

def expectation(a,m):return sum(v*m.get(i,0)*m.get(j,0) for (i,j),v in a.items())

def inner(a,b,m):return expectation(mul(a,b),m)

def inverse2(c):
    det=c[0][0]*c[1][1]-c[0][1]**2
    return [[c[1][1]/det,-c[0][1]/det],[-c[0][1]/det,c[0][0]/det]],det

def quadratic(v,a):return sum(v[i]*a[i][j]*v[j] for i in range(2) for j in range(2))

def integrand(theta,m):
    u={(1,0):math.cos(theta),(0,1):math.sin(theta)}
    v={(1,0):-math.sin(theta),(0,1):math.cos(theta)}
    x={(1,0):1};y={(0,1):1}
    V=[power(u,2),mul(u,v)];A=power(v,2);T=power(u,3)
    C=[[inner(a,b,m) for b in V] for a in V]
    ci,det=inverse2(C)
    av=[inner(A,a,m) for a in V]
    variance=inner(A,A,m)-quadratic(av,ci)
    tau2=inner(T,T,m)-(inner(T,x,m)**2+inner(T,y,m)**2)/m[2]
    assert min(det,variance,tau2)>0
    return (1/(2*math.pi*m[2]))*(1/(2*math.pi*math.sqrt(det)))*tau2**(2/3)*variance/2

def coefficient(L,N,n):
    m=moments(L,N)
    val=sum(integrand(2*math.pi*(i+.5)/n,m) for i in range(n))*2*math.pi/n
    return math.gamma(7/6)/(24**(1/3)*math.sqrt(math.pi))*val

def roots2(a,b,c):
    if abs(a)<1e-12:return [-c/b] if abs(b)>1e-12 else []
    disc=b*b-4*a*c
    if disc<0:return []
    return [(-b-math.sqrt(disc))/(2*a),(-b+math.sqrt(disc))/(2*a)]

def classify_numeric(k,s,a,b,c):
    # Extra critical roots obey Ax+Bz+C=0 after eliminating x^2-1/4.
    A=a-12*k*b/a;B=b/2-6*k*c/a;C=-12*k*s/a
    if abs(A)<1e-8:return None
    q=-B/A;t=-C/A
    zs=roots2(6*k*q*q+a*q+b/2,12*k*q*t+a*t,6*k*(t*t-.25))
    count=0
    for z in zs:
        x=q*z+t
        h=2*k*x**3-1.5*k*x-.5*k+s*z*z/2+a*(x*x-.25)*z/2+b*x*z*z/2+c*z**3/6
        if abs(h)<1e-7 or abs(h+k)<1e-7:return None
        if -k<h<0:count+=1
    return count

out={}
for L in (2,3,4,8,24):
    N=max(20,math.ceil(3*L))
    m=moments(L,N)
    c1=coefficient(L,N,512);c2=coefficient(L,N+10,1024)
    out[str(L)]={"c_d2":c2,"refinement_difference":abs(c2-c1),"mu4_minus_3mu2sq":m[4]-3*m[2]**2,"direction_integrand_ratio_axis_diagonal":integrand(0,m)/integrand(math.pi/4,m)}

# Separate arbitrary-precision image-sum derivatives; do not subtract near-equal binary64 numbers.
with localcontext() as ctx:
    ctx.prec=170
    L=Decimal(24);q=(-L*L/2).exp();z=Decimal(1);nums=[Decimal(-1),Decimal(3),Decimal(-15)]
    for n in range(1,4):
        x=L*n;ex=(-x*x/2).exp();z+=2*ex
        he=[x*x-1,x**4-6*x*x+3,x**6-15*x**4+45*x*x-15]
        nums=[a+2*b*ex for a,b in zip(nums,he)]
    a=-nums[0]/z;m4=nums[1]/z
    anis=m4-3*a*a
    out['side24_decimal']={"anisotropy_m4_minus_3a2":str(anis),"leading_anisotropy_ratio":str(anis/(2*L**4*q)),"P3_exact":str(-Fraction(620813376,35)),"P3q":str((-Decimal(620813376)/35)*q)}

rng=random.Random(20261003);tested=0;fail=[]
for _ in range(20000):
    k=rng.uniform(.2,2);a=rng.uniform(-4,4);b=rng.uniform(-4,4);c=rng.uniform(-4,4)
    if abs(a)<.02:continue
    B=b-a*a/(12*k);s=-abs(B)/2-rng.uniform(.02,4)
    D=(c-a*b/(4*k)+a**3/(72*k*k))/2
    threshold=-(s-B)**2*(B+2*s)/(12*k)
    pred=(2 if D*D<threshold else 1) if s>B else (1 if D*D>threshold else 0)
    got=classify_numeric(k,s,a,b,c)
    if got is None:continue
    tested+=1
    if pred!=got:fail.append([k,s,a,b,c,pred,got])
out['cubic_classifier']={"tested":tested,"mismatches":len(fail),"examples":fail[:3]}

# Integrate the cusp deficit directly, splitting at the exact change of branch.
y=1.7;d=.8;s0=(6*d/abs(y))**.25
direct=-(y*y*s0+36*d*d/(7*s0**7))
formula=-(8/7)*6**.25*abs(y)**1.75*abs(d)**.25
out['cusp_deficit']={"direct":direct,"formula":formula,"difference":abs(direct-formula)}
print(json.dumps(out,indent=2))
