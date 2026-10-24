import math, random
# 1. fourth row identity: (6/r^2)[psi'(-a)+psi'(a)-2(psi(a)-psi(-a))/r] == (3/(4a^3)) int_{-a}^{a}(a^2-s^2) psi'''(s) ds, r=2a
random.seed(1)
c=[random.uniform(-1,1) for _ in range(7)]
psi=lambda s: sum(c[i]*s**i for i in range(7))
dpsi=lambda s: sum(i*c[i]*s**(i-1) for i in range(1,7))
d3psi=lambda s: sum(i*(i-1)*(i-2)*c[i]*s**(i-3) for i in range(3,7))
a=0.37; r=2*a
lhs=(6/r**2)*(dpsi(-a)+dpsi(a)-2*(psi(a)-psi(-a))/r)
N=200000; h=2*a/N
rhs=(3/(4*a**3))*sum((a**2-s**2)*d3psi(s)*h for s in [-a+(i+0.5)*h for i in range(N)])
print("fourth row identity", lhs, rhs)
# kernel mass
print("kernel mass", (3/(4*a**3))*(2*a**3-2*a**3/3))
# 2. constant chain
tau=1.7
f=lambda k: k**(4/3)*(2*math.pi)**-0.5/tau*math.exp(-72*k*k/tau/tau)
K=20; N=400000; h=K/N
I=sum(f((i+0.5)*h)*h for i in range(N))
print("4*36*I/tau^(4/3) =", 4*36*I/tau**(4/3), " Gamma(7/6)/(24^(1/3)sqrt(pi)) =", math.gamma(7/6)/(24**(1/3)*math.sqrt(math.pi)))
# 3. half Gaussian moment
g=lambda y: y**(4/3)*math.exp(-y*y/2)/math.sqrt(2*math.pi)
J=sum(g((i+0.5)*h)*h for i in range(N))
print("half-gauss 4/3 moment", J, 2**(-1/3)*math.gamma(7/6)/math.sqrt(math.pi))
# 4. 4x4 Jacobian determinant
r=0.3
M=[[0.5,0.5,0,0],[-1/r,1/r,0,0],[0,0,-1/r,1/r],[12/r**3,-12/r**3,6/r**2,6/r**2]]
def det(m):
    n=len(m)
    if n==1: return m[0][0]
    return sum((-1)**j*m[0][j]*det([row[:j]+row[j+1:] for row in m[1:]]) for j in range(n))
print("4x4 det", det(M), "12/r^4 =", 12/r**4)
# 5. scalar-block determinant identity det[[a,b^T],[b,A]] = a det A - b^T adj(A) b, with singular A test
A=[[1,2,3],[2,4,6],[1,0,1]]  # singular (row2=2*row1)
b=[0.3,-0.7,1.1]; al=0.9
H=[[al]+b]+[[b[i]]+A[i] for i in range(3)]
def adj(A):
    n=len(A)
    return [[(-1)**(i+j)*det([row[:i]+row[i+1:] for k,row in enumerate(A) if k!=j]) for j in range(n)] for i in range(n)]
adjA=adj(A)
quad=sum(b[i]*adjA[i][j]*b[j] for i in range(3) for j in range(3))
print("block det", det(H), al*det(A)-quad)
# 6. polynomial tail exponents
d=3; beta=d+8+0.5
print("A4 shell exponent d+4-beta/2 =", d+4-beta/2, " M_j exponent (d+8-beta)/2 =", (d+8-beta)/2, " interp exponent d/2+3-beta/2 vs -1+(d+8-beta)/2:", d/2+3-beta/2, -1+(d+8-beta)/2)
