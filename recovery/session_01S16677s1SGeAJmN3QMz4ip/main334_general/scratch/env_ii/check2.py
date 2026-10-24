import math
d,C,c=2,1.0,0.7
H=lambda b,k: C*(1+abs(b)+k)**(2*d)*math.exp(-c*(b*b+k*k))
def simpson(f,a,b,n=400):
    h=(b-a)/n; s=f(a)+f(b)
    for i in range(1,n): s+=f(a+i*h)*(4 if i%2 else 2)
    return s*h/3
# substitute k=t^3 to remove k^-2/3: k^-2/3 dk = 3 dt
def mass(G):  # int_R int_0^inf k^-2/3 G(b,k) db dk
    return simpson(lambda b: simpson(lambda t: 3*G(b,t**3),1e-6,2.2,300),-8,8,320)
I_SG=mass(H)
alpha,a=1.0,1.0
m=a**(-2/3)*alpha/(alpha-2/3)
def HF(b,k):  # E[A^-2 H(b/A,k/A)], A density alpha a^-alpha x^(alpha-1) on (0,a); sub x=e^y
    return simpson(lambda y: alpha*a**(-alpha)*math.exp(y*(alpha-2))*H(b/math.exp(y),k/math.exp(y)),-18,math.log(a),400)
print("I_SG",I_SG,"m",m,"lhs",mass(HF),"m*I_SG",m*I_SG)
for (b,k) in [(0.0,1e-3),(1e-4,1e-4),(3.0,5.0)]:
    print((b,k),max(math.exp(-2*y)*H(b/math.exp(y),k/math.exp(y)) for y in [i/100-14 for i in range(2800)]))
