import math
d,C,c=2,1.0,0.7
H=lambda b,k: C*(1+abs(b)+k)**(2*d)*math.exp(-c*(b*b+k*k))
def simpson(f,a,b,n=400):
    h=(b-a)/n; s=f(a)+f(b)
    for i in range(1,n): s+=f(a+i*h)*(4 if i%2 else 2)
    return s*h/3
def mass(G,n=240):
    return simpson(lambda b: simpson(lambda t: 3*G(b,t**3),0.0,2.2,n),-8,8,n)
I_SG=mass(H)
for alpha,a in [(3.0,1.0),(4.0,0.7)]:
    m=a**(-2/3)*alpha/(alpha-2/3)
    def HF(b,k):
        return simpson(lambda y: alpha*a**(-alpha)*math.exp(y*(alpha-2))*H(b/math.exp(y),k/math.exp(y)),-16,math.log(a),600)
    print(alpha,a,"lhs",mass(HF),"m*I_SG",m*I_SG)
