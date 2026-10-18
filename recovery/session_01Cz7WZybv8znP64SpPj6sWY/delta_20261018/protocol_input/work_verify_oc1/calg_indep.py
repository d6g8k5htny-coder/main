# Independent check: ORIGINAL (1),(4),(6),(8),(12) at d=2, L=24 vs ORIGINAL (32).
# Moments by direct spectral lattice sum (no Poisson). Angular integral by Gauss-Legendre via mp.quad.
import mpmath as mp
mp.mp.dps = 200
L = mp.mpf(24)
h = 2*mp.pi/L
M = 400  # q_m^2/2 at m=400: (2pi*400/24)^2/2 ~ 5.5e3 -> e^-5483 negligible at 200 dps
w = [mp.e**(-(h*m)**2/2) for m in range(0, M+1)]
def S(k):
    return w[0]*(0 if k>0 else 1) + 2*mp.fsum(((h*m)**(2*k))*w[m] for m in range(1, M+1))
z = S(0); a = S(1)/z; b = S(2)/z; c = S(3)/z
k4 = b-3*a*a; k6 = c-15*a*b+30*a**3
print("a-1 =", mp.nstr(a-1, 6), " k4 =", mp.nstr(k4, 6), " k6 =", mp.nstr(k6, 6))
def integrand(th):
    e = (mp.cos(th), mp.sin(th)); v = (-mp.sin(th), mp.cos(th))
    B = [[a*a*((1 if i==j else 0)+2*e[i]*e[j]) + (k4*e[i]**2 if i==j else 0) for j in range(2)] for i in range(2)]
    detB = B[0][0]*B[1][1]-B[0][1]*B[1][0]
    Binv = [[B[1][1]/detB, -B[0][1]/detB], [-B[1][0]/detB, B[0][0]/detB]]
    D = [e[i]*(a*a+k4*v[i]**2) for i in range(2)]
    q = sum(D[i]*Binv[i][j]*D[j] for i in range(2) for j in range(2))
    F = (3*a*a + k4*(v[0]**4+v[1]**4) - q)/2
    V = 6*a**3 + 9*a*k4*(e[0]**4+e[1]**4) + (k6-k4**2/a)*(e[0]**6+e[1]**6)
    return F*V**(mp.mpf(2)/3)/mp.sqrt(detB)
I = mp.quad(integrand, mp.linspace(0, 2*mp.pi, 9))
C = mp.gamma(mp.mpf(7)/6)/(mp.cbrt(24)*mp.sqrt(mp.pi))/((2*mp.pi)**2*a)*I
c2inf = 2/(3*mp.pi*mp.sqrt(3))*mp.cbrt(mp.mpf(3)/2)*mp.gamma(mp.mpf(7)/6)/mp.sqrt(mp.pi)
print("C_alg(2,24) =", mp.nstr(C, 50))
print("c_2,inf     =", mp.nstr(c2inf, 50))
print("C/c2inf - 1 =", mp.nstr(C/c2inf-1, 8))
print("lower endpoint 0.07340691930603427103 - C =", mp.nstr(mp.mpf('0.07340691930603427103')-C, 6))
print("upper endpoint 0.07340691930603427104 - C =", mp.nstr(mp.mpf('0.07340691930603427104')-C, 6))
print("predicted 3/4 k4 + 5/72 k6 + (a-1)*(-1+...)? :", mp.nstr(mp.mpf(3)/4*k4 + mp.mpf(5)/72*k6, 8))
