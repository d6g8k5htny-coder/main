"""Sanity checks for the project-map report (exploratory; not a certificate)."""
import mpmath as mp
mp.mp.dps = 40
print('mpmath', mp.__version__)

def S(L, M=60):   # image-sum normalizer, 1D: sum_j exp(-(L j)^2/2)
    return mp.fsum(mp.e**(-(L*j)**2/2) for j in range(-M, M+1))
def Z(L, M=400):  # Fourier normalizer, 1D: sum_j exp(-omega^2 j^2/2), omega=2pi/L
    w = 2*mp.pi/L
    return mp.fsum(mp.e**(-(w*j)**2/2) for j in range(-M, M+1))
def Theta(u, M=400):  # Theta(u)=sum exp(-pi u n^2)
    return mp.fsum(mp.e**(-mp.pi*u*n*n) for n in range(-M, M+1))

for L in (mp.mpf(1)/2, 1, 3, 24):
    lhs = Z(L); rhs = L/mp.sqrt(2*mp.pi)*S(L)
    print('L', L, 'Z_{1,L}', mp.nstr(lhs, 25), ' (L/sqrt(2pi)) S_{1,L}', mp.nstr(rhs, 25),
          ' Theta(2pi/L^2)', mp.nstr(Theta(2*mp.pi/L**2), 25), ' diff', mp.nstr(lhs-rhs, 3))

# 1D normalized periodized covariance vs Jacobi theta_3(pi t/L, q), q=exp(-2pi^2/L^2)
for L in (3, 24):
    q = mp.e**(-2*mp.pi**2/L**2)
    for t in (mp.mpf('0.3'), mp.mpf('1.7'), mp.mpf(L)/3):
        k_img = mp.fsum(mp.e**(-(t+L*j)**2/2) for j in range(-60, 61))/S(L)
        k_th = mp.jtheta(3, mp.pi*t/L, q)/mp.jtheta(3, 0, q)
        print('L', L, 't', t, 'image-sum k_L', mp.nstr(k_img, 20), ' theta3 ratio', mp.nstr(k_th, 20), ' diff', mp.nstr(k_img-k_th, 3))

# de Bruijn-Newman normalization: H_0(z)=int_0^inf Phi(u) cos(zu) du = (1/8) xi(1/2+iz/2)
def Phi(u, N=30):
    return mp.fsum((2*mp.pi**2*n**4*mp.e**(9*u) - 3*mp.pi*n**2*mp.e**(5*u))*mp.e**(-mp.pi*n**2*mp.e**(4*u)) for n in range(1, N+1))
def H(t, z):
    return mp.quad(lambda u: mp.e**(t*u*u)*Phi(u)*mp.cos(z*u), [0, 0.5, 1, 2, 4])
def xi(s):
    return s*(s-1)/2*mp.pi**(-s/2)*mp.gamma(s/2)*mp.zeta(s)
mp.mp.dps = 25
for z in (0, 10, 2*mp.zetazero(1).imag):
    print('z', mp.nstr(z, 15), 'H_0', mp.nstr(H(0, z), 15), ' xi/8', mp.nstr(xi(mp.mpf(1)/2+1j*z/2)/8, 15))

# exact lifetime identity for consecutive zeros a<c of H: int_a^c H = -(1/2) int_a^c (x-a)(c-x) H''(x) dx
# check on H_0 between its first two zeros (z = 2*gamma_1, 2*gamma_2): RHS uses H'' = -dH/dt (backward heat eq.)
g1, g2 = 2*mp.zetazero(1).imag, 2*mp.zetazero(2).imag
lhs = mp.quad(lambda x: H(0, x), [g1, g2])
dHdt = lambda x: mp.quad(lambda u: u*u*Phi(u)*mp.cos(x*u), [0, 0.5, 1, 2, 4])   # = d/dt H_t at t=0 = -H''
rhs = mp.mpf(1)/2*mp.quad(lambda x: (x-g1)*(g2-x)*dHdt(x), [g1, g2])
print('int_{2g1}^{2g2} H_0 =', mp.nstr(lhs, 15), ' (1/2) int (x-a)(c-x) dH/dt =', mp.nstr(rhs, 15))
