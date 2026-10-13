# Independent float evaluation of Astra's d=3, L=4 ratio c_{3,4}/c_{3,inf}, by a different rule.
import sys, math, random; sys.argv = ['x']
import pjc                                   # merged #297 checker (main 2f8d721), moments only
from decimal import Decimal as D
q2, q4, q6, _ = pjc.q_derivatives(D(4))
a, m4, m6 = float(-q2), float(q4), float(-q6)
b = a*a; k = m4 - 3*b; k6 = m6 - 15*m4*a + 30*a**3; h = k6 - k*k/a; A_ = 9*a*k
c1 = 1/b; kap = k/(2*b*(2*b+k)); c0 = c1 - 4*b/((2*b+k)*(5*b+k))
beta0 = c0 - 2*kap; betas = c1 - 2*kap
detH = b**3*(2*b+k)**2*(5*b+k)
D0 = 29/6 - math.sqrt(6)
const = 2*math.sqrt(3)/(a**1.5*math.sqrt(detH)*6**(2/3)*D0)
Tmin = 6*a**3 + A_ + h; Tmax = 6*a**3 + A_/3 + h/9
print('a=%.12f k=%.12f k6=%.12f h=%.12f  beta0=%.10f  Tmin=%.10f Tmax=%.10f' % (a, k, k6, h, beta0, Tmin, Tmax))

def inv3(B):
    (p,q,r),(_,s,t),(_,_,u) = B
    cof = [[s*u-t*t, r*t-q*u, q*t-r*s],[r*t-q*u, p*u-r*r, q*r-p*t],[q*t-r*s, q*r-p*t, p*s-q*q]]
    det = p*cof[0][0] + q*cof[0][1] + r*cof[0][2]
    return [[x/det for x in row] for row in cof], det

def frame(th, ph):
    st, ct, sp, cp = math.sin(th), math.cos(th), math.sin(ph), math.cos(ph)
    u = (st*cp, st*sp, ct); w = (ct*cp, ct*sp, -st); v = (-sp, cp, 0.0)
    return u, w, v

def precision(u, w, v):
    cols = [[1-z*z for z in u], [w[i]*w[i]-v[i]*v[i] for i in range(3)], [2*w[i]*v[i] for i in range(3)]]
    return [[(c0 if i == j == 0 else c1 if i == j else 0.0) - kap*sum(cols[i][m]*cols[j][m] for m in range(3)) for j in range(3)] for i in range(3)]

def eig_CJ(C, detB):
    s1 = C[0][0]-C[1][1]-C[2][2]
    s2 = -(C[0][0]*C[1][1]-C[0][1]**2) - (C[0][0]*C[2][2]-C[0][2]**2) + (C[1][1]*C[2][2]-C[1][2]**2)
    s3 = 1/detB
    p = lambda x: ((x-s1)*x+s2)*x - s3
    lo, hi = 0.0, 8.0
    for _ in range(200):
        mid = (lo+hi)/2
        if p(mid) < 0: lo = mid
        else: hi = mid
    lam = (lo+hi)/2
    m = (lam - s1)/2; prod = s3/lam; d2 = m*m - prod
    d = math.sqrt(max(d2, 0.0))
    return lam, m+d, m-d          # lambda, mu, nu

def cone_moment(lam, mu, nu, npsi=512):      # (14) by direct periodic trapezoid in psi, no series
    tot = 0.0
    for j in range(npsi):
        psi = 2*math.pi*(j+0.5)/npsi
        be = mu*math.cos(psi)**2 + nu*math.sin(psi)**2
        tot += 3*lam*lam - 4*lam*be + 8*be*be - 8*be**2.5/math.sqrt(lam+be)
    return tot/npsi/2

def R_of(th, ph, want=False):
    u, w, v = frame(th, ph)
    B = precision(u, w, v); C, detB = inv3(B)
    lam, mu, nu = eig_CJ(C, detB)
    Du = cone_moment(lam, mu, nu)
    T = 6*a**3 + A_*sum(z**4 for z in u) + h*sum(z**6 for z in u)
    R = const*T**(2/3)*Du/math.sqrt(detB)
    return (R, T, Du, detB, lam, mu, nu) if want else R

# sanity: reference cone moment and MC check of the half factor at an anisotropic point
print('D(5/3,1,1) by psi-trapezoid = %.12f  (29/6-sqrt6 = %.12f)' % (cone_moment(5/3, 1, 1), D0))
random.seed(20261005)
R0, T0, D_ax, detB_ax, lam_ax, mu_ax, nu_ax = R_of(1e-9, 0.3, True)
N = 400000; acc = 0.0
for _ in range(N):
    z0, z1, z2 = random.gauss(0,1), random.gauss(0,1), random.gauss(0,1)
    qv = lam_ax*z0*z0 - mu_ax*z1*z1 - nu_ax*z2*z2
    if qv > 0: acc += qv*qv
print('axis: (lam,mu,nu)=(%.6f,%.6f,%.6f) D by (14)=%.6f  MC half-moment=%.4f (N=%d)' % (lam_ax, mu_ax, nu_ax, D_ax, acc/N/2, N))
print('axis R=%.10f T=%.10f D=%.10f ; face diag R=%.10f ; body diag R=%.10f' % (R0, T0, D_ax, R_of(math.pi/2, math.pi/4), R_of(math.atan(math.sqrt(2)), math.pi/4)))

# Gauss-Legendre nodes on [0,1]
def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n+1):
        x = math.cos(math.pi*(i-0.25)/(n+0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for kk in range(2, n+1):
                p0, p1 = p1, ((2*kk-1)*x*p1 - (kk-1)*p0)/kk
            dp = n*(x*p1 - p0)/(x*x-1)
            dx = p1/dp; x -= dx
            if abs(dx) < 1e-15: break
        xs.append(x); ws.append(2/((1-x*x)*dp*dp))
    return [(1+x)/2 for x in xs], [w/2 for w in ws]

def ratio(nz, nphi):
    zs, wz = gauss_legendre(nz)
    tot = 0.0
    for z, wgt in zip(zs, wz):
        th = math.acos(z)
        tot += wgt*sum(R_of(th, (math.pi/2)*(j+0.5)/nphi) for j in range(nphi))/nphi
    return tot
for nz, nphi in ((16, 16), (32, 32), (48, 64)):
    r = ratio(nz, nphi)
    print('GL%d x trap%d: ratio = %.12f   coefficient = %.12f' % (nz, nphi, r, r*0.04177593184059834334))
print('published ratio interval (0.967424069599, 0.969025287026); coefficient (0.040415041992, 0.040481934343)')

# uniform-bound checks on a fine parameter grid (finite differences)
Rstar, M1, M2 = 1.143532183656, 4.161419802375, 44.544592614265
n = 160; hstep = (math.pi/2)/n; maxR = 0; minT = 9; maxT = 0; d1 = 0; d2 = 0; eps = 1e-3
for i in range(n+1):
    th = i*hstep
    for j in range(n+1):
        ph = j*hstep
        Rv, Tv = R_of(th, ph, True)[:2]
        maxR = max(maxR, Rv); minT = min(minT, Tv); maxT = max(maxT, Tv)
        if 0 < i < n and 0 < j < n:
            Rt1, Rt2 = R_of(th+eps, ph), R_of(th-eps, ph)
            Rp1, Rp2 = R_of(th, ph+eps), R_of(th, ph-eps)
            d1 = max(d1, abs(Rt1-Rt2)/(2*eps), abs(Rp1-Rp2)/(2*eps))
            d2 = max(d2, abs(Rt1-2*Rv+Rt2)/eps**2, abs(Rp1-2*Rv+Rp2)/eps**2)
print('grid %dx%d: max R = %.6f (R* = %.6f)  T in [%.6f, %.6f] (claimed [%.6f, %.6f])' % (n, n, maxR, Rstar, minT, maxT, Tmin, Tmax))
print('max |dR| = %.4f (M1 = %.3f)   max |d2R| = %.4f (M2 = %.3f)' % (d1, M1, d2, M2))
