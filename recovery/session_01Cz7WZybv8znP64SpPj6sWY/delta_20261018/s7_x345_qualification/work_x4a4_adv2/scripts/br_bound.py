# Independent check (adversarial #2, X4/A4): d=2 torus field, axial frame rotated by angle phi.
# Computes Cov_Q(D^beta f(z), A_r) for |beta|<=4 on a z-grid as r -> 0, with 30-digit arithmetic,
# plus the Sigma^{-1}-Cauchy-Schwarz bound on the pin-regression term.
import sympy as sp, mpmath as mp, itertools
mp.mp.dps = 40
L = 6
x1,x2 = sp.symbols('x1 x2', real=True)
norm = sum(sp.exp(-sp.Rational(1,2)*((L*n1)**2+(L*n2)**2)) for n1 in range(-2,3) for n2 in range(-2,3))
K = sum(sp.exp(-sp.Rational(1,2)*((x1+L*n1)**2+(x2+L*n2)**2)) for n1 in range(-2,3) for n2 in range(-2,3))/norm
cache = {}
def dK(i,j):
    if (i,j) not in cache:
        cache[(i,j)] = sp.lambdify((x1,x2), sp.diff(K,x1,i,x2,j) if (i or j) else K, 'mpmath')
    return cache[(i,j)]
# A linear functional = list of (coef, point(2), multiindex(i,j)) meaning sum coef * d^{i,j} f(point)
def cov(F,G):
    s = mp.mpf(0)
    for (a,p,(i,j)) in F:
        for (b,q,(k,l)) in G:
            # Cov(d^{ij} f(p), d^{kl} f(q)) = (-1)^{k+l} d^{i+k, j+l} K(p-q)
            s += a*b*(-1)**(k+l)*dK(i+k,j+l)(p[0]-q[0], p[1]-q[1])
    return s
def dirderiv(order_dirs, p, coef=1):
    # product of directional derivatives along list of unit vectors, at point p
    terms = {}
    for choice in itertools.product(range(2), repeat=len(order_dirs)):
        c = mp.mpf(coef)
        for v,ch in zip(order_dirs, choice): c *= v[ch]
        mi = (choice.count(0), choice.count(1))
        terms[mi] = terms.get(mi,0) + c
    return [(c,p,mi) for mi,c in terms.items()]
def run(phi, r):
    u = (mp.cos(phi), mp.sin(phi)); w = (-mp.sin(phi), mp.cos(phi))
    a = (-r/2*u[0], -r/2*u[1]); c = (r/2*u[0], r/2*u[1])
    f_ = lambda p: [(mp.mpf(1),p,(0,0))]
    fx = lambda p,s=1: dirderiv([u],p,s)
    fy = lambda p,s=1: dirderiv([w],p,s)
    U0 = [(mp.mpf(1)/2,a,(0,0)),(mp.mpf(1)/2,c,(0,0))]
    U1 = [(1/r,c,(0,0)),(-1/r,a,(0,0))]
    U2 = fx(c,1/r)+fx(a,-1/r)
    U3 = fx(a,6/r**2)+fx(c,6/r**2)+[(-12/r**3,c,(0,0)),(12/r**3,a,(0,0))]
    V0 = fy(a,mp.mpf(1)/2)+fy(c,mp.mpf(1)/2)
    V1 = fy(c,1/r)+fy(a,-1/r)
    U = [U0,U1,U2,U3,V0,V1]
    A = dirderiv([w,w],a)
    S = mp.matrix(6,6)
    for i in range(6):
        for j in range(6): S[i,j] = cov(U[i],U[j])
    Si = mp.inverse(S)
    cUA = mp.matrix([cov(Ui,A) for Ui in U])
    varQA = cov(A,A) - (cUA.T*Si*cUA)[0]
    worst_cross = mp.mpf(0); worst_B = mp.mpf(0); worst_ratio = mp.mpf(0)
    for zx in [mp.mpf(k)*L/7 for k in range(7)]:
        for zy in [mp.mpf(k)*L/7+mp.mpf('0.13') for k in range(7)]:
            z = (zx,zy)
            for order in range(5):
                for i in range(order+1):
                    D = [(mp.mpf(1),z,(i,order-i))]
                    cDU = mp.matrix([cov(D,Ui) for Ui in U])
                    worst_cross = max(worst_cross, max(abs(t) for t in cDU))
                    pin = (cDU.T*Si*cUA)[0]
                    covQ = cov(D,A) - pin
                    worst_B = max(worst_B, abs(covQ/varQA))
                    cs = mp.sqrt((cDU.T*Si*cDU)[0]*(cUA.T*Si*cUA)[0])
                    if cs>0: worst_ratio = max(worst_ratio, abs(pin)/cs)
    return mp.norm(Si, 2) if False else max(abs(t) for t in Si), varQA, worst_cross, worst_B, worst_ratio
for phi in [mp.mpf(0), mp.mpf('0.7')]:
    for r in [mp.mpf('0.2'), mp.mpf('0.02'), mp.mpf('0.002')]:
        maxSi, varQA, wc, wB, wr = run(phi, r)
        print('phi=%s r=%s max|Sigma^-1_ij|=%s VarQ(A)=%s max|Cov(D^b f(z),U_k)|=%s max|D^b B_r|=%s max|pin|/CS=%s' % (
            mp.nstr(phi,2), mp.nstr(r,3), mp.nstr(maxSi,5), mp.nstr(varQA,5), mp.nstr(wc,5), mp.nstr(wB,5), mp.nstr(wr,5)))
