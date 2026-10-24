import mpmath as mp
mp.mp.dps = 60
L = mp.mpf(24); h = 2*mp.pi/L; al = h**2/2
M = 400
def S(k): return mp.fsum((h*m)**(2*k)*mp.e**(-al*m*m) for m in range(-M, M+1))
z = S(0); a = S(1)/z; b = S(2)/z; c = S(3)/z
beta = b-a*a; nu3 = c-b*b/a; k4 = b-3*a*a; k6 = c-15*a*b+30*a**3
w1 = mp.e**(-al); w2 = mp.e**(-4*al)
print('a,b,c', mp.nstr(a,15), mp.nstr(b,15), mp.nstr(c,15))
print('beta,nu3', mp.nstr(beta,15), mp.nstr(nu3,15))
print('witness a', mp.nstr(2*h**2*w1/z,6), 'beta', mp.nstr(2*h**4*w1/z**2,6), 'nu3', mp.nstr(72*h**6*w1*w2/(z*(w1+4*w2)),6))
lamH = min(beta, 2*a*a); LamH = b + a*a
print('lambda_H', mp.nstr(lamH,15), 'Lambda_H', mp.nstr(LamH,15))
# d=2 joint Hessian covariance in coordinates (H11,H12,H22): Cov(Hij,Hkl)=E[QiQjQkQl]
def E4(i,j,k,l):
    cnt=[0,0]
    for t in (i,j,k,l): cnt[t]+=1
    mom={0:1,2:a,4:b}
    if cnt[0]%2 or cnt[1]%2: return mp.mpf(0)
    return mom[cnt[0]]*mom[cnt[1]]
mins={'B':mp.inf,'G':mp.inf,'V':mp.inf}; maxs={'G':-mp.inf}
for s in range(0,2001):
    t = mp.pi*2*s/2000
    e=(mp.cos(t),mp.sin(t)); u=(-mp.sin(t),mp.cos(t))
    # linear functionals of H: H_ee, H_eu, H_uu as v^T H w = sum v_i w_j H_ij
    def lin(v,w):
        return {(i,j):v[i]*w[j] for i in range(2) for j in range(2)}
    fs=[lin(e,e),lin(e,u),lin(u,u)]
    C=mp.matrix(3,3)
    for p in range(3):
        for q in range(3):
            C[p,q]=mp.fsum(fs[p][(i,j)]*fs[q][(k,l)]*E4(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2))
    # B(e) = Cov(He) in standard coords: He components = sum_j H_ij e_j
    B=mp.matrix(2,2)
    for p in range(2):
        for q in range(2):
            B[p,q]=mp.fsum(e[j]*e[l]*E4(p,j,q,l) for j in range(2) for l in range(2))
    # compare with M2
    B2=mp.matrix([[a*a*(1+2*e[0]**2)+k4*e[0]**2, a*a*2*e[0]*e[1]],[a*a*2*e[0]*e[1], a*a*(1+2*e[1]**2)+k4*e[1]**2]])
    assert mp.mnorm(B-B2,1) < mp.mpf(10)**-40
    eb=min(mp.eigsy(B)[0])
    Y=C[0:2,0:2]; G=C[2,2]-(C[2,0:2]*mp.inverse(Y)*C[0:2,2])[0,0]
    V=6*a**3+9*a*k4*(e[0]**4+e[1]**4)+(k6-k4**2/a)*(e[0]**6+e[1]**6)
    mins['B']=min(mins['B'],eb); mins['G']=min(mins['G'],G); maxs['G']=max(maxs['G'],G); mins['V']=min(mins['V'],V)
print('min eig B', mp.nstr(mins['B'],12), ' b0=lamH/2', mp.nstr(lamH/2,12))
print('Gamma_e range', mp.nstr(mins['G'],12), mp.nstr(maxs['G'],12), ' gamma0', mp.nstr(lamH/2,12), 'LamH', mp.nstr(LamH,12))
print('min V', mp.nstr(mins['V'],12), ' v0=nu3/4', mp.nstr(nu3/4,12))
