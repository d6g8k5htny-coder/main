import mpmath as mp
mp.mp.dps = 600
def mom(L):
    L = mp.mpf(L)
    h = 2*mp.pi/L; al = h*h/2
    cut = 2*mp.log(10)*mp.mp.dps + 50
    K = int(mp.sqrt(cut/al)) + 2
    ms = list(range(-K, K+1))
    w = {m: mp.exp(-al*m*m) for m in ms}
    z = mp.fsum(w.values())
    p = {m: w[m]/z for m in ms}
    q2 = {m: (h*m)**2 for m in ms}
    a = mp.fsum(p[m]*q2[m] for m in ms)
    b = mp.fsum(p[m]*q2[m]**2 for m in ms)
    c = mp.fsum(p[m]*q2[m]**3 for m in ms)
    return dict(L=L,h=h,al=al,z=z,p=p,q2=q2,a=a,b=b,c=c,ms=ms,K=K)
lg = lambda x: mp.log10(abs(x))
print("L  K  log10[(b+a^2)/beta]  log10[(b+2a^2)/beta]  log10[(c+b^2/a)/nu3]  log10[(c+3b^2/a)/nu3]  nu3/(72h^6e^-4a/z)  z")
for L in ["0.3","0.5","1","2","3","4","6","8","12"]:
    r = mom(mp.mpf(L)); a,b,c,z,h,al = r['a'],r['b'],r['c'],r['z'],r['h'],r['al']
    beta = b - a*a; nu3 = c - b*b/a
    lead = 72*h**6*mp.exp(-4*al)/z
    print(L, r['K'], mp.nstr(lg((b+a*a)/beta),5), mp.nstr(lg((b+2*a*a)/beta),5),
          mp.nstr(lg((c+b*b/a)/nu3),6), mp.nstr(lg((c+3*b*b/a)/nu3),6),
          mp.nstr(nu3/lead, 12), mp.nstr(z, 8), "beta=", mp.nstr(beta,6), "nu3=", mp.nstr(nu3,6))
# identity checks
print("identities")
for L in ["0.5","1","2","3","6","12"]:
    r = mom(mp.mpf(L)); a,b,c = r['a'],r['b'],r['c']; p,q2,ms = r['p'],r['q2'],r['ms']
    B = mp.fsum(p[k]*p[l]*(q2[k]-q2[l])**2 for k in ms for l in ms)/2
    D = mp.fsum(p[k]*p[l]*q2[k]*q2[l]*(q2[k]-q2[l])**2 for k in ms for l in ms)/2
    print(L, "relerr beta", mp.nstr(abs(B-(b-a*a))/abs(B),5), "relerr ac-b2", mp.nstr(abs(D-(a*c-b*b))/abs(D),5))
