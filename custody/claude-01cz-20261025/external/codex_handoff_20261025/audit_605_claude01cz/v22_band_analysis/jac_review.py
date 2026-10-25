import sympy as sp
r,s,eta=sp.symbols('r s eta',positive=True)
raw=sp.symbols('Fm D1m D2m Fp D1p D2p gx1 gx2 gy1 gy2'); Fm,D1m,D2m,Fp,D1p,D2p,gx1,gx2,gy1,gy2=raw
U0=(Fm+Fp)/2; U1=(Fp-Fm)/r; U2=(D1p-D1m)/r; U3=6*(D1m+D1p-2*U1)/r**2; V0=(D2m+D2p)/2; V1=(D2p-D2m)/r
Z=[U0, s*(U1-r**2*U3/24), s**2*U2, s**3*U3, s*V0, s**2*V1, s*gx1, s*gx2, s*(gy1-gx1)/eta, s*(gy2-gx2)/eta]
J=sp.Matrix([[sp.diff(z,x) for x in raw] for z in Z]); d=sp.simplify(J.det())
print('det =',sp.factor(d),'; |det| == 12 s^13 r^-5 eta^-2:',sp.simplify(sp.Abs(d)-12*s**13/(r**5*eta**2))==0)
e=sp.symbols('e',positive=True); print('in e: ',sp.simplify((12*s**13/(r**5*eta**2)).subs(eta,e/s)))
# density: floor c s^12 => det Cov >= s^120 => pZ <= C s^-60 ; pT = |det| pZ
print('p_T power of s:',13-60,' ; with eta=e/s ->',13-60+2)
