"""C027 station machinery (post G-C1/G-C2 supersession): exact limit coefficients per rational station.
Typed rule corrected: all 9-pin mean-Hessian entries are O(r); typed_inf reads r-coefficient signs."""
import sys, math
sys.path.insert(0,'.')
import c026_series as cs
from fractions import Fraction as Fr
Mt=(Fr(0),Fr(0)); St=(Fr(1),Fr(0)); B=Fr(6,5)
FUN6=[((0,0),Mt),((1,0),Mt),((0,1),Mt),((0,0),St),((1,0),St),((0,1),St)]
MON6=[(0,0),(1,0),(0,1),(2,0),(1,1),(3,0)]; G6E=[0,1,1,0,1,1]; C6E=[0,1,1,2,2,3]
MON9A=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2)]
MON9B=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(0,3)]
G9E=[0,1,1,0,1,1,0,1,1]; C9E=[0,1,1,2,2,2,3,3,3]
Phi=lambda x:0.5*(1+math.erf(x/math.sqrt(2)))
def _Trows(W,ge,ce,vecs):
    LS=cs.LS
    return [sum((vecs[p].shift(ge[p])*W[j][p] for p in range(len(W)) if W[j][p]!=0),LS()).shift(-ce[j]) for j in range(len(W))]
def _Krows(OUT,FUN,W,ge,ce):
    K=[]
    for (ao,po) in OUT:
        Sop=[cs.cser(ao,FUN[p][0],(po[0]-FUN[p][1][0],po[1]-FUN[p][1][1])) for p in range(len(FUN))]
        K.append(_Trows(W,ge,ce,Sop))
    return K
def station(Yt, NM6=17, K6=13, NM9=12, K9=6, want_finite_r=None):
    LS,LSc=cs.LS,cs.LSc
    cs.NM=NM6
    Spp6,Vh6,W6,G6,G06=cs.build_frame(FUN6,MON6,G6E,C6E)
    X6=cs.neumann_inv(G6,G06,K=K6)
    OUTY=[((0,0),Yt),((1,0),Yt),((0,1),Yt)]
    KYr=_Krows(OUTY,FUN6,W6,G6E,C6E)
    KY=cs.mmul(KYr,X6)
    vals6=[LSc(B),LS(),LS(),LSc(B)-LS(3,[Fr(1,6)]),LS(),LS()]
    u6=_Trows(W6,G6E,C6E,vals6)
    mo=[sum((KY[i][j]*u6[j] for j in range(6)),LS()) for i in range(3)]
    Soo=cs.mat(3)
    for i,(ai,pi) in enumerate(OUTY):
        for j,(aj,pj) in enumerate(OUTY): Soo[i][j]=cs.cser(ai,aj,(pi[0]-pj[0],pi[1]-pj[1]))
    KKt=cs.mmul(KY,cs.mT(KYr))
    So=[[Soo[i][j]-KKt[i][j] for j in range(3)] for i in range(3)]
    detS=So[1][1]*So[2][2]-So[1][2]*So[1][2]
    if detS.is0() or detS.off!=6: return dict(flag='detS_order', lam=0.0)
    c6v=detS.c[0]
    adj=[[So[2][2],LS()-So[1][2]],[LS()-So[1][2],So[1][1]]]
    mg=[mo[1],mo[2]]; Sfg=[So[0][1],So[0][2]]
    qn=mg[0]*(adj[0][0]*mg[0]+adj[0][1]*mg[1])+mg[1]*(adj[1][0]*mg[0]+adj[1][1]*mg[1])
    if qn.is0(): qv=Fr(0)
    elif qn.off==6: qv=qn.c[0]/c6v
    elif qn.off>6: qv=Fr(0)
    else: return dict(flag='q_order', lam=0.0)
    cross=Sfg[0]*(adj[0][0]*mg[0]+adj[0][1]*mg[1])+Sfg[1]*(adj[1][0]*mg[0]+adj[1][1]*mg[1])
    muN=mo[0]*detS-cross-LSc(B)*detS
    if muN.is0(): m=Fr(0)
    elif muN.off==9: m=Fr(6)*muN.c[0]/c6v
    elif muN.off>9: m=Fr(0)
    else: return dict(flag=None, lam=0.0, why='window_escape', q=float(qv))
    s2N=So[0][0]*detS-(Sfg[0]*(adj[0][0]*Sfg[0]+adj[0][1]*Sfg[1])+Sfg[1]*(adj[1][0]*Sfg[0]+adj[1][1]*Sfg[1]))
    if s2N.is0() or s2N.off!=12: return dict(flag='s2_order', lam=0.0)
    st2=Fr(36)*s2N.c[0]/c6v; st=math.sqrt(float(st2)); mf=float(m)
    Pw=Phi(-mf/st)-Phi((-1-mf)/st)
    out=dict(flag=None, q=float(qv), c6=float(c6v), m=mf, st=st, Pw=Pw)
    if Pw<1e-12:
        out.update(lam=0.0, why='Pwin0'); return out
    # full v*(r) series when interior (function identity with the mp pipeline); clipped constant otherwise
    def sdiv(A,Bs):
        off=A.off-Bs.off; c=[]; a=A.c
        for k in range(len(a)):
            x=a[k]
            for j in range(min(k,len(c))):
                bi=k-j
                if bi<len(Bs.c): x-=c[j]*Bs.c[bi]
            c.append(x/Bs.c[0])
        return cs.LS(off,c)
    if Fr(-1)<m<Fr(0): vser=sdiv(muN,detS)          # = (mu_t - b)(r) as a series
    else: vser=cs.LS(3,[min(max(m,Fr(-1)),Fr(0))*Fr(1,6)])  # clipped: (v*-b) = clip * r^3/6
    cs.NM=NM9
    FUN9=FUN6+[((0,0),Yt),((1,0),Yt),((0,1),Yt)]
    built=False; basis_used=None
    for MB,bname in [(MON9A,'A'),(MON9B,'B')]:
        try:
            Spp9,Vh9,W9,G9,G09=cs.build_frame(FUN9,MB,G9E,C9E)
            if cs.frmat_det(G09)==0: continue
            built=True; basis_used=bname; break
        except (AssertionError, StopIteration):
            continue
    if not built:
        out.update(flag='frame9_degenerate_both', lam=0.0); return out
    out['basis']=basis_used
    X9=cs.neumann_inv(G9,G09,K=K9)
    H9=[((2,0),Mt),((1,1),Mt),((0,2),Mt),((2,0),St),((1,1),St),((0,2),St),((2,0),Yt),((1,1),Yt),((0,2),Yt)]
    K9r=cs.mmul(_Krows(H9,FUN9,W9,G9E,C9E),X9)
    vals9=[LSc(B),LS(),LS(),LSc(B)-LS(3,[Fr(1,6)]),LS(),LS(),LSc(B)+vser,LS(),LS()]
    u9=_Trows(W9,G9E,C9E,vals9)
    mh=[sum((K9r[i][j]*u9[j] for j in range(9)),LS()) for i in range(9)]
    dM=mh[0]*mh[2]-mh[1]*mh[1]; dS=mh[3]*mh[5]-mh[4]*mh[4]; dY=mh[6]*mh[8]-mh[7]*mh[7]
    def lead2(d):
        if d.is0() or d.off>2: return Fr(0)
        if d.off<2: raise AssertionError("det order %d"%d.off)
        return d.c[0]
    cM,cS,cY=lead2(dM),lead2(dS),lead2(dY)
    tr=mh[0]+mh[2]
    if tr.is0() or tr.off>1: trc=Fr(0)
    elif tr.off==1: trc=tr.c[0]
    else: raise AssertionError("tr order %d"%tr.off)
    typed=(cM>0) and (cS<0) and (cY<0) and (trc<0)
    num=abs(cM*cS*cY) if typed else Fr(0)
    lam=(math.exp(-float(qv)/2)/(2*math.pi*math.sqrt(float(c6v))))*Pw*float(num)/3.230979
    out.update(lam=lam, cM=float(cM), cS=float(cS), cY=float(cY), tr_coef=float(trc), typed=bool(typed),
               margin=(min(abs(float(cM)),abs(float(cS)),abs(float(cY))) if typed else 0.0))
    if want_finite_r:
        out['finite']={str(r):dict(detM=dM.ev(r),detS=dS.ev(r),detY=dY.ev(r)) for r in want_finite_r}
    return out
