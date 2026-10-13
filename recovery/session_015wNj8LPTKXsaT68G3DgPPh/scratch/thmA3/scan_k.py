import sys, multiprocessing
exec(open('t3.py').read())
CG=Fr('22424320.655069')
r0, r1 = Fr(31,2**14), Fr(1,512)
if len(sys.argv)>1 and sys.argv[1]=='small': r0, r1 = Fr(31,2**17), Fr(1,4096)
LAW=[None]
def job(box):
    bb,kb=box
    if LAW[0] is None: LAW[0]=band_law(r0,r1)
    best=None
    for x0,th in ((Fr(1,64),Fr(1,32)),(Fr(1,1024),Fr(1,256))):
        res=box_bound(LAW[0], bb, kb, r0, r1, x0=x0, theta=th)
        if best is None or res['total']<best['total']: best=res
    return (bb,kb,float(best['total']/CG),float(best['main']/CG),float(best['tail']/CG))
ks=[Fr(1,2)+Fr(j,8) for j in range(13)]
kbs=list(zip(ks[:-1],ks[1:]))
bbs=[(Fr(0),Fr(1,16)),(Fr(1,16),Fr(1,4)),(Fr(1,4),Fr(1,2)),(Fr(1,2),Fr(1))]
boxes=[(bb,kb) for kb in kbs for bb in bbs]
with multiprocessing.Pool(4) as p:
    out=p.map(job, boxes)
for bb,kb,t,m,tl in out:
    print('k [%.4f,%.4f] b [%.4f,%.4f]  total/cG %.4f  main %.4f tail %.2e'%(float(kb[0]),float(kb[1]),float(bb[0]),float(bb[1]),t,m,tl))
