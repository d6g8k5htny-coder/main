import sys, multiprocessing
exec(open('t3.py').read())
CG=Fr('22424320.655069')
F=Fr
GRID = (((F(1,2),F(3,4)), (0, F(1,4), 1)), ((F(3,4),F(1)), (0, F(1,4), 1)), ((F(1),F(5,4)), (0, F(1,4), 1)),
        ((F(5,4),F(3,2)), (0, F(1,8), F(1,2), 1)), ((F(3,2),F(13,8)), (0, F(1,8), F(3,8), 1)),
        ((F(13,8),F(7,4)), (0, F(1,16), F(3,16), F(1,2), 1)), ((F(7,4),F(15,8)), (0, F(1,16), F(3,16), F(7,16), 1)),
        ((F(15,8),F(31,16)), (0, F(1,32), F(3,32), F(7,32), F(15,32), 1)),
        ((F(31,16),F(63,32)), (0, F(1,64), F(3,64), F(7,64), F(15,64), F(31,64), 1)),
        ((F(63,32),F(2)), (0, F(1,512), F(5,1024), F(11,1024), F(11,512), F(41,1024), F(19,256), F(9,64), F(1,4), F(15,32), F(15,16), 1)))
boxes=[((F(g[i]),F(g[i+1])),kb) for kb,g in GRID for i in range(len(g)-1)]
print('boxes', len(boxes))
r0, r1 = (F(31,2**14), F(1,512)) if sys.argv[1]=='512' else (F(31,2**17), F(1,4096))
LAW=[None]
def job(box):
    bb,kb=box
    if LAW[0] is None: LAW[0]=band_law(r0,r1)
    res=box_bound(LAW[0], bb, kb, r0, r1, x0=F(1,64), theta=F(1,32))
    return (bb,kb,float(res['total']/CG),float(res['tail']/CG))
with multiprocessing.Pool(4) as p:
    out=p.map(job, boxes)
for bb,kb,t,tl in sorted(out, key=lambda z:-z[2])[:14]:
    print('k [%.5f,%.5f] b [%.5f,%.5f]  total/cG %.5f tail %.1e'%(float(kb[0]),float(kb[1]),float(bb[0]),float(bb[1]),t,tl))
