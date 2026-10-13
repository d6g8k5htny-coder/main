import sys, json, time
sys.argv=['x']; sys.path.insert(0,'.')
import remote3 as R
from math import pi
OUT='/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/r2_verify.json'
out={}
# (1) allocation
for x in [(10.0,0,0),(2.0,1.0,0),(0.2,0.2,0)]:
    c=R.Conditioning(x,0.0,1.0); nodes=c.cone_nodes(); tw=sum(w for w,_ in nodes)
    for N in (40000,160000):
        pos=0; reused_w=0; tot=0
        for w,_ in nodes:
            n=max(1,int(round(N*w/tw))); tot+=n
            if pos>=N: reused_w+=w
            pos+=n
        out['alloc %s N=%d'%(x,N)]={'nodes':len(nodes),'allocated':tot,'overshoot':tot/N-1,'reused_weight_frac':reused_w/tw}
print(json.dumps(out,indent=0),flush=True)
# (2) grid: per-cell contributions on Rc=6 h=0.4 grid, and shifted-centre grid at Rc=6
def cells(b,k,zs,z0,Rc,h,shift=0.0):
    n1=int(round(2*Rc/h)); nr=int(round(Rc/h)); h1,hr=2*Rc/n1,Rc/nr
    far=R.Conditioning((10.0,0,0),b,k); fb=far.weighted(zs); res=[]
    for i in range(n1):
        x1=-Rc+(i+0.5)*h1+shift
        for j in range(nr):
            r=(j+0.5)*hr; vol=2*pi*r*h1*hr
            try: c=R.Conditioning((x1,r,0.0),b,k)
            except ValueError: continue
            E=c.weighted(zs); res.append((x1,r,sum(vol*(c.density*E[q]-far.density*fb[q])/z0 for q in range(4))))
    return res
z0=36*R.m3(0.0); t=time.time()
for seed in (2026,2027):
    zs=R.standard_normals(40000,seed)
    base=cells(0.0,1.0,zs,z0,6.0,0.4)
    full=sum(v for _,_,v in base); inner=sum(v for x1,r,v in base if abs(x1)<=5 and r<=5)
    sh=sum(v for _,_,v in cells(0.0,1.0,zs,z0,6.0,0.4,shift=0.2))
    out['seed%d'%seed]={'Rc6':full,'Rc6_cells_inside_Rc5_box':inner,'shell':full-inner,'Rc6_shifted_h/2':sh,'shift_minus_base':sh-full}
    print(seed,out['seed%d'%seed],round(time.time()-t),flush=True)
    json.dump(out,open(OUT,'w'))
print('done')
