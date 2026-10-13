import sys, json, time
sys.argv=['x']; sys.path.insert(0,'.')
import remote3 as R
from math import pi
OUT='/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/r3_verify.json'
def holes(b,k,zbatches,z0,Rc=6.0,h=0.4,x_far=(10.0,0.0,0.0)):
    n1=int(round(2*Rc/h)); nr=int(round(Rc/h)); h1,hr=2*Rc/n1,Rc/nr
    far=R.Conditioning(x_far,b,k); far_b=[far.weighted(zs) for zs in zbatches]
    tot=[0.0]*len(zbatches)
    for i in range(n1):
        x1=-Rc+(i+0.5)*h1
        for j in range(nr):
            r=(j+0.5)*hr; vol=2*pi*r*h1*hr
            try: c=R.Conditioning((x1,r,0.0),b,k)
            except ValueError: continue
            for bi,zs in enumerate(zbatches):
                E=c.weighted(zs)
                tot[bi]+=sum(vol*(c.density*E[q]-far.density*far_b[bi][q])/z0 for q in range(4))
    return tot
zb=[R.standard_normals(40000,2026+i) for i in range(4)]
z0=36*R.m3(0.0); out={}; t=time.time()
for name,kw in [('h0.4',dict(h=0.4)),('h0.6',dict(h=0.6)),('h0.3',dict(h=0.3)),('Rc5',dict(Rc=5.0,h=0.4))]:
    out[name]=holes(0.0,1.0,zb[:2],z0,**kw); print(name,out[name],round(time.time()-t),flush=True)
    json.dump(out,open(OUT,'w'))
lam=R.far_field(0.0)['lambda']; rp={}
for i in range(4):
    p=R.ray_profile(0.0,1.0,zb[i],z0,lam)
    rp[i]={n:[(r['rho'],r.get('ratio_to_far')) for r in rows] for n,rows in p.items()}
    out['rays']=rp; json.dump(out,open(OUT,'w')); print('ray batch',i,round(time.time()-t),flush=True)
print('done')
