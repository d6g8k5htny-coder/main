import sys, multiprocessing
exec(open('t3.py').read())
CG=Fr('22424320.655069')
print('boxes', len(BOXES))
bl=bands(); print('bands', len(bl))
sel=[b for b in bl if b[1] in (Fr(1,4096), Fr(1,2048), Fr(1,1024), Fr(1,512))]
def job(args):
    (r0,r1),box = args
    law=LAWS.get((r0,r1))
    if law is None:
        law=band_law(r0,r1); LAWS[(r0,r1)]=law
    res=box_bound(law, box[0], box[1], r0, r1, x0=COMBO[0], theta=COMBO[1])
    return (r1, box, float(res['total']/CG), float(res['tail']/CG))
LAWS={}
with multiprocessing.Pool(4) as p:
    out=p.map(job, [(band, box) for band in sel for box in BOXES], chunksize=len(BOXES))
for band in sel:
    rows=sorted([o for o in out if o[0]==band[1]], key=lambda z:-z[2])
    print('r1', band[1], 'top:', ['k[%s,%s] b[%s,%s] %.5f' % (str(o[1][1][0]),str(o[1][1][1]),str(o[1][0][0]),str(o[1][0][1]),o[2]) for o in rows[:4]])
