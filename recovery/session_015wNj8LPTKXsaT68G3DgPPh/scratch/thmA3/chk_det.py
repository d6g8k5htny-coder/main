import time
exec(open('t3.py').read())
for p in (8, 16, 32, 64):
    for b in (Fr(0), Fr(1,2), Fr(1)):
        t=time.time(); v=contact_moment(p, 0, b); dt=time.time()-t
        print(p, float(b), 'mid %.6e width %.3e  root %.4f  [%.2fs]' % (float(v.mid()), float(v.width()), float(v.hi)**(1/p) if v.hi>0 else -1, dt))
