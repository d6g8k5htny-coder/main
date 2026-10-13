import time
exec(open('t3.py').read())
for (r0, r1) in ((Fr(31,2**17), Fr(1,4096)), (Fr(31,2**14), Fr(1,512)), (Fr(15,4096), Fr(1,256)), (Fr(15,1024), Fr(1,64))):
    t=time.time(); law=band_law(r0, r1); t1=time.time()-t
    for bb, kb in (((Fr(0),Fr(1,512)),(Fr(63,32),Fr(2))), ((Fr(0),Fr(1,4)),(Fr(1,2),Fr(1)))):
        for (x0, th) in ((Fr(1,64),Fr(1,32)),(Fr(1,1024),Fr(1,256))):
            t=time.time(); res=box_bound(law, bb, kb, r0, r1, x0=x0, theta=th); t2=time.time()-t
            print(float(r1), [float(x) for x in bb+kb], float(x0), 'total %.6g main %.6g tail %.3g zt %.6f m3lo %.6f corr %.3g KB %.6f g0 %.6f I3 %.6f A %.6g B %.4g C %.4g D %.4g eps %.2g eta %.2g  [%.1fs law %.1fs]' % (tuple(float(res[k]) for k in ('total','main','tail','zt','m3lo','corr','KB','g0','I3','termA','termB','termC','termD','eps','eta'))+(t2, t1)))
