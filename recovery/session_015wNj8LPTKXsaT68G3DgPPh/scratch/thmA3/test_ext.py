import time, multiprocessing
exec(open('t3.py').read())
bl=bands()
sel=[bl[0], bl[1], bl[-1], bl[-2]]
def job(band):
    t=time.time(); rec=band_record(band); return band, rec, time.time()-t
with multiprocessing.Pool(4) as p:
    for band, rec, dt in p.imap_unordered(job, sel):
        mx=max(Fr(x) for x in rec['bound']); zm=min(Fr(x) for x in rec['zfloor'])
        print([str(x) for x in band], 'max %.6g zmin %.6f tail_max %s  [%.0fs]' % (float(mx), float(zm), rec['tail_max'], dt), flush=True)
