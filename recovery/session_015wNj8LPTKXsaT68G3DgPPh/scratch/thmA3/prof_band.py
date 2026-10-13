import cProfile, pstats, time
exec(open('t3.py').read())
bl=bands()
band=[b for b in bl if b[1]==Fr(1,2**20)][0]
t=time.time(); law=band_law(*band); print('law', time.time()-t)
t=time.time()
cProfile.run('[box_bound(law, bb, kb, band[0], band[1], x0=COMBO[0], theta=COMBO[1]) for bb, kb in BOXES[:8]]', 'prof2.out')
print('8 boxes', time.time()-t)
p=pstats.Stats('prof2.out'); p.sort_stats('cumulative').print_stats(30)
