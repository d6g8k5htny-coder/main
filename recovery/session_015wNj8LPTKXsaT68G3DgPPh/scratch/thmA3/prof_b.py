import cProfile, pstats, time
exec(open('t3.py').read())
r0, r1 = Fr(31,2**14), Fr(1,512)
law=band_law(r0, r1)
bb, kb = (Fr(0),Fr(1,512)),(Fr(63,32),Fr(2))
cProfile.run('box_bound(law, bb, kb, r0, r1, x0=Fr(1,64), theta=Fr(1,32))', 'prof.out')
p=pstats.Stats('prof.out'); p.sort_stats('tottime').print_stats(20)
