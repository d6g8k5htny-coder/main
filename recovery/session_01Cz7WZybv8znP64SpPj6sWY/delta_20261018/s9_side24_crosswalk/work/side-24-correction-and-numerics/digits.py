import sys
sys.argv=['x']
exec(open('<SCRATCH>/s9/work/side-24-correction-and-numerics/numerics.py').read().split("for kk, v in out.items()")[0])
print('059(25) printed :', abs25); print('059(25) computed:', mp.nstr(R1*cref, 50))
print('059(27) printed :', abs27); print('059(27) computed:', mp.nstr(24**3*R1*cref, 50))
print('063(3) printed :', f063_3); print('063(3) computed:', mp.nstr(mp.mpf(r063_1)-R1, 125))
print('063(2) printed :', a063_2[-12:], ' computed tail:', mp.nstr(mp.mpf(r063_1)*cref, 285)[-14:])
sref=mp.nstr(cref,140,strip_zeros=False); s24=mp.nstr(c24,140,strip_zeros=False)
print('cref decimals 110-125:', sref[2+109:2+125]); print('c24  decimals 110-125:', s24[2+109:2+125])
# position check: decimal place index of first difference
for i,(x,y) in enumerate(zip(sref[2:],s24[2:]),1):
    if x!=y: print('first differing decimal place =',i, x, y); break
print('enclosure lower/upper shared prefix:', enc['dimensions']['3']['lower'][:21], enc['dimensions']['3']['upper'][:21])
