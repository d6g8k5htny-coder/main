import sys, multiprocessing
exec(open('t3.py').read())
F=Fr
BANDS=((F(31,2**17), F(1,4096)), (F(31,2**14), F(1,512)))
LAWS={}
def law_of(i):
    if i not in LAWS: LAWS[i]=band_law(*BANDS[i])
    return LAWS[i]
def bound(bb, kb, i):
    r0, r1 = BANDS[i]
    return box_bound(law_of(i), bb, kb, r0, r1, x0=F(1,64), theta=F(1,32))['total']
TOP=((F(0),F(1,512)),(F(63,32),F(2)))
def targets():
    return [bound(TOP[0], TOP[1], i)*(1+F(1,10000)) for i in range(2)]
def design(args):
    kb, tg = args
    grid=[F(0)]
    b=F(0)
    while b<1:
        ok=lambda bh: all(bound((b,bh),kb,i)<=tg[i] for i in range(2))
        if ok(F(1)):
            grid.append(F(1)); break
        lo, hi = b, F(1)          # lo feasible-ish (b itself), hi infeasible
        lo_n, hi_n = int(b*1024), 1024
        while hi_n-lo_n>1:
            mid=(lo_n+hi_n)//2
            if ok(F(mid,1024)): lo_n=mid
            else: hi_n=mid
        nb=F(max(lo_n, int(b*1024)+1),1024)
        grid.append(nb); b=nb
    return (kb, grid)
if __name__=='__main__':
    tg=targets()
    print('targets', [float(t) for t in tg], flush=True)
    KB=[(F(3,2),F(13,8)),(F(13,8),F(7,4)),(F(7,4),F(15,8)),(F(15,8),F(31,16)),(F(31,16),F(63,32)),(F(63,32),F(2))]
    with multiprocessing.Pool(4) as p:
        for kb, grid in p.imap_unordered(design, [(kb, tg) for kb in KB]):
            print([str(x) for x in kb], len(grid)-1, [str(x) for x in grid], flush=True)
