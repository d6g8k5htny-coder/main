import sys, time
import fcore2 as F
from ia import isqrt, dn, up
from drive2 import setup
k = float(sys.argv[1]); which = sys.argv[2]; tol = float(sys.argv[3])
sig, (Ma, Mb, Mc) = setup(k)
fn = {'V': F.V_box, 'H2': F.H2_box, 'H': F.H_box}[which]
sd = (sig[0][1], sig[1][1], sig[2][1]); L = 8.0; n0 = 8
edges = [[-L * s[1] + 2 * L * s[1] * i / n0 for i in range(n0 + 1)] for s in sig]
cells = [((edges[0][i], edges[0][i + 1]), (edges[1][j], edges[1][j + 1]), (edges[2][l], edges[2][l + 1])) for i in range(n0) for j in range(n0) for l in range(n0)]
cat = {}
for cell in cells[::7]:            # a deterministic 1/7 subsample of the initial cells
    stack = [(cell, 0)]
    while stack:
        box, dep = stack.pop()
        I = fn(box, Ma, Mb, Mc)
        if I[1] - I[0] > tol and dep < 60:
            w = [(box[i][1] - box[i][0]) / sd[i] for i in range(3)]
            d = w.index(max(w)); l_, h_ = box[d]; mid = 0.5 * (l_ + h_)
            for half in ((l_, mid), (mid, h_)):
                nb = list(box); nb[d] = half; stack.append((tuple(nb), dep + 1))
        else:
            Bl, Bh, ul, uh = F.box_geometry(box)
            I1 = fn(box, Ma, Mb, Mc, second=False)
            if ul <= 0 <= uh:
                c = 'straddle u=0, B>0' if Bl > 0 else ('straddle u=0, B<0' if Bh < 0 else 'straddle u=0, B~0')
            elif I[1] - I[0] >= I1[1] - I1[0]:
                c = 'first-order (other)'
            else:
                c = 'second-order'
            n, s = cat.get(c, (0, 0.0)); cat[c] = (n + 1, s + (I[1] - I[0]))
tot = sum(v[1] for v in cat.values())
for c, (n, s) in sorted(cat.items(), key=lambda t: -t[1][1]):
    print('%-24s leaves %7d  width sum %.3e  (%.1f%%)' % (c, n, s, 100 * s / tot))
