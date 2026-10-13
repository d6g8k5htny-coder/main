import math, random
import fcore2 as F
from test_boxes import ref, Ma, Mb, Mc, sa2, sb2, sc2
random.seed(5); bad = 0; imp = 0; n = 0
for t in range(3000):
    sz = 10 ** random.uniform(-2.5, -0.5)
    b0 = random.uniform(-4, 0.3); a0 = random.uniform(-3, 3) * math.sqrt(sa2); c0 = random.uniform(-3, 3) * math.sqrt(sc2)
    box = ((a0, a0 + sz * math.sqrt(sa2)), (b0, b0 + sz * math.sqrt(sb2)), (c0, c0 + sz * math.sqrt(sc2)))
    Bl, Bh, ul, uh = F.box_geometry(box)
    s, wl, wh, Tl, Th = F._wT(ul, uh)
    if Bl >= 0 or s == 0 or Tl >= F._half_cube(Bl)[1] or (Bh < 0 and Th < F._half_cube(Bh)[0]):
        continue                     # only boundary-straddling boxes
    n += 1
    I = F.H2_box(box, Ma, Mb, Mc); I1 = F.H2_box(box, Ma, Mb, Mc, second=False)
    r = ref(box, 'H2', 40)
    tolr = 1e-6 * abs(r) + 1e-15
    if not (I[0] - tolr <= r <= I[1] + tolr):
        r2 = ref(box, 'H2', 120)
        if not (I[0] - 1e-7 * abs(r2) - 1e-15 <= r2 <= I[1] + 1e-7 * abs(r2) + 1e-15):
            bad += 1; print('FAIL', box, I, r, r2)
    if I[1] - I[0] < 0.5 * (I1[1] - I1[0]): imp += 1
print('boundary-straddling H2 boxes', n, 'bad', bad, 'improved >2x', imp)
