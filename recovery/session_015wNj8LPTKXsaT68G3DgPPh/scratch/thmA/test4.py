import time, json
from fractions import Fraction as Fr
import theorem_a as ta
bl = ta.bands()
for i in (0, 60, 128):
    t0 = time.time()
    rec = ta.band_record(bl[i])
    mx = max(rec['boxes'], key=lambda b: Fr(b['bound']))
    print(i, rec['r'], 'max', mx, 'ratio %.5f' % (float(Fr(mx['bound'])) / 5324360.4426), 'time %.1f' % (time.time() - t0))
