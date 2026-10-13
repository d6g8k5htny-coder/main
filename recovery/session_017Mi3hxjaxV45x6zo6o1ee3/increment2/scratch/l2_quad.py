import sys; sys.argv=['x']
import pjc
from pjc import D, q_derivatives, coefficient_d2, fmt
q2,q4,q6,_ = q_derivatives(D(2)); m2,m4,m6 = -q2,q4,-q6
k = {2: m2, 4: m4-3*m2*m2, 6: m6-15*m4*m2+30*m2**3}
prev = None
for N in (1024, 2048, 4096):
    c = coefficient_d2(k, N)
    print(N, fmt(c, 24), '' if prev is None else 'diff ' + fmt(abs(c-prev), 3), flush=True)
    prev = c
