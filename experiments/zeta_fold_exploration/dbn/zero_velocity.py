"""Exact zero velocity at t = 0: for a simple zero z_j of H_t, dz_j/dt = -H_t/H_z = H_zz/H_z.
Evaluated with the independent tanh-sinh implementation in indep_ht.py for zeros #1, #2, #5.
Expected: -0.5796306, -0.7549955, -0.4654455 (the finite-difference values in
check_dbn_basic.out agree; the 300-zero symmetric real-zero ODE sum, valid for t > Lambda and
at t = 0 only under RH, agrees to its truncation level of about 1.5e-4: a consistency check).
Stdout saved as results/dbn/zero_velocity.out.
"""
import mpmath as mp

from indep_ht import Hder

mp.mp.dps = 40
for k in [1, 2, 5]:
    z = 2*mp.zetazero(k).imag
    print(k, mp.nstr(Hder(0, z, 2)/Hder(0, z, 1), 12))
