#!/usr/bin/env python3
"""Explicit constants for Theorem A of [LP] in the plane (d = 2), for the torus kernel K_L with L >= 10 in every frame:
for 0 < r <= r_*, b in [0, 1], k in [1/2, 2],

    1 - p_r(b, k, R)  <=  Q^W(G_r^c)  <=  C(r_*) r^3,

with G_r the good event of [LP] section 7 / [CAP] (1) and C(r_*) certified here (NOTE.md Theorem E).  The proof bounds
Q^W(G_r^c) = E_Q[W_r 1{G_r^c}] / Z_r from above by an exact Gaussian regression near r = 0:

  * covariances of the preconditioned pin and target functionals are exact power series in r, obtained from the Taylor
    coefficients of the kernel at 0 (every negative power cancels identically, checked); the torus remainder is bounded
    uniformly in r and in the frame by Cauchy estimates on a polydisc;
  * the failure event is reduced to {lambda <= (4/(3k)) r M_3^2} u {r M_4 > 3k/10}; M_3 is bounded through exact averaging
    identities of the pinned jets and M_4 through the Taylor series at the midpoint;
  * the main term is integrated in lambda = -f_yy(M) against its Gaussian density after an exact regression on w = f_yy(M);
    the remaining expectations are closed-form incomplete Gaussian moments plus Hoelder-controlled perturbations;
  * the normalizer Z_r is bounded below in closed form; the rare branches are Gaussian tails.
All arithmetic is exact rational interval arithmetic (outward rounding to 2^-160); standard library only.
"""
import argparse
import json
import math
import multiprocessing
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term')
MUT = None
PBITS = 160
L_MIN = 10
NX = 60                    # Taylor order in x of the covariance series (exact); the rest is a bounded tail
NB = 24                    # truncation order of the Taylor tails of the 4-jets at the midpoint (the rest is bounded)
X0 = Fr(1, 64)             # threshold for the quadratic near-branch analysis
THETA = Fr(1, 32)          # splitting parameter in the max-moment lemma
THETA_E = Fr(1, 16)        # splitting parameter for the (r nu + v)^2 factor
R_STARS = (Fr(1, 4096), Fr(1, 2048), Fr(1, 1024), Fr(1, 512), Fr(1, 256), Fr(1, 128), Fr(1, 64))
BAND_B = (Fr(0), Fr(1))
BAND_K = (Fr(1, 2), Fr(2))


def require(cond, msg):
    if not cond:
        print(json.dumps({'check': 'FAIL', 'reason': msg, 'mutant': MUT}))
        sys.exit(1)


# ----------------------------------------------------------------------------- rational intervals (from Math-#195 / #203)
