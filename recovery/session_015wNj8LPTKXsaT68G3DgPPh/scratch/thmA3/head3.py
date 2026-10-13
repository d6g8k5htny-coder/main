#!/usr/bin/env python3
"""Explicit cap-route constants for Theorem A of [LP] in dimension d = 3, for the torus kernel K_L with L >= 10 in every
frame: for 0 < r <= r_*, b in [0, 1], k in [1/2, 2],

    1 - p_r(b, k, R)  <=  Q^W(G_r^c)  <=  C3(r_*) r^3,        Z_r / r^2 >= z3(r_*),

with G_r the good event of [LP] section 7 / [CAP] (1) (partial-block operator norms on the cylinder
D = [-2r, 2r] x ball(0, 2r)) and C3(r_*), z3(r_*) certified here (NOTE.md Theorem E3).  The proof bounds
Q^W(G_r^c) = E_Q[W_r 1{G_r^c}] / Z_r from above by an exact Gaussian regression near r = 0:

  * covariances of the eight preconditioned pins and the 31 targets are exact power series in r, obtained from the Taylor
    coefficients of the kernel at 0 (every negative power cancels identically, checked); the torus remainder is bounded
    uniformly in r and in the frame by Cauchy estimates on a polydisc in C^3;
  * the failure event is reduced to {lambda_1 <= (4/(3k)) r M_3^2} u {r M_4 > 3k/10}; M_3 is bounded blockwise through exact
    averaging identities of the pinned jets and M_4 through the Taylor series at the midpoint (Frobenius norms);
  * the main term is integrated over the transverse Hessian B = -D_y^2 f(M) in eigenvalue coordinates after an exact
    regression on its three entries, against an angle-free Gaussian majorant of its density; the remaining expectations are
    closed-form incomplete Gaussian and chi moments plus Hoelder-controlled perturbations;
  * the normalizer Z_r is bounded below through a Loewner-order coupling with the contact law, whose moments
    E[det(B)^a tr(B)^c 1{B > 0}] are in closed form; the rare branches are Gaussian and chi tails.
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
MUTANTS = ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term', 'no-lambda2')
MUT = None
PBITS = 160
L_MIN = 10
NX = 60                    # Taylor order in x of the covariance series (exact); the rest is a bounded tail
NB = 24                    # truncation order of the Taylor tails of the 4-jets at the midpoint (the rest is bounded)
X0 = Fr(1, 64)             # default threshold for the quadratic near-branch analysis (records use COMBOS)
THETA = Fr(1, 32)          # default splitting parameter in the max-moment lemma (records use COMBOS)
THETA_E = Fr(1, 16)        # splitting parameter for the ||r nu - v||^2 factor
QS = (2, 4, 8, 16)         # Hoelder exponents of the rare branches: E[W 1_E] <= ||W||_q P(E)^(1 - 1/q), smallest bound used
R0_COUPLING = 12           # radius of the Loewner coupling of the transverse Hessian with the contact law (floor, section 6)
R_STARS = (Fr(1, 4096), Fr(1, 2048), Fr(1, 1024), Fr(1, 512), Fr(1, 256), Fr(1, 128), Fr(1, 64))
BAND_B = (Fr(0), Fr(1))
BAND_K = (Fr(1, 2), Fr(2))


def require(cond, msg):
    if not cond:
        print(json.dumps({'check': 'FAIL', 'reason': msg, 'mutant': MUT}))
        sys.exit(1)


# ----------------------------------------------------------------------------- rational intervals (from Math-#195 / #203 / #206)
