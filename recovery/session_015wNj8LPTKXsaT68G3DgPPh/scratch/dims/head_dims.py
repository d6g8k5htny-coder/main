#!/usr/bin/env python3
"""The cap-failure coefficient of [LP] Theorem A's proof route in every dimension: certified values in d = 3.

Theorem G_d (NOTE.md): for the d-dimensional six-pin law of [LP] and the good event G_r of [LP] section 7 / [CAP] (1)
(partial-block operator norms), Q^W(G_r^c) = c_G^(d)(b, k) r^3 + o(r^3), and for the reference kernel

    c_G^(d)(b, k) = R_d(b) R(b) J^(d)(k),

R_d the hard-direction factor of Math-#184, R the planar factor of Math-#203, J^(d)(k) = (216 k^3)^-1 E[ int_{max(a,c)}^{8M^2}
(s - a)(s - c) ds ] with M = max(12k, 2|v|, ||Omega||_op, ||Y||_op) over the full transverse blocks.

This script (standard library, exact rational interval arithmetic):
  * checks the d = 3 contact law of the jets exactly (rational covariances of the reference kernel);
  * certifies R_3(0) = (32 + 28 sqrt2)/17 and R(0) = 1/(2 sqrt(pi));
  * brackets J^(d)(k) for d = 2, 3 from Gaussian moments and chi-square tails (self-contained), checks that the d = 2
    bracket contains the certified planar J(k) of [G] = Math-#203, and encloses J^(3)(k) by the monotone coupling
    J^(2) <= J^(3) <= J^(2) + tail;
  * evaluates c_G^(3)(0, k); floating R_3(b) R(b) and Monte Carlo J^(3)(k) are recorded as controls.
"""
import argparse
import json
import math
import os
import random
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = ('no-y-tail', 'r3-one', 'no-bp', 'frob-omega-off')
MUT = None
PBITS = 160
K_GRID = (Fr(1, 2), Fr(3, 4), Fr(1), Fr(3, 2), Fr(2))
# [G] Math-#203 at a388580, RESULTS.json (git blob e3a8e25a49fb572f0e3fb95c88f070ef9fa9d731): certified planar J(k)
JG = {Fr(1, 2): ('301744.4364470241', '301744.4616663935'),
      Fr(3, 4): ('995454.6123240485', '995454.6129283207'),
      Fr(1): ('2359292.9611053703', '2359292.9611122569'),
      Fr(3, 2): ('7962621.6645357431', '7962621.6645378716'),
      Fr(2): ('18874366.3403946311', '18874366.3403964820')}


def require(cond, msg):
    if not cond:
        print(json.dumps({'check': 'FAIL', 'reason': msg, 'mutant': MUT}))
        sys.exit(1)


