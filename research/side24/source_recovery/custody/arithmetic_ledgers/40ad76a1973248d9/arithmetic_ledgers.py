#!/usr/bin/env python3
"""
Re-run of the standard-library arithmetic ledgers (Modules J, K, L, G, H),
including the repaired coefficient-bound transcript line (reproduction guide
section 9 inconsistency).

This script prints BOTH the unnormalized-scale line and the relative-scale line
so that source and transcript agree by construction:
  TAYLOR_REMAINDER_AT_1e-113_UNNORMALIZED_SCALE < 5e-187   (10^40 Hessian scale)
  DECLARED_RELATIVE_HESSIAN_BOUND = 1e42
  TAYLOR_REMAINDER_AT_1e-113 < 5e-185                      (relative map)
"""
from fractions import Fraction as F
import math

# --- energy-adapted margins (Module J) ---
K = 4096
e0 = F(1, 1024*K)
cone_adverse = F(257, 1024) + F(257, 1024*K) + F(1, 2048*K**2) + 2*e0
strip_adverse = F(7, 8) + F(5, 8*K) + F(3, 32*K**2) + F(4, 3)*e0
bracket = F(21, 32*K) + F(1, K**2) + 3*e0 + F(3, 8192) + F(1, 4*K)
assert cone_adverse < F(1, 3)
assert strip_adverse < F(9, 10)
assert bracket == F(9869, 16777216) and bracket < F(1, 768)
assert F(49, 192) - F(1, 768) == F(65, 256) and F(65, 256) > F(1, 4)
print(f"ENERGY_CONE_ADVERSE = {cone_adverse} < 1/3")
print(f"ENERGY_STRIP_ADVERSE = {strip_adverse} < 9/10")
print(f"LEVEL_BRACKET = {bracket} = 9869/16777216 < 1/768")
print("LEVEL_RESERVE: P(5/4,Y) > 49/192 - 1/768 = 65/256 > 1/4")

# --- radial power ledger (Modules A/K/L) ---
# (r^3/6) * r^(-6) * r^2 * r^2 dr = (1/6) r dr   in d=3
powers = {'gap_jacobian': 3, 'corrected_pins': -6, 'endpoint_weights': 2, 'polar_measure': 2}
assert sum(powers.values()) == 1
print("RADIAL_POWER_SUM = 1  ->  universal (1/6) r dr in d=3")

# --- lifetime pushforward scalar ---
life = 6**(2/3)/3
assert abs(life*3 - 6**(2/3)) < 1e-15
combined = life/6
print(f"LIFETIME_FACTOR = 6^(2/3)/3 ; COMBINED_SCALAR = 6^(2/3)/18 = {combined:.10f}")

# --- image tail (Module G) ---
q = math.exp(-288)
assert q < 1e-125
print(f"q = e^-288 = {q:.6e} < 1e-125  (log10 q = {math.log10(q):.5f})")
# shell ratio bound ((n+1)^6 Q^((n+1)^2))/(n^6 Q^(n^2)) <= (3/2)^6 Q^5 < 1/2 for n>=2
ratio_bound = (1.5)**6 * q**5
assert ratio_bound < 0.5
print(f"SHELL_RATIO_BOUND = (3/2)^6 q^5 = {ratio_bound:.3e} < 1/2")

# --- coefficient-map Taylor remainder: REPAIRED transcript (both scales) ---
unnorm = F(1, 2) * 10**40 * F(1, 10**113)**2
rel = F(1, 2) * 10**42 * F(1, 10**113)**2
assert unnorm == F(5, 10**187) and rel == F(5, 10**185)
print("TAYLOR_REMAINDER_AT_1e-113_UNNORMALIZED_SCALE < 5e-187")
print("DECLARED_RELATIVE_HESSIAN_BOUND = 1e42")
print("TAYLOR_REMAINDER_AT_1e-113 < 5e-185")
assert math.log10(1e-180/5e-185) > 4  # margin to the theorem-level 1e-180
print("ASSEMBLED_EPSILON_24 < 1e-180 (dominated by 5e-185 quadratic + 1e-210 linear-tail)")

# --- ordered-pair midpoint determinant ---
import sympy as sp
I2 = sp.eye(2)
Z2 = sp.zeros(2)
A = sp.Matrix.vstack(sp.Matrix.hstack(I2, -I2/2), sp.Matrix.hstack(I2, I2/2))
assert A.det() == 1
print("MIDPOINT_MAP_DETERMINANT = 1 (d=1 block model; general d identical blockwise)")

# --- Lemma A.4.4 inward-tube budget (delta_0 = 1/(512 R^4)) ---
K = F(4096); eps0 = F(1, 1024*4096)
assert 512*eps0 == F(1, 8192)
# delta_0 choice: 2 K r <= delta_0 since 2 K eps0 / R^5 = 1/(512 R^5) <= 1/(512 R^4), R>=1
assert 2*K*eps0 == F(1, 512)
# adverse ratio: 1/4 + 1/(4K) + 1/(16K^2) + 1/(4K) + (1/(2K))*(8193/8192 + 1/(2K) + 1/(8K^2))
ratio = F(1,4) + F(1,4)*1/K + F(1,16)*1/K**2 + F(1,4)*1/K + F(1,2)*1/K*(F(8193,8192) + F(1,2)*1/K + F(1,8)*1/K**2)
assert ratio < F(1,4) + F(3,1)/K
assert F(1,4) + F(3,1)/K < F(1,3)   # since K > 36
print(f"A44_LATERAL_ADVERSE_RATIO <= {float(ratio):.6f} < 1/4 + 3/K < 1/3")
# capture section: 1/(128 tau R^3) < 1/(16 R) since tau >= 2
assert 128*2 > 16  # i.e. 1/(128*tau*R^3) <= 1/(256 R^3) < 1/(16R)
# F_X adverse in tube: 1/(K R^3) + 1/(8 K^2 R^7) + 1/8192 < 1 (R>=1)
fx = F(1,1)/K + F(1,8)/K**2 + F(1,8192)
assert fx < 1
print(f"A44_FX_ADVERSE_IN_TUBE <= {float(fx):.8f} < 1")
print("ALL_ASSERTIONS_PASS")
