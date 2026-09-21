# Stage-0 Periodized Bargmann–Fock Susceptibility Probe

## Purpose

This experiment tests a prerequisite for the proposed infinite-volume stabilization route: whether
a simple conditioned positive-excursion cluster has an \(L\)-stable expected area away from
critical height.

It does **not** estimate the canonical q₀ pairing probability.

## Configuration

- Torus sizes: \(L=12,24,48\)
- Levels: \(b=1.2,0.8,0.4,0.2\)
- Grid spacing: \(\Delta x=0.25\)
- Samples: 300 per \((L,b)\) cell
- Condition: \(f(0)=b+0.25\)
- Connectivity: periodic four-neighbor
- Seed: 20260714

## Results

| b | L=12 mean area | L=24 mean area | L=48 mean area | L48/L24 |
|---:|---:|---:|---:|---:|
| 1.2 | 6.934 | 6.742 | 7.404 | 1.098 |
| 0.8 | 13.566 | 14.928 | 16.290 | 1.091 |
| 0.4 | 33.814 | 61.973 | 67.448 | 1.088 |
| 0.2 | 48.307 | 130.194 | 230.327 | 1.769 |

The \(L=48/L=24\) ratios are close to one for \(b=1.2,0.8,0.4\), while the \(b=0.2\)
ratio remains large. The cluster-radius and wrap statistics in the CSV show the same qualitative
transition.

## Adjudication

**Measured, Stage-0 only.** The data support finite susceptibility at the higher fixed levels and
move the likely scaling frontier toward \(b\downarrow0\). They do not prove PALM-SUSC or
PALM-CAMP and do not measure \(q_L\).

The experiment omits:

1. maximum and saddle gradient constraints;
2. Hessian typing;
3. determinant Palm weights;
4. the nearby saddle pin;
5. persistence-partner evaluation;
6. continuum critical-point localization.

A full Stage-1 test should use spectral conditional sampling of the maximum–saddle jet, Palm
reweighting by Hessian determinants, adaptive critical-point localization, and a merge-tree
calculation. A uniform grid cannot directly resolve the theorem range \(r\le0.025\) economically.
