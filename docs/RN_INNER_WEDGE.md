# Fixed-axis RN bound on a finite-width inner annular wedge

For the normalized SIDE24 Gaussian field, fix `r=1/20`, `b=6/5`,
`M=(-1/40,0)`, `S=(1/40,0)`, and the original six-pin values
`(b,0,0,b-1/48000,0,0)`. On the region

```
W = { (rho*cos(2*pi*t), rho*sin(2*pi*t)) :
      1/10 <= rho <= 11/100, -1/1024 <= t <= 1/1024 },
```

the typed RN integrand, integrated over the full height window
`[b-1/48000,b]`, satisfies the following bounds, **conditional on the
imported fixed-radius H3 floor** `Z >= 0.0077592917375327855`:

```
0 <= I(y) < 1/100000                 for every y in W,
area(W) = 21*pi/5120000,
integral_W I(y) dy < 33/256000000000.
```

Signed turns describe the same wedge as `[0,1/1024]` together with
`[1023/1024,1]`. The pins stay on the x-axis. Varying the moving point's
angle does not establish an all-orientation theorem for the pins.

This is a scoped author candidate with exact replay. The full annulus
`1/10 <= |y| <= 5`, its remote-budget assembly, all-small-radius bounds,
actual weighted-Palm losses, event identification and required independent
acceptance remain open. There is no canonical status change, no original
prize closure and no organizational-independence credit. In particular,
elder pairing is not identified with selected gradient-branch adjacency.

## Proof and exact accounting

The [density-weighted moment lemma](RN_DENSITY_MAJORANT.md) retains the
coupling between explained and residual Gaussian variance. If the same
six-pin conditional three-jet law satisfies `det(Sigma)>=D>0` and full-mark
Mahalanobis energy `q>=Q>=0`, it gives

```
I(y) <= (1/48000)*512*(8+Q)^3*exp(-Q/2)
        / (Z_lo*(2*pi)^(3/2)*sqrt(D)).
```

The six-pin energy bound below 8 is an existing imported proof. Every
original Hessian coordinate has normalized torus variance at most 4;
nonzero image tails are retained. The lemma uses Holder exponents `(3,3,3)`
and assumes no independence between Hessians. It multiplies the determinant
moment and density before using the monotonicity of `(8+q)^3 exp(-q/2)`.

The [spatial determinant proof](RN_WEDGE_DETERMINANT.md) supplies the missing
premises throughout W. Twelve adjacent rectangles partition
`[9999/100000,11/100] x [-7/10000,7/10000]`, which contains W. Each rectangle
has a degree-six Taylor enclosure with exact rational arithmetic at 256
bits. The covariance determinant is formed as a common polynomial before
spatial substitution; all determinant perturbation terms from the L2
remainder are retained. Each rectangle proves

```
det(Sigma) > 8/10^25,     q > 103.
```

The gradient-coordinate lower bound for q holds at every height, so the
entire source height window is covered. This is a continuum enclosure,
not an inference from a point mesh. Positive determinant and the source
law's positive-semidefinite covariance establish nonsingularity; the
independent entrywise interval hull need not be positive definite.

Substituting `D=8/10^25`, `Q=103` gives an exact interval upper strictly
below `1/100000`. The twelve Cartesian areas are not added to the annular
budget. Their common cap applies to the contained wedge; its single polar
cell has exact area `pi*(rho_hi^2-rho_lo^2)*Delta(t)`. The cover ledger has
zero pending cells, and `pi<22/7` gives the stated rational integral bound.
The earlier four N6 pilot squares remain separate; none is added here.

## Replay and evidence

```
python tools/rn_inner_wedge_check.py
python -O tools/rn_inner_wedge_check.py
python -m pytest -q tests/test_rn_density_majorant.py tests/test_rn_n6_inputs.py tests/test_rn_side24_wedge.py tests/test_rn_inner_wedge.py
```

The checker reconstructs and compares the exact candidate at
`research/rn/candidates/inner_wedge_20260920_v1.json`. It authenticates the
previously delivered N6 archive, its original dependency bytes and current
source eligibility before executing the authored Taylor adapter. Historic
RN5 programs remain data. Fresh source checks precede warm-cache reuse and
repeat after the spatial calculation; a cached module is not current custody.
The original H3 floor and pin-energy result remain explicit imported inputs.

Controls exercise the coupled sixth-moment inequality, determinant cross
term/signs, spatial cancellation, all determinant-error products, refusal of
a coarse rectangle, positivity, source exclusion, gaps/overlaps, a weak child
hidden by optimistic aggregates, pending cells, duplicate Jacobians and
incorrect area factors. Failed larger probes are preserved in the registered
delivery. Technical reconstruction and organizational independence are
recorded separately; same-provider review adds zero independence credit.
