# H3 whole-radius band and RN degree-six local bounds

For the normalized SIDE24 Gaussian field, the new fixed-axis H3 result is

```text
(17/10) r² <= Z(r) <= (349/100) r²,       0 < r <= 1/20,
100/(349 r²) <= 1/Z(r) <= 10/(17 r²).
```

Here `K(x,y)=k(x)k(y)`, with
`k(s)=sum_j exp(-(s+24j)^2/2)/sum_j exp(-(24j)^2/2)`.
The points are `M=(-r/2,0)` and `S=(r/2,0)`, with observations
`f(M)=6/5`, `f(S)=6/5-r³/6`, and both gradients zero.
`Z(r)` is the conditional expectation of `|det H_M det H_S|` multiplied
by the indicator that M is a maximum and S is a saddle. The statement covers
every positive radius in the interval, under the canonical Gaussian
regression law at these exact pins. It does not extend to other orientations.

The lower proof uses three independent Gaussian line sectors:
`X=f(x,0)`, `Y=fy(x,0)` and `E=fyy(x,0)+m2 f(x,0)`, where `m2=-k''(0)`.
The pins condition X and Y separately and leave E unchanged. Exact
fourth-derivative integral remainders control the axial Hessians; transverse
midpoint/difference independence and a positive-part determinant inequality
give a typed lower coefficient exceeding `17/10`. This handles the typing
boundary directly, rather than replacing a limit by a finite-radius claim.

For the cancellation-free pin vector
`V=(fM,fxM,fyM,(fxS-fxM)/r,(fyS-fyM)/r,
(fS-fM)/r³-(fxS+fxM)/(2r²))`, with
`v=(6/5,0,0,0,0,-1/6)`, covariance transport gives
`vᵀ Cov(V)⁻¹v <= 1266466/160083 < 8` throughout the band.
The upper proof collects equal covariance powers before interval evaluation,
retains full Taylor remainders, and certifies twenty complete radius cells
`[j/400,(j+1)/400]`, `j=0,...,19`. These are continuum enclosures, not
twenty point evaluations. Both proofs retain kernel normalization and all
omitted image tails.

At `r=1/20`, the band floor is `17/4000=0.00425`. The separately pinned
fixed-radius floor `0.0077592917375327855` is stronger there and remains the
premise used by the RN calculation below. Neither that floor nor an existing
RN budget is silently replaced by the new whole-band constants. The earlier
sharper single-radius ceiling also remains a separate result.

The RN successor uses degree-six spatial Taylor transport at fixed
`r=1/20`, `b=6/5`. All four admitted closed squares cover the full centered
height mark `t in [-1/96000,1/96000]`, equivalently
`v in [b-1/48000,b]`. Three portable witnesses per square bound determinant
moments of orders four at M and S and order two at the moving point: twelve
witnesses in total. Hölder, the three-jet density/window cap and the original
H3 floor give a uniform upper U for the nonnegative typed spatial integrand.

For a square of half-width h, its area is `4h²`, so its integral is at most
`4h² U`. All short bounds below are exact outward rational displays.

| Square center | Half-width h | Integrand upper U | Whole-square integral upper |
|---|---:|---:|---:|
| `(1,1)`, original pilot | `1/4000` | `< 3.81e-6` | `< 9.525e-13` |
| `(1,1)`, wider pilot | `1/20` | `< 3.07e-5` | `< 3.07e-7` |
| `(1/10,0)`, inner x-axis | `1/10000000` | `< 1.18e-303` | `< 4.72e-317` |
| `(0,1/10)`, inner y-axis | `1/100000` | `< 3.38e-5` | `< 1.352e-14` |

On the original square, the earlier N0 upper was approximately `5.589e-6`;
the new upper improves it by more than 30%. The wider pilot has 200 times
the original half-width. The other three admitted squares failed the N0
sufficient pivot test. N6 still refuses the larger half-widths `1/10`,
`1/1000000` and `1/10000` at the respective centers. Those three attempts
remain pending, with their reasons retained; failure of a sufficient pivot
test is not a field counterexample.

The N6 proof uses normalized derivatives through order 18, a 51-target
six-pin lift, exact rational preconditioning and signed common-monomial
aggregation. It retains the order-seven L2 and conditional-mean remainders,
the original-Hessian mean correction `+A0*w`, and the density Jacobian once.
Actual marginal positivity follows by invertible congruence, without assuming
that every independent entry choice in an interval matrix is positive
definite. The extremely small x-axis cap stays strictly positive: a proved
rational exponential bound replaces floating-point underflow to zero.

These squares are local certificates. The original and wider pilots overlap;
their integral bounds must not be added as a partition. Both inner-axis
squares cross the annulus boundary. Their whole-square caps do not assign
annulus area or establish a complete spatial sum. A full RN result still
needs a valid covering partition, exact boundary accounting, zero pending
cells and the remaining integral-budget assembly. All-radius RN transport,
all-angle H3, the full 24-jet chart and elder-pairing/branch-adjacency event
transfer remain separate obligations. This delivery closes none of those
program-level items or the full q0 theorem.

The immutable repository archive is
`research/campaigns/h3_rn_n6_20260920_v1.zip`, SHA-256
`73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21`.
Its original `MANIFEST.json` has SHA-256
`81770e26ec654d3b462f06cebaa20d0dd282cfd4df10ac593c7887ea0d292b40`.
Within that archive, the mathematical sources are `H3_BAND_THEOREM.md`,
`h3_floor/PROOF.md`, `h3_ceiling/PROOF.md`, `pin_energy/PROOF_v2.md` and
`rn_n6/PROOF.md`. Exact local bounds, refusals and witness hashes are in
`rn_n6/results_final/RUN.json`; each H3 folder retains its exact certificate.
The manifest binds these original files without rewriting their scope.

After integration, run the portable wrapper from the repository root:

```sh
python tools/h3_rn_n6_check.py
```

The default route runs five mathematical jobs in normal and optimized Python:
ten executions with 34 expected output comparisons. It checks the immutable
payload and declared repository/source dependencies, including non-Python
source documents. The archive also retains the authors' test receipts and
negative controls. Replay success verifies the recorded computation; it is
not formal proof-assistant validation or external acceptance. This is
source-exposed, same-provider work with zero organizational independence
credit. Canonical scientific status is unchanged and
`original_prize_closed` remains false.
