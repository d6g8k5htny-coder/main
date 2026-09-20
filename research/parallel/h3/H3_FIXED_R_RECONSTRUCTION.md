# Fixed-r H3 floor: exact source-exposed reconstruction

This author-side candidate reconstructs the lower-bound computation at
`r = 1/20`, `b = 6/5` for the normalized side-24 two-dimensional Gaussian
field. It is not peer review, earns zero organizational independence credit,
and changes no scientific status. It supplies no spatial cover, all-small-r
floor, historical upper-endpoint replay, or closure of the RN uniform lemma.
R17 target: `PARALLEL-H3-RUNG-FLOOR-20260920-v1`; claim
`15219ecb-5b2e-4f44-9d58-d349cd244866`.

## Exact conclusion

Let `Q` condition on the raw six pins

`(f(M), fx(M), fy(M), f(S), fx(S), fy(S)) = (b, 0, 0, b-r³/6, 0, 0)`,

where `M=(-r/2,0)`, `S=(r/2,0)`. Define

`Z = E_Q[|det H_M det H_S| 1{H_M negative definite, H_S saddle}]`.

The reconstructed source box proves the exact imported floor

`Z >= 0.0077592917375327855`.

The interval evaluation of its box contribution is contained in the following
outward rational decimal bracket (these are exact terminating decimals):

`[0.0077592917375327858692, 0.0077592917375327858693]`.

Thus the source's displayed imported decimal is strictly below the recomputed
box contribution. No historical rounding defect is established. The report
retains full rational endpoints and uses those, never displayed approximations,
for every comparison.

A separately labeled successor changes only the first upper box edge from
`18/25` to `721/1000`. Its contribution is enclosed by

`[0.0077593156978542075136, 0.0077593156978542075137]`,

which separately proves the conservative floor `0.007759315`. This does not
alter the original frozen source, nor does it change any repository consumer.

## Source custody and source exposure

The H3 prose member is
`intake/rn_source/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md`,
7,003 bytes, SHA-256
`6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa`,
inside RN5 archive SHA-256
`28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e`.

The historical algorithm is
`K3_SIDE24_LB/UPPER2D/H3_closure/h3_rung_floor.py`, 16,960 bytes,
SHA-256 `094d382451ff37e8d3123e7203c743f359cb0af1c2c12cf336adf74fc0b25d8f`,
in active Drive archive `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`, 30,148,285 bytes,
SHA-256 `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`.
The archive was reconstructed from the existing verified local chunk store.
`Store.object_chunks` checks each chunk and, at iterator exhaustion, the total
byte count and reconstructed archive SHA. The complete join exhausts it before
any ZIP member is read. Target members are checked against logical exclusions
before access; their exact hashes are checked again. No vault, legacy or
quarantine body was opened. Historical source was read, not executed.

The author is exposed to the historical proof and source and uses the same
safe-box strategy. The implementation is a new exact-arithmetic reconstruction,
not an independent review lineage. It reuses the repository's certified
interval primitives, normalized torus covariance and interval conditioning.
All loaded repository Python source identities are retained and checked for
change during the run. Source identities authenticate bytes, not authority.

## Pin-law and determinant scaling

Use the invertible linear transformation

```
V0 = f(M)
V1 = fx(M)
V2 = fy(M)
V3 = (fx(S)-fx(M))/r
V4 = (fy(S)-fy(M))/r
V5 = (f(S)-f(M))/r³ - (fx(S)+fx(M))/(2r²).
```

The stipulated raw pins give exactly `(b,0,0,0,0,-1/6)`. Conversely,

```
fx(S) = r V3 + V1
fy(S) = r V4 + V2
f(S)  = r³ V5 + V0 + r(r V3 + 2 V1)/2.
```

The checker verifies this exact rational roundtrip. Conditioning on V is
therefore the same Gaussian disintegration as conditioning on the raw pins.

Set `q=fyy`, `alpha=fxx/r`, `beta=fxy/r` at each station. Then

`H = [[r alpha, r beta], [r beta, q]]`,

so `det H = r(alpha q-r beta²) = r D`, and
`det H_M det H_S = r² D_M D_S`. A separate formal polynomial calculation
checks this coefficient by coefficient over exact Fractions, before any
Gaussian integral. Removing `r²` is a rejected mutation. The result remains
in the original Hessian units; there is no hidden normalization factor.

## Certified law and safe box

Order the soft coordinates as
`Y=(q_M,q_S,alpha_M,alpha_S,beta_M,beta_S)`. Form their joint covariance with
V directly from the normalized periodized kernel using the current exact
torus adapter. The full image tail and normalization denominator are retained.
The existing kernel proof establishes a positive semidefinite Gaussian
derivative Gram law; the six positive pin LDL pivots verify invertibility for
this conditioning. Schur-complement solves enclose the actual conditional
mean and covariance without a point-valued inverse.

Six further positive soft LDL pivots yield enclosures of a lower-triangular
factor `T=L sqrt(D)`, so the actual law admits `Y=mu+T U`, with independent
standard normal coordinates U. Each exact covariance member follows the same
LDL recurrences, whose denominators have positive lower bounds. Interval
dependency can widen these enclosures, but cannot discard the actual law.

The original rational box is

```
low  = (-7, -8, -6, -8, -12/5, -8)
high = (18/25, 2, 6, 8, 12/5, 8).
```

Interval evaluation proves `q_M<0`, `q_S<0`, `alpha_M<0`, `alpha_S>0`,
`D_M>0`, `D_S<0` everywhere on this entire box. Hence M is a strict maximum
and S a saddle there. In particular `|D_M D_S|=-D_M D_S` on the box. Outside
it the defining integrand for Z is nonnegative. Therefore

`Z >= r² E[-D_M D_S 1{U in B}]`.

The successor box is checked afresh for all the same strict signs. No station
sampling or Monte Carlo enters either certificate. Exact-torus corrections
smaller than the working enclosure width remain contained; this does not
replay the historical separate strict nonplanarity diagnostic.

## Integral without quadrature

Each soft coordinate is affine in U, each D is quadratic, and `-D_M D_S` is
a quartic. The implementation expands its 93 retained interval coefficients
with outward rounding after operations. For each multi-index e, independence
of the actual standard Gaussian coordinates gives

`E[prod_i U_i^e_i 1_B] = prod_i I(e_i; low_i, high_i)`.

Signed truncated moments are evaluated by

```
I0 = Phi(high) - Phi(low)
I1 = phi(low) - phi(high)
In = low^(n-1) phi(low) - high^(n-1) phi(high) + (n-1) I(n-2).
```

This follows by integrating `phi'(u)=-u phi(u)` by parts. Only degrees 0–4
are needed. The certified repository Phi and normal-density routines provide
containing intervals; exact rational endpoints and outward rounding preserve
containment throughout the finite sum. No unbounded quadrature or heuristic
remainder is used. Polynomial size is bounded by 210 monomials and degree 4.

## Reproduction and limitations

```
python3.11 -B verify_h3_floor.py --repo /absolute/path/to/research-main \
  --output /absolute/path/outside/repository/fresh-report.json
```

The output parent must exist; existing reports are refused, and repository
outputs are forbidden. `run_validation.py` runs normal and optimized modes,
checks identical report bytes, and executes five real CLI mutations in both
modes: wrong six-pin window, unsafe box, determinant-product sign, oversized
floor and missing r² scale. Every mutation must exit 1 with no report. It also
checks symmetric odd moments and the degree-2/4 integration-by-parts formulas.
No model-generated confidence score or majority vote is used.

Executed validation: 47 positive/in-process checks, five in-process rejected
controls, two successful normal/optimized CLI runs with byte-identical reports
and standard output, and ten rejected CLI mutations (five in each mode).
The separate overwrite guard refused an existing output and retained its exact
hash. `validation-receipt.json` and the process logs record actual exits.
The report payload SHA-256 is
`5f1cd0c8176f89b4b975b20f30c0e8d819c17a1eb488bd121558edfe2f27b2ff`;
the complete report-file SHA-256 is
`fcbd332394df44fa400dee83200de7291bfa8d68bdb58a90c7f53540926f6ef3`.
The preliminary scratch `result.json` is not part of the delivered certificate;
only the named `validated-*` reports are included in the final manifest.

This candidate can replace the *computationally unreplayed fixed-r lower-floor
dependency* after the sole repository writer verifies and integrates it. That
engineering change must not erase authorship, source exposure, kernel/source
dependencies, scientific-status boundaries, or independent-review obligations.
The old source's upper interval, its neighboring rungs and all-r band floor
are outside this delivery.
