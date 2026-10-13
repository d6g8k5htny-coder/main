# D5 nested microdisk: angle-free determinant bound and the axial approach

Object: CL-D5-MICRODISK-AXIAL-20260928-v1. Author: Claude (Anthropic), Claude Code.
Disposition: AUTHOR-SIDE CONDITIONAL REDUCTION; nonauthor review requested.
Scientific effect: NONE. No D5, RN, lifetime, status, graph or prize promotion.

## 1. Sources and the gap

All in `d6g8k5htny-coder/Math-`:

| Source | Role |
|---|---|
| `reviews/d5_short_edge_20260928/NOTE.md` (merged in #103 at `582c05b`) | Critical-segment bound and transverse-cone lemma, sine of the angle >= 1/2 |
| `reviews/pin_micro_covariance_20260926/NOTE.md` | Covariance/density envelope (3.1), (4.2) on the microdisk; author-side, unreviewed |
| `reviews/d5_pin_microdisk_20260927/NOTE.md` | Names the obstruction: the nested microdisk with Q -> 0 is not covered |

Pins: `M=(-r/2,0)`, `S=(r/2,0)`, `f(M)=b`, `f(S)=b-k r^3`, `grad f(M)=grad f(S)=0`,
with b, k in compact sets and `k >= k_- > 0`. Microdisk chart `X = M + r^2 (P,Q)`,
`rho = sqrt(P^2+Q^2) <= K`, `(P,Q) != 0`, `h^2 = Q^2 + r^2 P^2` (as in the covariance note).
Physical `d = |X-M| = r^2 rho`, `sigma = |sin angle(S-M, X-M)| = |Q|/rho`.

The #103 lemma needs `sigma >= 1/2`. Letting sigma vary gives a bound of order `sigma^-4 = rho^4/Q^4`,
which is not integrable as Q -> 0 at fixed P. So a determinant bound is still needed for the
axial approach, where it is the Gaussian density rather than the determinant that must supply the
smallness.

## 2. Deterministic bounds (C^{2,1}, planar, three exact critical points A, B, C)

Notation as in #103: `||H(x)-H(y)||_op <= L|x-y|` on a convex set containing the segments used,
`r=|B-A|`, `d=|C-A|`, `ell=|B-C|`, `Pi = |det H_A det H_B det H_C|`.

**(T2) Angle-dependent version.** For `0 < d <= r/4` and `sigma > 0`,

```
Pi <= (4225/16384) L^6 r^4 d^2 sigma^-4.
```

Proof: as in #103 with sigma kept. `|det H_A| <= L^2 r d/(4 sigma)`. Cramer with
`|alpha|,|beta| <= 1/sigma` gives `||H_A|| <= L(r+d)/(2 sigma)`, so
`||H_B|| <= L(r+d)/(2 sigma) + L r <= (13/8) L r/sigma`. The angle at C has sine `r sigma/ell`, so
`|det H_C| <= L^2 d ell^2/(4 r sigma) <= 25 L^2 r d/(64 sigma)`. The product constant is
`(1/4)(169/64)(25/64) = 4225/16384`.

**(T3) Angle-free soft-column bound.** For any positions of three critical points,

```
Pi <= (1/8) L^3 r d^2 ||H_A|| ||H_B|| ||H_C||.
```

Proof: for any 2x2 matrix H and unit v, `det H = ± det[Hv, Hv_perp]`, so by Hadamard
`|det H| <= ||Hv|| ||H||`. Take v along AC at A and at C (each column `<= L d/2` by the critical-segment
bound) and along BA at B (`<= L r/2`). Only segments AC and AB are used; no angle or `d <= r/4` condition.

## 3. Conditional proposition: the whole microdisk, all directions

Assume, uniformly for compact marks (b, k) with `k >= k_-` and all nonzero `(P,Q)` with `rho <= K`:

- (H1) density envelope, covariance note (4.2):
  `p_{grad f(X)|pins}(0) <= C r^-5 h^-2 exp(-c P^2/h^2)`;
- (H2) normalizer floor: `Z_r = E_{Q_r}[W_r] >= c_Z r^2`;
- (H3) conditional moments: `E_{Q_r}[L^6 + L^3 N^3 | grad f(X)=0] <= C (1 + |P|/h)^m` for a fixed m,
  where L is the Hessian-Lipschitz constant on triangle MSX and `N = max(||H_M||,||H_S||,||H_X||)`;
- (H4) validity of the determinant-weighted Kac–Rice identity.

**Claim.** The expected determinant-weighted witness count in `{rho <= K}` is `O(r^5)`.

Proof. The intensity is `iota(X) <= p(0) E[Pi | grad f(X)=0] / Z_r`, and physical area is `r^4 dP dQ`.

*Region I+II: `|Q| >= r|P|`.* Here `Q^2 <= h^2 <= 2Q^2`. Put `t=|P|/|Q|`, so `P^2/h^2 >= t^2/2`,
`|P|/h <= t` and `rho^6/Q^4 = Q^2 (1+t^2)^3`. By (T2), `Pi <= (1/3) L^6 r^8 Q^2 (1+t^2)^3`. Hence

```
iota <= C r^-5 Q^-2 e^{-c t^2/2} * r^8 Q^2 (1+t^2)^3 (1+t)^m / (c_Z r^2)
     = C' r (1+t^2)^3 (1+t)^m e^{-c t^2/2} <= C'' r.
```

The contribution is at most `C'' r * r^4 * pi K^2 = O(r^5)`.

*Region III: `|Q| < r|P|` (the axial approach).* Here `r^2P^2 <= h^2 <= 2 r^2 P^2`, so
`P^2/h^2 >= 1/(2r^2)`, `|P|/h <= 1/r` and `rho^2 <= 2P^2`. By (T3),
`Pi <= (1/8) L^3 N^3 r^5 rho^2`. Hence

```
iota <= C r^-5 (r^2 P^2)^-1 e^{-c/(2r^2)} * r^5 2P^2 (1+1/r)^m / (c_Z r^2)
     = C' r^-4 (1+1/r)^m e^{-c/(2r^2)},
```

which is bounded (no singularity at P or Q -> 0). The contribution is `O(r^{-m} e^{-c/(2r^2)})`, smaller than every power of r.

Summing gives `O(r^5)`. QED (conditional).

So, conditional on (H1)–(H4), item 2 of the microdisk note's obstruction list (the nested axis
`Q -> 0`) is not a determinant obstruction. Near the axis the mean term in (H1) costs
`exp(-c/(2r^2))`, and (T3) removes the `Q^-4` singularity.

## 4. Where (H3) comes from (sketch for review)

Assume C^3 sample paths with `E sup ||D^2 f||^6 + E sup ||D^3 f||^6 < infinity`. Under
`Q_r(. | grad f(X)=0)` the field is Gaussian.

- **Centered part.** Conditioning lowers covariance in PSD order, so by Jensen/Anderson its convex
  sup-norm moments are at most the unconditional ones.
- **Mean.** The mean is `E[.|pins] - Cov(., G_r|pins) Cov(G_r|pins)^-1 E[G_r|pins]`:
  - the first term is bounded (b, k compact, desingularized pin frame);
  - by (3.1) `|Cov(., G_r|pins)| <= C h` and `||Cov(G_r|pins)^-1|| <= (c h^2)^-1`;
  - by §4 of the covariance note `|E[G_r|pins]| <= C(|P| + h)`.

  So the mean is `O(1 + |P|/h)`, and (H3) holds with m = 6. Similarly, (H2) follows from
  `|det H_M det H_S| = 36 k^2 r^2 S0^2 + O(r^3)` given the pins, positive conditional variance of `S0`,
  and `k >= k_-`.

**Guardrail.** Unscaled height marks break (H2)–(H3). With a fixed O(1) gap `f(S)-f(M)`, the pins force
`E[f_xxx|.] ~ gap/r^3`.

Net effect: on the microdisk, D5's pin-neighbourhood count reduces to the covariance lemma (3.1)/(4.2)
plus regularity and Kac–Rice validity. The same argument extends to the whole pin chart
`p^2+q^2 <= 1/16` if (3.1) is extended beyond bounded (P,Q). Its §2 minor bounds use only `|p| <= 1/4`;
the Taylor remainder at `K ~ 1/(4r)` is not proved.

## 5. Finite evidence (not a continuum proof)

`triangle_bounds_probe.py` (standard library) builds random exact-rational cubics with three prescribed
critical points (exact null-space solve and exact criticality check). L is estimated from below by
dense direction sampling, so a pass with the lower estimate is a pass for the true L.

Seed 20260928; the output is byte-identical in normal and `-O` modes.

| Check | Triangles | Max ratio Pi / bound |
|---|---|---|
| #103 lemma (T1) | 679 | 0.018844 |
| (T2), general form | 1599 | 0.083392 |
| (T3) | 1599 | 0.922652 (nearly sharp) |

The 1599 triangles include 400 nearly collinear ones (sine of the angle from `1e-4` to `1e-1`).

`--mutant-noncritical` (criticality dropped) violates all three bounds by more than 10^6 and exits 1.

## 6. Not established here

(H1) itself (author-side, unreviewed); its extension beyond bounded (P,Q); C^3 regularity of the
periodized field with sixth sup-moments; Kac–Rice validity; the collar to the reviewed annulus;
intermediate scales outside the pin chart; multiple-witness collisions. D5 stays open.
