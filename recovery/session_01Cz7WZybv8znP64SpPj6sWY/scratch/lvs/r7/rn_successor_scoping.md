# Scoping note: round 7, third slice (RN `PinnedLawInterface2`), read-only, 2026-10-13 ~03:55Z

Author: Anthropic Claude, session 01Cz. Scientific effect NONE. Not a proposal yet (the Lifetime slice is proposed first,
main#307 6137849133); this note fixes what the RN probe would have to state, against `Specifications/RN.lean` at
Math- `8acc4856` (= `9fb0a4a0` = the module at `077a168b`).

## What exists in RN.lean (lines at 8acc4856)
- `PinnedLawInterface d [NeZero d] L Ω [MeasurableSpace Ω]` (339): `F`, `smooth`, `periodic`, `Q`, `Q_prob`, `Q_pinned`,
  `W`, `W_nonneg`, `W_typed`, `Z_pos`, `QW`, `QW_eq`, `QW_prob`, `Λ : ℕ → ℝ → ℝ → E d → E d → ℝ` (bare kernel).
- Counts: `windowCount` (393), `windowCountAll` (404), `windowCountIndexSum` (415), `allHeightCount` (424),
  `torusWindowCount` (454), `nearPinPairCount` (470); regions `remoteRegion` (434), `torusMinusPins` (444), `collar` (498),
  `omegaD` (509); means `meanWindowCount` (518), `meanAllHeightCount` (529).
- The 19 interface-blocked Props (548–1001) all quantify over every `I : PinnedLawInterface d L Ω`.

## Successor fields (design note §1), with the Lean shapes the Lifetime probe already validated
1. `Q_regression`: as in `Lifetime2_probe.lean` (pin observations `Fin 2 ⊕ (Fin 2 × Fin d)`, jet covariances from the
   RN module's own covariance kernel — RN.lean has no `periodicCovariance`; the kernel `K_L` would have to be defined in
   RN.lean (theta sum, as Lifetime.lean:95–107 does) or taken as a structure field with its own Gaussian characterisation).
   DECISION NEEDED: define `periodicCovariance` in RN.lean (duplication is the packet's convention, Lifetime.lean:67) or
   add a field. Preferred: define it (anchor rn-fixed-remote-window §1/§2 covariance clause), so `Q_regression` is a
   closed statement about explicit matrices.
2. `pinObsCov_posDef` (source §3 uniform inverse covariance), as in the Lifetime probe.
3. `W_det`: `W r b k R ω = typedWeight d r R (F ω)` with `typedWeight` as in the Lifetime probe (RN.lean has `hessian`
   (137) and `HessianHasInertia` (159)); this replaces the free `W` + `W_typed` by a defined object (keep the original
   fields untouched; the successor adds `W_det` tying them).
4. `Z_floor`: `∃ zStar > 0, ∃ rStar > 0, ∀ r ∈ (0, rStar], ∀ b k R, zStar * r^2 ≤ ∫ ω, W r b k R ω ∂(Q r b k R)`
   (lifetime-parent (5.5) line 180; d5-dimension-lift:280). An assumption of a source theorem: `does_not_claim` names it.
5. `kacRice_window` / `kacRice_allHeight`: `meanWindowCount d L Ω I r b k R j A = ∫ x in A, ∫ y in Ioo (b - k r^3) b,
   Λ' r j b k R x y` (and all-height). `Λ'` needs the conditional density of `(∇f(x), f(x))` given the pins at `(0, y)`
   and `E_Q[W · F_j(H_x) | ∇f(x) = 0, f(x) = y]`. Lean shape: Gaussian density from the regression covariance of the
   `(d+1)`-jet at `x` given the pins (`gaussDensity` as Side24.lean:520 has it, over `Fin (d+1)`), times an integral
   against `multivariateGaussian` of the Hessian coordinates given pins AND the `x`-jet (a second Schur step). Heavy but
   mechanical; the Side24 slice's `condHessianCov` is the template. The field would then be `Λ_def : Λ j b k u x = Λ' …`
   tying the bare kernel to the defined object, and the Kac-Rice identities as fields (Mathlib has no Kac-Rice).
6. `count_finite`/`count_integrable`: a.s. finiteness of the counted sets and integrability of the counts under `QW`
   (lifetime-parent:288). Fields, named as assumptions.

## Expected effect
19 Props become determined by fields; none becomes provable soon: the first-moment bounds reproduce d5-dimension-lift /
rn-annulus-bridge sections (weeks each); Palm-route second moments need marked Kac-Rice for pairs (weeks each);
`WitnessCollisionNearPinPairsTarget` is OPEN in the source (RN.lean scope note), so no field set may make it provable:
its successor would be `specified` with that said in `does_not_claim`.

## Order of work if routed
(a) probe `RN2_probe.lean` (fields 1–4 and 6 first; 5 as a second probe); (b) design read; (c) four-file slice as for
Side24/Lifetime (append to RN.lean, rows, docs), stacked serially. Estimate: a session-evening for (a)+(c) once (b) is
in; counts would move 146 → 165 Props, 11 → 12 structures (after the Lifetime slice).

## Addendum (~04:25Z): both RN probes elaborate at the pin

- `probes/RN2_probe.lean` (fields 1–4 and 6: regression law, nondegenerate observation covariance, defined typed weight,
  normalizer floor on compact windows, a.s. finite critical set, measurable counts): exit 0.
- `probes/RN2b_probe.lean` (field 5): the contact kernel `Λ'_j(x; b, k, R)(y)` is DEFINED from the regression law's
  Gaussian parameters with two Schur steps (the `(d+1)`-jet at `x` given the pins; the three Hessians at `M`, `S`, `x`
  given pins and jet), a Gaussian density on a finite index type (`gaussDensityOn`), a Sylvester-inertia predicate for
  matrices (`MatHasInertia`, the matrix analogue of `HessianHasInertia`), the typed weight `F_d(H_M) F_(d-1)(H_S) F_j(H_x)`
  on Hessian coordinates, and Mathlib's `multivariateGaussian`; the Kac-Rice identities for the window and all-height mean
  counts are fields of `PinnedLawInterface3`. Exit 0.
- Reader items for the eventual design read: (a) sign/scale convention `Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1++l2} K(x-y)`;
  (b) the Hessian coordinates are the raw upper-triangular entries (no `sqrt 2` scaling), consistent with `symOfH`;
  (c) `pinDesc` reuses `site3` with a dummy third point for the pins (cosmetic; a packet version would use a two-site
  helper); (d) the kernel's `Z` argument is the normalizer `∫ W dQ`, so `Z_floor` of RN2 is what keeps `Z⁻¹` meaningful;
  (e) the Kac-Rice identity is a field (Mathlib `d13f23b7` has no Kac-Rice formula), to be named as the source's
  theorem assumed in `does_not_claim`; (f) `Λ` of the parent interface stays the free limit kernel of Theorem A (2),
  which is a statement about `r → 0`, so no `Λ_def` field is proposed; the mean measures themselves become determined.
- Nothing is proposed by this addendum; the Lifetime slice (main#307 6137849133) is the next proposal in line.
