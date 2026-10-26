# V33–V40: D1 mandatory reading rules and global marked Kac–Rice interface

**Audit:** 2026-10-24. **Scope:** independent local calculations and conditional logical reconstructions; NOT a peer-reviewed end-to-end mathematical certificate. No refuted/candidate results are admitted; these are scoped proof *components* explicitly consumed by the reconciled project.

**Mandatory authoritative reading rule:** Original P = `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` must be consumed with (E1) `ERRATUM_CONGRUENCE.md`, (E2) `reviews/d1_section9_borel_repair_20260925/REPAIR.md` **as a full replacement for original P §9**, and (E3) `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md` §5 (W1 and embedding radius). The reconciliation states that these all belong to the same claim identity and upstream source changes trigger downstream revalidation.

## V33 — corrected Hessian congruence (exact independent matrix identity)

Write `H=[[r*alpha, r*beta^T],[r*beta,A]]` for positive `r`, with symmetric `A`. For `D=diag(r^(-1/2),I)`, direct multiplication gives

`D H D = [[alpha, sqrt(r)*beta^T],[sqrt(r)*beta,A]]`.

Also `det(D H D)=det(D)^2 det(H)=r^(-1)det(H)`. Both were rechecked symbolically for transverse dimensions 1–3, with a deliberate countercheck showing the original `diag(sqrt(r),I)` gives a different leading alpha-block and the wrong determinant power. This resolves a **displayed formula error in the original**, not the Gaussian asymptotics.

## V34 — embedding and convergence-scope correction

The product cylinder `[-2r,2r] x closed-ball(0,2r)` lies inside the Euclidean radius `2*sqrt(2)*r` ball. A flat `L`-torus exponential chart is injective on radius `<L/2`; hence the sufficient explicit condition is `r < L/(4 sqrt(2))`. This is exact geometry. The corrected parent only obtains errors of normalized block Hessians tending to zero **in probability relative to A_M**, and `A_M -> A_0` **in distribution**. Slutsky allows the joint distributional conclusion; asserting `A_M -> A_0` in probability would be stronger and is not established.

## V35 — Borel global elder event (conditional topological verification)

For smooth functions on a compact connected manifold, define

`d_f(M)=sup_{gamma(0)=M, f(gamma(1))>f(M)} min_t f(gamma(t))`, empty supremum `=-∞`.

On a Morse/distinct-critical-value locus, strict superlevel connectivity `d_f(M)>h` admits a countable collection of rational polygonal path tests (with locally finite atlas charts and strict clearance). Evaluation of `f` along a fixed finite polygon is continuous in the `C^0` topology, so a countable union of such tests is Borel. A real-valued function whose strict superlevel events are Borel is measurable; equality with `f(S)` is Borel. This checks the logic of the source's measurability reduction, **conditional** on Morse genericity, chart approximation and the stated persistent-homology elder convention. It does not prove genericity for the Gaussian field.

## V36 — C²-valued conditional Gaussian kernel (standard-regression check)

For `G(t)=(∇f(x),∇f(y))` and nonsingular `Σ_G(t)`, the projected residual is

`R_t=f-Cov(f,G(t)) Σ_G(t)^(-1) G(t)`.

Its cross-covariance with `G(t)` is zero by direct expansion, and Gaussianity gives independence. The conditional mean is `Cov(f,G(t)) Σ_G(t)^(-1) z`. Extending the construction to `C²(X)` requires a smooth Gaussian sample version and continuity/uniform nondegeneracy on compact separated pair domains. These are *imports*, not proved merely by this algebraic computation. In particular the remote source-byte integrity and the full-field tightness checks have not been reproduced.

## V37 — finite measure/monotone-class transfer (conditional but mathematically exact)

The corrected §9 introduces finite Borel measures `μ,ν` on `D × C²(X)` and proves they agree on bounded continuous cylinder functions, assuming the continuous-weight version of Kac–Rice. A countable dense set of function-and-derivative evaluations through order 2 together with a countable location basis generates the Borel σ-algebra on the separable space. The class of bounded functions with equal `μ` and `ν` integrals contains constants and the determining algebra and is stable under bounded pointwise limits (finiteness!). By the functional monotone-class theorem it contains every bounded Borel mark, including the global elder mark and type indicators. No false lower-semicontinuity or independence assertion is used.

**Unverified imported theorem:** applicability of Armentano–Azaïs–León Theorems 2.2/7.1 under all necessary conditions is accepted from the project review, not independently checked against the complete v3 paper in this audit. Conditional identity is established; *not* a new general Kac–Rice theorem.

## V38 — exact radial and lifetime Jacobian ledger

The ordered location map `M=z-h/2`, `S=z+h/2` has absolute Jacobian 1. In `h=ru` polar coordinates, `dh=r^(d-1) dr dσ(u)`, where `dσ` is **non-normalized** surface area. For heights `f(M)=b` and `f(S)=b-kr³`, the absolute two-height Jacobian is `r³` at fixed r. The centered-pin density scales as `12 r^(-(d+3)) π_r`, and the endpoint determinant normalizer supplies `Z_r=r² (Z_r/r²)`. Their powers multiply to

`r^(d-1) * r³ * 12r^(-(d+3)) * r² = 12r`.

Thus the candidate intensity is `r A_r dr db dk dσ` with `A_r=12π_r Z_r/r²`. Because roles are an ordered max/saddle pair, **no factor 1/2** is justified. At fixed gap `ell=kr³`, `r dr/dell=(1/3) ell^(-1/3) k^(-2/3)`; checked by differentiation for arbitrary d and symbolic power balancing.

The conditional **pair-count identity** still depends on V37 and the source pin-density construction; the jacobian check itself is exact.

## V39 — all-mark cutoff, Gaussian domination and off-diagonal exhaustion

Near-pair restriction `r≤r0` maps precisely to `k≥ell/r0³`. Assume the source's unnormalized envelope

`0≤A_r≤H(b,k)=C(1+|b|+k)^(2d) exp[-c(b²+k²)]`.

Then `H(b,k) k^(-2/3)` is integrable over `b∈R, k>0`. Near k=0, `∫_0^1 k^(-2/3) dk=3`; the Gaussian tail controls infinity. The exact untruncated Gaussian test integral is `∫_0^∞ k^(-2/3) exp(-c k²) dk = (1/2) c^(-1/6) Γ(1/6)` for `c>0`.

Since the spatial pair intensity is bounded by `C r dr` near zero, `∫_0^{r0} r dr=r0²/2`; nonnegative monotone convergence permits exhaustion by compact off-diagonal subsets. With the displayed unnormalized majorant and pointwise convergence, dominated convergence proves the leading near density limit. **Important limitation:** none of this alone proves the *source's* uniform Gaussian moment estimate yielding `H` or the full global elder theorem; no globally normalized probability floor is used.

## V40 — bounded fixed-far intensity (conditional)

For pairs separated by at least fixed `r0`, assume that the **original** full-pin Gaussian covariance is uniformly nonsingular on the compact separated domain. Conditional Hessian-product moments are bounded by `C(1+|b|)^(2d)`. The original pinned density at heights `(b,b-ell)` is bounded by `C exp(-c b²)` uniformly for `0<ell≤1`. Since `(b,ell)↦(b,b-ell)` has determinant magnitude 1, the total far height-density integral is finite and bounded independent of ell. This proves the **implication** `0≤ν_eld^far≤ν_cand^far≤C`, conditional on full Kac–Rice and covariance/moment hypotheses; not a shrinking-annulus certificate.

## Evidence and remaining gaps

- Exact algebra: V33, V34 (geometric part), V38, V39 (integrals only).
- Source-grounded logical/probabilistic reconstructions: V35, V36, V37, V39, V40 with the stated imports.
- Not checked: authenticated upstream raw Git bytes; direct full-text external Kac–Rice paper v3 theorem hypotheses; the C²-kernel continuity in all required norms; Gaussian all-field moment uniformity; full D1/D2 end-to-end proof; peer review or Lean proof.
- No scientific status was promoted. Previously verified local checks whose **source-explicit** proof lineage has changed are recorded in `LINEAGE_IMPACT_v11_to_v12.json` for reinspection.

## Pinned primary records

1. Reconciliation: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md
2. Required §9 replacement: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/reviews/d1_section9_borel_repair_20260925/REPAIR.md
3. Required matrix erratum: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md
4. Parent: https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md
