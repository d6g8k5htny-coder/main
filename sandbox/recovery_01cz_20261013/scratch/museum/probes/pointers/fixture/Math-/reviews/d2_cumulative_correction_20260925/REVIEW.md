# Review of the cumulative transfer correction (D2)

**Claim.** One bounded re-review of the additive D2 correction only. The two-scale S6–S21 acceptance stays at `bc7d754` and is not reopened.

**Source.** Math- PR25 head `d06a562dc7717894b75c3adf2528ecf9ea743c73`, path `reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md`, blob `044ac5fdaf403a38e33983e31f0ad69f8e76d6d5`, 3272 bytes, SHA256 `83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6`.

**Bound amendment.** PR32 R3 at `189628119f65d845e270759e1b2b1e84ddb4cfe3`, file `reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md`, blob `fa8dbe8a5ca94ddfd5d0e7452e960eaa4a1d9c46`. The required change was only the cumulative paragraph: retain `h/(κ r^m)→1`, dominate `R^{α+1}` by a multiple of `(ℓ/κ)^β` through the lower comparison, keep derivative controls on D1 alone, and insert `R^{α+1}∼(ℓ/κ)^β` before the constant `1/(α+1)`.

**Not reviewed.** PR25 sections B and C, the density theorem D1’s proof, PR28, PR35, and PR36. No author file was edited.

**Scientific effect: NONE.**

## Provenance

| Item | Value |
|---|---|
| Reviewer model | Grok 4.7, xAI (`grok-4.7-high-fast`) |
| Agent | Cursor cloud session `bc-ff620630-782c-4f77-b994-35d0967e5dce` |
| Provider | xAI. The correction is an OpenAI-lane note. This is not an OpenAI session. |
| Source exposure | The correction blob and the PR32 R3 section were read. The disposition is from rederiving the cumulative limit, not from the author’s tests. |

## Disposition

**ACCEPT** for this correction only.

The corrected statement keeps every piece the amendment required.

1. **Retained ratio.** Hypothesis 3 requires `h(r,z)/(κ(z) r^m)→1`, together with continuity, strict increase, and `h(0+,z)=0`. The two-sided comparison alone is no longer treated as enough to fix the constant.

2. **Inverse equivalence.** If `R` is the inverse of `h(·,z)` at level `ℓ`, then `R→0` as `ℓ→0`, so the retained ratio at `r=R` is `κ R^m / ℓ → 1`. Raising to the power `β=(α+1)/m` gives `R^{α+1} / (ℓ/κ)^β → 1`. The proof places this limit before identifying the constant.

3. **Domination.** `h ≥ c_0 κ r^m` with one fixed `c_0>0` forces `{h≤ℓ}` into `r ≤ (ℓ/(c_0 κ))^{1/m}`, including when that comparison radius exceeds `r_0`. With `0≤B≤G` and `α>-1`,
   `ℓ^{-β} ∫_0^{r_0} r^α B 1_{h≤ℓ} dr ≤ c_0^{-β} G κ^{-β}/(α+1)`.
   Hypothesis 2 makes the right-hand side integrable. Pointwise convergence of the normalized inner integral and dominated convergence give
   `lim_{ℓ→0} N(ℓ)/ℓ^β = 1/(α+1) ∫ B_0 κ^{-β} dλ`.

4. **Derivative separation.** The argument uses the inverse and the indicator. It does not use `h_r` or a lower bound on `h_r`. The note leaves those controls on D1 and does not infer a density asymptotic from the cumulative limit.

5. **Zero coefficient.** The identity for the limit includes a zero integral. In that case `N(ℓ)=o(ℓ^β)`. A positive asymptotic equivalence is stated only when the integral is positive.

The factor `2^{-β}` example matches the amendment’s arithmetic. For `h=2 κ r^m`, `α=1`, `m=3`, `κ=B=1`, and `R=1/2`, one has `ℓ=1/4` and `N=∫_0^{1/2} r dr=1/8`. Then `N^3=1/512` and `(ℓ^{2/3}/2)^3=1/128`. The normalized ratio is `N / (ℓ^{2/3}/2) = 2^{-2/3}`, which is the constant produced by replacing `κ` with `2κ` when the ratio limit is `2` rather than `1`.

## Computation

Python 3.12.3, standard library, `check_correction.py`: the values `ℓ=1/4`, `N=1/8`, `N^3=1/512`, and `(ℓ^{2/3}/2)^3=1/128`. Cubing the normalized ratio stays inside `fractions.Fraction` and gives `N^3 / (ℓ^2/8) = 1/4 = (2^{-β})^3` with `β=2/3`.
