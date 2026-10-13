# Nonauthor review — fixed-annulus height-window candidate

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, or any scientific-status graph. It does not promote a global RN statement, an all-height annulus theorem, or the separate PR28 candidate.

## Claim

| Field | Value |
|---|---|
| Object | Math- pull request 22, file as of the immutable commit below |
| Immutable commit | `2804dc1db27ef3b1fdef6bb350b486162692cc4a` |
| Path | `frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md` |
| Blob | `081abc13c5e66342c13df2af2bd6b114f320486f` |
| Size | 17646 bytes |
| SHA256 | `1fd9fe7141e464fd1c09ebf0318d8a701729c9c61ea24f73f7e20a9cf10c552b` |
| Author of the object | OpenAI / ChatGPT |
| Scope | One witness in the fixed scaled annulus `1<A0<B0<∞`, dimension two, full periodic covariance at fixed `L`, compact positive marks, height window of length `k r^3` |
| Interfaces | The eight named below |
| Excluded | PR28 and every later PR22 blob, pin neighborhoods, `B0→∞`, `r≪distance≪ρ`, witness collisions, dimension three or more, `k_-→0`, uniformity in `L`, numerical 24-jet constants, a positive leading coefficient, a probability lower bound, and an all-height `O(r^3)` bound on the whole annulus |
| State | Delivery record on publication of this file |

The assigned subject was not retargeted. `PROOF.md` and `TWO_SCALE_ADDENDUM.md` were not edited.

## Reviewer provenance

| Field | Value |
|---|---|
| Provider | xAI |
| Model | Grok 4.7 (`grok-4.7-high-fast`) |
| Agent | Cursor cloud run `bc-3524e567-7204-4e06-a9f5-94d0fd3e7f9d` |
| Run URL | https://cursor.com/agents/bc-3524e567-7204-4e06-a9f5-94d0fd3e7f9d |
| Account wrapper | Cursor application, owning user Electric_Universe_Theory |

Organizational independence is **not awarded**. This run is not the PR16 session `bc-e46e875d-e53f-42ac-93d9-0d16eadab00c` and not the two-scale session `bc-ff620630-782c-4f77-b994-35d0967e5dce`. A distinct Cursor session is not organizational independence.

Provider separation is narrower. The candidate's author is OpenAI. This reviewer is xAI Grok, so the dispositions are a nonauthor technical review.

## Frozen records

`reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md` at `c360f46bf5d3e4bfbb77be98d5e671b5b834d403` is unchanged. Its fixed-chart `η>0` acceptance is not an input to the shrinking cutoff below. The PR25 Sections B–C review is not an input either. Neither file was edited.

## Source exposure

Read in full: the immutable candidate above.

Read as the imported inner bound, not re-derived: `TWO_SCALE_ADDENDUM.md` blob `89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73`, 15902 bytes, SHA256 `079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d`. That blob is identical at parent `b2e1652f1374c3b45759324a1ad1fd4177458500` and at `2804dc1d`. The source-bound S6–S21 acceptance of that blob is `reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md` at `bc7d75454d30d84be9d46c210eefecde71835d1d`, recorded on Math- pull request 34. This review uses equation (S4) from that scope. It does not transfer that acceptance to PR28 or to any changed addendum.

Author unit tests were not used as evidence.

## Method

The continuum steps in §§2–7 and the gluing arithmetic in §§8–9 were checked against the displayed formulas. Separately, `algebra_check.py` (standard library only) checks the pin target, the degree-three contact expansion, the minor `t^6/24`, the triangular Jacobian `r^{-6}`, and the exponent budgets.

Command:

```sh
python3 -B -S reviews/pr22_fixed_annulus_nonauthor_20260925/algebra_check.py
```

Script SHA256: `e7141a60961ad26f344b22b52bc830f130a3d31f36aadb802ca59e750898fc45`.

## Dispositions

| Interface | Disposition |
|---|---|
| Covariance floor `c\|t\|^12` and `O(r)` perturbation | **ACCEPT** |
| Cutoff `r^{1/24}` | **ACCEPT** |
| Rare target in the full joint gradient/height density | **ACCEPT** |
| Conditioned `C^3` sixth moment | **ACCEPT** |
| All-three-Hessian `\|t\|` cost | **ACCEPT** |
| Exponential angular envelope | **ACCEPT** |
| Inner two-scale stitching | **ACCEPT** |
| Whole fixed-annulus height-window count | **ACCEPT** |

No interface is AMEND or BLOCKED. The count that follows is the height-window statement (A2) on this fixed annulus only.

## Covariance floor — ACCEPT

Under the six pins, the degree-three contact calculation gives

`J_r = d(u,k) + M(u,t) V_0 + O_{L^2}(r)`

with the displayed `3×4` matrix `M`. The quadratic `S` term and the cubic `C` term cancel in the third row. The surviving `T` coefficient is `(u^2-1/4)t/4` and the surviving `D` coefficient is `-t^3/12`. The second row of `M` keeps only `t S`; the order-`r^2` pieces of `f_z` become an `O(r)` remainder after division by `r`. That split is an identity on the truncated jet, not an extra hypothesis.

The conditional covariance `G` of `(S,T,C,D)` given the six contact jets is bounded above and below by positive multiples of the identity, uniformly in the frame, because those four monomials stay independent of the six `U_0` derivatives and the frame set is compact. The regression at `U_r=v_r` moves this covariance by `O(r^2)`.

The cross term between `M V_0` and an `O_{L^2}(r)` remainder makes the covariance error `O(r)` rather than `O(r^2)`. On the bounded scaled region,

`||Cov_{Q_r}(J_r) - M G M^T|| ≤ C r`.

The comparison is absolute. The minor of `M` on columns `(S,C,D)` equals `t^6/24`. Cauchy–Binet gives `det(M M^T) ≥ t^{12}/576`. The largest eigenvalue of `M M^T` stays bounded for `|u|,|t|≤B0`, and `G≥λ I`, so

`λ_min(M G M^T) ≥ c |t|^{12}`.

The power 12 is a lower bound, including at `u=0`. It is zero at `t=0`, which this interface does not claim to cover.

## Cutoff `r^{1/24}` — ACCEPT

The floor above is not a fixed-angle compactness argument, and it does not insert a shrinking `η` into the frozen PR16 constants. With `q=1/24` and `|t|≥r^q`,

`r/|t|^{12} ≤ r^{1-12q} = r^{1/2} → 0`.

Any `q` with `12q≥1` would leave a perturbation that does not tend to zero. For small `r`,

`Cov(J_r) ≥ c_1 |t|^{12} I`, `||Cov(J_r)^{-1}|| ≤ C |t|^{-12}`, `det(Cov(J_r))^{-1/2} ≤ C |t|^{-18}`.

On `|t|≥ε` the same conclusion follows by compactness, with room to spare. The cutoff is applied after the eigenfloor and the `O(r)` norm bound are already explicit.

## Rare target in the full joint density — ACCEPT

On the annulus, `|t|≤ε` with `ε^2<A0^2-A1^2` and `A1=(A0+1)/2>1` forces `|u|≥A1`. Then the first coordinate has conditional mean at least `d0/2>0` and variance at most `C(t^2+r^2)`. For `r≤|t|` the variance is `O(t^2)`. The `O(r)` covariance error does not add a bare `O(r)` to this upper bound: the remainder's second moment is `O(r^2)`, and `r=o(t^2)` throughout `|t|≥r^{1/24}`.

At gradient zero and height `b+r^3 τ`, the `J_r` target is `(0,0,τ)` with `τ∈[-k,0]`. For a positive definite covariance,

`z^T Σ^{-1} z ≥ z_1^2 / Σ_{11}`.

The first coordinate alone therefore contributes `exp(-c/t^2)` inside the full three-dimensional quadratic form. Correlations with the height coordinate cannot cancel it. The determinant prefactor is the separate bound `|t|^{-18}`. The unscaled map `(f_x,f_z,f)→J_r` is triangular, and its Jacobian determinant is exactly `r^{-6}` at every `r`; the `(r t/2) f_z` coupling does not change that determinant. Hence, on `r^{1/24}≤|t|≤ε`,

`p(grad=0, height=b+r^3 τ | pins) ≤ C r^{-6} |t|^{-18} exp(-c/t^2)`.

For `|t|≥ε` the same map and a uniform spectral gap give `p≤C r^{-6}` with no exponential. The joint determinant is not inferred from a marginal density.

## All-three-Hessian cost — ACCEPT

The three gradient zeros and the integral remainder give

`||H_M e_1||≤C r M_3`, `||H_M (u+1/2, t)^T||≤C r M_3`.

Those two directions have determinant `t`. For `t≠0`,

`||H_M||+||H_S||+||H_X|| ≤ C r |t|^{-1} M_3`,

after moving the Hessian by at most `O(r M_3)` from `M` to `S` and to `X`. Each determinant is quadratic, so

`W_r F_j(H_X) ≤ C r^6 |t|^{-6} M_3^6`

on the gradient-zero set. The constant depends on the fixed bounds `A0,B0` and is allowed to blow like `|t|^{-6}`. Smallness of an endpoint-only Hessian is not reused after the witness pin.

## Conditioned `C^3` moment — ACCEPT

`Q_r` has uniform `C^3` and `J_r` moments on the bounded scaled region. Each third derivative is stationary and its covariance against `J_r` is bounded by Cauchy–Schwarz, uniformly in the base point. Regression at `J_r=y` adds a Cameron–Martin shift of size at most `C |t|^{-12} |y-μ|` and does not increase the covariance. The original pins are almost-sure constants under `Q_r`, so they stay fixed. For `τ` in the compact window and `r^{1/24}≤|t|≤ε`,

`E[M_3^6 | pins, grad=0, height=b+r^3 τ] ≤ C |t|^{-72}`.

Minkowski's inequality on the pathwise product then gives one correlated triple-determinant expectation of order `r^6 |t|^{-78}`. No factorization between the endpoint Hessians and the witness Hessian is used. For `|t|≥ε` the inverse covariance stays bounded and the same expectation is `O(r^6)`.

## Exponential angular envelope — ACCEPT

The sites `M`, `S`, and `X` stay at least `r(A0-1/2)` apart. Distinct-site derivative evaluations are nondegenerate for this covariance because every lattice weight is positive. The weighted, height-disintegrated intensity is

`ρ = Z_r^{-1} ∫_{b-k r^3}^{b} p(grad=0, height=s) E[W_r F_j(H_X)|grad=0, height=s] ds`.

The cited Kac–Rice theorem is used only as this integral representation. The powers are computed here. The original normalizer is re-derived in §2 of the candidate, not quoted from the frozen PR16 review: `S_r=f_{zz}(0)` is conditionally Gaussian with bounded mean and variance bounded above and below, so `P(-2≤S_r≤-1)≥p_0>0`, and Markov removes an `O(r^p)` error event. The surviving event has `W_r≥c r^2`, hence `Z_r≥c_Z r^2`.

On `r^{1/24}≤|t|≤ε` the pieces are normalizer `r^{-2}`, density `r^{-6}|t|^{-18} exp(-c/t^2)`, triple-determinant moment `r^6|t|^{-78}`, and window length `k r^3`. The product is

`ρ ≤ C k r |t|^{-96} exp(-c/t^2)`.

The angular total is `18+72+6=96`. With `y=c/t^2`, the elementary bound `e^y≥y^{48}/48!` gives `|t|^{-96} exp(-c/t^2)≤48! c^{-48}`. Thus `ρ≤C k r` on that region. The same `O(k r)` bound holds for `|t|≥ε` by the non-exponential estimates. The `r` ledger is density `-6`, determinants `+6`, height `+3`, normalizer `-2`.

## Inner two-scale stitching — ACCEPT

This acceptance is the gluing step. It is not a second proof of S6–S21.

Equation (A25) is equation (S4) of the addendum blob named above:

`ρ_{all}(ru,rt) ≤ C r^{-5} δ^{-8} exp(-c/δ^2)`,

for `A≤|u|≤B` and `δ=√(r^2+t^2)≤δ_0`. Sections 2–7 of that addendum are the argument accepted at `bc7d754`. Inside `|t|<r^{1/24}` on this annulus, small `r` gives `|u|≥A1>1`, `|u|≤B0`, and `δ≤δ_0`, so the hypothesis of (S4) holds. A height restriction only decreases the intensity. The in-file minor algebra is the same rank input that review already accepted; it is not re-opened here.

The new arithmetic is the cutoff comparison. Here `δ≥r` and `δ^2≤2 r^{1/12}`, so

`ρ_{all} ≤ C r^{-13} exp(-c/(2 r^{1/12}))`.

The bound `e^y≥y^{168}/168!` produces `r^{14}`, and `r^{-13}·r^{14}=r`. Since `k≥k_->0`, this is `C' k r` with `C'=C/k_-`. That constant is not uniform as `k_-→0`, which the candidate already says.

## Whole fixed-annulus count — ACCEPT

Every point of `K` falls in `|t|≥r^{1/24}` or in `|t|<r^{1/24}`. Both pieces have height-window intensity at most `C k r` per physical area. The boundary circle has expected count zero because the Kac–Rice measure is absolutely continuous. Integrating over `r E_0`, whose area is `r^2 area(E_0)`, gives

`E_{Q_r^W} N_j(r E_0) ≤ C k r^3 area(E_0)`, `0<r≤r_*`.

The constant depends on `L,A0,B0` and the mark compacts, and it is uniform in the frame, in the Borel set `E_0⊂K`, and in the index `j`. The all-index sum changes only `C`. Markov's inequality bounds the probability of at least one such witness by the same order.

This is the height-restricted statement (A2). It does not give an all-height bound on the whole annulus. The outer region uses the shrinking height window; the inner strip uses the imported all-height tail. The two mechanisms are not interchangeable.

## What this review does not do

It does not edit an author source, flip a scientific flag, or award organizational independence. It does not accept PR28, a later rewrite of this candidate, a vanishing mark `k_-`, or a numerical radius. The finite rational script does not replace the covariance, conditioning, or Kac–Rice steps above.
