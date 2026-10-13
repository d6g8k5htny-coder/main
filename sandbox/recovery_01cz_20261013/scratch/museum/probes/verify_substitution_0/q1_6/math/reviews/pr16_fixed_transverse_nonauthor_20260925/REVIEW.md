# Nonauthor review — fixed-transverse count candidate

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, or any scientific-status graph. It is not permission to promote the candidate.

## Claim

| Field | Value |
|---|---|
| Object | Math- pull request 16 |
| Immutable commit | `32b80ee085dc6a40113d1e46e333cda50d57ba21` |
| Path | `reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md` |
| Blob | `024d927779f79fadaf932541ea6474f2995f8c50` |
| Size | 9902 bytes |
| SHA256 | `f64c954245f17bcbc64a5ccf58a60689cf2a2fd85362f8321c0f64ff66584e36` |
| Author of the object | OpenAI / ChatGPT, 25 September 2026 |
| Scope | Exact fixed-transverse dimension-two chart only: `A>1`, `B<∞`, `\|v\|≥η>0`, compact marks, qualitative `C` and `r_*` |
| Interfaces | R1, R2, R3, R4, R5 as named in Math- issue 24 |
| Excluded | Axis, thin belt, `η→0`, pin or witness collision, intermediate scales, dimension three, numerical constants, historical 24-jet / all-cell certificate, positive leading coefficient, probability lower bound |
| Also excluded | Math- issue 23, pull requests 19, 21, and 22, and any edit of the reviewed proof |
| State | ACTIVE on publication of this file. Existing 120-minute stale-claim convention runs from this commit's timestamp. |
| Write scope | `reviews/pr16_fixed_transverse_nonauthor_20260925/` only |

Issue 24 asked for an acknowledgment on the issue before ACTIVE. Creating that comment returned HTTP 403, `Resource not accessible by integration`. This file is the claim and the result.

## Reviewer provenance

| Field | Value |
|---|---|
| Provider | xAI |
| Model | Grok 4.7 (run id `grok-4.7-high-fast`) |
| Agent | Cursor cloud run `bc-e46e875d-e53f-42ac-93d9-0d16eadab00c` |
| Run URL | https://cursor.com/agents/bc-e46e875d-e53f-42ac-93d9-0d16eadab00c |
| Account wrapper | Cursor application, owning user Electric_Universe_Theory |

Organizational independence is **not awarded**. A distinct Cursor session is not organizational independence. The shared workspace account does not separate this run from the account that committed the OpenAI text.

Provider independence is different from that organizational fact. The candidate's author is OpenAI. This reviewer is xAI Grok, not an OpenAI session, so the dispositions below are a nonauthor technical review rather than an OpenAI self-review.

## Source exposure

Read in full: the immutable candidate above, Math- pull request 16 body and the three review-request comments, Math- issue 24, and the two meta-framework pull request 6 comments `5839347079` and `5839348136`.

Also read, as coordination rather than as proof: governance- pull request 4 `REVIEW_TOPOLOGY.md` at `6da327b520574ba88de27e70c6456e55326d5dbe`, and governance- pull request 3 file `amendments/20260925-math16-transverse-review.md` at `a1fc2c1f5b1a90d08c4b3cd20b418eeecf8fca1a`. That note is a `cursor[bot]` review whose model is UNKNOWN. It already records R1–R5 ACCEPT. It was not used as analytic evidence. The author's `test_transverse_repair.py` and `test_finite_pin_oracle.py` were not run and are not evidence for these dispositions.

## Method

The continuum argument was checked step by step against the displayed formulas. Separately, `algebra_check.py` (standard library only) expands the displayed cubic in exact rationals. A passing run of that script checks finite identities only.

Command:

```sh
python3 -B -S reviews/pr16_fixed_transverse_nonauthor_20260925/algebra_check.py
```

## Dispositions

| Interface | Disposition |
|---|---|
| R1 endpoint regression and full `Z_r/r^2` floor | **ACCEPT** |
| R2 joint gradient/height normalization and minor `v^6/24` | **ACCEPT** |
| R3 three critical gradients imply all three Hessians `O(r M_3)` | **ACCEPT** |
| R4 uniform conditional `C^3` sixth moment after the witness pins | **ACCEPT** |
| R5 weighted marked Kac-Rice ledger | **ACCEPT** |

No interface is AMEND, COUNTEREXAMPLE, or BLOCKED. A failed interface would have blocked the scoped count. None failed.

Scoped count statement, at this chart only:

`E_(Q_r^W) N_j(rE) ≤ C k r^3 area(E)` for `0<r≤r_*`,

is **ACCEPT** as a qualitative theorem on the declared domain. This acceptance does not award organizational independence and does not authorize a status flip.

## R1 — ACCEPT

The centered transform `U_r` evaluates to the stated target `(b−k r^3/2, −k r^2, 0, 12k, 0, 0)` on every field that meets the six pins. The fourth slot is the reason: `(6/r^2)·(0−2·(−k r^2))=12k`. On a one-variable jet, the same fourth slot has constant term exactly `φ'''(0)`, with the next term `r^2 φ^{(5)}(0)/40`. That is the contact identification `U_0=(f,f_x,f_xx,f_xxx,f_z,f_xz)` with target `(b,0,0,12k,0,0)`.

The periodized covariance has a strictly positive weight on every lattice mode. A finite list of derivative evaluations is therefore nondegenerate whenever those evaluations are linearly independent as distributions. Distinct multi-indices are independent, so both the six contact functionals and the appended `a=f_zz(0)` are jointly nondegenerate in every orthonormal frame. The frame circle and the mark rectangle are compact, the limiting covariance depends continuously on them, and a positive continuous function on a compact set has a positive minimum. For small `r` the transformed covariances stay uniformly invertible. Regression through `U_r`, rather than through the raw colliding pins, is what keeps `C^m` moments bounded.

On the displayed cubic, `f_xx(M)/r=−6k` and `f_xx(S)/r=6k` hold exactly. The mixed terms are `f_xz(M)=−q r/2` and `f_xz(S)=q r/2`, which is the stated `O(r)` size, and

`det H_M / r = −6 k a + 3 k c r − q^2 r/4`,

`det H_S / r = 6 k a + 3 k c r − q^2 r/4`.

The product of the leading terms is `−36 k^2 a^2`. The weight uses absolute determinants, so the limit density is `36 k^2 a^2`. For `k>0` and small `r`, `H_M` is negative definite and `H_S` has index one exactly on the side `a<0`, up to an error that is absorbed by uniform integrability away from the null set `a=0`. The conditional law of `a` given `U_0` is a nondegenerate Gaussian whose variance is the Schur complement of a positive-definite jet covariance, hence bounded below on the compact parameter set, while the conditional mean stays bounded because the target `(b,12k)` stays in a fixed compact. Therefore `E[a^2 1_{a<0} | U_0]` has a positive minimum `z_0`, and `Z_r ≥ z_* r^2` for small `r`. The floor uses the full endpoint weight. It does not import a numerical certificate.

## R2 — ACCEPT

The normalized witness

`J_r = (f_x/r^2, f_z/r, (f−b−(r v/2) f_z)/r^3)`

was expanded on the cubic. The first and third components are exactly the displayed limits, including the finite-pin shifts `u^2−1/4` and `2u^3−3u/2−1/2`. The second component is exactly

`a v + r·((q/2)(u^2−1/4)+c u v+(d/2) v^2)`,

so its limit is `a v`, as written. The height subtraction cancels the quadratic `a` term and the cubic `c` term; the surviving `q` coefficient is `q/4` and the surviving `d` coefficient is `−1/12`.

The Jacobian matrix of `(J_1,J_2,J_3)` in `(a,c,d)`, with the finite-`r` entries of `J_2` retained, has determinant exactly `v^6/24`. On `|v|≥η` this is at least `η^6/24`. Those three jets are unpinned, and `k` enters only as a pinned mean shift. The map is therefore a submersion uniformly on the fixed chart. The conditional law of `(a,q,c,d)` given `U_0` is a nondegenerate Gaussian, so the pushforward law of `J_r` has covariance bounded below for small `r`.

The raw map `(f_x,f_z,f)←J_r` is triangular with determinant `r^2·r·r^3=r^6`. At a critical point of height `b+r^3 τ` the `J_r`-target is `(0,0,τ)`. The Gaussian density of `J_r` is bounded on that compact target set, and

`p(grad f=0, f=b+r^3 τ | pins) ≤ C r^{−6}`.

The height factor `r^3` remains inside the later integral. It is not dropped in this interface.

## R3 — ACCEPT

This step is deterministic and uses only the three gradient zeros. From `grad f(S)=grad f(M)=0` and the integral remainder,

`H_M (S−M) = O(r^2 M_3)`.

Since `S−M=(r,0)`, both `f_xx(M)` and `f_xz(M)` are `O(r M_3)`. Expanding `f_z` from `M` to `X=(r u, r v)` and using `f_z(M)=f_z(X)=0` gives

`v f_zz(M) = −(u+1/2) f_xz(M) + O(r M_3)`.

On the chart, `|v|≥η` and `|(u,v)|≤B`, so `f_zz(M)=O(r M_3)`. Hessian Lipschitz control on the segments, whose lengths are `O(r)`, moves this bound to `S` and to `X`. In dimension two each determinant is quadratic in the Hessian, and three of them produce

`W_r |det H_X| ≤ C r^6 M_3^6`

on the event `grad f(X)=0`. The index indicator only shrinks the left side. The constant depends on `η` and `B`, which the candidate already allows. The estimate is false in the form `O(r M_3)` if `η=0`, and that case is outside the claimed chart.

## R4 — ACCEPT

`J_r` is a linear image of the field, and `Q_r` is Gaussian, so the regression coupling

`F + Cov(F,J_r) Cov(J_r)^{−1} (target−J_r)`

realizes the conditional law given `J_r=(0,0,τ)`. The endpoint pin functionals are almost-sure constants under `Q_r`, so their covariance with `J_r` vanishes and the pins survive.

Under `Q_r`, section 2 supplies uniform `L^2` bounds on derivatives and section 3 supplies uniform `L^2` bounds on `J_r`. Cauchy-Schwarz therefore bounds `sup |Cov(D^3 f, J_r)|` uniformly on the chart. The limiting covariance of `J_r` is positive definite by the minor in R2, so its inverse stays bounded for small `r`. The target `(0,0,τ)` with `τ∈[−k,0]` is compact. The conditional field is the original Gaussian plus a Cameron-Martin shift of bounded `C^3` size and with a smaller covariance, so

`E[M_3^6 | pins, J_r=(0,0,τ)] ≤ C`

uniformly for small `r`. No independence between the endpoint determinants and the witness determinant is required. Combined with the pathwise bound in R3,

`E[W_r |det H_X| 1_{index=j} | pins, grad=0, height=t] ≤ C r^6`.

## R5 — ACCEPT

The quantity `E_{Q_r}[W_r N_j(rE)]` is the intensity of critical points in `rE`, with height in `(b−k r^3, b)` and index `j`, weighted by the endpoint factor `W_r`. At each fixed `r>0` the witness chart stays a positive distance from both pins because `A>1`, so the three-point jet is nondegenerate and the conditional Kac-Rice formula applies. The weight is an integrable field functional: absolute determinants are continuous at singular matrices after the usual truncation, and the Gaussian moments from R4 close the limit.

The resulting integral is at most

`area(rE) · (k r^3) · (C r^{−6}) · (C r^6) = C k r^5 area(E)`,

because `area(rE)=r^2 area(E)`. Division by the full normalizer `Z_r≥z_* r^2` yields `C k r^3 area(E)`. Markov's inequality then bounds the probability of at least one such witness by the same order. That step does not reverse an event probability into an expectation.

The cited Armentano–Azaïs–León framework is used only as the expected-integral representation. The powers and the conditioning are the argument checked here. The ledger stops at the fixed chart. It does not extend to a vanishing transverse coordinate, a collision, an intermediate scale, or a 24-jet certificate.

## Boundary checks that stayed inside scope

The minor `v^6/24` is zero at `v=0`, so the uniform density bound really does need `η>0`. The candidate already excludes that limit. The same cubic satisfies all six pins while keeping the `−1/4` axial corrections, which is the finite-pin repair this chart uses. Those facts support R2 and R3 on the stated domain. They are not an extension of the domain.

## What this review does not do

- It does not edit `TRANSVERSE_BOUND_CANDIDATE.md` or any other author file.
- It does not accept Math- pull request 9, the axis or thin-belt charts, pull request 22's tube extension, or a program-level RN closure.
- It does not treat the governance- `cursor[bot]` ACCEPT, or any unittest, as the proof.
- It does not award organizational independence.
- It does not flip a scientific status bit.

## Computation record

`python3 -B -S reviews/pr16_fixed_transverse_nonauthor_20260925/algebra_check.py` exited 0. Script SHA256: `9f3580d4e6e99b7f3c7409a97123a2a1d20fa63f45a4cae0b8cbbe1046b98c38`. All printed checks passed, including the six pins, the exact `J_1` and `J_3`, the order-`r` formula for `J_2`, both the limiting and finite-`r` minors `v^6/24`, the `U_4` contact term, the two `det/r` expansions, and the exponent identity `2+3−6+6=5`, then `5−2=3`.
