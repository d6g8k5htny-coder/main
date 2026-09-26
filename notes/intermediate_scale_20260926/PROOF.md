# Intermediate physical scale: uniform envelope between the fixed annulus and the fixed remote window

**Object:** D5-INTERMEDIATE-SCALE-20260926-v1.
**Author of this note:** xAI / Grok 4.7, Cursor cloud session `bc-235357c8-228c-432e-9139-26ec8b5eab2f`.
**Disposition:** additive nonauthor derivation. Independent re-review of this note is open.
**Scientific effect:** NONE. No parent proof is edited. No global RN, elder, lifetime, or 24-jet status is changed.

## 1. The overlap this note fills

Two reviewed statements bound one extra critical point in the between-pin height window, in dimension two, for the exact periodized Gaussian field at a fixed torus size L.

The fixed scaled annulus, source `frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md`, blob `081abc13c5e66342c13df2af2bd6b114f320486f`, SHA256 `1fd9fe7141e464fd1c09ebf0318d8a701729c9c61ea24f73f7e20a9cf10c552b`, reviewed in `reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md`, gives physical-area intensity at most `C(B0) k r` on the scaled annulus `1<A0≤|y|≤B0<∞`. The constant depends on the fixed outer scaled radius `B0`.

The fixed remote window, source `frontiers/remote_window_20260924/PROOF.md`, SHA256 `a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7`, reviewed on main issue 76, gives physical-area intensity at most `C(ρ) k r^3` on the region at distance at least a fixed `ρ>0` from the pins. The constant depends on that fixed exclusion.

The open overlap is the intermediate physical scale

    r ≪ |x| ≪ 1,

equivalently a growing scaled radius `B(r)=|x|/r→∞` whose physical radius `|x|=r B(r)` still tends to zero. Pin neighborhoods and mutually shrinking witness separations are not treated.

**Theorem (uniform intermediate envelope).** Fix L>0 and the centered variance-one field on `R^2/(L Z^2)` with the full covariance `K_L`. Let marks `b` and `k` range over fixed compacts with `0<k_-≤k≤k_+<∞`. Condition on the six pins at `M=-(r/2)e_1` and `S=(r/2)e_1` exactly as in the two sources above, and write `Q_r^W` for the endpoint-determinant reweighting. Let `ρ_j^W(x)` be the `Q_r^W`-intensity, per physical area, of index-j critical points whose height lies strictly between the two pinned heights.

There exist `C, s_0, δ_0>0`, depending only on `L` and those mark compacts, such that for every frame, every such mark, every index `j`, and every `x` with

    0 < r ≤ δ_0 |x| ≤ δ_0 s_0,

one has

    ρ_j^W(x) ≤ C k r^3 |x|^{-6}.

The constant `C` does not depend on `x`, on `r`, or on the frame. No numerical value of `C`, `s_0`, or `δ_0` is claimed.

**Proposition (transverse sharpening).** On the cone `|x·e_1| ≤ |x|/2` the same hypotheses give the stronger majorant

    ρ_j^W(x) ≤ C k r^3 |x|^{-4}.

**Proposition (covariance floor is not uniform).** On the same transverse cone the conditional covariance `Σ(x)` of `(∇f(x), f(x))` under the contact pin law satisfies

    c |x|^{12} ≤ det Σ(x) ≤ C |x|^{12}

for small `|x|`. In particular its smallest eigenvalue tends to zero as `|x|→0`. A lower bound on `λ_min(Σ(x))` that is independent of `|x|` is false. The intensity majorant above is uniform only after the displayed power of `|x|` is kept.

## 2. Imports, and what is not imported

The endpoint-only normalizer is taken from the reviewed remote argument, which does not use a positive exclusion radius: under the six pins alone,

    Z_r / r^2 = z_0 + O(r),    inf z_0 > 0,

with the infimum over the mark compacts and over frames. Thus `c r^2 ≤ Z_r ≤ C r^2` for small `r`. The same two-sided bound is the normalizer `(A8)` of the fixed-annulus source. Remote conditioning is not inserted into `Z_r`.

The count is the height-disintegrated weighted Kac–Rice formula already used in both sources. For each fixed `r`, a point with `|x|≥ r/δ_0` stays a definite fraction of `|x|` away from both pins, so the witness jet is a distinct site from the pins and the fixed-`r` nondegeneracy used in those sources applies. Uniformity in the intermediate location is the new step.

Armentano–Azaïs–Léon, arXiv:2304.07424v3, is used only as the integral representation those reviews already checked. This note does not reopen their hypotheses.

Not used: elder selection, a cap event, any 24-jet certificate, dimension `d≥3`, and any bound that sends `k_-` to zero.

## 3. Contact coupling at an intermediate point

Let `U_0=(f,f_x,f_xx,f_xxx,f_z,f_xz)(0)` and let `v_0=(b,0,0,12k,0,0)`. The six-pin regression at scale `r` couples to the contact law `U_0=v_0` with

    ||F_r - F_0||_{L^p(C^q)} ≤ C_{p,q} r^2

for every fixed `p,q`, uniformly in the frame. This is the coupling already recorded for the fixed annulus and the fixed remote window; its proof is Fourier summability of `K_L` and does not need `|x|` bounded below.

The two fields do not differ by `O(r^2)` in the scaled jets that matter here. Their pin jets agree through first order up to the explicit gaps

    f(0)=b-k r^3/2+O_{L^p}(r^4),
    ∇f(0)=O_{L^p}(r^2),
    f_xx(0)=O_{L^p}(r^2),  f_xz(0)=O_{L^p}(r^2),

under `Q_r`, against exact contact values. Evaluating at distance `s=|x|` therefore costs

    |Δf(x)| ≤ C(r^3 + r^2 s + r^2 s^2),
    |Δ∇f(x)| ≤ C(r^2 + r^2 s)

in every fixed `L^p`. Throughout `r≤δ_0 s` with `δ_0` and `s` small, these errors are absorbed into the Taylor remainders of Sections 4 and 6. The constants below are stated for the contact law; the finite-`r` law obeys the same bounds for `δ_0` small enough.

## 4. Two exact jet minors

Write `s=|x|`, `x=(x_1,x_2)`, and `σ=x_2/s` when `s>0`. Derivatives below are contact-conditional at the origin. The pinned values are `f=b`, `∇f=0`, `f_xx=0`, `f_xz=0`, `f_xxx=12k`. The free cubic derivatives `(S,T,C,D)=(f_zz,f_xxz,f_xzz,f_zzz)` and the axial derivatives `Q=f_xxxx`, `R=∂_1^5 f` are part of a finite jet whose conditional covariance, given `U_0`, satisfies `λ I ≤ G ≤ Λ I`. The constants `λ,Λ` depend on `L` and on the list of monomials, not on `x` or `r`. Distinct monomials are independent because every Fourier weight of `K_L` is positive; compactness of the frame set makes `λ` and `Λ` uniform.

The Taylor map from `(S,T,C,D)` into `(f_{x_1}, f_{x_2}, f-b)` has Gram determinant

    det(M_3 M_3^T) = x_2^8 (9 x_1^4 + 4 x_1^2 x_2^2 + x_2^4) / 576.

The quartic form in parentheses is at least `(x_1^2+x_2^2)^2`. Hence

    det(M_3 M_3^T) ≥ (x_2^8 s^4) / 576.

The Taylor map on the three axial monomials `(T,Q,R)` has determinant `x_1^{10}/5760`. Cauchy–Binet therefore yields

    det(M_ax M_ax^T) ≥ x_1^{20} / 5760^2.

Adding further derivative columns increases the Gram matrix in the Loewner order, so the conditional covariance `Σ_♯` of the degree-9 Taylor polynomial satisfies both lower bounds multiplied by `λ^3`. The integral remainder after degree 9 is `O(s^{10})` in the value and `O(s^9)` in the gradient, in `L^2`. Its covariance is `O(s^{18})` on the gradient block. On the event `s≤s_0` this perturbation is at most half the main determinant in the sense of Section 5, because `λ_min(Σ_♯)≥ c s^{16}` follows from `det Σ_♯≥ c s^{20}` and `λ_max(Σ_♯)≤ C s^2` (the transverse derivative `f_{x_2}` has its value at the origin pinned, so its variance at distance `s` is `O(s^2)`). The resulting conditional covariance `Σ(x)` of `(∇f(x), f(x))` obeys

    det Σ(x) ≥ c x_2^8 s^4,     det Σ(x) ≥ c x_1^{20},

and therefore

    det Σ(x) ≥ c s^{20}

for every direction, once `s≤s_0`. On the cone `|x_1|≤s/2` one has `|x_2|≥(√3/2)s`, so the first bound is `det Σ(x)≥ c s^{12}`. The matching upper bound on that cone is Hadamard plus the scalings `Var(f_{x_1})≤ C s^4`, `Var(f_{x_2})≤ C s^2`, `Var(f)≤ C s^4`, improved by the explicit cubic Gram computation: the degree-12 piece is already `Θ(s^{12})` and higher jets contribute `O(s^{14})`. Thus `c s^{12}≤ det Σ(x)≤ C s^{12}` on that cone for small `s`. This is the second proposition. A spectral floor independent of `s` would require `det Σ` bounded below by a positive constant, which fails.

## 5. Density envelope

Let `m` and `v` be the contact conditional mean and variance of `f_{x_1}(x)`. The pinned cubic gives

    m = 6 k x_1^2 + O(s^3),

and the free cubic row gives

    v ≤ Λ (x_1^2 x_2^2 + x_2^4/4) + C s^6 ≤ C s^4 (σ^2 + σ^4 + s^2).

On `|x_1|≥s/2` one has `m≥ c k_- s^2` for small `s`, hence the Gaussian quadratic form of `Σ(x)` at gradient zero satisfies

    q(x) ≥ m^2 / v ≥ c / (σ^2 + s^2).

The Gaussian density of a nondegenerate 3-vector obeys

    p(∇f=0, f=b+h) ≤ C (det Σ)^{-1/2} exp(-q/2).

Split on `σ`.

- If `σ^2 ≤ s`, then `q≥ c/(2s)` and `det Σ≥ c s^{20}`, so the product is at most `C s^{-10} exp(-c'/(2s))`, which is bounded by `C s^{-6}`.
- If `σ^2 ≥ s` and `|x_1|≥s/2`, then `det Σ≥ c s^{12} σ^8`, so `(det Σ)^{-1/2}≤ C s^{-6} σ^{-4}`, while `q≥ c'/(2σ^2)`. The function `σ^{-4} exp(-c'/(4σ^2))` is bounded for `σ∈(0,1]`.
- If `|x_1|≤s/2`, then `det Σ≥ c s^{12}` and `(det Σ)^{-1/2}≤ C s^{-6}`.

In all three regimes, for every height in a fixed compact,

    p(∇f(x)=0, f(x)=b+h) ≤ C s^{-6}.

The same bound holds for the finite-`r` law when `δ_0` is small, by the coupling in Section 3: the mean and covariance of the scaled vector move by `o(1)` on `r≤δ_0 s`.

## 6. Endpoint weight and conditional moments

Under the two gradient pins, each endpoint Hessian has the block form `H_i=[[r α_i, r β_i],[r β_i, A_i]]` with `|α_i|` and `|β_i|` controlled by a third-derivative norm `M` on a ball of radius `O(s)`, and with `|A_i|` controlled by the same norm. For `r≤1`,

    |det H_i| ≤ C r (1+M)^2.

The witness determinant is at most `C(1+M)^2`. Therefore

    W_r F_j(H_x) ≤ C r^2 (1+M)^6.

The field `M`, as a supremum of finitely many third derivatives on that ball, is a Lipschitz function of a finite jet. Conditional on the 3-vector `Y=(∇f(x),f(x))`, its law is Gaussian up to that Lipschitz image. The regression shift of any one derivative against `Y` has size at most `C|z|`, where `z=Σ^{-1/2}(Y-E Y)`, because the variance explained by `Y` cannot exceed the unconditional variance. Unconditional third-derivative moments are finite and frame-uniform by the Fourier bound. Hence

    E[(1+M)^6 | Y=y] ≤ C (1+|z|)^6,

and the density factor `exp(-|z|^2/2)(1+|z|)^6` is bounded. Combined with Section 5,

    p(y) E[W_r F_j(H_x) | Y=y] ≤ C r^2 s^{-6}

uniformly for `y` equal to gradient zero and height in the window. The original normalizer contributes `Z_r^{-1}≤ C r^{-2}`. The window has length `k r^3`. Therefore

    ρ_j^W(x) ≤ C k r^3 s^{-6},

which is the theorem.

## 7. Transverse sharpening

Restrict to `|x_1|≤s/2`, so `|x_2|≥(√3/2)s`. On the cubic contact jet the linear combination

    -(2 x_1/x_2^2) f_{x_1} - (2/x_2) f_{x_2} + (6/x_2^2)(f-b)

equals `S=f_zz(0)` exactly: the coefficients of `T,C,D` vanish and the pinned cubic in `k` cancels. The degree-4 and higher remainder, divided by `x_2` or `x_2^2`, is `O(s^2)` in `L^2` on this cone. Hence

    Var(S | ∇f(x), f(x)) ≤ C s^4,

and at a window height the conditional mean of `S` is `O(r^3/s^2+s^2)`.

Separately, gradients vanish at `M`, `S`, and `x`. The direction from `M` to `x` makes an angle bounded below with `e_1` on this cone, and `|x-M|≤ C s`. The integral form of the gradient difference gives `||H_x||≤ C s M` and `||H_M||≤ C s M`. Thus `|det H_x|≤ C s^2 (1+M)^2`. Inserting this gain in place of the crude `|det H_x|≤ C(1+M)^2` multiplies the Section 6 bound by `s^2` and produces

    ρ_j^W(x) ≤ C k r^3 s^{-4}.

The curvature isolation is not required for this power. It is the reason a still smaller power is plausible: `E[S^2|Y]` is `O(s^4)` rather than `O(1)`, and the scaling recorded in Section 8 is consistent with one further power of `s` from the witness determinant. That further power is not needed for the theorem or the proposition.

## 8. What is uniform, and what is not

The fixed-remote argument obtains a constant by compactness of `{|x|≥ρ}` in the covariance of `(U_0,∇f(x),f(x))`. Section 4 shows that this covariance determinant is of order `|x|^{12}` on a whole cone, so the compactness constant diverges at least like a negative power of `ρ`. That is a counterexample to uniformity of the spectral floor, not a counterexample to an intensity bound.

The intensity majorant that replaces it on the whole intermediate region is the explicit multiple `C k r^3 |x|^{-6}` of the theorem. One constant `C` covers every intermediate radius. On a subregion `|x|≥ρ` the bound is `C ρ^{-6} k r^3`, which is the fixed-remote shape with the divergence rate written down. The transverse cone improves the rate to `|x|^{-4}`.

The fixed-annulus shape is physical intensity `O(k r)`. At a point with `|x|=B r` and `B→∞`, the theorem gives `C k r^3 (B r)^{-6} = C k B^{-6} r^{-3}`. For each fixed `B` this is a large multiple of `k r`, so it does not reprove the annulus theorem. It limits how fast the annulus constant may grow if the same majorant is used at large scaled radius: no growth faster than the displayed power is forced by this argument, and none is ruled out.

A matching lower bound of order `|x|^{-6}` is not claimed. Direct scaling of the cubic jet on the transverse ray is compatible with a product `p(y) E[S^2 |det H_x|]` of order `|x|` or smaller, which would put the intensity inside `O(k r^3)`. Turning that scaling into a theorem needs a uniform upper bound on the conditional fourth moment of the witness determinant of order `|x|^6`, beyond the operator-norm bound `|det H_x|≤ C |x|^2 (1+M)^2` used here. That moment is left open. Until it is proved, the uniform statement is the envelope in the theorem, not a `ρ`-independent multiple of `k r^3`.

## 9. Exclusions

Still open, and not touched here: pin neighborhoods; witnesses within a bounded scaled radius of `M` or `S`; two witnesses whose separation tends to zero; heights outside the between-pin window; dimension three or more; `k_-→0`; a numerical constant; a 24-jet certificate; elder selection. The fixed-annulus and fixed-remote theorems are used only through the interfaces named in Section 2. Their reviews are not reopened and are not extended beyond those interfaces.
