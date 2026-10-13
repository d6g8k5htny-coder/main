# Witness collisions in the fixed-remote height window: a second factorial moment and matching probability asymptotics

**Object:** CL-D5-REMOTE-COLLISION-20260928-v1.
**Author:** Anthropic Claude (Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3`).
**Disposition:** author-side proof candidate. **Nonauthor review is required.**
**Scientific effect:** NONE. No `STATUS`, `PROOF_INDEX` verdict, `GRAPH`, claim, prize or source body changes.

## 1. Results

### Setting

Use the exact model, observations and weight of the reviewed fixed-remote theorem, `frontiers/remote_window_20260924/PROOF.md` (RN-FIXED-REMOTE-WINDOW-20260924-v1, Git blob `b383bfcc88ec4ad497dff01fb6640e429ba24a84`, "Theorem A" there).

- **Field.** Fix `d>=2` and `L>0`. `f` is the centered variance-one Gaussian field on `X=R^d/(LZ^d)` with the exact normalized periodized Gaussian covariance.
- **Marks.** Compact `B=[b_-,b_+]` and `K=[k_-,k_+]` with `0<k_-`.
- **Pins.** For an orthonormal frame with axis `u`: `M=-(r/2)u`, `S=(r/2)u`, `f(M)=b`, `f(S)=b-kr^3`, `grad f(M)=grad f(S)=0`.
- **Laws.** `Q_r` is Gaussian regression on these `2(d+1)` observations.
  `W_r=F_d(H_M)F_(d-1)(H_S)`, `Z_r=E_(Q_r)W_r`, and `dQ_r^W=(W_r/Z_r)dQ_r`, where `F_j(H)=|det H|` on nonsingular `H` with `j` negative eigenvalues and `0` otherwise.
- **Region and window.** Fix `0<rho<L/4` and put `D_rho={x: dist(x,0)>=rho}` and `I_r=(b-kr^3,b)`.
- **Count.** For Borel `E` contained in `D_rho` and an index `j`,

      N_j(E) = #{x in E: grad f(x)=0, index H_x=j, f(x) in I_r},
      N(E)   = sum_j N_j(E).

The reviewed Theorem A gives

    | E_(Q^W) N_j(E) - k r^3 integral_E Lambda_j | <= C r^4 |E|,               (A)

with a continuous kernel `Lambda_j` that is bounded above and away from zero on `D_rho`. It states explicitly that (A) is **not** a matching lower bound on the event probability. Its §6 handles pairs only at a **fixed** separation `eta>0` and records that shrinking separation is uncontrolled. This is the open "witness-collision" region of the D5 graph (`math.rn-region.witness-collision`, "eta->0 mutual witness separation"). The present note treats that region **only when both witnesses lie in the fixed remote region `D_rho`**; see §8.

### Theorem C (near-diagonal witness pairs)

Fix `0<eta_0<=rho/2`. There are `C` and `r_*>0`, depending on `d,L,rho,eta_0,B,K`, such that for `0<r<=r_*`, all marks and frames, all indices `i,j`, and every Borel `E` contained in `D_rho`:

    E_(Q_r^W) #{(x,x') ordered: x!=x' in E, |x-x'|<eta_0,
                 grad f(x)=grad f(x')=0, index H_x=i, index H_x'=j,
                 f(x),f(x') in I_r}
       <= C r^5 |E|.                                                          (C)

The bound is uniform over all separations below `eta_0`, and pairs at arbitrarily small distance are included.

### Corollary D (second factorial moment)

With the same constants,

    E_(Q_r^W) N(E)(N(E)-1) <= C r^5 |E|,
    Q_r^W{ N(E) >= 2 } <= C r^5 |E|.                                          (D)

### Corollary E (matching probability asymptotics, fixed remote region)

For every index `j` and Borel `E` contained in `D_rho`,

    | Q_r^W{ N_j(E) >= 1 } - k r^3 integral_E Lambda_j(x) dx | <= C r^4 |E|.  (E)

The same holds for `N(E)` with `Lambda=sum_j Lambda_j`. In particular, for `E` of positive volume the `Q_r^W`-probability that the remote region contains an additional window critical point is asymptotic to `k r^3 integral_E Lambda`.

### Corollary F (lower bound on the whole torus in every fixed `d`; matching upper bound only for `d=2`, conditional on one pending import)

For all small `r`,

    Q_r^W{ some critical point x != M,S has f(x) in I_r } >= c r^3,           (F-)

with `c=(k_-/2) inf integral_(D_rho) Lambda_j > 0`, for a fixed index `j` (or with `Lambda=sum_j Lambda_j` throughout, applying (E) to the total count). The infimum is over marks and frames. `r_*` may need to be reduced. The factor `1/2` is needed: an asymptotic `A r^3 + O(r^4)` with a possibly negative `O(r^4)` term only licenses constants strictly below `A`.

**Upper half, planar only.** For `d=2`, consume the global single-witness first moment (I5) of `frontiers/intermediate_window_20260928`. That source is Math-#107, still pending integration; its nonauthor review record is merged as `reviews/d5_intermediate_window_claude_20260928/` via Math-#109. Markov then gives the matching upper bound `<= C r^3`, so the order of the event probability is exactly `r^3`. This half is **conditional** on that import.

For `d>2` the upper half stays conditional on a dimension-matched global first moment. #107 does not supply one, because (I5) is planar.

The lower half (F-) uses only (A) and (D) and holds in every fixed `d>=2`. Theorem C and Corollaries D and E also keep their fixed-`d` scope.

## 2. Dependencies

| Consumed | Where | Status |
|---|---|---|
| Theorem A (A), the kernel `Lambda_j`, and the uniform remote nondegeneracy argument | `frontiers/remote_window_20260924/PROOF.md` §§1–5 | ACCEPT on the fixed-rho interfaces (main#76 review and reconciliation) |
| Uniform `Q_r` moments `sup E_Q||f||_(C^m)^p<infinity` | parent `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (4.1) | A4 ACCEPT (Math-#106) |
| Gradient-pin averages (5.1) and block identity (5.3) | same parent, §5 | A3 ACCEPT (Math-#106) |
| Normalizer floor `Z_r >= z_* r^2` | same parent, (5.5) | A3 ACCEPT (Math-#106) |
| Weighted Kac–Rice, general Borel sets | Armentano–Azaïs–León, arXiv:2304.07424v3, Thms 2.1, 2.2, 7.1, Remarks 7–8 | primary source; same use as parent §9 and the remote §4 |

Nothing else is imported. In particular the cap theorem, elder selection and #107 are **not** used for Theorem C or Corollaries D–E.

## 3. The pair Kac–Rice formula

Let `D'` be an open neighbourhood of `D_rho` at distance `>=rho/2` from `0`. Let `T` be the open off-diagonal set of `D' x D'`, and define `G(x,x')=(grad f(x), grad f(x'))`, which maps into `R^(2d)`.

Under `Q_r` (a Gaussian law with bounded mean), the covariance of `G(x,x')` is positive definite at every point of `T`:

- the functionals are gradients at two distinct sites, both separated from the pins;
- by positivity of every Fourier weight of the periodized kernel (parent §2), distinct derivative functionals at distinct sites are linearly independent modulo the `2(d+1)` pin functionals.

On compact subsets of `T`, continuity and compactness bound the density of `G` above. The derivative of `G` is block diagonal, `diag(H_x, H_x')`, so its Jacobian is `|det H_x||det H_x'|`.

For the Theorem 7.1 mark, use the truncated weight

    g_m = min(W_r, m) 1{H_x in O_i} 1{H_x' in O_j} 1{f(x) in I_r} 1{f(x') in I_r}.

Here `O_j` is the open set of nonsingular symmetric matrices with `j` negative eigenvalues.

The witness determinants are **not** part of the mark. Theorem 7.1's Jacobian factor `Delta=|det H_x||det H_x'|` supplies them, and `Delta 1{H_x in O_i}1{H_x' in O_j} = F_i(H_x)F_j(H_x')`, so each witness determinant appears exactly once. The mark satisfies hypotheses (a)–(b) of Theorem 7.1:

- `W_r` is a continuous nonnegative functional of the field;
- indicators of the open sets `O_i`, `O_j` and `I_r` are lower semicontinuous.

So Theorem 7.1 of arXiv:2304.07424v3 applies. Its hypothesis (c) holds by Remark 8, and Theorem 2.1's hypotheses hold by Remark 7. Monotone convergence in `m`, together with the nondegenerate joint density of the four heights and gradients (below), then gives the ordered-pair formula for every Borel `B` contained in `T`:

    E_Q[ sum_{(x,x') in B: G=0} W_r 1{H_x in O_i} 1{H_x' in O_j} 1{f(x),f(x') in I_r} ]
      = integral_B integral_(I_r) integral_(I_r)
          p_(grad f(x),grad f(x'),f(x),f(x'))(0,0,y,y')
          E_Q[ W_r F_i(H_x) F_j(H_x') | grad f(x)=grad f(x')=0, f(x)=y, f(x')=y' ]
          dy dy' dx dx'.                                                     (3.1)

Divide by `Z_r` to obtain the `Q_r^W` count. The endpoint weight appears once, and each witness determinant appears once. This is the `m=2` case of the remote §6 formula, now on the whole off-diagonal set rather than `|x-x'|>=eta`.

## 4. Divided-difference coordinates, uniform through collision

Write `x'=x+delta e`, with `0<delta<eta_0` and `e` a unit vector. Define

    G_delta   = (grad f(x+delta e) - grad f(x)) / delta,
    D3_delta  = [ f(x+delta e) - f(x) - (delta/2) e.(grad f(x)+grad f(x+delta e)) ] / delta^3,
    V_delta   = ( grad f(x), G_delta, f(x), D3_delta )  in R^(2d+2).                    (4.1)

**Exact identities.** Let `g(t)=f(x+te)`. Then:

- `G_delta` is the average of `D^2 f(x+te)e` over `t` in `[0,delta]`.
- By the trapezoid (Peano) rule,

      D3_delta = -(1/12) integral_0^delta w_delta(t) g'''(t) dt,
      w_delta(t) = 6 t(delta-t)/delta^3 >= 0,   integral w_delta = 1.            (4.2)

**Limits.** As `delta` tends to 0, `V_delta` tends to

    V_0 = (grad f(x), H_x e, f(x), -(1/12) d_e^3 f(x)),

in every `L^p`, uniformly in `x` and `e`.

**Change of variables.** The map from `(grad f(x), grad f(x'), f(x), f(x'))` to `V_delta` is linear and block lower triangular. Its diagonal blocks are `I_d`, `delta^(-1) I_d`, `1` and `delta^(-3)`. So

    p_(grad f(x),grad f(x'),f(x),f(x'))(0,0,y,y') = delta^(-d-3) p_(V_delta)(0,0,y,(y'-y)/delta^3),   (4.3)

and on `{grad f(x)=grad f(x')=0}` we have `G_delta=0` and `D3_delta=(f(x')-f(x))/delta^3`.

**Lemma 1 (uniform floor).** There are `c_1>0`, `C_1` and `r_1>0` such that

    c_1 I <= Cov_(Q_r)(V_delta) <= C_1 I,    |E_(Q_r) V_delta| <= C_1,

for all `0<r<=r_1`, `0<=delta<=eta_0`, `x` in `D_rho`, unit `e`, frames and marks.

*Proof.*

1. **Continuity.** Every coordinate of `(U_r, V_delta)` is an integral of a derivative of `f` against a probability kernel that converges to a point mass as `r` or `delta` tends to 0. Here `U_r` is the frame transform of the pins; it is invertible for `r>0`, and its `r=0` limit is the contact jet `U_0` (parent §3). So the joint covariance is a continuous function on the compact set
   `[0,r_1] x [0,eta_0] x D_rho x S^(d-1) x O(d)`.
2. **Positive definiteness for `delta>0`.** For `r>0`, `U_r` is an invertible transform of the actual endpoint observations `(f, grad f)` at `M` and `S`. Only at `r=0` is it the contact jet `U_0` at `0`. In both cases the pin functionals are supported within distance `r/2<=rho/4` of `0`, whereas `x` and `x'` are at distance `>=rho/2` from `0`. So the functionals are, up to an invertible map, distinct derivative functionals at distinct sites: `M`, `S` (or `0` at `r=0`), `x` and `x'`.
3. **Positive definiteness at `delta=0`.** At `x`, the functionals `f`, `grad f`, `H e` and `d_e^3 f` have distinct derivative orders 0, 1, 2 and 3. `H -> He` maps `Sym_d` onto `R^d` for every unit `e` (checked exactly in `collision_exact_check.py`), so the order-2 block has full rank. Together with the pin functionals, which are the endpoint observations at `M`, `S` for `r>0` and the contact jets at `0` for `r=0`, supported at distinct sites, the positive-Fourier-weight argument makes the covariance positive definite.
4. **Conclusion.** Continuity of `lambda_min` and compactness give the floor. The `Q_r` covariance is the Schur complement on `U_r`, which inherits the floor. The target `v_r` is bounded, so the means are bounded. ∎

**Lemma 2 (conditional moments and density).** For every `p` there is `C_p` such that, for all parameters above, `y` in `I_r` and real `t`,

    E_(Q_r)[ K^p | V_delta=(0,0,y,t) ] <= C_p (1+|t|)^p,
    p_(V_delta)(0,0,y,t) <= C_p (1+|t|)^(-p),                                             (4.4)

where `K = C_d(1+||f||_(C^3(X)))`.

*Proof.* Under `Q_r`, the pair `(f, V_delta)` is jointly Gaussian.

- `Var V_delta<=C_1`, so every cross-covariance `|Cov_Q(D^alpha f(z), V_delta)|` is bounded by Cauchy–Schwarz and (4.1) of the parent.
- Regress `f` on `V_delta`. The conditional mean moves by at most `C(1+|y|+|t|)` in `C^3`.
- The centered residual has moments bounded by those of `f - E f` plus `C` times those of `V_delta`.
- The Gaussian density bound follows from Lemma 1. ∎

## 5. Pathwise bounds

**Lemma 3 (endpoint weight).** Pathwise, `W_r <= r^2 K^(2d)` for `r<=1`.

*Proof.* By parent (5.1), `|alpha_i|` and `||beta_i||` are at most `M3/2<=K`, and `||A_i||<=K`. By (5.3),
`|det H_i| = r|alpha_i det A_i - r beta_i^T adj(A_i) beta_i| <= r K^d`. Multiply the two endpoints. ∎

**Lemma 4 (a critical pair forces two small Hessian directions).** If `grad f(x)=grad f(x+delta e)=0`, then

    |H_x e|, |H_(x') e| <= K delta / 2,
    |det H_x|, |det H_(x')| <= (delta/2) K^d.                                               (5.1)

*Proof.*

1. `0 = grad f(x+delta e)-grad f(x) = integral_0^delta H(x+te) e dt`, so
   `delta H_x e = -integral_0^delta (H(x+te)-H_x) e dt`, and `|H_x e| <= delta^(-1) integral_0^delta K t dt = K delta/2`.
2. The same argument from `x'` in direction `-e` gives the bound for `H_(x') e`.
3. For symmetric `H` and unit `e`: `|det H| = product sigma_i <= sigma_min ||H||^(d-1) <= |He| ||H||^(d-1)`, and `||H||<=K`. ∎

Both witnesses of a colliding pair therefore carry a small Hessian direction. This is what compensates the `delta^(-d)` singularity of the gradient density.

## 6. Proof of Theorem C

**Rewrite the near pairs.** Restrict (3.1) to `B_near={(x,x') in E x E: 0<|x-x'|<eta_0}` and use polar coordinates `x'=x+delta e`, `dx'=delta^(d-1) d delta de`. Substituting (4.3) and `t=(y'-y)/delta^3`, so that `dy'=delta^3 dt`,

    E_(Q^W) T_near(E)
     <= Z_r^(-1) integral_E dx integral_(S^(d-1)) de integral_0^eta_0 delta^(d-1) d delta
          * delta^(-d-3) * delta^3 integral_(I_r) dy integral_(J_(y,delta)) dt
          p_(V_delta)(0,0,y,t) E_Q[ W_r |det H_x| |det H_x'| | V_delta=(0,0,y,t) ],

where `J_(y,delta)=(I_r-y)/delta^3` is an interval of length `k r^3/delta^3`.

**Bound the integrand.** On the conditioning event, Lemmas 3–4 give

    W_r |det H_x| |det H_x'| <= (1/4) r^2 delta^2 K^(4d).

By (4.4) with `p` large, `p_V (1+|t|)^(4d)` is bounded and integrable in `t`. Hence

    integral_(I_r) dy integral_J p_V E[ ... | V ] dt <= C r^2 delta^2 * k r^3 * min(1, k r^3/delta^3).

**Collect the powers.** The `delta` exponent is `(d-1)-(d+3)+3+2 = 1`, independent of `d`. With `Z_r>=z_* r^2`,

    E_(Q^W) T_near(E) <= C r^3 |E| integral_0^infinity delta min(1, k r^3/delta^3) d delta
                      = C r^3 |E| (3/2) (k r^3)^(2/3)
                      <= C' r^5 |E|.

The last integral is evaluated exactly by splitting at `delta_0=(k r^3)^(1/3)`: it equals `delta_0^2/2 + k r^3/delta_0 = (3/2) delta_0^2`. This proves (C). ∎

**Remark (where the gain comes from).** Pairs closer than about `r` cost only one window factor `r^3`, because the second height is then automatically in the window. They are rare because of their small area, `O(r^2)`. For pairs farther apart than `r`, the second window supplies the extra factor `r^3/delta^3`. Both effects give `r^3 * r^2`.

## 7. Separated pairs and the corollaries

**Lemma 5 (separated pairs).** On `B_sep={(x,x') in E x E: |x-x'|>=eta_0}`,

    E_(Q^W) T_sep(E) <= C r^6 |E|^2.

*Proof.* The site pairs lie in a compact set of distinct, separated sites. As in Lemma 1, the joint density of `(grad f(x), grad f(x'), f(x), f(x'))` under `Q_r` is bounded, and conditional `K`-moments are bounded. By Lemma 3, and since `|det H|<=K^d`, the integrand in (3.1) is at most `C r^2`. Integrating the two windows gives `(k r^3)^2`, and dividing by `Z_r` gives `r^6`. This is the `m=2`, `eta=eta_0` estimate of the remote §6, (15)–(16), re-derived here for mixed indices. ∎

**Proof of Corollary D.** Sum Theorem C and Lemma 5 over the finitely many index pairs. This gives `E N(E)(N(E)-1) <= C r^5|E| + C r^6|E|^2`.

The second term is absorbed explicitly: `r^6|E|^2 = r^5|E| * (r|E|) <= r_* L^d * r^5|E|`, using `r<=r_*` and `|E|<=L^d`. So `E N(E)(N(E)-1) <= C' r^5|E|`, uniformly in `E`, including sets of tiny volume. For integers `N>=0`, `1{N>=2} <= N(N-1)/2`. ∎

**Proof of Corollary E.** For integers `N>=0`, `N - N(N-1)/2 <= 1{N>=1} <= N`.

Apply this to `N_j(E)`. Since `0<=N_j(E)<=N(E)` and `n -> n(n-1)` is nondecreasing on the nonnegative integers, `N_j(E)(N_j(E)-1) <= N(E)(N(E)-1)`, so (D) for the total count also bounds the index-`j` factorial moment. Take expectations and use (A) and (D):

    k r^3 integral_E Lambda_j - C r^4|E| - C r^5|E| <= Q^W{N_j(E)>=1} <= k r^3 integral_E Lambda_j + C r^4|E|.  ∎

**Proof of Corollary F.** The event contains `{N_j(D_rho)>=1}`. By (E), its probability is at least `k r^3 integral_(D_rho) Lambda_j - C r^4 >= r^3 (2c - C r)`, with `c` as above. After reducing `r_*` so that `C r_* <= c`, this is at least `c r^3`. For `d=2`, the upper half is Markov applied to the (I5) first moment and is conditional on that import. For `d>2` no upper half is claimed. ∎

## 8. Scope and non-claims

**Established:** the witness-collision region **inside the fixed remote region `D_rho`**, meaning pairs at all separations below `eta_0`, including coincidence limits, at first-order factorial-moment level. Also a **matching lower bound** on the window-event probability, which the remote theorem explicitly did not supply.

**Not established:**

- **A torus-wide second factorial moment, or `math.rn-region.witness-collision` in its full torus-wide sense.** Pairs with either witness in the local, collar or intermediate regions are not estimated here. Corollary F's lower bound is global only because the global event contains the remote event.

- **Pairs with a point near the pins or at intermediate scales.** One or both witnesses in `|x| < rho` would need the collision estimate combined with the #105 and #107 machinery. The remote nondegeneracy of Lemma 1 is not uniform as `rho` tends to 0.
- **All-height pairs.** Only window pairs are counted.
- **Higher factorial moments, a Poisson limit, or an explicit numerical constant.**
- **Anything about elder selection.** D1 is separate.
- **Uniformity** as `k -> 0`, as marks grow, or as `d` or `L` vary.
- Any status transition.

## 9. Literature and method

The divided-difference change of variables near the diagonal is the standard tool for two-point Kac–Rice formulas at short range. See:

- S. Ladgham, R. Lachièze-Rey, *Local repulsion of planar Gaussian critical points*, SPA 2023, which gives the exact second factorial moment in small balls;
- D. Beliaev, V. Cammarota, I. Wigman, *No repulsion between critical points for planar Gaussian random fields*, ECP 2020;
- L. Gass, M. Stecconi, *The number of critical points of a Gaussian field: finiteness of moments*, PTRF 2024;
- S. Muirhead, *A second moment bound for critical points of planar Gaussian fields in shrinking height windows*, SPL 160 (2020), arXiv:1901.11336, which treats unconditioned stationary fields in large domains.

None of these treats the pinned, determinant-weighted law `Q_r^W`, and none extracts the two-window factor `min(1,r^3/delta^3)`. Their theorems are not imported. Only the method, divided differences plus Kac–Rice, is shared. No priority claim is made from this bounded search. It used Consensus and the arXiv records, and a Drive search found the program's July 2026 literature pass on three-point machinery, which is consistent with this assessment.

## 10. Finite controls

`collision_exact_check.py` uses the Python standard library only and exact arithmetic. It checks:

- the trapezoid identity (4.2), including the constant `1/12` and the kernel mass;
- the divided-difference average for `G_delta`;
- the Jacobian `delta^(-d-3)` of (4.3), for `d=2,3`;
- surjectivity of `H -> He` for exact rational unit vectors, `d=2,3`;
- the determinant inequality `det(H)^2 <= |He|^2 ||H||_F^(2(d-1))` on 3,000 exact rational matrices, and the equality `|det H| = |He| ||H||_op^(d-1)` on diagonal families with `e` along the smallest eigenvalue (so Lemma 4 is sharp);
- the fold example `f=t^3/3 - eps t`: its two critical points `x=-sqrt(eps)`, `x'=sqrt(eps)` are at distance `delta=2 sqrt(eps)`, with `f(x')-f(x) = -delta^3/6 = -(1/12) f''' delta^3`, which is (4.2) with equality;
- the `delta` and `r` power ledgers;
- the exact integral `(3/2) a^(2/3)` for `a=q^3`;
- the Bonferroni inequalities for `N<=60`.

Five mutants, altering the Jacobian exponent, the trapezoid constant, a dropped determinant factor, the window factor, and the Bonferroni sign, must fail. These checks do not prove the continuum or Gaussian steps.

## Appendix A. Details for the continuum steps held in review 5342481786

These spell out standard steps; none changes a statement.

**A1. Theorem 7.1 hypotheses for the pair process.**
- Take `T` (the open off-diagonal set of `D' x D'`) as parameter set, `X(x,x') = G(x,x') = (grad f(x), grad f(x'))` (in AAL's notation both the parameter dimension and the value dimension equal `2d`), and the auxiliary process `Z = (f, Df, D^2 f)` with continuous paths on the torus.
- Theorem 2.1's (i): `f` has smooth paths, so `X` is `C^1`.
- Theorem 2.1's (ii): under `Q_r`, `X(t)` is Gaussian with covariance continuous in `t` and positive definite on `T` (§3), so its density is continuous and locally uniformly bounded on compact subsets of `T`.
- Theorem 2.1's (iii) is Remark 7: the conditional law given `X(t)=v` is a Gaussian regression, continuous in `v` for the `C^1` topology.
- Theorem 7.1's (a)–(b): `g_m(t,Z)` is a product of the continuous nonnegative `min(W_r,m)`, which depends only on `Z` at the fixed pins, with indicators of the open sets `{H_x in O_i}`, `{H_x' in O_j}`, `{f(x), f(x') in I_r}`. `(t,Z) -> H_x, f(x)` is jointly continuous, so each indicator is lower semicontinuous in `t` and in `Z` (uniform convergence on compacts), and so is the product.
- Theorem 7.1's (c) is Remark 8, since `(X, Z)` is jointly Gaussian under `Q_r`.

**A2. Removing the truncation.**
- `g_m` increases to `g = W_r 1{O_i}1{O_j}1{I_r}` pointwise.
- The left side of (3.1) is an expected sum of nonnegative terms, each increasing in `m`.
- The right side is `integral_B p(0) E[Delta g_m | X=0] dt`, whose integrand increases in `m`, first inside the conditional expectation and then in `t`.
- Monotone convergence applies on both sides, so (3.1) holds for `g`, as an equality of extended nonnegative numbers. The subsequent estimates show it is finite.
- Height disintegration uses the nondegenerate joint density of `(X(t), f(x), f(x'))`, again from §3/Lemma 1.

**A3. Uniform `L^p` continuity of `V_delta` through `delta=0` (Lemma 1).**
- Each coordinate of `V_delta` has the form `integral D^alpha f(x+te)[...] mu_delta(dt)`, with `mu_delta` a probability measure on `[0,delta]` (a point mass, the uniform law, or the trapezoid weight `w_delta`). As `delta` tends to 0, `mu_delta` tends to the point mass `delta_0`.
- Covariances of such coordinates are `integral integral D^alpha D^beta K_L((x+te)-(y+se)) mu(dt) nu(ds)`. `K_L` is smooth (periodized Gaussian), so these are jointly continuous in `(delta, x, e)`, including `delta=0`. The same holds for cross-covariances with the pin functionals `U_r`, jointly with `r` in `[0,r_1]` (parent §3 / A2 of Math-#106).
- `V_delta - V_0` is centered Gaussian, so `L^2` continuity gives `L^p` continuity for every `p`.
- Positive definiteness at every parameter point is parent §2 (A1 of Math-#106, accepted): distinct derivative functionals at distinct sites have positive-definite covariance because every Fourier weight `a_n>0`. At `delta=0` the order-2 functionals `(He)_i` are distinct because `H -> He` is onto.

**A4. Lemma 2.**
- Write `Sigma = Cov_Q(V_delta)`, with `c_1 I <= Sigma <= C_1 I`, and `A(z) = Cov_Q(D^alpha f(z), V_delta) Sigma^(-1)`.
- `|Cov_Q(D^alpha f(z), V_delta)| <= (Var_Q D^alpha f(z))^(1/2) C_1^(1/2)`, which is uniformly bounded by parent (4.1). So `sup_z |A(z)| <= C`.
- Conditioning on `V_delta = w` shifts the mean of `D^alpha f` by `A(z)(w - E_Q V_delta)`, of size at most `C(1+|w|)`.
- The residual `f - E_Q[f | V_delta]` is Gaussian and independent of `V_delta`. Its `C^3` sup norm has moments bounded by those of `f - E_Q f`, plus `C` times those of `V_delta`.
- Hence `E_Q[K^p | V_delta = w] <= C_p (1+|w|)^p`. The density bound is the Gaussian density with `Sigma >= c_1 I` and bounded mean.

## Revision history

- **v1**, `a4d4f74`: initial candidate.
- **v2**: clarifications from the OpenAI/ChatGPT review [5342280263](https://github.com/d6g8k5htny-coder/Math-/pull/110#pullrequestreview-5342280263) at `a4d4f74`. That review accepts the fixed-remote near-diagonal mechanism, its `O(r^5)` order, and Corollaries E and F at their stated scopes. It is source-exposed on the remote dependency. The clarifications:
  - Corollary D now states the absorption `r^6|E|^2 <= r_* L^d r^5|E|` explicitly.
  - Corollary E now states `N_j(N_j-1) <= N(N-1)`.
  - §1 and §8 now say that this does not close a torus-wide second factorial moment or the torus-wide witness-collision region.
  - No mathematical change was made.
- **v3**: adds Appendix A, which spells out the continuum steps held in the xAI/Harper review [5342481786](https://github.com/d6g8k5htny-coder/Math-/pull/110#pullrequestreview-5342481786): the Theorem 7.1 hypotheses for the pair process, truncation removal, `L^p` continuity through `delta=0`, and Lemma 2. That review accepts the finite identities, the (3.1) mark and the Lemma 1 mechanism. No statement changes.
- **v4**: amendments from the OpenAI foreground review ([comment 5876143293](https://github.com/d6g8k5htny-coder/Math-/pull/110#issuecomment-5876143293)):
  - Corollary F's explicit constant is now `c=(k_-/2) inf integral Lambda_j`, not the infimum itself.
  - The matching upper half of F is restricted to `d=2`, because #107's (I5) is planar. For `d>2` it remains conditional on an unsupplied dimension-matched first moment.
  - Lemma 1 step 2 now distinguishes the endpoint observations (for `r>0`) from the contact jet (at `r=0`).
  - Theorem C, Corollaries D and E, and (F-) are unchanged in scope.
