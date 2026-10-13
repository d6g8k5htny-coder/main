# D5 collar between the pin disks and the fixed annulus

Object: CL-D5-COLLAR-20260928-v1. Author: Anthropic Claude (Claude Code session
`session_017Mi3hxjaxV45x6zo6o1ee3`).
Disposition: AUTHOR-SIDE CANDIDATE; nonauthor review required before any reliance.
Scientific effect: NONE. No D5, RN, lifetime, `lemma_closed`, status, graph or prize promotion.

## 1. Sources and the gap

All paths are in `d6g8k5htny-coder/Math-`.

| Source | Blob | Role |
|---|---|---|
| `frontiers/rn_annulus_bridge_20260925/PROOF.md` (PR28, merged via PR44) | `6f317515…` | Fixed annulus `A<=|y|<=B`, `A>1`: `E_W N_j <= C r^3 area`; normalizer (5); weighted Kac–Rice (16) |
| `reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md` (xAI) | `559d7224…` | R1–R7 ACCEPT, including R5 `Z_r>=c_Z r^2` and R6 weighted Kac–Rice |
| `reviews/d5_pin_neighborhood_20260926/REVIEW.md` (xAI) | `11a6b8d0…` | "Overlap and stitch": leaves `1/4<|s|` with `rho<A` open |
| `reviews/d5_pin_disk_all_directions_20260928/NOTE.md` | this PR | Pin disks `|X-M|,|X-S|<=r/4` |

**Setting.**

- The same six pins, law `Q_r`, weight `W_r=F_2(H_M)F_1(H_S)` and normalizer `Z_r` as PR28, with the
  all-height count `N_j`.
- Scaled coordinates about the midpoint: `X = r y`, `y=(p-1/2, q)`, `M=r(-1/2,0)`, `S=r(1/2,0)`.
  Equivalently `X = M + r(p,q)`.
- For fixed `A>1` the **collar** is

```
C_A = { |y| <= A,  |X-M| >= r/4,  |X-S| >= r/4 }.
```

The **gap.** PR28 excludes pin neighbourhoods and `A -> 1/2`; the pin disks are in `d5_pin_disk_all_directions_20260928`.
Between them lies `C_A`, which contains the segment between the pins. At its midpoint `p=1/2`, all three
`(S0,T,R4)` minors of the degree-four frame vanish, so the degree-four chart loses rank there.

## 2. Exact degree-five frame

`collar_exact_check.py` builds the degree-five Taylor model at M with the 15 free jets of orders 2–5:
`S0=f_zz`, `T,C,D` (third order), `R4,E1,E2,E3,F4` (fourth), and `R5,F41,F32,F23,F14,F05` (fifth).
The pins hold exactly with

```
f_xx  = -6kr + r^2 R4/12 + r^3 R5/30,
f_xxx = 12k  - r R4/2   - 3 r^2 R5/20,
f_xz  = -rT/2 - r^2 E1/6 - r^3 F41/24.
```

With `Y=(f_x(X)/r^2, f_z(X)/r)`, the degree-five model gives `Y = (6kp(p-1), 0) + B J` exactly. Every entry
of B carries a factor of q or r. The three minors are exact:

```
m(S0,C) = -q^3/2,
m(R4,T) =  r^2 p^2 (p-1)^2 (2p-1)/24,
m(R5,T) =  r^3 p^2 (p-1)^2 (5p^2+5p-4)/240.
```

The polynomials `2p-1` and `5p^2+5p-4` have no common root. The script proves
`100(2p-1)^2 + (5p^2+5p-4)^2 >= 1/70` for all real p. (At `p=1/2` the value is `1/16`; the fifth-order
column rescues the midpoint.)

## 3. Collar covariance and density lemma

Fix `delta=1/8`. Define the **strip** `Sigma = C_A ∩ {|q|<=delta}` and the **interior** `C_A ∩ {|q|>=delta}`.
The script checks that `|p(p-1)|>=1/6` on the strip.

**(a) Rank.** By Cauchy–Binet, `det(BB^T)` is at least the sum of the three squared minors, so

```
det(BB^T) >= q^6/4 + r^4 a(p)^2 + r^6 b(p)^2 >= c (q^6 + r^6)       (r<=1),
```

where `a(p)=p^2(p-1)^2(2p-1)/24` and `b(p)=p^2(p-1)^2(5p^2+5p-4)/240`. The last inequality uses
`p^2(p-1)^2>=1/36` on the strip and the separation bound. On the strip `tr(BB^T) <= C(q^2+r^2)`, since every
entry carries a factor of q or r and p is bounded. Hence `sigma_min(B)^2 >= c (q^2+r^2)^2` on the strip. On
the interior, `det >= delta^6/4` and the trace is bounded, so `sigma_min(B)>=c_delta`.

**(b) Jets.** The 21 derivative monomials of order ≤5 are distinct one-point functionals. By the Fourier
positivity argument accepted in PR28 R2 and (A5), their covariance is positive definite. So the Schur
covariance of J given the six pins has a uniform floor `lambda_*` for `r<=r_*`, because the pins are an
invertible transform of `U_r -> U_0`.

**(c) Remainder.** Degree-six Taylor and pin-solve corrections contribute `O_{L^p}(r^3)` to `Y1` and
`O_{L^p}(r^4)` to `Y2`, uniformly on `C_A`. On the strip this is `o(q^2+r^2)`.

**(d) Covariance bounds.** From (a)–(c), for `r<=r_*`,

```
strip:    c (q^2+r^2)^2 I <= Cov(Y|pins),   Var(Y_i|pins) <= C (q^2+r^2);
interior: c_delta I <= Cov(Y|pins) <= C I.
```

**(e) Density.** Adding the bounded conditional jet means gives
`E[Y1|pins] = 6kp(p-1) + O(|q|+r)`, so `|E Y1| >= k_-/2` on the strip for small `delta` and r. Then
`mu^T Cov^-1 mu >= (E Y1)^2/Var(Y1) >= c/(q^2+r^2)`. With the physical Jacobian `r^-3` this gives

```
p_{grad f(X)|pins}(0) <= C r^-3 (q^2+r^2)^-2 exp(-c/(q^2+r^2))   on the strip,
p_{grad f(X)|pins}(0) <= C r^-3                                   on the interior.
```

## 4. Deterministic three-Hessian bounds

**(T4) Symmetric three-vertex bound.** Let A, B, C be critical points of a C^{2,1} function with
Hessian-Lipschitz constant L on their triangle. Let a, b, c be the side lengths and K the area. Then

```
|det H_A det H_B det H_C| <= L^6 (abc)^4 / (512 K^3).
```

*Proof.* At each vertex, the two critical-segment columns give `|det H_V| sin(theta_V) <= L^2 l_1 l_2/4`.
Multiply over the vertices and use `sin(theta_V) = 2K/(l_1 l_2)`. ∎

The script checks T4 on 599 seeded random exact-rational critical cubics: max ratio 0.2579. A noncritical
mutant violates it.

**In the collar,** `K = r^2|q|/2` (base `MS=r`, height `r|q|`) and every side is at most `C_A r`, so
T4 gives `Pi <= C L^6 r^6 |q|^-3`. For `|q|<r` use the angle-free (T3) bound of the pin-disk note:
`Pi <= (1/8)L^3 r d^2 N^3 <= C L^3 N^3 r^3`.

## 5. Conditional moments

The regression-mean argument of PR28 R3 and the pin-disk note's §4 applies here. It uses the Cov bounds
of §3, `|E Y|<=C` and (A5) moments. The conditional mean of `D^2f`, `D^3f` on the triangle after
`grad f(X)=0` is bounded by `C(q^2+r^2)^{-3/2}` on the strip and by `C_delta` on the interior. Hence

```
E_{Q_r}[L^6 + L^3 N^3 | grad f(X)=0] <= C (q^2+r^2)^{-9}     (strip),   <= C_delta   (interior).
```

## 6. Proposition (candidate)

Assume the Kac–Rice framework of PR28 R6: X stays at distance `>=r/4` from both pins, where the gradient
law is nondegenerate by §3. Use the normalizer (5), accepted as R5. Then for every fixed `A>1` there are
`C, r_*` such that, uniformly in marks, frames and `j`,

```
E_{Q_r^W} N_j(C_A) <= C r^3,     0<r<=r_*.
```

*Proof.* The intensity satisfies `iota <= p(0) E[Pi|grad f(X)=0]/Z_r`, with physical area element `r^2 dp dq`.

- **Interior:** `iota <= C r^-3 * C_delta r^6 / (c_Z r^2) = C r`.
- **Strip with `|q|>=r`:**
  `iota <= C r^-3 q^-4 e^{-c/(2q^2)} * C r^6 |q|^-3 (2q^2)^-9 / (c_Z r^2) = C r poly(1/|q|) e^{-c/(2q^2)} <= C' r`.
- **Strip with `|q|<r`:** `iota <= C r^-3 r^-4 e^{-c/(2r^2)} * C r^3 r^-18 / (c_Z r^2) = C r^-24 e^{-c/(2r^2)}`,
  which is bounded and smaller than every power of r.

Integrating over `C_A`, whose physical area is at most `pi A^2 r^2`, gives `O(r^3)`. ∎

## 7. Assembly (conditional)

The pin disks (`d5_pin_disk_all_directions_20260928`: `O(r^3)` each), the collar (this note) and PR28's
annulus `A<=|y|<=B` share `Q_r`, `W_r`, `Z_r` and the all-height count, and they tile `{|y|<=B}`.
Together they would give

```
E_{Q_r^W} N_j(r {|y|<=B} \ {M,S}) <= C r^3.
```

This is conditional on this note and the pin-disk note both being reviewed; each is author-side. This
closes the "overlap and stitch" gap named in `d5_pin_neighborhood_20260926/REVIEW.md` **only as a
candidate**. It does not cover:

- growing B or intermediate physical distances `r << |X| << 1`;
- witness–witness collisions;
- elder selection;
- `k -> 0`;
- `d>=3`.

D5 is **not** closed.

## 8. Review requested

1. The three exact minors and the separation bound (`collar_exact_check.py`).
2. The trace and rank argument of §3(a) at the midpoint.
3. The remainder orders in §3(c).
4. Proof and constant of (T4).
5. The §6 strip integration, especially `|q|<r`.
6. Whether PR28's R6 Kac–Rice argument transfers to the collar unchanged.
