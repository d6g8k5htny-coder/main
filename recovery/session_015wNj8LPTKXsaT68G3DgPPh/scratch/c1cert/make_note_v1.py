"""Generate NOTE.md and README.md of CL-CU-CUSP-COEFFICIENT-20261001 from RESULTS.json (numbers rounded outward)."""
import json
import pathlib
import sys
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

PK = pathlib.Path('/home/user/wt_cusp/frontiers/cusp_coefficient_certified_20261001')
R = json.loads((PK / 'RESULTS.json').read_text())
V = R['values']
CTL = json.loads(pathlib.Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else None


def iv(p, sig):
    """Outward rounding of a published decimal interval to `sig` significant digits."""
    lo, hi = Decimal(p[0]), Decimal(p[1])
    def rd(x, mode):
        if x == 0:
            return Decimal(0)
        q = Decimal(1).scaleb(x.adjusted() - sig + 1)
        return x.quantize(q, rounding=mode)
    a, b = rd(lo, ROUND_FLOOR), rd(hi, ROUND_CEILING)
    return '[%s, %s]' % (a, b)


def width(p):
    lo, hi = Decimal(p[0]), Decimal(p[1])
    return float((hi - lo) / abs(hi))


d1, d2, d3 = V['d=1'], V['d=2'], V['d=3']
P = V['parts']
rules = R['rules']
nb = rules['d=3']

note = f"""# Certified cusp coefficient `c1` of Theorem CU (Math-#207) for the Gaussian kernel, `d = 1, 2, 3`

**Object:** CL-CU-CUSP-COEFFICIENT-20261001-v1.
**Author:** Anthropic Claude (Claude Code session `session_015wNj8LPTKXsaT68G3DgPPh`), 1 October 2026.
**Disposition:** certified numerical enclosures of the explicit Gaussian integral (CU.2) of Math-#207. Author-side; no
review yet. Scope claim: Math-#207 comment 5929273376.
**Scientific effect:** NONE. No `STATUS`, `PROOF_INDEX`, `GRAPH`, claim, catalog, prize or source-body change. Same GitHub
account as every lane; zero organizational-independence credit. Nothing in Math-#207, #214, #216 or #218 is touched.

## 0. Statement

Math-#207 ([CU], head `826b8be`, `PROOF.md` blob `f6df5a73`) proves, at candidate status, the second-order term of the
elder lifetime law, `nu_eld(ell) = c ell^(-1/3) + c1 ell^(1/4) + o(ell^(1/4))`, with

    c1 = -(192/7) 2^(1/4) integral_(S^(d-1)) integral_R pi_0(u; v_0(b,0)) E_0[ |Y|^(7/4) |Delta|^(1/4) 1{{A < 0}} | b ] db dsigma(u),     (CU.2)

where `Y = (f4/12) Delta - gamma^T adj(A) gamma/4` and `Delta = det A`, built from the zero-gap contact jets `f4 = d_u^4 f(0)`,
`gamma = grad_Theta d_u^2 f(0)` and `A = D_Theta^2 f(0)` ([CU] (0.1)). [CU] evaluates (CU.2) by floating quadrature and lists
certified numerics as not claimed. This packet certifies it.

**Theorem (certified enclosures).** For the Gaussian kernel `K(z) = exp(-|z|^2/2)` on `R^d`, the integral (CU.2) and the
candidate coefficient `I^cand = (3^(1/4)/2) c1` of [CU] (CU'.1) satisfy

| `d` | `c1` | `I^cand` | `I^cand - c1` |
|---|---|---|---|
| `1` | `{iv(d1['c1'], 14)}` | `{iv(d1['I_cand = (3^(1/4)/2) c1'], 14)}` | `{iv(d1['I_cand - c1'], 14)}` |
| `2` | `{iv(d2['c1'], 12)}` | `{iv(d2['I_cand = (3^(1/4)/2) c1'], 12)}` | `{iv(d2['I_cand - c1'], 12)}` |
| `3` | `{iv(d3['c1'], 9)}` | `{iv(d3['I_cand = (3^(1/4)/2) c1'], 9)}` | `{iv(d3['I_cand - c1'], 9)}` |

With the leading coefficients `c_(d,ref)` of `coefficients/side24_v1` (1) ([SIDE24]), enclosed here from their closed
form,

| `d` | `c_(d,ref)` | `c1 / c_(d,ref)` |
|---|---|---|
| `2` | `{iv(d2['c_(d,ref)'], 12)}` | `{iv(d2['c1 / c_(d,ref)'], 12)}` |
| `3` | `{iv(d3['c_(d,ref)'], 12)}` | `{iv(d3['c1 / c_(d,ref)'], 9)}` |

`RESULTS.json` holds the certificate's own intervals (16 digits, after widening by `1e-12` relative); the tables round them
further outward.

**Consequences.**
- **[CU]'s floating values are confirmed to all their digits.** [CU] §8 (remark 3) reports `c1 = -0.26939883` (`d = 2`)
  and `-0.21184835` (`d = 3`), `c = 0.0734069193` and `0.0417759318`, `c1/c = -3.669938` and `-5.071062`, and `I^cand`,
  `I^cand - c1` to seven digits. Each is the correct rounding of the certified midpoint (`controls.py`, §7). The `d = 3`
  referee Monte Carlo `-0.211848 +- 0.000024` is `0.01` standard errors from it.
- **The reference-kernel correction ratio is certified.** In `d = 3` the relative correction of [CU] (CU.1) is
  `(c1/c) ell^(7/12)`, with `c1/c_(3,ref)` in `{iv(d3['c1 / c_(d,ref)'], 9)}`. This is the Gaussian kernel. The SIDE24
  covariance differs by `1 + O(e^(-288))` ([CU] §8, remark 3), a transfer this packet does not bound (§8).
- **`d = 1` is a closed form.** `c1^(1) = -(384/7) 2^(1/4) 12^(-9/4) (2 pi)^-2 sqrt(4 pi/3) 30^(7/8) 2^(7/8)
  Gamma(11/8)/sqrt(pi)`. It agrees with Math-#214's `C_1` (#216's table: `-0.227606`). As [CU]'s author notes (#207 comment
  5925314397), (CU.2) at `d = 1` is #214's coefficient.

## 1. What is evaluated

The integral (CU.2) as [CU] defines it.
- `E_0[. | b]` is the expectation over the contact jets under the zero-gap target `v_0(b, 0)`: `f(0) = b`,
  `d_u f(0) = d_u^2 f(0) = d_u^3 f(0) = 0`, `grad_Theta f(0) = grad_Theta d_u f(0) = 0`.
- `pi_0(u; v_0(b, 0))` is the density of those pins. By [LP] §15 and [CU] (CU.2) it factorizes as
  `pi_0 = p_(f, V_u)(b, 0) p_G(0) phi_tau(0)`, where `G = grad f(0)` and `V_u = H_f(0) u`.
- `d sigma` is the surface measure on `S^(d-1)`, with `|S^0| = 2`, `|S^1| = 2 pi`, `|S^2| = 4 pi`.

The kernel is `K(z) = exp(-|z|^2/2)`, which is isotropic, so the `u`-integral is `|S^(d-1)|` times the value at `u = e_1`.
`Cov(d^a f, d^b f) = (-1)^|b| d^(a+b) K(0)`, and `d^g K(0) = prod_i (-1)^(g_i/2) (g_i - 1)!!` for even `g`, 0 otherwise.

## 2. Reduction (Lemma R)

**R.1 (the zero-gap contact law).** Given the pins,
- `f4 ~ N(-3b, 24)`;
- `gamma ~ N(0, 2 I_(d-1))`;
- `A = -b I_(d-1) + G`, with `G` symmetric, `G_jj ~ N(0, 2)` and `G_jk ~ N(0, 1)` for `j < k`;
- all of these independent;
- `pi_0 = (2 pi)^(-(d+1)) 12^(-1/2) e^(-3b^2/4)`.

*Proof.*
- **Odd jets.** `(f_x, f_y, f_xxx, gamma)` are uncorrelated with the even jets (`K` is even).
  - Each `gamma_j = f_xxy_j` has variance 3 and covariance `-1` with `f_y_j`. Its covariances with `f_x`, `f_xxx` and the
    other `gamma`'s vanish, so given the pins its variance is `3 - 1 = 2`.
  - `tau^2 = Var(f_xxx | grad f = 0) = 15 - 9 = 6`, so `phi_tau(0) = (12 pi)^(-1/2)`, and `p_G(0) = (2 pi)^(-d/2)`.
- **Even jets.** `W = (f, f_xx, f_xy_j)` has covariance `[[1, -1], [-1, 3]] (+) I_(d-1)`, with
  `W^(-1)_(11) = 3/2` and determinant 2, so `p_W(b, 0) = (2 pi)^(-(d+1)/2) 2^(-1/2) e^(-3b^2/4)`.
  - Regression on `W = (b, 0, 0)` gives `E[f4] = (3, -15).(3/2, 1/2) b = -3b` and `Var f4 = 105 - 81 = 24`.
  - It gives `E[A_jj] = -b`, `Var A_jj = 3 - 1 = 2`, `Cov(A_jj, A_kk) = 1 - 1 = 0` and `Var A_jk = 1`.
  - It gives `Cov(f4, A_jj) = -3 - (-3) = 0`.
  - All other covariances vanish by parity of the individual indices.
- Multiplying the three factors gives `pi_0`. ∎

**R.2 (Mellin form).** For `0 < p < 2`, `|y|^p = K^(-1) integral_0^oo (1 - cos(omega y)) omega^(-1-p) d omega`, with
`K = pi/(2 Gamma(1+p) sin(pi p/2))`. At `p = 7/4`, `K = pi/(2 Gamma(11/4) sin(7 pi/8))`.

All integrands below are nonnegative before the real part is taken, so Tonelli applies. Hence
`E|Y|^(7/4) = K^(-1) integral (1 - Re E e^(i omega Y)) omega^(-11/4) d omega`.

**R.3 (`d = 1`).** Here `A` is empty and `Y = f4/12`. The `b`-weight `e^(-3b^2/4)` makes `f4` a centered normal of variance
`24 + 9 (2/3) = 30`. With `E|Z|^(7/4) = 2^(7/8) Gamma(11/8)/sqrt(pi)` this gives the closed form of §0.

**R.4 (`d = 2`).** On `A = -a < 0` we have `Delta = A` and `adj A = 1`.
- Given `(a, b)`, `(f4/12) A ~ N(ab/4, a^2/6)` and `E e^(-i omega gamma^2/4) = (1 + i omega)^(-1/2)`.
- The `b`-weight `pi_0 phi_(-b,2)(-a)` equals `(2 pi)^-3 12^(-1/2) (4 pi)^(-1/2) e^(-3a^2/16) e^(-(b - a/4)^2)`, and
  `integral db` gives the factor `sqrt(pi) e^(i omega a^2/16 - omega^2 a^2/64)`.
- Then `integral_0^oo a^(1/4) e^(-beta a^2) da = Gamma(5/8)/(2 beta^(5/8))`, for `Re beta > 0` and the principal power,
  with `beta(omega) = 3/16 - i omega/16 + 19 omega^2/192`.
- Hence
      c1^(2) = -(192/7) 2^(1/4) (2 pi)^-2 12^(-1/2) Gamma(5/8) K^(-1)/4 . J,
      J = integral_0^oo omega^(-11/4) [ beta(0)^(-5/8) - Re((1 + i omega)^(-1/2) beta(omega)^(-5/8)) ] d omega.
- With `omega = u^4`, `beta = (3/16)(1 - i omega/3 + 19 omega^2/36)` and `C0 = (3/16)^(-5/8)`, we get
  `J = 4 C0 integral_0^oo g2(u) du`, where `g2(u) = (1 - Re H(u^4))/u^8` and
  `H(omega) = (1 + i omega)^(-1/2)(1 - i omega/3 + 19 omega^2/36)^(-5/8)`.

**R.5 (`d = 3`).**
- `A < 0` has eigenvalues `-a1 <= -a2 < 0`. The laws of `G` and `gamma` are rotation invariant.
- On `2 x 2` symmetric matrices `dA = |lambda_1 - lambda_2| d lambda_1 d lambda_2 d theta` (the Jacobian is exactly
  `|lambda_1 - lambda_2|`), with `theta` in `[0, pi)`, so `integral_(A<0) dA = pi integral_(a1>a2>0) (a1 - a2) da1 da2`.
- In the eigenframe `gamma^T adj(A) gamma = -(a2 gamma1^2 + a1 gamma2^2)`. Given `(A, b)`,
      E e^(i omega Y) = e^(-i omega b a1 a2/4 - omega^2 (a1 a2)^2/12) (1 - i omega a1)^(-1/2) (1 - i omega a2)^(-1/2).
- The `b`-weight is `e^(-(5/4)(b - S/5)^2 + S^2/20 - (a1^2 + a2^2)/4)`, with `S = a1 + a2`, so `integral db` gives
  `sqrt(4 pi/5) e^(S^2/20 - (a1^2 + a2^2)/4) e^(-i omega a1 a2 S/20 - omega^2 (a1 a2)^2/80)`. Hence
      c1^(3) = -(192/7) 2^(1/4) 4 pi (2 pi)^(-11/2) 12^(-1/2) (1/2) sqrt(4 pi/5) pi K^(-1) T3,
      Phi(omega) = (1 - i omega a1)^(-1/2) (1 - i omega a2)^(-1/2) exp(-i omega a1 a2 (a1 + a2)/20 - 23 omega^2 (a1 a2)^2/240).
- Normalize by `a1 = s^4`, `a2 = a1 v^4` and `omega = t^4/a1`. Then `da1 da2 = 16 s^7 v^3 ds dv` and
  `omega^(-11/4) d omega = 4 a1^(7/4) t^(-8) dt`, and
      T3 = 64 integral_0^oo ds integral_0^1 dv integral_0^oo dt  s^20 v^4 (1 - v^4) e^(-s^8 E(v)) Gt,
      E(v) = (2v^8 - v^4 + 2)/10 >= 3/16,     Gt = (1 - Re Phi)/t^8.

**R.6 (safe form).**
- Write `R = Re log Phi` and `I = Im log Phi`. Then `1 - Re Phi = -expm1(R) + e^R 2 sin^2(I/2)`. With
  `E1(x) = expm1(x)/x`, `L(x) = log1p(x)/x`, `A(x) = atan(x)/x` and `Sinc(x) = sin(x)/x`,
      Rt = -(1/4)(L(t^8) + v^8 L(t^8 v^8)) - 23 a1^2 v^8/240,      It = (1/2)(A(t^4) + v^4 A(t^4 v^4)) - a1^2 v^4 (1 + v^4)/20,
      Gt = -E1(t^8 Rt) Rt + e^(t^8 Rt) Sinc(t^4 It/2)^2 It^2/2.
- `g2` is the same with `R/omega^2 = -(1/4)L(omega^2) - (5/16) q L(omega^2 q)`, `q = 7/6 + 361 omega^2/1296`, and
  `I/omega = -(1/2)A(omega) + (5/8)A(omega/v)/v`, `v = 3 + 19 omega^2/12`.
- No step divides by a small quantity, so the near-zero cancellation of `1 - Re Phi` is exact.
- `cusp.py` writes both integrands once, against real-interval and complex-box operations.

## 3. Rigorous quadrature

**Lemma Q (Gauss–Legendre on an ellipse).** Let `f` be analytic in the open Bernstein ellipse `c + h E_rho` of `[a, b]`
(`h = (b - a)/2`) with `|f| <= M` there. Then
    |integral_a^b f - Q_n f| <= h . 4 M (1 + 1/(4n^2 - 1)) rho^(-2n) / (1 - rho^(-2)).
*Proof.*
- On `[-1, 1]` the Chebyshev coefficients satisfy `|a_k| <= 2 M rho^(-k)`.
- `Q_n` is exact up to degree `2n - 1`, and exact on odd `T_k` by symmetry.
- For even `k`, `|integral T_k| <= 2/(k^2 - 1)` and `|Q_n T_k| <= sum w = 2`.
- Summing over `k >= 2n` gives the bound. ∎

The nodes are certified in exact rational arithmetic (`gauss.py`):
- Newton steps on the grid `2^-200`, then a strict sign change of `P_n` at `r -+ 2^-150`.
- The weights `2(1 - x^2)/(n P_(n-1)(x))^2` are enclosed by an exact midpoint value plus a Lipschitz term.
- Nodes are one ulp wide and weights are tight to `2e-16` relative.

**Lemma A (certifying analyticity).**
- The bound `M` is taken over complex boxes covering the closed ellipse: slabs along the real axis, and in `d = 3` real
  sub-intervals of the other variables.
- There are two representations:
  - the safe form of R.6, with the integral-hull enclosures `E1(z) = int e^(tau z)`, `L(z) = int (1 + tau z)^-1`,
    `A(z) = int (1 + tau^2 z^2)^-1`, `Sinc(z) = int cos(tau z)`, or their closed forms for `|z| >= 1` (`cx.py`);
  - the direct form `(1 - (Phi + Phi*)/2)/t^8`, bounded through `|z^a| = |z|^a`, where every base of a principal power must
    avoid `(-oo, 0]`.
- A representation that evaluates without exception on a box is analytic on a neighbourhood of it: every denominator and
  every logarithm or power argument has been checked away from its singular set.
- Every slab meets the real axis in a segment where the representation equals the integrand. By the identity theorem it is
  the integrand's analytic continuation on that slab.
- The bound is therefore valid even when different slabs use different representations.

**Lemma B (`d = 3` boxes).**
- The `(s, t)`-plane up to `s <= 3`, `t <= 15` is cut into 247 boxes. The `s`- and `v`-breakpoints include `c/t` near
  the origin, because for large `t` the integrand has branch points at `|v| = 1/t` and grows for `arg s > pi/16`.
- `v` is innermost. On a box `S x T`,
  `|I - Q| <= |(I_S - Q_S) I_T I_V f| + |Q_S (I_T - Q_T) I_V f| + |Q_S Q_T (I_V - Q_V) f|`. Each term takes Lemma Q in its
  own variable, with the other variables real, and the positive weights summing to `|S|` and `|T|`.
- Since `f3 >= 0`, a box is enclosed by `[0, volume . sup f3]` when that is below `1e-12` ({nb['crude']} boxes). The other
  {nb['gauss_legendre']} use Gauss–Legendre ({nb['nodes']} nodes), each to `1e-11`.
- **Tails.**
  - For `t > 15`, `integral = wgt (15^-7/7 + theta 15^-9/9)` with `|theta| <= 1`, since `|Phi| <= t^-2` on the real axis.
    The `s`-integral `integral s^20 e^(-s^8 E) ds = Gamma(21/8)/(8 E^(21/8))` is exact.
  - For `s > 3`, `Gt <= min(E[Y^2]/(2 a1^2), 2 t^-8)` gives less than `1e-280`.
- In `d = 2`, 15 `u`-panels up to `u = 15` are each taken to `5e-16`. The tail is
  `15^-7/7 +- 1.5 . 15^-14/14`, since `|Re H(u^4)| <= (36/19)^(5/8) u^-7`.

## 4. Arithmetic

- **`ia.py`** is byte-identical to Math-#217's (`frontiers/c8_elder_failure_coefficient_20261001/ia.py`). It provides
  binary64 with one-ulp outward rounding, `pi` and `ln 2` from exact rational series, and `exp(-y)` with an a priori Taylor
  bound.
- **`elem.py`** adds `log`/`log1p` (`2 atanh` series), `atan` (two halvings plus the alternating series), `sin`/`cos`
  (reduction mod `pi/2`, alternating series; interval ranges include `+-1` at enclosed critical points), and the four
  kernels with their monotonicity.
- **`consts.py`** computes `log Gamma` by Stirling's series at `q + 10`, with the remainder bounded by the first omitted
  term, together with the exact rational product `prod (q + j)`.
- No libm transcendental is trusted.
- **Results are rounded outward.**
  - Both representations of each bound are reduced to a single float.
  - Every sum is an interval sum.
  - `RESULTS.json` widens by `1e-12` relative and rounds outward to 16 digits.
- `--check` recomputes everything. It requires each recomputed interval to lie inside the published one, and the published
  one to be at most `3e-12` relatively looser. It also requires identical rule data (panels, box counts, node counts).

## 5. Parts

| quantity | enclosure |
|---|---|
| `integral_0^oo g2(u) du` (`d = 2`) | `{iv(P['d=2: int_0^oo g2(u) du'], 14)}` |
| `P2` (so that `c1^(2) = P2 . integral g2`) | `{iv(P['d=2: P2'], 13)}` |
| `T3` (`d = 3`) | `{iv(P['d=3: T3'], 11)}` |
| `T3`: boxes / `t > 15` tail | `{iv(P['d=3: boxes'], 11)}` / `{iv(P['d=3: tail t > 15'], 4)}` |
| `P3` (so that `c1^(3) = P3 . T3`) | `{iv(P['d=3: P3'], 13)}` |
| `K = pi/(2 Gamma(11/4) sin(7 pi/8))` | `{iv(P['K = pi/(2 Gamma(11/4) sin(7 pi/8))'], 13)}` |
| `Gamma(5/8)`, `Gamma(11/4)` | `{iv(P['Gamma(5/8)'], 13)}`, `{iv(P['Gamma(11/4)'], 13)}` |

`d = 2` panel rules, as `[a, b, n, rho]`: {', '.join('[%g, %g, %d, %g]' % tuple(r) for r in rules['d=2 panels [a, b, n, rho]'])}.

## 6. Verification

`python3 -B -S certificate.py --check --procs N` runs a self-test, recomputes everything (about 4 minutes on four cores)
and compares the result with `RESULTS.json`.

The self-test checks:
1. Gauss–Legendre exactness on even monomials for `n = 4 ... 48`.
2. The real-interval safe forms against the direct complex-power formulas, in floating point, at ten points.
3. A deliberately coarse 6-point rule on `[0.85, 1.15]`, whose certified enclosure must contain a 40-point floating
   reference. This checks the error bound itself.
4. The `t`-tail against its floating formula.

The assembly also compares `integral g2` and `T3` with coarse floating quadratures of the safe form in libm floating
point. This guards normalizations and factors.

Five seeded defects must each be rejected by the self-test (exit 1): `kernel-A` (a `1e-7` change in `A`), `gl-weight`
(weights `x (1 + 1e-9)`), `bound-scale` (error bounds `x 1e-6`), `phase-sign` (the sign of the `a1^2` phase term in `f3`),
and `tail-drop`. The workflow `.github/workflows/cusp-coefficient-certified.yml` runs, for `-B -S` and `-B -O -S`: the
manifest and pin check, `--check`, the five mutants, and a clean-tree check.

## 7. Floating controls (`controls.py`; not part of the certificate)

{CTL['table'] if CTL else '(see controls.py)'}

## 8. What this does not do

- **Gaussian kernel only.** The periodized SIDE24 covariance (side `L = 24`) changes the one-site covariances by factors
  `1 + O(e^(-L^2/2)) = 1 + O(e^(-288))` ([CU] §8, remark 3). A transfer in the manner of Math-#217's Lemma T is possible
  but is not done here. In particular the SIDE24 value `c_(3,24)` and the ratio `c1/c_(3,24)` are not certified, only
  the reference-kernel ones.
- **Not a review of Theorem CU.** The numbers are the integral (CU.2) defines. That `c1` is the coefficient of
  `ell^(1/4)` in `nu_eld` is [CU]'s statement (author-side; it depends on Math-#191 and Math-#198), consumed, not
  reviewed.
- No `c2` (Math-#216), no `d >= 4`, no rate.
- Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## 9. Provenance

- **Sources** (`SOURCE_MAP.json`):
  - [CU] Math-#207 `frontiers/cusp_second_order_20261001/PROOF.md` at head `826b8be` (blob `f6df5a73`, unmerged; recorded,
    not checked on this tree);
  - [SIDE24] `coefficients/side24_v1/PROOF.md` (1);
  - [LP] `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` §15.

  The last two are verified on this tree by the workflow.
- **Files:**
  - the certificate: `certificate.py` (drivers, self-test, mutants, publication), `cusp.py` (integrands), `d3.py`
    (`d = 3` boxes and tails), `gauss.py`, `cx.py`, `elem.py`, `ia.py`, `consts.py`;
  - `controls.py`;
  - `RESULTS.json`, `SOURCE_MAP.json`, `SOURCE_FILES.json` (the manifest).
"""

readme = f"""# Certified cusp coefficient `c1` (Math-#207, (CU.2)), Gaussian kernel

`CL-CU-CUSP-COEFFICIENT-20261001-v1`. Author-side certified numerics; scientific effect NONE. See `NOTE.md`.

| `d` | `c1` | `c1 / c_(d,ref)` |
|---|---|---|
| `1` | `{iv(d1['c1'], 14)}` | |
| `2` | `{iv(d2['c1'], 12)}` | `{iv(d2['c1 / c_(d,ref)'], 12)}` |
| `3` | `{iv(d3['c1'], 9)}` | `{iv(d3['c1 / c_(d,ref)'], 9)}` |

Replay: `python3 -B -S certificate.py --check --procs N` (self-test, then a full recomputation against `RESULTS.json`).
Standard library only.
"""
(PK / 'NOTE.md').write_text(note)
(PK / 'README.md').write_text(readme)
print('NOTE', len(note), 'README', len(readme))
