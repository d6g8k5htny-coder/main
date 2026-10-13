"""Generate NOTE.md and README.md of the packet from RESULTS.json (numbers are never typed by hand)."""
import json, sys
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
PK = '/home/user/wt_fail/frontiers/c8_elder_failure_coefficient_20261001'
R = json.load(open(PK + '/RESULTS.json'))
res = R['results']
CTL = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None


def rd(s, sig, mode):
    x = Decimal(s)
    if x == 0:
        return Decimal(0)
    e = x.adjusted() - (sig - 1)
    return x.quantize(Decimal(1).scaleb(e), rounding=mode)


def sh(iv, sig=7):
    """Outward rounding of a published enclosure to sig significant digits, shown as [lo, hi]."""
    lo, hi = rd(iv[0], sig, ROUND_FLOOR), rd(iv[1], sig, ROUND_CEILING)
    return '[%s, %s]' % (fmtd(lo), fmtd(hi))


def fmtd(d):
    s = format(d, 'f') if abs(d.adjusted()) < 7 else format(d, 'E')
    return s


def sub_iv(x, y):
    """Exact outward difference of two published decimal intervals."""
    return [str(Decimal(x[0]) - Decimal(y[1])), str(Decimal(x[1]) - Decimal(y[0]))]


def mid(iv):
    return (float(iv[0]) + float(iv[1])) / 2


def relw(iv):
    return (float(iv[1]) - float(iv[0])) / (2 * abs(mid(iv)))


K = [('1/2', 'k=0.5'), ('1', 'k=1'), ('2', 'k=2')]
rows_J = []
for lab, key in K:
    r = res[key]
    rows_J.append('| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |' % (
        lab, sh(r['J_fail'], 8), sh(r['J_1'], 8), sh(r['J_2'], 6), sh(r['share_J2_over_Jfail'], 5), sh(r['near_palm_excess'], 5)))
rows_a = []
for lab, key in K:
    for b in ('0', '1'):
        c = res[key]['coefficients']['b=%s' % b]
        rows_a.append('| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |' % (
            lab, b, sh(c['p_b(0)/z_0'], 8), sh(c['alpha_1'], 7), sh(c['alpha_2'], 5), sh(c['a_fail = alpha_1 + alpha_2'], 7)))
rows_parts = []
for lab, key in K:
    p = res[key]['parts']
    rows_parts.append('| `%s` | `%s` | `%s` (`%s`) | `%s` | `%s` | `%s` |' % (
        lab, sh(p['E[2(B_-)^3]'], 11), res[key]['E[2T] exact'], fmtd(rd(str(float(Decimal(res[key]['E[2T] exact'].split('/')[0]) / Decimal(res[key]['E[2T] exact'].split('/')[1]) if '/' in res[key]['E[2T] exact'] else Decimal(res[key]['E[2T] exact']))), 10, ROUND_FLOOR)),
        sh(p['E[V] boxes'], 9), sh(p['E[H_2] boxes'], 8), fmtd(rd(p['tail bound'][1], 2, ROUND_CEILING))))
CF = res['C_fail']
leaves = R['leaves']
print('J rows', *rows_J, sep='\n')
print('alpha rows', *rows_a, sep='\n')
print('parts', *rows_parts, sep='\n')
print('Ik', sh(CF['I_k'], 7), 'C', sh(CF['C_fail^{B,K}'], 6), 'Ib', sh(CF['I_b'], 10), 'const', sh(CF['const'], 10), 'Wtot', sh(CF['int_K W(k) dk'], 9))
print('leaves', leaves)
print('rel widths: Jfail', [round(relw(res[k]['J_fail']), 8) for _, k in K], 'J2', [round(relw(res[k]['J_2']), 6) for _, k in K], 'C', relw(CF['C_fail^{B,K}']))


def ctl(key, default='(see controls.py)'):
    return CTL.get(key, default) if CTL else default


r1 = res['k=1']['coefficients']['b=0']
r2 = res['k=2']
D3 = res['d=3 (conditional on Math-#175 (L1) and Math-#184 Theorem 1)']
TT = res['torus model, every L >= 10, every frame (Lemma T)']
sys.dont_write_bytecode = True
sys.path.insert(0, PK)
import torus as _TOR
_RAW = _TOR.transfer(_TOR.E_I2_upper())
_rr = _RAW['pi0p_ratio']
RATIO_DEV = '%.1e' % max(1.0 - _rr[0], _rr[1] - 1.0)
EI2 = '%.3e' % _TOR.E_I2_upper()
EPS = '%.2e' % _RAW['eps']
# the displayed (rounded) planar intervals must be identical for the torus model
for lab, key in K:
    assert sh(res[key]['J_1'], 8) == sh(sub_iv(TT[key]['J_fail'], TT[key]['J_2']), 8), (key, 'J_1')
    for q in ('J_fail', 'J_2'):
        sig = {'J_fail': 8, 'J_2': 6}[q]
        assert sh(res[key][q], sig) == sh(TT[key][q], sig), (key, q)
    for b in ('0', '1'):
        for q, qt, sig in (('alpha_1', 'alpha_1', 7), ('alpha_2', 'alpha_2', 5), ('a_fail = alpha_1 + alpha_2', 'a_fail', 7)):
            assert sh(res[key]['coefficients']['b=%s' % b][q], sig) == sh(TT[key]['b=%s' % b][qt], sig), (key, b, q)
assert sh(CF['C_fail^{B,K}'], 6) == sh(TT['C_fail^{B,K}'], 6)
rows_d3 = []
for lab, key in K:
    e = D3['%s, b=0' % key]
    rows_d3.append('| `%s` | `%s` | `%s` | `%s` |' % (lab, sh(e['alpha_1^(3)'], 7), sh(e['alpha_2^(3) = nu^(3)(2)'], 5), sh(e['a_fail^(3)'], 7)))
NOTE = f"""# Certified planar near cluster coefficients `alpha_1`, `alpha_2` and the elder-failure constant `C_fail^{{B,K}}`

**Object:** CL-C8-ELDER-FAILURE-COEFFICIENT-20261001-v1.
**Author:** Anthropic Claude (Claude Code session `session_015wNj8LPTKXsaT68G3DgPPh`), 1 October 2026.
**Disposition:** certified numerical enclosures. Binary64 interval arithmetic with outward rounding encloses explicit
finite-dimensional Gaussian integrals that the sources define. Author-side. Nonauthor reads at `b54ea71` (OpenAI /
Codex): Slices A and C ACCEPT (review 5378890193), Slice B ACCEPT (review 5379043497); v1.3 applies their three
non-blocking clarifications (the `chi^2` coordinates, outward comparison brackets, the `Phi`-series bound).
**Scientific effect:** NONE. No `STATUS`, `PROOF_INDEX`, `GRAPH`, claim, catalog, prize or source-body change. Same GitHub
account as every lane; zero organizational-independence credit.

## 0. Statement

Planar model, reference (continuum) kernel `K(z) = exp(-|z|^2/2)`. `(a, beta, c) ~ N(0, diag(2, 2, 6))` are the odd
contact jets and `I_1`, `I_2` the near failure integrals of [NUM] §§2–3 (recalled in §1). Write `J_j(k) = E[I_j]` and
`J_fail = J_1 + J_2`. By [NUM] §2, from [CL] Theorem N or [CUB] Theorem F,

    alpha_j(b, k) = (p_b(0)/z_0) J_j(k),      p_b(0)/z_0 = phi(b/sqrt2) / (sqrt2 36 k^2 m_(2,b)),
    m_(2,b) = (b^2 + 2) Phi(b/sqrt2) + sqrt2 b phi(b/sqrt2),

with `nu(2) = alpha_2` ([NUM] §1) and `a_fail = alpha_1 + alpha_2` the selector-failure coefficient of [S] (S2).

**Theorem (certified enclosures).** The following intervals contain the exact values for the reference kernel. By Lemma T
(§6a) they also contain them for the torus model of [LP] and [S], with every side `L >= 10` and every frame. The torus
enclosures in `RESULTS.json` agree with these at the displayed precision. The certificate's own intervals (`RESULTS.json`,
10 digits) are rounded further outward here.

| `k` | `J_fail = J_1 + J_2` | `J_1` | `J_2` | `J_2 / J_fail` | near Palm excess `2J_2/(J_1 + 2J_2)` |
|---|---|---|---|---|---|
{chr(10).join(rows_J)}

| `k` | `b` | `p_b(0)/z_0` | `alpha_1` | `alpha_2 = nu(2)` | `a_fail = alpha_1 + alpha_2` |
|---|---|---|---|---|---|
{chr(10).join(rows_a)}

**Corollary 1 (the elder-failure constant).** On `B x K = [0, 1] x [1/2, 2]`, the coefficient of [S] §9 for the reference
kernel,

    C_fail^{{B,K}} = integral_(B x K x S^1) A_0^ref a_fail / (3 k^(5/3)) db dk dsigma(u) = const I_b I_k,
    const = 4 / (sqrt2 (2 pi)^(5/2) sqrt12),   I_b = integral_0^1 exp(-b^2) db,
    I_k = integral_(1/2)^2 exp(-12 k^2) k^(-5/3) J_fail(k) dk,

satisfies

    I_k in {sh(CF['I_k'], 7)},      C_fail^{{B,K}} in {sh(CF['C_fail^{B,K}'], 6)}.

By Lemma T the same interval holds for the torus model with every `L >= 10`. That is the model of [S] §9, which states
`nu_rej^{{B,K}}(ell) ~ C_fail^{{B,K}} ell^(2/3)` and `E N_rej^{{B,K}}(0, t] ~ (3/5) C_fail^{{B,K}} t^(5/3)`.

**Corollary 2 (`d = 3`, conditional on Math-#175 and Math-#184).**
- Math-#175 (OpenAI/Codex, open, AUTHOR_SIDE / HOLD) is [FIB]:
  - its Theorem F, (S3), identifies `(1 - p_r)/r^3 -> a_1 + a_2` in fixed `d >= 3`;
  - its (L1) is `C_fail^{{B,K}} = integral_(B x K x S^(d-1)) A_0 a_fail / (3k^(5/3))`.
- Math-#184 (open) is [HD]. Its Theorem 1 gives `a_j^(d) = R_d(b) alpha_j` with `R_d = N_d m_(2,b) / m_(d,b)`.

In `d = 3`, for the reference kernel, the derivation of §1 gives
`A_0^ref,(3) = 432 (2 pi)^-4 12^(-1/2) k^2 exp(-12k^2) exp(-3b^2/4) m_(3,b)`: the transverse jet `f_xy2` adds a factor
`(2 pi)^(-1/2)`, `p_G(0) = (2 pi)^(-3/2)` and `z_0 = 36 k^2 m_(3,b)`. Both `m_(3,b)` and `m_(2,b)` cancel. With
`N_3(b) = pi (2 pi)^(-1/2) g(b)` ([HD] (2), `m = 2`) we have `exp(-3b^2/4) phi(b/sqrt2) N_3(b) = exp(-b^2) g(b) / 2`, so

    C_fail^(3),{{B,K}} = 4 pi (4/sqrt2) (2 pi)^-4 12^(-1/2) I_b^(3) I_k,      I_b^(3) = (1/2) integral_0^1 exp(-b^2) g(b) db,

with `g(b) = (b^3 + 6b) Phi(b/sqrt2) + sqrt2 (b^2 + 4) phi(b/sqrt2)`. The certificate checks `N_3(0) = 2 sqrt2` ([HD] (3)) and
gives

    I_b^(3) in {sh(D3['I_b^(3)'], 7)},      C_fail^(3),{{B,K}} in {sh(D3['C_fail^(3),{B,K}'], 6)}.

At `b = 0`, `R_3(0) = (32 + 28 sqrt2)/17`:

| `k` | `alpha_1^(3)` | `alpha_2^(3) = nu^(3)(2)` | `a_fail^(3)` |
|---|---|---|---|
{chr(10).join(rows_d3)}

**Consequences.**
- **`nu(2)` and `a_fail` are certified.** At `(b, k) = (0, 1)`: `alpha_2 = nu(2)` is in `{sh(r1['alpha_2'], 5)}` and `a_fail` in
  `{sh(r1['a_fail = alpha_1 + alpha_2'], 7)}`. The two-point share `J_2/J_fail` decreases from
  `{100 * mid(res['k=0.5']['share_J2_over_Jfail']):.2f}%` at `k = 1/2` to `{100 * mid(res['k=2']['share_J2_over_Jfail']):.2f}%` at `k = 2`.
- **These are the model's constants.** By Lemma T the planar enclosures hold for the torus model of [LP] and [S] with
  every `L >= 10` and every frame, not only for the reference kernel.
- **[NUM]'s open digit is settled.** At `k = 2`, [NUM] observed that Gauss–Hermite and Monte Carlo disagree on `J_2` by
  2.6% and said the third digit of `alpha_2(2)` was not established. The certified `J_2(2)` is in `{sh(r2['J_2'], 6)}`. It
  excludes the GH60 value `8.641` and is consistent with the Monte Carlo value `8.875 +- 0.073`. The Gauss–Hermite
  sequence had not converged there.
- **The cap route overstates the lifetime difference by about `1.5 x 10^5`.** Math-#215 (open) certifies
  `nu_cand - nu_eld <= C_{{B,K}}(r_*) ell^(2/3)` through the cap criterion, with `C_{{B,K}}(1/4096) = 402.5297`. As
  `r_* -> 0`, that route can give no less than `Gamma_2 = 307.51` (floating). The leading constant of the difference is
  `C_fail^{{B,K}}`, about `0.00273`: `{ctl('ratio_CBK')}` times smaller than `C_{{B,K}}(1/4096)` and `{ctl('ratio_Gamma')}` times
  smaller than `Gamma_2`.
- **The same holds in `d = 3`, conditionally.** `C_fail^(3),{{B,K}}` is about `0.00441`. That is `{ctl('ratio_CBK3')}` times
  smaller than Math-#215's `C_{{B,K}}^(3)(1/4096) = 1205.7089` and `{ctl('ratio_Gamma3')}` times smaller than
  `Gamma_3 = 540.97`.

## 1. What is evaluated

[NUM] §3 (from [CUB] (C2)–(C3) and [CL] (3.12)) defines, with `B = beta - a^2/(12k)`, `D = (c - a beta/(4k) +
a^3/(72k^2))/2`, the typed domain `s < -|B|/2`, the weight `w_0 = 9k^2 (4s^2 - B^2)` and

    g(s) = -2s^3 + 3Bs^2 - B^3 - 12k D^2,

the classification: `n = 2` iff `s > B` and `g(s) > 0`; `n = 1` iff `g(s) < 0`; `n = 0` iff `s <= B` and `g(s) > 0`.
Then `I_j = integral_(s < -|B|/2) w_0 1{{n = j}} ds`. [NUM] §2 gives the prefactor `p_b(0)/z_0`. Here `p_b(0)` is the density
at `0` of `A = f_zz | (f, f_xx, f_xz) = (b, 0, 0) ~ N(-b, 2)`, and `z_0 = 36 k^2 m_(2,b)`.

The reference integrand of [S] §9 is `A_0^ref = 12 pi_0 z_0`. [LP] §15 factorizes `pi_0 = p_(f,V_u)(b, 0) p_G(0)
phi_tau(12k)`. With the jet covariances of [NUM] §2 these factors are:
- `(f, f_xx)` has covariance `[[1, -1], [-1, 3]]`, and `f_xz` is independent with variance `1`. So
  `p_(f,V_u)(b, 0) = exp(-3b^2/4) / (2 pi sqrt2 sqrt(2 pi))`.
- `p_G(0) = 1/(2 pi)`.
- `tau^2 = Var(f_xxx | grad f = 0) = 15 - 9 = 6`.

Hence `pi_0 = (2 pi)^-3 12^(-1/2) exp(-3b^2/4 - 12k^2)` and

    A_0^ref = 432 (2 pi)^-3 12^(-1/2) k^2 exp(-12 k^2) exp(-3b^2/4) m_(2,b).

In `A_0^ref a_fail / (3 k^(5/3))`, `m_(2,b)` and `k^2` cancel, `exp(-3b^2/4) phi(b/sqrt2) / sqrt2 = exp(-b^2) / (2 sqrt pi)`,
and the circle gives `2 pi`. This leaves `const I_b I_k` with the `const` above.

## 2. Reduction to one root (Lemma 1)

Put `x = -s` and `P(x) = (x + B)^2 (2x - B)`. Expanding, `g(s) = P(x) - T` with `T = 12k D^2`. The typed domain is
`x > |B|/2`.

- **`B >= 0`.** `P` increases on `x >= B/2` from `P(B/2) = 0`. So `n = 1` exactly on `(B/2, x_b)`, and `n = 2` never
  occurs (it needs `x < -B`).
- **`B < 0`.** `P` decreases on `(|B|/2, |B|)` from `|B|^3/2` to `0`, then increases.
  - If `T < |B|^3/2`: `n = 2` on `(|B|/2, x_m)` and `n = 1` on `(x_m, x_b)`.
  - Otherwise: `n = 1` on `(|B|/2, x_b)`.

Here `x_b >= max(-B, B/2)` is the **big root** and `x_m in [|B|/2, |B|]` the **middle root** of `P = T`. In every case
the failure set `{{n >= 1}}` is the single interval `|B|/2 < x < x_b` (up to null sets). With the antiderivative
`9k^2 (4x^3/3 - B^2 x)`:

    I_fail = I_1 + I_2 = 3k^2 H,     H   = |B|^3 + 4 x_b^3 - 3B^2 x_b = 2(B_-)^3 + 2T + V,   V = -3B(2x_b - B)(x_b + B),
    I_2 = 3k^2 H_2,                  H_2 = |B|^3 + 4 x_m^3 - 3B^2 x_m   on {{B < 0, T < |B|^3/2}}, 0 elsewhere.

The identity `2T + V = (x + B)(2x - B)^2 = 4x^3 - 3B^2 x + B^3` holds at either root. The controls `(0, -2k, 0)` (`H_2 = 16k^3`,
`I_2 = 48k^5`) and `(0, 0, c)` (`H = 2T = 24k D^2`, `I_1 = 72k^3 D^2`) agree with [NUM] §3.

**Scaled coordinates.** `a^ = a / sqrt k ~ N(0, 2/k)`, `c^ = sqrt(k) c ~ N(0, 6k)`. Then `B = beta - a^2/12` and
`T = 3u^2` with `u = c^ + d` and `d = -a^ beta/4 + a^3/72`, so `J_fail(k) = 3k^2 E[H]` and `J_2(k) = 3k^2 E[H_2]` with
`k`-free integrands. Exactly, `E[2T] = 6 E[u^2] = 6 (6k + 1/(4k) + 5/(216 k^3))`. Also `E[2(B_-)^3] = 4 integral_0^inf
phi_(sqrt(2/k))(a) g(a^2/12) da` with `g(m) = E[(m - beta)_+^3]`.

## 3. Monotonicity and the `(B, w)`-calculus (Lemma 2)

Let `r = B/x`, so `r in [-1, 2]` on the big root and `r in [-2, -1]` on the middle root. Let `w = |u| = (T/3)^(1/2)`.
Implicit differentiation of `P(x) = T` gives:
- `x_T = 1/(6x(x + B))` (positive on the big root, negative on the middle root);
- `x_B = -(x - B)/(2x)`;
- `r_B = (2x - B)(x + B)/(2x^3)`;
- `H_T = (4 - r^2)/(2(1 + r))`;
- `V_T = -r(r + 4)/(2(1 + r))`;
- `H_B / x^2 = -(3/2)(r - 2)^2 (r + 1)` for `B > 0` and `-(3/2)(r + 2)(r^2 - r + 2)` for `B < 0`;
- `V_B = -(3/2) x^2 (r - 2)^2 (r + 1)`.

Therefore:
- `H` is non-increasing in `B` and non-decreasing in `T`.
- `V` is non-increasing in `B`. In `T` it is non-increasing for `B > 0` and non-decreasing for `B < 0`.
- `H_2` is non-increasing in both.
- On the big root, `x` increases in `T` and, for fixed `T`, is unimodal in `B` with minimum `x = B` at `B = (T/4)^(1/3)`.
  `r` is non-decreasing in `B`, increasing in `T` for `B < 0` and decreasing for `B > 0`.
- On the middle root, `x` and `r` are non-increasing in both `B` and `T`.

In `(B, w)` with `sigma = +1` on the big root and `-1` on the middle root:

    V_w  = -sigma sqrt3 x^(3/2) r (4 + r)(2 - r)^(1/2)
    V_BB = x p(r),           p(r) = 6 + 3r - 9r^2/2 - 3r^3/4 + 3r^4/4
    V_Bw = -sigma (sqrt3/2) x^(1/2) e(r),    e(r) = (2 - r)^(3/2)(4 + 2r + r^2)    (decreasing on r <= 2)
    V_ww = g(r) = -r(4 - r + r^2)                                                    (strictly decreasing)

- The `T`-derivatives blow up as `T -> 0` for `B < 0` (`V_TT ~ T^(-3/2)`). The `(B, w)`-Hessian is bounded: its entries are
  `x p(r)`, `x^(1/2) e(r)` and `g(r)` with `r` in a bounded range. At the triple point `(0, 0)` it is bounded but
  discontinuous, and `V` is `C^(1,1)` there, which is all the integral remainder needs.
- `V` depends on `u` only through `w`, with a kink at `u = 0` when `B < 0` (`V_w(B, 0) = 9|B|^(3/2)`).
- On the support boundary `T = |B|^3/2` the middle root is `|B|/2` (`r = -2`). There `H_2 = 0`,
  `H_2,B = -(3/2) x^2 (r + 2)(r^2 - r + 2) = 0` and `H_2,T = (4 - r^2)/(2(1 + r)) = 0`. So the zero extension of `H_2` is
  `C^1` and piecewise `C^2`.

All formulas were checked against finite differences of the root map (relative agreement `6e-6` at step `1e-4`) and
symbolically.

## 4. Box enclosures (Lemma 3)

For a box `Q` of jet coordinates, let `[B_l, B_h] x [T_l, T_h]` contain its image. The ranges of `B` and `d` are exact
up to outward rounding: `d` is linear in `beta`, and its `a`-extremes are at the ends or at `a = +-sqrt(6 beta)`. The
`u`-range is the `c`-range plus the `d`-range. Two enclosures of `integral_Q F dmu` (`F = V`, `H`, `H_2`) are formed and intersected; an empty
intersection would raise.

- **First order.** `I_1 = mu(Q) [min F, max F]`. The extremes sit at rectangle corners by Lemma 2, so the corner roots
  suffice (four big roots, or two middle roots).
- **Second order** (only if the `u`-range of `Q` avoids `0`, so `w = s u` with a fixed sign `s`). Take a float point
  `(B*, w*)` of the rectangle and Taylor's formula with integral remainder along the segment to `(B, w)` (the rectangle
  is convex):

      F = F* + F_B* dB + F_w* dw + 1/2 (h_BB dB^2 + 2 h_Bw dB dw + h_ww dw^2).

  - Each averaged Hessian entry `h` lies in the interval range `S` of the second derivative over the rectangle,
    computed from the corner ranges of `x` and `r`, with `p` evaluated in mean-value form. For `H_2` on boxes meeting
    the support boundary, the ranges are hulled with `0`.
  - Hence `integral_Q F dmu` lies in

        F* m + F_B* E[dB] + s F_w* E[du] + 1/2 (S_BB E[dB^2] + S_ww E[du^2]) + s c E[dB du] +- delta (E[dB^2] E[du^2])^(1/2),

    where `c` and `delta` are the centre and radius of `S_Bw`.
  - The six moments are exact Gaussian polynomial moments (orders `a <= 6`, `beta, c <= 2`). The second-order part
    has width `O(diam(S) h^2 mu(Q))`, so the scheme is third order.
- **Partial moments.** `M_n = (n - 1) s^2 M_(n-2) + s^2 (x_0^(n-1) phi(x_0) - x_1^(n-1) phi(x_1))`. Cell masses are
  differences of upper tails `Q(t)` on the side away from `0`, so they keep relative precision far out.
- **Roots.** The roots are bracketed by Newton iteration from a Cardano / trigonometric start. The bracket is then
  verified: `P(a) - T < 0 < P(b) - T` for the big root, reversed for the middle root. The sign test is a binary64
  evaluation of `(x + B)^2 (2x - B) - T` with an a priori rounding bound.
- **Refinement.** An initial `8^3` grid covers `[-8 sigma, 8 sigma]^3`. Boxes are bisected depth-first along their
  standardized widest side until the enclosure width is at most `10^-8`. Partial sums are rounded outward, and the
  initial cells are dealt to processes in a fixed pattern.

## 5. Exact and one-dimensional parts; tails

- `E[2T]` is the exact rational above.
- `E[2(B_-)^3]` is a one-dimensional integral over `4000` cells, each by second-order Taylor expansion about the
  midpoint.
  - `G(a) = g(a^2/12)` has `G'' = g''(m) a^2/36 + g'(m)/6`.
  - `g' = 3[(m^2 + 2) Phi(m/sqrt2) + sqrt2 m phi(m/sqrt2)]` and `g'' = 6[m Phi(m/sqrt2) + sqrt2 phi(m/sqrt2)]` are increasing,
    so `G''` is bounded by its values at the cell ends.
  - The tail `a > 12 sigma` uses `g(m) <= m^3 + 6m + sqrt2 (m^2 + 4)/sqrt(2 pi)`.
- **Outside the box.** Lemma 1 and `x_b <= |B| + (T/2)^(1/3)` (since `P(|B| + t) >= 2t^3`) give
  `|V|, H, H_2 <= 17|B|^3 + 8T`. This is a polynomial in `|a|, |beta|, |c|`, integrated over the complement by the union
  bound with exact tail moments.

These are expectations over the scaled law. `J_fail = 3k^2 (E[2(B_-)^3] + E[2T] + E[V])` and `J_2 = 3k^2 E[H_2]`. The
tail bound enters `E[V]` with both signs and `E[H_2]` from above.

| `k` | `E[2(B_-)^3]` | `E[2T]` exact | `E[V]` boxes | `E[H_2]` boxes | tail bound |
|---|---|---|---|---|---|
{chr(10).join(rows_parts)}

## 6. The `k`-integral

`I_k` is a four-dimensional box integral in the original coordinates `(a, beta, c, k)`, with weight
`W(k) = 3k^(1/3) exp(-12k^2)` (that is, `3k^2 exp(-12k^2) k^(-5/3)`):

    B = beta - a^2/(12k),   u = sqrt(k) c - a beta/(4 sqrt k) + a^3/(72 k^(3/2)).

- Each 4D box maps into a box of scaled coordinates, so the rectangle machinery of §4 applies unchanged.
- The Taylor moments are polynomials in `(a, beta, c)` times `k^(p/2)`, `p = -6..2`. Their `k`-factors
  `K_p = integral W(k) k^(p/2) dk` are enclosed on the `4096` elementary dyadic cells of `[1/2, 2]` by midpoint (below)
  and trapezoid (above) rules on `24` sub-cells. This is valid because every `f(k) = k^q exp(-12k^2)` with
  `q in [-8/3, 4/3]` is convex on `[1/2, 2]`: `f''/f = (24k - q/k)^2 - q/k^2 - 24 >= 87 - 16/3 - 24` for `q >= 0` and
  `>= 120` for `q < 0`.
- `k`-cells are dyadic and never finer than elementary.
- The tail is bounded uniformly in `k` by `17|B|^3 + 8T <= 68|beta|^3 + (68/216 + 576/5184) a^6 + 144 c^2 + 9 a^2 beta^2`.

Values:
- `int_K W(k) dk` is in `{sh(CF['int_K W(k) dk'], 9)}`;
- `const` is in `{sh(CF['const'], 10)}`;
- `I_b = sqrt(pi)(Phi(sqrt2) - 1/2)` is in `{sh(CF['I_b'], 10)}`;
- `I_k` (boxes plus tail) is in `{sh(CF['I_k'], 7)}`.

`I_b^(3)` is integrated on `100000` cells of `[0, 1]`. On each cell `exp(-b^2)` decreases and `g` increases (both positive),
so the cell integral lies between `h exp(-b_1^2) g(b_0)` and `h exp(-b_0^2) g(b_1)`.

## 6a. Lemma T: transfer to the torus

[LP] and [S] work on the torus `R^2/(L Z^2)` with
`K_L(z) = sum_n exp(-|z + Ln|^2/2) / sum_n exp(-|Ln|^2/2)` ([LP] §2, by Poisson summation). Everything certified in the
plane is a function of **one-site** jet covariances of order at most `6`:
- the odd conditional law of `(a, beta, c)` given `(f_x, f_z, f_xxx) = (0, 0, 12k)`;
- the even conditional law of `A = f_zz` given `(f, f_xx, f_xz) = (b, 0, 0)`, which gives `p_b(0)` and `z_0`;
- the factors of `pi_0` ([LP] §15).

`K_L` is even, so odd and even jets stay exactly uncorrelated.

**Lemma T.** For every side `L >= 10` and every frame:
1. **Covariances.** Every one-site covariance of order `<= 6` differs from its reference value by at most
   `delta <= {fmtd(rd(TT['jet covariance deviation delta (order <= 6)'][1], 3, ROUND_CEILING))}`. In a frame `R`,
   `d^g K_L(0) = sum_n (d^g G)(R^T L n)/Theta_L`, `|R^T L n| = L|n|`, and `|He_g1(y1) He_g2(y2)| <= Hh_g1(|y|) Hh_g2(|y|)` with
   `Hh_j(t) = sum_k j!/(k!(j-2k)! 2^k) t^(j-2k)`. The shell `|n|_inf = j` has `8j` points with `|n| >= j`. Each product
   `Hh_a Hh_c e^(-t^2/2)` decreases for `t^2 > a + c`, so the bound at `L = 10` holds for every `L >= 10`.
2. **Regressions.** Interval regressions with every entry widened by `delta` give the torus conditional laws:
   - odd: `N(12k v, S_c)` with `||diag(2,2,6)^(-1/2)(S_c - diag(2,2,6))diag(2,2,6)^(-1/2)|| <= eps = {EPS}`;
   - even: `N(c_1 b, s^2)`;
   - the ratio `(pi_0 p_b(0))^L/(pi_0 p_b(0))^ref` lies in `1 +- {RATIO_DEV}`, uniformly in `(b, k)` in the band;
   - `z_0` cancels in `A_0 a_fail` on the torus as well.
3. **Expectations.** Let `S_c = S_0^(1/2)(I + E)S_0^(1/2)` with eigenvalues `e_i` of `E`, and `mu~ = S_0^(-1/2) mu`.
   Coordinate-free, `1 + chi^2 = det(I - E^2)^(-1/2) exp(mu~^T (I - E)^(-1) mu~)`. In an orthonormal eigenbasis of `E`,
   with `mu~_i` the coordinates of `mu~` in that basis (not the original jet coordinates), this reads

       1 + chi^2 = prod (1 - e_i^2)^(-1/2) exp(sum mu~_i^2/(1 - e_i)).

   For `eps^2 <= 0.006` we have `(1 - eps^2)^(-3/2) <= 1.01` and `(1 - eps^2)^(-3/2) - 1 <= 2 eps^2`. With
   `e^m - 1 <= m e^m`, this gives `chi^2 <= 1.01 m e^m + 2 eps^2`, where `m = |mu~|^2/(1 - eps)`. The certificate checks
   `eps^2 <= 0.006` and obtains `chi^2 <= {fmtd(rd(TT['chi^2 bound'][1], 3, ROUND_CEILING))}`. By Cauchy–Schwarz, `|J^L - J| <= (E[I^2] chi^2)^(1/2)`, using
   `I = 3k^2 H <= 12 Pi_4` (the tail polynomial of §6, `k <= 2`), with `E[I^2] <= {EI2}`. This bounds `|J^L - J|` by
   `{fmtd(rd(TT['|J^L - J| bound (all k, J_fail and J_2)'][1], 3, ROUND_CEILING))}` for every `k` in `[1/2, 2]`, for `J_fail`
   and for `J_2` (`0 <= I_2 <= I_fail`).
4. **Constants.** Hence `C_fail^L` lies in `(1 +- {RATIO_DEV})(C_fail^ref +- const I_b K_(-4)/3 . |J^L - J|)`, with
   `K_(-4)/3 = integral e^(-12k^2) k^(-5/3) dk`. The torus `alpha_j`, `a_fail` at the grid points use the torus
   `p_b(0)/z_0`.

`torus.py` computes all of this in interval arithmetic. `RESULTS.json` lists the torus enclosures under "torus model".
Lemma T is planar: Corollary 2 stays a reference-kernel statement.

## 7. Arithmetic

Every `+, -, *, /, sqrt` is IEEE binary64 round-to-nearest, widened outward by one ulp. No libm transcendental is
trusted:
- `pi` comes from Machin's series with a rational remainder, and `ln 2` from `sum 1/(j 2^j)`.
- `exp(-y)`: `y = n ln2 + r` with `r in [0, 0.75]`, a degree-18 Taylor polynomial with the a priori bound
  `(56u + 10^-19) e^0.75`, then exact scaling by `2^-n`. The observed relative width is below `4e-13` for `y <= 700`.
- `Phi(t)` for `t < 2` is `1/2 + phi(t) sum t^(2n+1)/(2n+1)!!`, with an a priori summation bound and a geometric tail.
- `Q(t) = 1 - Phi(t)` for `t >= 2` is `phi(t) R(t)`, where `R` is Laplace's continued fraction
  `1/(t + 1/(t + 2/(t + ...)))`. It has positive elements, so consecutive approximants bracket `R`.

libm `pow`, `acos` and `cos` only seed iterations whose results are verified.

## 8. Verification

- **`--check`** (CI, under `-B -S` and `-B -O -S`) runs two stages:
  - a self-test: every box enclosure, and its first- and second-order parts separately, on `13` fixed 3D boxes and
    `2` 4D boxes contains a pure-Python Gauss–Legendre value (`12`–`16` points per axis; relative slack `1e-9`, or
    `1e-6` on the `C^1` support boundary of `H_2`), and the `k`-moments contain Gauss–Legendre values;
  - the full replay.
- The replayed enclosures must lie inside the published ones and be at most `2 x 10^-7` relatively (plus one unit of
  the 10th digit) tighter. The leaf counts must agree:
  - `V`: `{leaves['V k=0.5']:,}`, `{leaves['V k=1']:,}`, `{leaves['V k=2']:,}` at `k = 1/2, 1, 2`;
  - `H_2`: `{leaves['H2 k=0.5']:,}`, `{leaves['H2 k=1']:,}`, `{leaves['H2 k=2']:,}`;
  - `I_k` (4D): `{leaves['I_k 4D']:,}`.
- A full replay takes about 10 minutes on four cores.
- **Mutants** exit 1 at the self-test:
  - `vw-sign` flips `V_w`;
  - `ww-drop` drops `V_ww`;
  - `corner-swap` swaps the monotonicity corners of `V`;
  - `mid-branch` gives `H_2` the big-root sign of `V_w`;
  - `k-weight` drops `k^(1/3)` from `W`.
- **Development checks** (not in the packet):
  - `1,200` random 3D boxes and `250` random 4D boxes, `0` failures against `numpy` Gauss–Legendre;
  - the full-`H` integrand, boxed directly, gives an enclosure of `J_fail(1/2)` that intersects the split one
    (`controls.py`).

## 9. Comparisons (floating controls, `controls.py`)

{ctl('table_num', '(table)')}

## 10. What this does not do

- **Torus sides `L >= 10` only.** Lemma T covers every side `L >= 10` and every frame; `L < 10` is not covered. Corollary 2
  (`d = 3`) is for the reference kernel only.
- **Conditional identification.**
  - The certified numbers are the Gaussian integrals that [NUM] (from [CL] Theorem N, [CUB] Theorems C and F) and
    [S] §9 define.
  - That `a_fail` is the selector-failure coefficient and `C_fail^{{B,K}}` the lifetime constant are statements of
    [S]: AUTHOR_SIDE / HOLD, consumed, not reviewed.
  - `nu(1) = alpha_1 + k integral_X Lambda` also contains the remote part of [RM], which is not evaluated here.
  - Corollary 2 is conditional in addition on [FIB] (Math-#175, open, AUTHOR_SIDE / HOLD) and [HD] (Math-#184,
    open). Its pointwise values are given only at `b = 0`; `R_3(1)` would need `m_(3,1)`, which is not certified here.
- **Coverage.**
  - Pointwise coefficients only at `k = 1/2, 1, 2` and `b = 0, 1`, plus the band integral `C_fail^{{B,K}}`.
  - No other band and no `k -> 0`. `d = 3` is covered only through Corollary 2, and there is nothing for `d >= 4`.
- No rate, no finite-`r` statement.
- Not a review of [NUM], [S], [CL], [CUB], [LP] or Math-#215.

## 11. Provenance

- **Sources:** `SOURCE_MAP.json` pins [NUM], [S], [LP], [CL], [CUB] and the [NUM] results on `main` `3e0a91b`.
  Math-#215 (`d748c2a`, open) is pinned for comparison constants only. Math-#175 (`701dff4`) and Math-#184 (`d37ff5d`),
  both open, are pinned for Corollary 2.
- **Packet files:**
  - `certificate.py` (drivers, exact parts, tails, assembly, self-test, replay), `boxes.py` (roots, Taylor boxes),
    `torus.py` (Lemma T) and `ia.py` (interval arithmetic), all standard-library Python;
  - `controls.py` (floating controls);
  - `RESULTS.json`, the certificate's output.
- **Manifest:** `SOURCE_FILES.json`.
"""
open(PK + '/NOTE.md', 'w').write(NOTE)
print('NOTE written', len(NOTE))

README = f"""# Certified planar near cluster coefficients and the elder-failure constant

Certified interval enclosures, for the reference planar kernel, of the near cluster coefficients `alpha_1`, `alpha_2`
(`nu(2) = alpha_2`) of [NUM] at `k = 1/2, 1, 2`, `b = 0, 1`, and of the constant `C_fail^{{B,K}}` of [S] §9 on
`[0, 1] x [1/2, 2]`. Details and the method are in [NOTE.md](NOTE.md).

- `J_fail(1/2)` is in `{sh(res['k=0.5']['J_fail'], 8)}`, `J_fail(1)` in `{sh(res['k=1']['J_fail'], 8)}` and `J_fail(2)` in
  `{sh(res['k=2']['J_fail'], 8)}`.
- `J_2(1/2)` is in `{sh(res['k=0.5']['J_2'], 5)}`, `J_2(1)` in `{sh(res['k=1']['J_2'], 5)}` and `J_2(2)` in
  `{sh(res['k=2']['J_2'], 5)}`. The last settles the Gauss–Hermite / Monte Carlo disagreement that [NUM] reported at
  `k = 2`.
- `C_fail^{{B,K}}` is in `{sh(CF['C_fail^{B,K}'], 5)}`, about `1.5 x 10^5` below the cap-route constant
  `C_{{B,K}}(1/4096) = 402.53` of Math-#215.
- In `d = 3`, conditionally on Math-#175 and Math-#184, `C_fail^(3),{{B,K}}` is in `{sh(D3['C_fail^(3),{B,K}'], 5)}`.

The planar values hold for the reference kernel and, by a torus-transfer lemma, for the torus model of [LP] and [S] with
every side `L >= 10` and every frame.

**Method.** A third-order box scheme:
- second-order Taylor expansion in `(B, w = |u|)`, where the Hessian of the failure integrand is bounded and explicit;
- interval Hessian ranges times exact Gaussian moments, intersected with first-order monotone bounds;
- `k` as a fourth box coordinate for `C_fail`.

**Arithmetic.** Standard-library Python and binary64 interval arithmetic; no transcendental library value is trusted.

**Replay.** `python3 -B -S certificate.py --check` (CI in both interpreter modes, with five mutants). Floating controls:
`python3 -B certificate.py --controls`.

**Scientific effect:** NONE. Author-side; conditional on the sources' identifications ([NUM] from [CL]/[CUB], [S]).
Same GitHub account as every lane; zero organizational-independence credit.
"""
open(PK + '/README.md', 'w').write(README)
print('README written', len(README))
