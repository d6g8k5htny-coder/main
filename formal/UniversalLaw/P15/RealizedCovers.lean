/-!
# P15 realized covers — exact counts of the 816-label benchmark

Informal source (Layer 0, byte-pinned): `Math-` commit
`760340e921ac4ceda296b8118da936f1133e956e`,
`frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md`, 11467 bytes, SHA-256
`c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9`
(object P15-REALIZED-COVERS-20260924-v1, author OpenAI / ChatGPT), sections 4–6.

What this module establishes: the exact integer and rational arithmetic that the
source displays for the six-block benchmark (`a_i = 1`, `d_i = 408`, `H` = all
four-block subsets of six indices) — block and ground-set sizes, the two
minimal-forbidder counts of display (P10), the palette counts `816`, `815·3 < 2448`
and `818`, the elementary palette formula (P9) evaluated at the displayed inputs,
the ceiling identity `⌈n_i/a_i⌉ = d_i + 1` at finitely many instances, and the
union-bound rational of display (P11), and the positivity of the displayed cover
price `6/(40900^409)` as positivity of its numerator and of the power. Binomial
coefficients are computed by the exact multiplicative recursion and ceilings by
`(a + b − 1) / b`; both are defined locally so that the library has no
dependencies. Core Lean's `Rat` is not kernel-decidable by `decide` (its
`DecidableEq` instance does not reduce), so (P11) is stated cross-multiplied over
`Nat` with the rational form in the docstring. Every proof is `decide` except the
positivity of `40900 ^ 409`, which is core `Nat.pow_pos` applied to `0 < 40900`.

What this module does **not** establish: Proposition P1 (completeness of the
minimal forbidders), display (P2), Theorem P2 (that the full-block family covers
`O_K(D)` iff `K >= K_H(d)`), the palette formula (P9) itself, the probability
semantics of `mu_p`, the union bound and independence step behind (P11), the
hazard comparison (P4)–(P5), display (P8), or anything about the unrestricted
P15 prize, which the source itself leaves open. Nothing here changes any Layer 0
status.
-/

namespace UniversalLaw.P15

/-- Binomial coefficient by the exact multiplicative recursion
`C(n, k+1) = C(n, k) · (n − k) / (k + 1)` (each product is divisible by `k + 1`,
and `n − k` truncates to `0` once `k ≥ n`). Kept local: no dependencies. -/
def choose (n : Nat) : Nat → Nat
  | 0 => 1
  | k + 1 => choose n k * (n - k) / (k + 1)

/-- Ceiling of `a / b` for `b > 0`: `⌈a/b⌉ = (a + b − 1) / b`. -/
def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Source: "Take six blocks, a_i=1 and d_i=408. Thus EACH original block has 409
vertices and the original ground set has 2454 vertices" (the source runs
"has409" and "has2454" together). `n_i = a_i d_i + 1 = 409`, `6 · 409 = 2454`. -/
theorem benchmark_block_and_ground_sizes : 1 * 408 + 1 = 409 ∧ 6 * 409 = 2454 := by decide

/-- Source: "H consists of all 15 four-block subsets" (source: "all15"):
`C(6, 4) = 15`. -/
theorem four_block_supports_fifteen : choose 6 4 = 15 := by decide

/-- Source (P10): "6*binom(409,2)=500616 internal pairs, 15*409^4=419743994415
crossing transversal quadruples." Only the two integer values are checked; that
these count the minimal forbidders is Proposition P1. -/
theorem minimal_forbidder_counts :
    6 * choose 409 2 = 500616 ∧ 15 * 409 ^ 4 = 419743994415 := by decide

/-- Source: "Use 408 labels on macro-indices {0,2,4} and 408 different labels on
{1,3,5}, giving K_H(408,...,408)=816." Checked: `408 + 408 = 816`. That this
palette satisfies (P6), and that no smaller one does, is not checked. -/
theorem palette_816_from_two_label_classes : 408 + 408 = 816 := by decide

/-- Source: "at 815 it does not, because omitting one vertex from each block
leaves 2448 vertices, each color holds at most 3, and 815*3<2448.
Pairwise-disjoint local palettes would have used 2448 labels." Checked:
`2454 − 6 = 2448`, `815 · 3 < 2448`, `6 · 408 = 2448`. That each colour class
holds at most three vertices is the combinatorial step, not checked. -/
theorem palette_815_fails_by_counting : 2454 - 6 = 2448 ∧ 815 * 3 < 2448 ∧ 6 * 408 = 2448 := by
  decide

/-- Source: "The whole ground set, unlike the cover-avoiding sets, needs 818
colors, since 2454/3=818." Checked: the exact quotient `2454 / 3 = 818` with
`818 · 3 = 2454`. The chromatic claim itself is not checked. -/
theorem whole_ground_chromatic_818 : 2454 / 3 = 818 ∧ 818 * 3 = 2454 := by decide

/-- Source (P9): "K_H(d)=max(max_i d_i, ceil(sum_i d_i/s))" for `H` = all
`(s+1)`-subsets of `b` indices. The section-6 benchmark is the case `b = 6`,
`s + 1 = 4`, `d_i = 408`: `max(408, ⌈6·408/3⌉) = 816`; with `d_i + 1 = 409` for
the whole ground set: `max(409, ⌈6·409/3⌉) = 818`. The conjunct `3 + 1 = 4` is
not displayed: it reads the four-block supports as `s + 1 = 4` with `s = 3`. Only
the right-hand side of (P9) is evaluated; the formula is not proved. -/
theorem elementary_palette_formula_instances :
    3 + 1 = 4 ∧ max 408 (ceilDiv (6 * 408) 3) = 816 ∧ max 409 (ceilDiv (6 * 409) 3) = 818 := by
  decide

/-- Source: "The macro triangle at unit demands needs 3 colors" (source:
"needs3"). The right-hand side of (P9) at `b = 3`, `s = 1`, `d_i = 1`:
`max(1, ⌈3/1⌉) = 3`. The chromatic claim is not checked. -/
theorem macro_triangle_three_colors : max 1 (ceilDiv (1 + 1 + 1) 1) = 3 := by decide

/-- Source: "the chromatic number of the ENTIRE ground set X is K_H(d+1), because
ceil(n_i/a_i)=d_i+1." With `n_i = a_i d_i + 1`, checked at five `(a_i, d_i)`
instances including the benchmark `(1, 408)`. The general identity for all
`a_i ≥ 1` is not checked. -/
theorem ceil_block_over_capacity_instances :
    ∀ p, p ∈ [(1, 1), (1, 408), (2, 3), (3, 5), (5, 2)] →
      ceilDiv (p.1 * p.2 + 1) p.1 = p.2 + 1 := by decide

/-- Source: "each block's occupancy probability is at most 409p=1/100" (source:
"at most409p") at `p = 1/40900`: `409 · (1/40900) = 1/100 ⇔ 409 · 100 = 40900`.
That the occupancy probability is at most `409p` (a union bound) is not checked. -/
theorem block_occupancy_one_percent : 409 * 100 = 40900 := by decide

/-- Source (P11): "Q=1-mu_p(D) <=500616/(40900^2)+15/(100^4) =2449227/8180000000
< 3/10000" and "the actual-good probability exceeds 0.9997". Cross-multiplied:
`500616/40900^2 + 15/100^4 = 2449227/8180000000` is
`(500616·100^4 + 15·40900^2)·8180000000 = 2449227·(40900^2·100^4)`;
`2449227/8180000000 < 3/10000` is `2449227·10000 < 3·8180000000`; and
`1 − 3/10000 = 9997/10000`. The union bound, the independence of block
occupancies and the probability semantics are not checked. -/
theorem low_failure_union_bound_arithmetic :
    (500616 * 100 ^ 4 + 15 * 40900 ^ 2) * 8180000000 = 2449227 * (40900 ^ 2 * 100 ^ 4) ∧
    2449227 * 10000 < 3 * 8180000000 ∧ 10000 - 3 = 9997 := by decide

/-- Source: "The cover price is exactly 6/(40900^409)>0" at `p_v = c_v = 1/40900`.
Checked: `6/(40900^409) > 0` as positivity of the numerator `6` (six full-block
generators) and of the denominator `40900^409`; the latter is core `Nat.pow_pos`
applied to `0 < 40900` (the kernel does not evaluate the power). The price
identity itself — that the six-generator cover costs exactly `6/(40900^409)` —
is not checked. -/
theorem cover_price_numerator_and_denominator_positive : 0 < 6 ∧ 0 < 40900 ^ 409 :=
  ⟨by decide, Nat.pow_pos (a := 40900) (n := 409) (by decide)⟩

end UniversalLaw.P15
