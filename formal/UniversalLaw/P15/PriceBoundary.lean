/-!
# P15 demand-one price boundary — exact finite arithmetic of the counterexample

Informal source (Layer 0, byte-pinned): `Math-` commit
`760340e921ac4ceda296b8118da936f1133e956e`,
`frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md`, 2266 bytes, SHA-256
`498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b`
(object P15-PRICE-BOUNDARY-20260924-v1, author OpenAI / ChatGPT).

The source exhibits the smallest one-block member of the realized-cover family
(`a_1 = d_1 = 1`, `X = {x, y}`, no macro edges) with `p_x = p_y = 1/2` and
transformed prices `c_x = c_y = 2/3`, and computes `mu_p(D) = 3/4`, the four
generator costs `1, 2/3, 2/3, 4/9` and the comparison `4/9 > 1/3`. This module
restates exactly that finite arithmetic: subsets of the two-coordinate ground set
are Boolean pairs, the product measure at `p = 1/2` is carried as integer
numerators over the denominator `4`, and prices at `c = 2/3` as integer
numerators over the denominator `9`. Core Lean's `Rat` is not kernel-decidable
by `decide` (its `DecidableEq` instance does not reduce), so each rational
statement is the equivalent cross-multiplied statement over `Nat` with the
rational form in the docstring. The kernel checks the integers; the alignment
review checks that each integer statement is the stated rational one.

What this module does **not** establish: the two logarithm inequalities the
source proves (`phi(1/2) = log 2 > 2/3` and `log(4/3) < 1/3`), the identity
`min(1, -log mu_p(D)) = log(4/3)`, the probability semantics of `mu_p` beyond
the displayed finite sum, that every cover of the obstruction `{X}` must contain
a generator inside `X`, that additional nonnegative-price generators cannot lower
the minimum, and therefore the source's conclusion that the same-palette
extension to `c <= phi(p)` is false in general. Nothing here bears on P15-B, on
Theorem F, on any prize, or on any Layer 0 status.
-/

namespace UniversalLaw.P15

/-- The ground set `X = {x, y}` of the smallest one-block member: a subset of `X`
is a pair of Booleans (`x ∈ U`, `y ∈ U`). These are its four subsets, in the
order `∅, {x}, {y}, X`. -/
def subsets2 : List (Bool × Bool) := [(false, false), (true, false), (false, true), (true, true)]

/-- Membership in the demand-one downset `D` on one block of capacity `a_1 = 1`
with no macro edges: at most one coordinate is present. -/
def good (U : Bool × Bool) : Bool := !(U.1 && U.2)

/-- Numerator of the product-measure weight of `U` at `p_x = p_y = 1/2`, over the
common denominator `2 · 2 = 4`: a present coordinate contributes the numerator of
`p = 1/2`, an absent one the numerator of `1 - p = (2 - 1)/2`; both are `1`. -/
def weightNum (U : Bool × Bool) : Nat := (if U.1 then 1 else 2 - 1) * (if U.2 then 1 else 2 - 1)

/-- Numerator of the generator price `∏_{v ∈ U} c_v` at `c_x = c_y = 2/3`, over
the common denominator `3 · 3 = 9`: a present coordinate contributes `2` (the
numerator of `2/3`), an absent one `3` (the empty factor `1 = 3/3`). -/
def priceNum (U : Bool × Bool) : Nat := (if U.1 then 2 else 3) * (if U.2 then 2 else 3)

/-- Total product-measure mass of all four subsets, in units of `1/4`. -/
def totalMassNum : Nat := subsets2.foldl (fun acc U => acc + weightNum U) 0

/-- Product-measure mass of the good sets, in units of `1/4`. -/
def goodMassNum : Nat := (subsets2.filter good).foldl (fun acc U => acc + weightNum U) 0

/-- Source: "Consider its smallest one-block member: a_1=d_1=1, X={x,y}, no macro
edges." The block-size formula `n_i = |X_i| = a_i d_i + 1` is not displayed in this
note; it is display (P1) of the realized-cover note (`p15-realized-covers`,
`P15_REALIZED_COVERS.md`: "n_i=|X_i|=a_i d_i+1."). Checked: at `a_1 = d_1 = 1` it
gives `2 = |{x, y}|`. -/
theorem smallest_member_block_size : 1 * 1 + 1 = 2 := by decide

/-- Source: "Its good sets are the empty set and singletons". Filtering the four
subsets by the capacity-one condition leaves exactly `∅`, `{x}`, `{y}`. -/
theorem demand_one_good_sets :
    subsets2.filter good = [(false, false), (true, false), (false, true)] := by decide

/-- Source: "its one-piece obstruction is exactly {X}". The only subset outside
`D` is `X = {x, y}` itself; with `I_1(D) = D` this is `O_1(D) = {X}`. -/
theorem demand_one_obstruction : subsets2.filter (fun U => !good U) = [(true, true)] := by decide

/-- Source: "Nevertheless mu_p(D)=3/4". At `p_x = p_y = 1/2` every subset has
weight `1/4`; the four weights sum to `1` (`4/4`) and the three good sets to
`3/4`. Cross-multiplied: `goodMassNum / 2^2 = 3/4 ⇔ goodMassNum · 4 = 3 · 2^2`. -/
theorem good_measure_three_quarters : totalMassNum = 2 ^ 2 ∧ goodMassNum * 4 = 3 * 2 ^ 2 := by
  decide

/-- Source: "The possible generator costs are 1, 2/3, 2/3, 4/9 for the empty set,
the two singletons, and X respectively" (the source runs "are1,2/3,2/3,4/9"
together). Over the denominator `9`: `9/9, 6/9, 6/9, 4/9`. -/
theorem generator_prices_ninths : 3 * 3 = 9 ∧ subsets2.map priceNum = [9, 6, 6, 4] := by decide

/-- Source: "covercost_c(O_1(D))=4/9 > 1/3 > log(4/3)". Checked here: the minimum
of the four displayed generator costs is `4/9` (numerator `4` over `9`), and
`4/9 > 1/3 ⇔ 1 · 9 < 4 · 3`. The identification of that minimum with the exact
optimal cover price, and the final comparison with `log(4/3)`, are not checked. -/
theorem minimum_generator_price_exceeds_third :
    (subsets2.map priceNum).foldl min 9 = 4 ∧ 1 * 9 < 4 * 3 := by decide

end UniversalLaw.P15
