import Mathlib

/-!
# P15 realized covers, price boundary, restricted and full transformed-price budgets (statements only)

Scientific effect: NONE. Formal progress of every declaration here: `specified`.
Organizational-independence credit of this transcription: 0 (Anthropic / Claude).

This module transcribes, as definitions and `Prop`-valued specifications, the finite
combinatorics of four Layer 0 notes of `d6g8k5htny-coder/Math-` at commit
`760340e921ac4ceda296b8118da936f1133e956e`:

* `frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md` (P15-REALIZED-COVERS-20260924-v1,
  source id `p15-realized-covers`): the realized capacity/clutter family, its downset `D` (P1),
  the actual local restrictions (P2), the local covers (P3)-(P5), the palette demand `K_H(d)`
  (P6), the full-block cover criterion (P7), the capped hazard bound (P8), the elementary
  palette formula (P9), and the six-block 816-label realization with its counts (P10) and its
  union bound (P11);
* `frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md` (P15-PRICE-BOUNDARY-20260924-v1,
  source id `p15-price-boundary`): the exact two-coordinate refutation of the same-palette
  transformed-price extension at demand one;
* `frontiers/price_budget_20260924/PROOF.md` (P15-TRANSFORMED-PRICE-BUDGET-20260924-v1, source
  id `p15-price-budget`): the restricted budget (T1) with factor `16/27` for `p_v <= 1/4` and
  `d_i >= 2`, its local displays (T2)-(T5), and the 816-label consequence (T7);
* `frontiers/full_price_20260924/PROOF.md` (P15-FULL-TRANSFORMED-PRICE-20260924-v1, source id
  `p15-full-price`): the full-cube budget (F2) with the constant `rho_star = 1/(3 - log(3e - 2))`,
  its enclosure (F3), the hazard interpolation bounds (F6)-(F8), the capacity worst case
  (F9)-(F11), the assembly (F12)-(F13), the sharpness and demand-one boundary of Section 5 and
  the rational certificates of Section 6.

Because the domain is finite combinatorics, everything is DEFINED rather than given by
interface fields: a realized family is a data record `RealizedData` (finite ground type,
block map, capacities, demands, clutter) and its side conditions are one `Prop`
(`IsRealizedFamily`), so that the concrete realizations of the sources (`benchmark816`,
`twoCoord`, `tripleBlock`, `demandOneBlock`) are plain definitions without embedded proofs.
Two conventions are made explicit where the sources use extended arithmetic: the hazard
`-log 0 = +infinity` is rendered by `cappedHazard`, which returns the cap `1` when the
probability is zero (the sources: "the trivial price-one cover handles the endpoint"), and
`phi(1) = 1` is rendered by a guard in `phi`. The palette demand `K_H(d)` is `sInf` of the
feasible palette sizes; the sources call it "the minimum size K", and that the feasible set is
nonempty (so that `sInf` is a minimum) is NOT proved here but stated as `PaletteNumberAttained`.
The cover cost is `sInf` of the prices of all generator covers, a finite nonempty set of reals.

What this module does not establish. It consists only of definitions and `Prop`s; it states
nothing as a result and proves nothing. It does not establish (P7), (P8), (T1), (F2), the
enclosure (F3), `rho_star < 6/7`, any count of (P10), the union bound (P11), the refutation of
the demand-one extension, or any of the displayed inequalities; it does not identify these
realized families with any earlier abstract P15 benchmark; it does not touch original P15-B
or the unrestricted P15 prize. The arithmetic skeleton of several displays (the two-coordinate
prices `9, 6, 6, 4` over `9`, `3/4` over `4`, `1/3 < 4/9`; the counts `500616` and
`419743994415`; `2449227/8180000000 < 3/10000`; `16/27 < 6/7`; the adjacency of the (F3)
endpoints and `upper * 7 < 6 * 10^20`) is the target of the core-Lean package
`formal/UniversalLaw/P15/*` on main and is not re-stated as a result here. Nothing here
promotes, reclassifies or discharges any claim, premise or obligation of the registers;
alignment of these statements with the sources is PENDING_INDEPENDENT_REVIEW. Finite
`Prop`s about the concrete realizations (`twoCoord`, `tripleBlock`, the palette numbers of small
clutters) are decidable in principle and are future kernel-proof targets; the `Prop`s about
`benchmark816` quantify over `2^2454` subsets and are finite but not decidable in practice.

Primary sources (Layer 0, byte-pinned): `d6g8k5htny-coder/Math-` commit
`760340e921ac4ceda296b8118da936f1133e956e`; sha256 of the four paths:
`c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9` (realized covers),
`498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b` (price boundary),
`3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535` (price budget),
`87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9` (full price).
-/

namespace UniversalLaw.Spec.P15

/-! ## Product measure, prices, the transformed price `phi` and the capped hazard -/

/-- Independent-coordinate weight of a subset `U` of a finite ground type `Y` under the
probability vector `p`: `prod_{v in U} p_v * prod_{v notin U} (1 - p_v)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "Select the ORIGINAL coordinates independently with arbitrary probabilities p_v in [0,1], and let 0<=c_v<=p_v."
Interface: none; `p` is any real vector (the sources take `p_v in [0,1]`, imposed as a hypothesis where used).
Does not claim: that these weights sum to one over all subsets (not stated separately). -/
noncomputable def weight {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ) (U : Finset Y) : ℝ :=
  (∏ v ∈ U, p v) * ∏ v ∈ Finset.univ \ U, (1 - p v)

/-- Product-measure mass `mu_p(A)` of a finite family `A` of subsets of a finite ground type:
the sum of the weights of its members.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "F_A(t)=-log mu_(p(t))(A),"
Interface: none.
Does not claim: anything; this is the probability that the random set of independently selected coordinates lies in `A`. -/
noncomputable def muOf {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    (p : Y → ℝ) : ℝ :=
  ∑ U ∈ A, weight p U

/-- Price of a generator `g`: `product_(v in g) c_v`; the empty generator has price `1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "A generator's price is product_(v in g)c_v, and family price is the sum of generator prices."
Interface: none.
Does not claim: anything. -/
noncomputable def genPrice {Y : Type} (c : Y → ℝ) (g : Finset Y) : ℝ :=
  ∏ v ∈ g, c v

/-- Price of a generator family `G`: the sum of its generator prices; the empty family has
price `0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "The empty generator has price1; the empty family has price0."
Interface: none.
Does not claim: anything. -/
noncomputable def famPrice {Y : Type} (c : Y → ℝ) (G : Finset (Finset Y)) : ℝ :=
  ∑ g ∈ G, genPrice c g

/-- The transformed price ceiling `phi(t) = min(1, -log(1 - t))` with `phi(1) = 1`. The guard
`t < 1` renders the source's convention at `t = 1` (Mathlib's `Real.log 0 = 0` would otherwise
give `0`); for `t > 1`, outside the source's domain, the value `1` is a placeholder.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "where phi(t)=min(1,-log(1-t)), phi(1)=1."
Interface: none.
Does not claim: any value outside `[0, 1]`. -/
noncomputable def phi (t : ℝ) : ℝ :=
  if t < 1 then min 1 (-Real.log (1 - t)) else 1

/-- The capped, scaled hazard `min(1, rho * [-log m])` of a probability `m`, with the sources'
convention that the hazard of a zero probability is `+infinity`, so that the cap `1` applies:
`cappedHazard rho m = 1` when `m <= 0`. With `rho = 1` this is `min(1, -log mu_p(D))` of (P8);
with `rho = 16/27` the right side of (T1); with `rho = rho_star` the right side of (F2).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "If some q_i=1 the trivial price-one cover handles the endpoint."
Interface: none.
Does not claim: anything; the branch at `m <= 0` is the sources' endpoint convention, not a result. -/
noncomputable def cappedHazard (ρ m : ℝ) : ℝ :=
  if 0 < m then min 1 (ρ * (-Real.log m)) else 1

/-! ## The realized capacity/clutter family -/

/-- Data of a realized family on a finite ground type `X`: `b` blocks, the block map `blk`
(the blocks `X_i` are its fibres, hence disjoint and covering), capacities `a_i`, demands `d_i`
and the clutter `H` of forbidden block supports. The side conditions of the sources are the
separate `Prop` `IsRealizedFamily`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "Choose positive integers a_i,d_i and disjoint original-coordinate blocks X_i with"
Interface: none; every field is a finite datum, nothing is a hypothesis field.
Does not claim: that the data satisfy the sources' side conditions (that is `IsRealizedFamily`). -/
structure RealizedData (X : Type) where
  /-- The number of blocks. -/
  b : ℕ
  /-- The block of each original coordinate. -/
  blk : X → Fin b
  /-- The capacities `a_i`. -/
  a : Fin b → ℕ
  /-- The demands `d_i`. -/
  d : Fin b → ℕ
  /-- The clutter `H` of forbidden block supports. -/
  H : Finset (Finset (Fin b))

/-- The original-coordinate block `X_i = {v : blk v = i}`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "disjoint original-coordinate blocks X_i"
Interface: none.
Does not claim: anything. -/
def block {X : Type} [Fintype X] (F : RealizedData X) (i : Fin F.b) : Finset X :=
  Finset.univ.filter (fun v => F.blk v = i)

/-- The side conditions of the realized family: `a_i >= 1`, `d_i >= 1` ("positive integers
a_i,d_i"), `|X_i| = a_i d_i + 1`, every edge of `H` has size at least `2`, and `H` is a clutter
(no edge properly contains another; edges are distinct as members of a `Finset`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    n_i=|X_i|=a_i d_i+1."
Interface: none; this `Prop` IS the interface of the sources, stated for a data record.
Does not claim: that any particular data record satisfies it. -/
def IsRealizedFamily {X : Type} [Fintype X] (F : RealizedData X) : Prop :=
  (∀ i, 1 ≤ F.a i) ∧ (∀ i, 1 ≤ F.d i) ∧ (∀ i, (block F i).card = F.a i * F.d i + 1) ∧
    (∀ e ∈ F.H, 2 ≤ e.card) ∧ (∀ e ∈ F.H, ∀ f ∈ F.H, e ⊆ f → e = f)

/-- The block support `supp(U) = {i : U intersect X_i is nonempty}`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "supp(U)={i:U intersect X_i is nonempty}.              (P1)"
Interface: none.
Does not claim: anything. -/
def supp {X : Type} (F : RealizedData X) (U : Finset X) : Finset (Fin F.b) :=
  U.image F.blk

/-- Membership in the downset `D` of display (P1): `|U intersect X_i| <= a_i` for every `i`, and
`supp(U)` contains no edge of `H`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "D={U subset X: |U intersect X_i|<=a_i for every i,"
Interface: none.
Does not claim: that `D` is decreasing or contains the empty set (facts of the sources, not stated separately). -/
def IsGood {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (U : Finset X) : Prop :=
  (∀ i, (U ∩ block F i).card ≤ F.a i) ∧ ∀ e ∈ F.H, ¬ e ⊆ supp F U

/-- The downset `D` as a set of subsets of `X`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "                   supp(U) contains no edge of H},"
Interface: none.
Does not claim: anything. -/
def downset {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Set (Finset X) :=
  {U | IsGood F U}

/-- `U in I_k(D)`: `U` is partitionable into at most `k` members of `D` (a family of `k`
pairwise disjoint good parts, possibly empty, with union `U`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "write I_k(D) for sets partitionable into at most k members of D"
Interface: none.
Does not claim: equivalence with coverings by at most `k` good sets (true for a decreasing family, not stated). -/
def IsKDecomposable {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (k : ℕ)
    (U : Finset X) : Prop :=
  ∃ parts : Fin k → Finset X, (∀ j, IsGood F (parts j)) ∧
    (∀ j j', j ≠ j' → Disjoint (parts j) (parts j')) ∧
    (Finset.univ : Finset (Fin k)).biUnion parts = U

/-- The obstruction `O_K(D) = 2^X \ I_K(D)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "O_k(D)=2^X\I_k(D) for its obstruction"
Interface: none.
Does not claim: anything. -/
def obstruction {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (K : ℕ) :
    Set (Finset X) :=
  {U | ¬ IsKDecomposable F K U}

/-- `G` is a generator cover of `O_K(D)`: every obstructed set contains some `g in G`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "A generator cover G of O_k(D) means every obstructed set contains some g in G."
Interface: none.
Does not claim: anything. -/
def IsGeneratorCover {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (K : ℕ)
    (G : Finset (Finset X)) : Prop :=
  ∀ U : Finset X, U ∈ obstruction F K → ∃ g ∈ G, g ⊆ U

/-- The exact optimal cover price `covercost_c(O_K(D))`: the infimum of the family prices of all
generator covers of `O_K(D)`. The set of covers is finite and nonempty (`{empty set}` always
covers), so the infimum is a minimum; that fact is not proved here.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "Therefore the EXACT optimal cover price is"
Interface: none.
Does not claim: that the infimum is attained (finite, not proved here); for `c` with negative entries, outside the sources' domain, `sInf` of an unbounded set is Mathlib's default. -/
noncomputable def coverCost {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (K : ℕ)
    (c : X → ℝ) : ℝ :=
  sInf {t : ℝ | ∃ G : Finset (Finset X), IsGeneratorCover F K G ∧ famPrice c G = t}

/-- The product measure `mu_p(D)` of the downset: the sum over `U in D` of
`prod_{v in U} p_v prod_{v notin U} (1 - p_v)`, written as a sum over `2^X` of the indicator of
`D` times the weight.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "For EVERY independent original probability vector p in [0,1]^X,"
Interface: none.
Does not claim: anything. -/
noncomputable def mu {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (p : X → ℝ) : ℝ :=
  ∑ U ∈ (Finset.univ : Finset X).powerset, Set.indicator (downset F) (weight p) U

/-- The actual local good probability `mu_p(D_i) = P(U intersect X_i in D)`, with
`D_i = D intersect 2^(X_i)`: the sum over subsets `S` of the block of the indicator of `D` times
the block-restricted weight.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "Write q_i=P(U intersect X_i notin D_i) for the ACTUAL capacity restriction."
Interface: none.
Does not claim: the identification `D_i = {S subset X_i : |S| <= a_i}` (that is `LocalRestriction`). -/
noncomputable def localGood {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (i : Fin F.b) : ℝ :=
  ∑ S ∈ (block F i).powerset,
    Set.indicator (downset F) (fun S => (∏ v ∈ S, p v) * ∏ v ∈ block F i \ S, (1 - p v)) S

/-- The local failure probability `q_i = P(U_i notin D_i) = 1 - mu_p(D_i)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "With q_i=P(U_i notin D_i),"
Interface: none.
Does not claim: anything. -/
noncomputable def localFailure {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (i : Fin F.b) : ℝ :=
  1 - localGood F p i

/-! ## Palette demand -/

/-- `K` is a feasible palette size for the clutter `H` and demands `d`: there are label sets
`P_i subset {0, ..., K-1}` with `|P_i| >= d_i` and empty intersection over every `H`-edge
(no label lies in `P_i` for all `i in e`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    intersection_(i in e) P_i = empty, for every e in H. (P6)"
Interface: none.
Does not claim: upward closure in `K` (true, not stated). -/
def IsPaletteFeasible (b : ℕ) (H : Finset (Finset (Fin b))) (d : Fin b → ℕ) (K : ℕ) : Prop :=
  ∃ P : Fin b → Finset (Fin K), (∀ i, d i ≤ (P i).card) ∧
    ∀ e ∈ H, ∀ ℓ : Fin K, ∃ i ∈ e, ℓ ∉ P i

/-- The palette demand `K_H(d)`: the least feasible palette size, as `sInf` of the feasible set
(Mathlib's `sInf` on `ℕ` is the least element of a nonempty set and `0` for the empty set). The
sources define it as "the minimum size K"; nonemptiness of the feasible set is the separate
`Prop` `PaletteNumberAttained`, not proved here.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "Let K_H(d) be the minimum size K of an integer palette universe admitting subsets P_i with |P_i|>=d_i and"
Interface: none.
Does not claim: that the feasible set is nonempty; the source's remark that one may take `|P_i| = d_i`. -/
noncomputable def paletteNumber (b : ℕ) (H : Finset (Finset (Fin b))) (d : Fin b → ℕ) : ℕ :=
  sInf {K : ℕ | IsPaletteFeasible b H d K}

/-- The feasible set of palette sizes is nonempty, i.e. `K_H(d)` is itself feasible (the minimum
is attained), for every realized family.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "By discarding excess labels within blocks, one may take |P_i|=d_i."
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis.
Does not claim: the statement (not proved here; disjoint palettes of sizes `d_i` are a witness in the sources' argument). -/
def PaletteNumberAttained {X : Type} [Fintype X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → IsPaletteFeasible F.b F.H F.d (paletteNumber F.b F.H F.d)

/-- The specified cover: the family `G = {X_1, ..., X_b}` of full blocks.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "the fixed family G={X_1,...,X_b} covers O_K(D) if and only if"
Interface: none.
Does not claim: anything. -/
def fullBlockFamily {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    Finset (Finset X) :=
  (Finset.univ : Finset (Fin F.b)).image (block F)

/-! ## Realized covers: Propositions P1, P2 and the local covers (P3)-(P5) -/

/-- `U` is an inclusion-minimal forbidden set: not good, and every proper subset obtained by
removing one element is good.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "The inclusion-minimal forbidden ORIGINAL sets are exactly"
Interface: none.
Does not claim: anything. -/
def IsMinimalForbidden {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (U : Finset X) : Prop :=
  ¬ IsGood F U ∧ ∀ v ∈ U, IsGood F (U.erase v)

/-- `U` is a transversal `{x_i : i in e}` of the edge `e`: its support is `e` and it meets each
block of `e` in exactly one coordinate.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "(b) for each e in H, EVERY transversal {x_i:i in e}, with x_i in X_i."
Interface: none.
Does not claim: anything. -/
def IsTransversal {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (e : Finset (Fin F.b)) (U : Finset X) : Prop :=
  supp F U = e ∧ ∀ i ∈ e, (U ∩ block F i).card = 1

/-- Proposition P1: the inclusion-minimal forbidden sets are exactly (a) the `(a_i+1)`-subsets
of a single block and (b) the transversals of the edges of `H`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "(a) every (a_i+1)-subset of a single X_i;"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis.
Does not claim: the characterisation (statement only); its completeness clause is what the source calls "completeness, not just existence of some witnesses". -/
def MinimalForbiddenCharacterisation {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ U : Finset X, IsMinimalForbidden F U ↔
    ((∃ i, U ⊆ block F i ∧ U.card = F.a i + 1) ∨ ∃ e ∈ F.H, IsTransversal F e U)

/-- Display (P2): the actual local restriction is `D_i = D intersect 2^(X_i) = {S subset X_i :
|S| <= a_i}`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "D_i=D intersect 2^(X_i)={S subset X_i: |S|<=a_i}.     (P2)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis.
Does not claim: the identity (statement only). -/
def LocalRestriction {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ (i : Fin F.b) (S : Finset X), S ⊆ block F i →
    (IsGood F S ↔ S.card ≤ F.a i)

/-- Section 3: `I_k(D_i) = {S : |S| <= a_i k}` for subsets `S` of the block `X_i`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    I_k(D_i)={S:|S|<=a_i k}."
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis.
Does not claim: the identity (statement only). -/
def LocalDecomposability {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ (i : Fin F.b) (k : ℕ) (S : Finset X), S ⊆ block F i →
    (IsKDecomposable F k S ↔ S.card ≤ F.a i * k)

/-- Display (P3): for `k = d_i` the only obstructed subset of `X_i` is `X_i` itself, so
`G_i = {X_i}` is an exact local obstruction cover.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    G_i={X_i}                                           (P3)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis.
Does not claim: the statement (statement only). -/
def LocalObstructionIsFullBlock {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    Prop :=
  IsRealizedFamily F → ∀ (i : Fin F.b) (S : Finset X), S ⊆ block F i →
    (¬ IsKDecomposable F (F.d i) S ↔ S = block F i)

/-- Display (P4): for `0 <= c_v <= p_v <= 1`, `price(G_i) = prod_{v in X_i} c_v <= prod p_v
<= q_i <= phi(q_i)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "       =P(U_i=X_i)<=q_i<=phi(q_i),                      (P4)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis; `p`, `c` with the displayed ranges.
Does not claim: the chain (statement only); the identity `prod p_v = P(U_i = X_i)` is the definition of the weight. -/
def LocalPriceChain {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ (p c : X → ℝ), (∀ v, 0 ≤ p v ∧ p v ≤ 1) → (∀ v, 0 ≤ c v ∧ c v ≤ p v) →
    ∀ i : Fin F.b, genPrice c (block F i) ≤ ∏ v ∈ block F i, p v ∧
      ∏ v ∈ block F i, p v ≤ localFailure F p i ∧
      localFailure F p i ≤ phi (localFailure F p i)

/-- Display (P5) (also (T6)): `mu_p(D) <= prod_i (1 - q_i)` and, when `mu_p(D) > 0`,
`sum_i q_i <= -log prod_i (1 - q_i) <= -log mu_p(D)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "        <=-log(product_i(1-q_i))<=-log mu_p(D).           (P5)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis; `p in [0,1]^X`.
Does not claim: the inequalities (statement only); independence of crossing events ("Crossing events are not declared independent"). -/
def IndependentBlockHazard {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ p : X → ℝ, (∀ v, 0 ≤ p v ∧ p v ≤ 1) →
    mu F p ≤ ∏ i, (1 - localFailure F p i) ∧
    (0 < mu F p →
      ∑ i, localFailure F p i ≤ -Real.log (∏ i, (1 - localFailure F p i)) ∧
      -Real.log (∏ i, (1 - localFailure F p i)) ≤ -Real.log (mu F p))

/-! ## Realized covers: the full-block cover criterion (P7) and the hazard bound (P8) -/

/-- Proposition P2, display (P7): the fixed family `G = {X_1, ..., X_b}` covers `O_K(D)` if and
only if `K >= K_H(d)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    K>=K_H(d).                                          (P7)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis; `K_H(d)` is `paletteNumber`.
Does not claim: the criterion (statement only); optimality of any other cover ("It does not prove that no different cover with the same budget works at fewer colors"); unrestricted prize optimality; disposition transcribed from LANDING_CLAIMS for p15-realized-covers: HOLD_WITH_DOMAIN; nonauthor review and acceptance are not claimed here. -/
def FullBlockCoverIff {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ K : ℕ,
    IsGeneratorCover F K (fullBlockFamily F) ↔ paletteNumber F.b F.H F.d ≤ K

/-- Display (P8): for `0 <= c_v <= p_v <= 1` and `K >= K_H(d)`,
`covercost_c(O_K(D)) <= min(1, -log mu_p(D))`, with the endpoint convention of `cappedHazard`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "covercost_c(O_K(D)) <= min(1,-log mu_p(D)).            (P8)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis; `p in [0,1]^X`; prices in the range `0 <= c_v <= p_v` only ("No assertion is made here for the additional prices p_v<c_v<=phi(p_v)").
Does not claim: the bound (statement only); anything for `c_v > p_v`; the unrestricted P15 prize; original P15-B; disposition transcribed from LANDING_CLAIMS for p15-realized-covers: HOLD_WITH_DOMAIN; nonauthor review and acceptance are not claimed here. -/
def CoverCostHazardBound {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ (p c : X → ℝ), (∀ v, 0 ≤ p v ∧ p v ≤ 1) → (∀ v, 0 ≤ c v ∧ c v ≤ p v) →
    ∀ K : ℕ, paletteNumber F.b F.H F.d ≤ K → coverCost F K c ≤ cappedHazard 1 (mu F p)

/-- Section 4 comparison: the chromatic number of the ENTIRE ground set `X` (the least `K` with
`X in I_K(D)`) is `K_H(d+1)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "the chromatic number of the ENTIRE ground set X is K_H(d+1), because ceil(n_i/a_i)=d_i+1."
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis.
Does not claim: the statement (statement only). -/
def WholeGroundChromaticNumber {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    Prop :=
  IsRealizedFamily F → ∀ K : ℕ,
    IsKDecomposable F K Finset.univ ↔ paletteNumber F.b F.H (fun i => F.d i + 1) ≤ K

/-! ## Realized covers Section 5: the elementary palette formula (P9) -/

/-- The clutter of all `(s+1)`-subsets of `b` block indices.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "A fully elementary case is H consisting of all (s+1)-subsets of b indices, 1<=s<=b."
Interface: none.
Does not claim: anything; for `s = b` the family is empty ("Empty H is allowed"). -/
def completeClutter (b s : ℕ) : Finset (Finset (Fin b)) :=
  (Finset.univ : Finset (Fin b)).powersetCard (s + 1)

/-- Display (P9): for `H` all `(s+1)`-subsets of `b` indices, `1 <= s <= b`, and positive
demands, `K_H(d) = max(max_i d_i, ceil(sum_i d_i / s))`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    K_H(d)=max(max_i d_i, ceil(sum_i d_i/s)).             (P9)"
Interface: none; the positivity of the demands is the sources' standing assumption ("positive integers a_i,d_i"), added as a hypothesis.
Does not claim: the formula (statement only); the matroid palette formula of Section 5 (not specified). -/
def ElementaryPaletteFormula : Prop :=
  ∀ (b s : ℕ) (d : Fin b → ℕ), 1 ≤ s → s ≤ b → (∀ i, 1 ≤ d i) →
    paletteNumber b (completeClutter b s) d =
      max (Finset.univ.sup d) ⌈((∑ i, d i : ℕ) : ℚ) / (s : ℚ)⌉₊

/-- Section 5: the triangle on three block indices (all `2`-subsets) at unit demands needs `3`
colors.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "at unit demands needs3 colors, although a formula considering only size-two edge loads would give2."
Interface: none.
Does not claim: the value (statement only; finite and decidable in principle). -/
def TriangleNeedsThree : Prop :=
  paletteNumber 3 (completeClutter 3 1) (fun _ => 1) = 3

/-! ## Realized covers Section 6: the six-block 816-label realization -/

/-- The six-block realization: ground type `Fin 6 × Fin 409` (six blocks of `409` coordinates,
`2454` in all), block map the first projection, `a_i = 1`, `d_i = 408`, and `H` all `15`
four-block subsets.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "Take six blocks, a_i=1 and d_i=408. Thus EACH original block has409 vertices and the original ground set has2454 vertices. H consists of all15 four-block subsets."
Interface: none; a concrete data record.
Does not claim: that it satisfies `IsRealizedFamily` (that is `Benchmark816IsRealized`); that any earlier six-block application had these coordinates ("it is not a statement that every earlier six-block application had409 vertices per block"). -/
def benchmark816 : RealizedData (Fin 6 × Fin 409) where
  b := 6
  blk := Prod.fst
  a := fun _ => 1
  d := fun _ => 408
  H := completeClutter 6 3

/-- The six-block data satisfy the side conditions of a realized family.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "Every D-color class contains at most one vertex per block and occupies at most three blocks."
Interface: none.
Does not claim: the statement (finite, decidable in principle; not proved here). -/
def Benchmark816IsRealized : Prop :=
  IsRealizedFamily benchmark816

/-- Section 6: `K_H(408, ..., 408) = 816` for `H` all four-block subsets of six.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    K_H(408,...,408)=816."
Interface: none.
Does not claim: the value (statement only); the arithmetic `max(408, ceil(2448/3)) = 816` is a core-Lean target on main, not here. -/
def Benchmark816Palette : Prop :=
  paletteNumber 6 (completeClutter 6 3) (fun _ => 408) = 816

/-- Section 6: the six full blocks cover the obstruction at `816` colors and not at `815`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "At816 colors it covers the obstruction; at815 it does not, because omitting one vertex from each block leaves2448 vertices, each color holds at most3, and815*3<2448."
Interface: none.
Does not claim: the statement (finite over `2^2454` subsets, not decidable in practice); `815*3 < 2448` is a core-Lean target on main. -/
def Benchmark816Cover : Prop :=
  IsGeneratorCover benchmark816 816 (fullBlockFamily benchmark816) ∧
    ¬ IsGeneratorCover benchmark816 815 (fullBlockFamily benchmark816)

/-- Section 6: the whole ground set needs `818` colors (`2454/3 = 818`): it is `818`-decomposable
and not `817`-decomposable.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "The whole ground set, unlike the cover-avoiding sets, needs818 colors, since2454/3=818."
Interface: none.
Does not claim: the statement (statement only); `2454/3 = 818` is a core-Lean target on main. -/
def Benchmark816WholeGround : Prop :=
  IsKDecomposable benchmark816 818 Finset.univ ∧ ¬ IsKDecomposable benchmark816 817 Finset.univ

/-- Display (P10): the minimal forbidden sets of the six-block realization are
`6 * binom(409, 2)` internal pairs (support of size one) and `15 * 409^4` crossing transversal
quadruples (support of size four), and nothing else.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    15*409^4=419743994415 crossing transversal quadruples. (P10)"
Interface: none.
Does not claim: the counts (statement only); the evaluations `500616` and `419743994415` are core-Lean targets on main; the source stresses these are "not a claim that the program enumerated more than four hundred billion sets". -/
def Benchmark816MinimalForbiddenCounts : Prop :=
  Set.ncard {U : Finset (Fin 6 × Fin 409) |
      IsMinimalForbidden benchmark816 U ∧ (supp benchmark816 U).card = 1} = 6 * Nat.choose 409 2 ∧
  Set.ncard {U : Finset (Fin 6 × Fin 409) |
      IsMinimalForbidden benchmark816 U ∧ (supp benchmark816 U).card = 4} = 15 * 409 ^ 4 ∧
  Set.ncard {U : Finset (Fin 6 × Fin 409) | IsMinimalForbidden benchmark816 U} =
    6 * Nat.choose 409 2 + 15 * 409 ^ 4

/-- Section 6: at `p_v = c_v = 1/40900` the cover price of the six full blocks is exactly
`6/(40900^409) > 0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "    6/(40900^409)>0."
Interface: none.
Does not claim: the identity (statement only; the six blocks are distinct members of the image). -/
def Benchmark816Price : Prop :=
  famPrice (fun _ => (1 : ℝ) / 40900) (fullBlockFamily benchmark816) = 6 / 40900 ^ 409 ∧
    (0 : ℝ) < 6 / 40900 ^ 409

/-- Display (P11): at `p_v = 1/40900`, `Q = 1 - mu_p(D) <= 500616/40900^2 + 15/100^4 =
2449227/8180000000 < 3/10000`, and the good probability exceeds `0.9997`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md sha256 c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9
Anchor: "       =2449227/8180000000 < 3/10000.                    (P11)"
Interface: none.
Does not claim: the bound (statement only; "only a union bound, not an equality of the global bad probability"); the two rational identities are core-Lean targets on main. -/
def Benchmark816UnionBound : Prop :=
  1 - mu benchmark816 (fun _ => (1 : ℝ) / 40900) ≤ 500616 / 40900 ^ 2 + 15 / 100 ^ 4 ∧
  (500616 / 40900 ^ 2 + 15 / 100 ^ 4 : ℝ) = 2449227 / 8180000000 ∧
  (2449227 / 8180000000 : ℝ) < 3 / 10000 ∧
  (9997 / 10000 : ℝ) < mu benchmark816 (fun _ => (1 : ℝ) / 40900)

/-! ## Price boundary: the exact two-coordinate refutation at demand one -/

/-- The smallest one-block member: `X = {x, y}`, `a_1 = d_1 = 1`, no edges.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "The prescribed cover is {{x,y}}."
Interface: none; a concrete data record.
Does not claim: anything. -/
def twoCoord : RealizedData (Fin 2) where
  b := 1
  blk := fun _ => 0
  a := fun _ => 1
  d := fun _ => 1
  H := ∅

/-- The probabilities `p_x = p_y = 1/2` of the refutation.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "Take independent original probabilities p_x=p_y=1/2 and prices c_x=c_y=2/3."
Interface: none.
Does not claim: anything. -/
noncomputable def twoCoordP : Fin 2 → ℝ := fun _ => 1 / 2

/-- The prices `c_x = c_y = 2/3` of the refutation.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "These are admissible TRANSFORMED prices: phi(1/2)=log 2>2/3."
Interface: none.
Does not claim: admissibility (that is part of `DemandOneRefutation`). -/
noncomputable def twoCoordC : Fin 2 → ℝ := fun _ => 2 / 3

/-- The four generator costs `1, 2/3, 2/3, 4/9` of the empty set, the two singletons and `X`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "The possible generator costs are1,2/3,2/3,4/9 for the empty"
Interface: none.
Does not claim: the values (statement only; as integers over `9` they are core-Lean targets on main). -/
def TwoCoordGeneratorPrices : Prop :=
  genPrice twoCoordC ∅ = 1 ∧ genPrice twoCoordC {0} = 2 / 3 ∧ genPrice twoCoordC {1} = 2 / 3 ∧
    genPrice twoCoordC Finset.univ = 4 / 9

/-- The exact refutation: the data are a realized family with `K_H(d) = 1` and one-piece
obstruction `{X}`; the prices are admissible transformed prices (`phi(1/2) = log 2 > 2/3`);
`mu_p(D) = 3/4`, `min(1, -log mu_p(D)) = log(4/3) < 1/3`; every cover costs at least `4/9`,
`covercost_c(O_1(D)) = 4/9 > 1/3 > log(4/3)`; hence the capped hazard is strictly below the
exact cover cost at the same palette.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "    covercost_c(O_1(D))=4/9 > 1/3 > log(4/3)."
Interface: none; everything is concrete.
Does not claim: the statement (statement only; the finite parts are decidable in principle, the two logarithm inequalities are analysis); that this contradicts original P15-B ("It does not contradict the original P15-B amalgamation"); anything about the `408`-scale or the unrestricted prize; the integral representation `log 2 = 2 integral_0^(1/3) 1/(1-t^2) dt` is not specified; disposition transcribed from LANDING_CLAIMS for p15-price-boundary: EXACT_COUNTEREXAMPLE; nonauthor review and acceptance are not claimed here. -/
def DemandOneRefutation : Prop :=
  IsRealizedFamily twoCoord ∧
  paletteNumber 1 ∅ (fun _ => 1) = 1 ∧
  (∀ U : Finset (Fin 2), U ∈ obstruction twoCoord 1 ↔ U = Finset.univ) ∧
  (∀ v, twoCoordC v ≤ phi (twoCoordP v)) ∧
  phi (1 / 2) = Real.log 2 ∧ (2 / 3 : ℝ) < Real.log 2 ∧
  mu twoCoord twoCoordP = 3 / 4 ∧
  cappedHazard 1 (mu twoCoord twoCoordP) = Real.log (4 / 3) ∧ Real.log (4 / 3) < (1 / 3 : ℝ) ∧
  (∀ G : Finset (Finset (Fin 2)), IsGeneratorCover twoCoord 1 G → 4 / 9 ≤ famPrice twoCoordC G) ∧
  coverCost twoCoord 1 twoCoordC = 4 / 9 ∧ (1 / 3 : ℝ) < 4 / 9 ∧
  cappedHazard 1 (mu twoCoord twoCoordP) < coverCost twoCoord 1 twoCoordC

/-- The elementary inequality `log(1 + x) < x` for `x > 0` used by the refutation.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b
Anchor: "using log(1+x)<x for x>0."
Interface: none.
Does not claim: the inequality (statement only; the source proves it, this module does not). -/
def LogOnePlusBelowIdentity : Prop :=
  ∀ x : ℝ, 0 < x → Real.log (1 + x) < x

/-! ## Price budget: the restricted transformed-price extension (T1)-(T5), (T7) -/

/-- Display (T2): `-log(1 - p) <= p/(1 - p)` for `0 <= p < 1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "    -log(1-p)=integral_0^p dt/(1-t) <= p/(1-p).               (T2)"
Interface: none.
Does not claim: the inequality (statement only); the integral representation is not specified. -/
def LogRatioBound : Prop :=
  ∀ p : ℝ, 0 ≤ p → p < 1 → -Real.log (1 - p) ≤ p / (1 - p)

/-- The exponent arithmetic of display (T3): for `a >= 1` and `n >= 2a + 1`,
`4^(a+1)/3^n <= (4/3)(4/9)^a <= 16/27`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "       <= (4/3)*(4/9)^a <=16/27.                          (T3)"
Interface: none.
Does not claim: the inequalities (statement only); the preceding ratio steps of (T3) (price over `prod_{v in S} p_v`) are stated through their consequence `RestrictedLocalBudget`; the case `a = 1, n = 3` giving `16/27` is a core-Lean target on main. -/
def RatioExponentBound : Prop :=
  ∀ a n : ℕ, 1 ≤ a → 2 * a + 1 ≤ n →
    (4 : ℝ) ^ (a + 1) / 3 ^ n ≤ (4 / 3) * (4 / 9) ^ a ∧ (4 / 3 : ℝ) * (4 / 9) ^ a ≤ 16 / 27

/-- Display (T4): under `a_i >= 1`, `d_i >= 2`, `0 <= p_v <= 1/4` and `0 <= c_v <= phi(p_v)`, the
full-block generator satisfies `price({X_i}) <= (16/27) q_i <= phi(q_i)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "    price({X}) <= (16/27)q <= phi(q).                     (T4)"
Interface: the family data `F` with `IsRealizedFamily F` and `d_i >= 2` as hypotheses.
Does not claim: the bound (statement only); the source notes "The extra demand hypothesis d>=2 is load-bearing" and that `p_v <= 1/4` "is a sufficient range, not an asserted sharp threshold". -/
def RestrictedLocalBudget {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → (∀ i, 2 ≤ F.d i) →
  ∀ (p c : X → ℝ), (∀ v, 0 ≤ p v ∧ p v ≤ 1 / 4) → (∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) →
    ∀ i : Fin F.b, genPrice c (block F i) ≤ (16 / 27) * localFailure F p i ∧
      (16 / 27) * localFailure F p i ≤ phi (localFailure F p i)

/-- The restricted transformed-price extension, display (T1): for every realized family with
`a_i >= 1`, `d_i >= 2`, independent `0 <= p_v <= 1/4`, EVERY `0 <= c_v <= phi(p_v)` and the SAME
`K >= K_H(d)`, `covercost_c(O_K(D)) <= min(1, (16/27)[-log mu_p(D)])`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "    covercost_c(O_K(D)) <= min(1,(16/27)*[-log mu_p(D)]).       (T1)"
Interface: the family data `F` with `IsRealizedFamily F` and `d_i >= 2` as hypotheses; `K_H(d)` is `paletteNumber`; the zero-probability endpoint follows the `cappedHazard` convention.
Does not claim: the bound (statement only); any retraction of the demand-one refutation; the unrestricted prize; new classical coloring theory; disposition transcribed from LANDING_CLAIMS for p15-price-budget-restricted: HOLD_WITH_DOMAIN; nonauthor review ("nonauthor review open") and acceptance are not claimed here. -/
def RestrictedTransformedPriceBudget {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) : Prop :=
  IsRealizedFamily F → (∀ i, 2 ≤ F.d i) →
  ∀ (p c : X → ℝ), (∀ v, 0 ≤ p v ∧ p v ≤ 1 / 4) → (∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) →
    ∀ K : ℕ, paletteNumber F.b F.H F.d ≤ K →
      coverCost F K c ≤ cappedHazard (16 / 27) (mu F p)

/-- The sufficient ratio `rho(a, d, p_*) = p_*^(ad - a) / (1 - p_*)^(ad + 1)` of display (T5).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "    rho(a,d,p_*)=p_*^(ad-a)/(1-p_*)^(ad+1).                (T5)"
Interface: none; natural-number subtraction `a*d - a` agrees with the source for `d >= 1`.
Does not claim: anything. -/
noncomputable def rhoTest (a d : ℕ) (pBound : ℝ) : ℝ :=
  pBound ^ (a * d - a) / (1 - pBound) ^ (a * d + 1)

/-- Display (T5) as a sufficient test, block by block: if a common upper bound `p_* < 1` has
`0 <= p_v <= p_*` for every coordinate `v` of the block `X_i` and `rho(a_i, d_i, p_*) <= 1`, then
for all `0 <= c_v <= phi(p_v)` the full-block price is at most `q_i`. The source: "This is only a
sufficient test; rho>1 is not a counterexample and must not be labeled infeasible."

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "If rho<=1 the same argument proves price({X})<=q for that block."
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis; the bound `p_*` is imposed only on the coordinates of the block under test (the source: "for any common upper bound p_*<1", "for that block"), the price bound `0 <= c_v <= phi(p_v)` on every coordinate as in (T1).
Does not claim: the statement (statement only); anything when `rho > 1` (the source: a block with `rho > 1` "must not be labeled infeasible"). -/
def RhoTestSufficient {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ (p c : X → ℝ) (pBound : ℝ), 0 ≤ pBound → pBound < 1 →
    (∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) →
    ∀ i : Fin F.b, (∀ v ∈ block F i, 0 ≤ p v ∧ p v ≤ pBound) →
      rhoTest (F.a i) (F.d i) pBound ≤ 1 →
      genPrice c (block F i) ≤ localFailure F p i

/-- Section 5 of the price budget: the cap test `q >= 2/3` gives `phi(q) = 1`, since `log 3 > 1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "or if q>=2/3 (then phi(q)=1, since log3>1)"
Interface: none.
Does not claim: the statement (statement only); the integral proof of `log 3 > 1` is not specified; the checker program of that section is not specified. -/
def CapTestAtTwoThirds : Prop :=
  1 < Real.log 3 ∧ ∀ q : ℝ, 2 / 3 ≤ q → q ≤ 1 → phi q = 1

/-- Section 4 of the price budget: for the six-block realization at `p_v = 1/40900`, the SAME
`816`-label full-block cover is valid for the FULL transformed range
`0 <= c_v <= log(40900/40899)` (which is `phi(1/40900)`): it covers `O_816(D)` and its price meets
the (T1) budget `min(1, (16/27)[-log mu_p(D)])`; consequently so does the optimal cover cost.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "    0<=c_v<=log(40900/40899),"
Interface: none; the specified cover is `fullBlockFamily benchmark816`; the last conjunct is the (T1) consequence for the infimum over all covers.
Does not claim: the statement (statement only); "No large witness enumeration" is involved, nor any claim about earlier P15 inputs. -/
def Benchmark816TransformedRange : Prop :=
  phi (1 / 40900) = Real.log (40900 / 40899) ∧
  IsGeneratorCover benchmark816 816 (fullBlockFamily benchmark816) ∧
  ∀ c : Fin 6 × Fin 409 → ℝ, (∀ v, 0 ≤ c v ∧ c v ≤ Real.log (40900 / 40899)) →
    famPrice c (fullBlockFamily benchmark816) ≤
        cappedHazard (16 / 27) (mu benchmark816 (fun _ => (1 : ℝ) / 40900)) ∧
      coverCost benchmark816 816 c ≤
        cappedHazard (16 / 27) (mu benchmark816 (fun _ => (1 : ℝ) / 40900))

/-- Display (T7) and the explicit rational price of Section 4: at maximal equal prices the union
price is `6 log(40900/40899)^409 <= 6/(40899^409)`; `c = p + p^2/2 = 81801/3345620000` for
`p = 1/40900` is admissible (`c < phi(p)`) and its cover price is exactly `6 c^409`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/price_budget_20260924/PROOF.md sha256 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535
Anchor: "    6*log(40900/40899)^409 <=6/(40899^409).                (T7)"
Interface: none.
Does not claim: the statements (statement only); the rational identity `1/40900 + (1/40900)^2/2 = 81801/3345620000` is exact arithmetic not re-proved here; admissibility rests on `integral_0^p 1/(1-t) dt > integral_0^p (1+t) dt`, which is not specified. -/
def Benchmark816RationalPrice : Prop :=
  6 * Real.log (40900 / 40899) ^ 409 ≤ (6 : ℝ) / 40899 ^ 409 ∧
  (1 / 40900 : ℝ) + (1 / 40900) ^ 2 / 2 = 81801 / 3345620000 ∧
  (81801 / 3345620000 : ℝ) < phi (1 / 40900) ∧
  famPrice (fun _ => (81801 : ℝ) / 3345620000) (fullBlockFamily benchmark816) =
    6 * (81801 / 3345620000) ^ 409

/-! ## Full price: constants (F1), the result (F2) and the enclosure (F3) -/

/-- `p_star = 1 - exp(-1)`, display (F1).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    p_star=1-exp(-1),"
Interface: none.
Does not claim: anything. -/
noncomputable def pStar : ℝ := 1 - Real.exp (-1)

/-- `h_star = 3 - log(3e - 2)`, display (F1); ASCII `3e-2` of the source means `3*e - 2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    h_star=3-log(3e-2),"
Interface: none.
Does not claim: `h_star > 1` (that is `HStarExceedsOne`). -/
noncomputable def hStar : ℝ := 3 - Real.log (3 * Real.exp 1 - 2)

/-- `rho_star = 1/h_star = 1/(3 - log(3e - 2))`, display (F1); written out so that it is
definitionally `1 / hStar`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    rho_star=1/h_star.                                   (F1)"
Interface: none.
Does not claim: the enclosure (F3) or `rho_star < 6/7` (those are `RhoStarEnclosure`, `RhoStarBelowSixSevenths`). -/
noncomputable def rhoStar : ℝ := 1 / (3 - Real.log (3 * Real.exp 1 - 2))

/-- Lower endpoint `0.84547981724898672067` of display (F3), as a rational over `10^20`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    0.84547981724898672067 < rho_star"
Interface: none.
Does not claim: anything. -/
noncomputable def rhoStarLower : ℝ := 84547981724898672067 / 10 ^ 20

/-- Upper endpoint `0.84547981724898672068` of display (F3), as a rational over `10^20`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "                            < 0.84547981724898672068 < 6/7. (F3)"
Interface: none.
Does not claim: anything; adjacency of the endpoints and `upper * 7 < 6 * 10^20` are core-Lean targets on main. -/
noncomputable def rhoStarUpper : ℝ := 84547981724898672068 / 10 ^ 20

/-- The headline result F, display (F2): for every realized family with `a_i >= 1`, `d_i >= 2`,
EVERY independent `p in [0,1]^X`, every `0 <= c_v <= phi(p_v)` and the SAME `K >= K_H(d)`,
`covercost_c(O_K(D)) <= min(1, rho_star [-log mu_p(D)])`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    covercost_c(O_K(D)) <= min(1,rho_star[-log mu_p(D)]).  (F2)"
Interface: the family data `F` with `IsRealizedFamily F` and `d_i >= 2` as hypotheses; `K_H(d)` is `paletteNumber`; the zero-probability endpoints follow the `cappedHazard` convention ("the price-one cap proves the assertion").
Does not claim: the bound (statement only); optimality of `rho_star` (that is `SharpnessOfRhoStar`); any unrestricted downset or prize statement (the source: "palette size remains K_H(d), which can depend on H and d"); the nonauthor review recorded in LANDING_CLAIMS (REVIEWED_SCOPED) is not reproduced or extended here. -/
def FullTransformedPriceBudget {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    Prop :=
  IsRealizedFamily F → (∀ i, 2 ≤ F.d i) →
  ∀ (p c : X → ℝ), (∀ v, 0 ≤ p v ∧ p v ≤ 1) → (∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) →
    ∀ K : ℕ, paletteNumber F.b F.H F.d ≤ K → coverCost F K c ≤ cappedHazard rhoStar (mu F p)

/-- Display (F3): `0.84547981724898672067 < rho_star < 0.84547981724898672068 < 6/7`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    0.84547981724898672067 < rho_star"
Interface: none.
Does not claim: the enclosure (statement only; the source's `full_price.py` Fraction procedure is not a kernel proof and is not specified); this is the next full-result target named in main's `docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md`, still at `specified`. -/
def RhoStarEnclosure : Prop :=
  rhoStarLower < rhoStar ∧ rhoStar < rhoStarUpper ∧ rhoStarUpper < 6 / 7

/-- The simpler rational factor: `rho_star < 6/7`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "Thus 6/7 is a simpler valid rational factor."
Interface: none.
Does not claim: the inequality (statement only); `16/27 < 6/7` is a core-Lean target on main. -/
def RhoStarBelowSixSevenths : Prop :=
  rhoStar < 6 / 7

/-! ## Full price Section 2: the hazard interpolation bounds (F4)-(F8) -/

/-- `A` is a proper decreasing family containing the empty set: `empty in A`, `X notin A`, and
`A` is closed under taking subsets.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "Let A be any proper decreasing family on n>=1 coordinates containing the empty set."
Interface: none.
Does not claim: anything. -/
def IsProperDownset {Y : Type} [Fintype Y] (A : Finset (Finset Y)) : Prop :=
  ∅ ∈ A ∧ Finset.univ ∉ A ∧ ∀ U ∈ A, ∀ V : Finset Y, V ⊆ U → V ∈ A

/-- `F_A(t) = -log mu_(p(t))(A)` with `p_i(t) = 1 - exp(-t_i)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    p_i(t)=1-exp(-t_i),     t in [0,1]^n,"
Interface: none.
Does not claim: anything. -/
noncomputable def hazardCoord {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    (t : Y → ℝ) : ℝ :=
  -Real.log (muOf A (fun i => 1 - Real.exp (-t i)))

/-- `H_A = F_A(1, ..., 1) = -log mu_(p_star, ..., p_star)(A)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    H_A=F_A(1,...,1)>0."
Interface: none.
Does not claim: positivity (that is `HazardStarPositive`). -/
noncomputable def hazardStar {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y)) : ℝ :=
  -Real.log (muOf A (fun _ => pStar))

/-- `H_A > 0` for every proper decreasing family on `n >= 1` coordinates containing the empty set.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "The probability is positive on this compact cube because the empty set is admitted and every p_i(t)<1."
Interface: `A` with `IsProperDownset A` as hypothesis.
Does not claim: the statement (statement only). -/
def HazardStarPositive : Prop :=
  ∀ (n : ℕ) (A : Finset (Finset (Fin n))), 1 ≤ n → IsProperDownset A → 0 < hazardStar A

/-- Display (F5) as separate coordinatewise concavity of `F_A` on `[0,1]` in each coordinate,
the other coordinates held fixed in `[0,1]`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "Only SEPARATE coordinatewise concavity is asserted; joint concavity is not needed or claimed."
Interface: `A` with `IsProperDownset A` as hypothesis.
Does not claim: the statement (statement only); joint concavity; the derivative formulas (F4)-(F5) themselves (not specified; concavity is their consequence). -/
def CoordinatewiseConcavity : Prop :=
  ∀ (n : ℕ) (A : Finset (Finset (Fin n))), 1 ≤ n → IsProperDownset A →
    ∀ (i : Fin n) (t : Fin n → ℝ), (∀ j, 0 ≤ t j ∧ t j ≤ 1) →
      ConcaveOn ℝ (Set.Icc (0 : ℝ) 1) (fun s : ℝ => hazardCoord A (Function.update t i s))

/-- Display (F6): `F_A(t) >= (prod_i t_i) H_A` on the cube `[0,1]^n`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    F_A(t) >= (product_i t_i) H_A.                       (F6)"
Interface: `A` with `IsProperDownset A` as hypothesis.
Does not claim: the inequality (statement only). -/
def HazardProductBound : Prop :=
  ∀ (n : ℕ) (A : Finset (Finset (Fin n))), 1 ≤ n → IsProperDownset A →
    ∀ t : Fin n → ℝ, (∀ j, 0 ≤ t j ∧ t j ≤ 1) → (∏ i, t i) * hazardStar A ≤ hazardCoord A t

/-- Display (F7): for `p in [0,1]^n` with `mu_p(A) > 0`, `-log mu_p(A) >= H_A prod_i phi(p_i)`;
when `mu_p(A) = 0` the source reads the inequality as an extended one and nothing is stated.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    -log mu_p(A) >= H_A product_i phi(p_i).              (F7)"
Interface: `A` with `IsProperDownset A` as hypothesis.
Does not claim: the inequality (statement only). -/
def HazardPhiBound : Prop :=
  ∀ (n : ℕ) (A : Finset (Finset (Fin n))), 1 ≤ n → IsProperDownset A →
    ∀ p : Fin n → ℝ, (∀ j, 0 ≤ p j ∧ p j ≤ 1) → 0 < muOf A p →
      hazardStar A * ∏ i, phi (p i) ≤ -Real.log (muOf A p)

/-- Display (F8): for transformed prices `0 <= c_i <= phi(p_i)`,
`prod_i c_i <= min(1, [-log mu_p(A)]/H_A)`, with the `cappedHazard` endpoint convention.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    product_i c_i <= min(1, [-log mu_p(A)]/H_A).          (F8)"
Interface: `A` with `IsProperDownset A` as hypothesis.
Does not claim: the inequality (statement only); that the full-ground generator covers any obstruction ("It does not assert that this generator alone covers an arbitrary obstruction"). -/
def FullGroundPriceBound : Prop :=
  ∀ (n : ℕ) (A : Finset (Finset (Fin n))), 1 ≤ n → IsProperDownset A →
    ∀ (p c : Fin n → ℝ), (∀ j, 0 ≤ p j ∧ p j ≤ 1) → (∀ j, 0 ≤ c j ∧ c j ≤ phi (p j)) →
      ∏ i, c i ≤ cappedHazard (1 / hazardStar A) (muOf A p)

/-- `phi(p_star) = 1`: at the reference vector `p_i = p_star`, `c_i = 1`, the two sides of (F7)
are equal.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "at every p_i=p_star and every c_i=1, the two sides of (F7) are equal"
Interface: none.
Does not claim: the identity (statement only). -/
def PhiAtPStar : Prop :=
  phi pStar = 1

/-- Exactness of `1/H_A`: a full-ground generator universally meets the capped hazard budget
(for all `p in [0,1]^n` and all transformed prices) precisely when `H_A >= 1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "a full-ground generator universally meets the capped hazard budget precisely when H_A>=1"
Interface: `A` with `IsProperDownset A` as hypothesis.
Does not claim: the equivalence (statement only); coverage of arbitrary `A` ("it does not infer coverage of arbitrary A"). -/
def FullGroundBudgetIff : Prop :=
  ∀ (n : ℕ) (A : Finset (Finset (Fin n))), 1 ≤ n → IsProperDownset A →
    ((∀ (p c : Fin n → ℝ), (∀ j, 0 ≤ p j ∧ p j ≤ 1) → (∀ j, 0 ≤ c j ∧ c j ≤ phi (p j)) →
        ∏ i, c i ≤ cappedHazard 1 (muOf A p)) ↔ 1 ≤ hazardStar A)

/-! ## Full price Section 3: capacity families and the sharp worst case (F9)-(F11) -/

/-- The capacity family `{S subset {0, ..., n-1} : |S| <= a}`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "For a capacity restriction A={S:|S|<=a} on n coordinates,"
Interface: none.
Does not claim: anything. -/
def capacityFamily (n a : ℕ) : Finset (Finset (Fin n)) :=
  (Finset.univ : Finset (Fin n)).powerset.filter (fun U => U.card ≤ a)

/-- The binomial lower tail `P(Bin(n, p) <= a) = sum_{j <= a} C(n, j) p^j (1 - p)^(n - j)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    H_(n,a)=-log P(Bin(n,p_star)<=a).                    (F9)"
Interface: none.
Does not claim: anything. -/
noncomputable def binomTail (n a : ℕ) (p : ℝ) : ℝ :=
  ∑ j ∈ Finset.range (a + 1), (Nat.choose n j : ℝ) * p ^ j * (1 - p) ^ (n - j)

/-- Display (F9): `H_(n,a) = -log P(Bin(n, p_star) <= a)`, i.e. the product measure of the
capacity family at `p_star` is the binomial tail.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    H_(n,a)=-log P(Bin(n,p_star)<=a).                    (F9)"
Interface: none.
Does not claim: the identity (statement only; a finite identity for each `n`, general in `n`). -/
def CapacityHazardIsBinomial : Prop :=
  ∀ n a : ℕ, hazardStar (capacityFamily n a) = -Real.log (binomTail n a pStar)

/-- Display (F10): for `S ~ Bin(2a+1, p)`,
`P(Bin(2a+3, p) <= a+1) - P(S <= a) = (1-p)^2 P(S = a+1) - p^2 P(S = a) = p(1-2p) P(S = a)`, and
this is negative for `1/2 < p < 1` (the source writes `p > 1/2`; at `p = 1` the difference is
`0`, so the strict inequality is stated for `p < 1`, which covers `p_star`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "       =p(1-2p)P(S=a)<0.                                (F10)"
Interface: none.
Does not claim: the identities (statement only; polynomial identities in `p`, a future kernel target for each fixed `a` and by induction in general). -/
def OddMajorityIdentity : Prop :=
  ∀ (a : ℕ) (p : ℝ),
    binomTail (2 * a + 3) (a + 1) p - binomTail (2 * a + 1) a p =
      (1 - p) ^ 2 * ((Nat.choose (2 * a + 1) (a + 1) : ℝ) * p ^ (a + 1) * (1 - p) ^ a) -
        p ^ 2 * ((Nat.choose (2 * a + 1) a : ℝ) * p ^ a * (1 - p) ^ (a + 1)) ∧
    binomTail (2 * a + 3) (a + 1) p - binomTail (2 * a + 1) a p =
      p * (1 - 2 * p) * ((Nat.choose (2 * a + 1) a : ℝ) * p ^ a * (1 - p) ^ (a + 1)) ∧
    (1 / 2 < p → p < 1 → binomTail (2 * a + 3) (a + 1) p - binomTail (2 * a + 1) a p < 0)

/-- Display (F11): for `a >= 1` and `n >= 2a + 1`, `H_(n,a) >= H_(3,1) = -log[3exp(-2) - 2exp(-3)]
= 3 - log(3e - 2) = h_star > 1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "       =3-log(3e-2)=h_star>1.                           (F11)"
Interface: none.
Does not claim: the statements (statement only); the source stresses "it is not a finite-grid extrapolation". -/
def CapacityWorstCase : Prop :=
  (∀ a n : ℕ, 1 ≤ a → 2 * a + 1 ≤ n →
      hazardStar (capacityFamily 3 1) ≤ hazardStar (capacityFamily n a)) ∧
  hazardStar (capacityFamily 3 1) = -Real.log (3 * Real.exp (-2) - 2 * Real.exp (-3)) ∧
  -Real.log (3 * Real.exp (-2) - 2 * Real.exp (-3)) = hStar ∧
  1 < hStar

/-- The last step of (F11): `e > 2`, `e^2 - (3e - 2) = (e - 1)(e - 2) > 0`, hence
`3exp(-2) - 2exp(-3) < exp(-1)` and `h_star > 1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    e^2-(3e-2)=(e-1)(e-2)>0,"
Interface: none.
Does not claim: the statements (statement only). -/
def HStarExceedsOne : Prop :=
  2 < Real.exp 1 ∧
  Real.exp 1 ^ 2 - (3 * Real.exp 1 - 2) = (Real.exp 1 - 1) * (Real.exp 1 - 2) ∧
  0 < (Real.exp 1 - 1) * (Real.exp 1 - 2) ∧
  3 * Real.exp (-2) - 2 * Real.exp (-3) < Real.exp (-1) ∧
  1 < hStar

/-! ## Full price Sections 3-4: local price bound (F12) and global assembly (F13) -/

/-- Display (F12): for each block with `d_i >= 2` and positive local good probability, under
`0 <= c_v <= phi(p_v)`, `price({X_i}) <= rho_star [-log mu_p(D_i)]`, and the price is at most
one.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    price({X_i}) <= rho_star [-log mu_p(D_i)],            (F12)"
Interface: the family data `F` with `IsRealizedFamily F` and `d_i >= 2` as hypotheses.
Does not claim: the bound (statement only); `price <= q_i` ("The inequality price<=q_i is NOT required and is in general stronger than what is proved here"). -/
def LocalPriceBoundFull {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → (∀ i, 2 ≤ F.d i) →
  ∀ (p c : X → ℝ), (∀ v, 0 ≤ p v ∧ p v ≤ 1) → (∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) →
    ∀ i : Fin F.b, 0 < localGood F p i →
      genPrice c (block F i) ≤ rhoStar * (-Real.log (localGood F p i)) ∧
      genPrice c (block F i) ≤ 1

/-- Display (F13): `mu_p(D) <= prod_i mu_p(D_i)` and, when `mu_p(D) > 0`,
`sum_i [-log mu_p(D_i)] <= -log mu_p(D)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    sum_i [-log mu_p(D_i)] <= -log mu_p(D).               (F13)"
Interface: the family data `F` with `IsRealizedFamily F` as hypothesis; `p in [0,1]^X`.
Does not claim: the inequalities (statement only; "without independence of crossing failures"). -/
def GlobalAssemblyHazard {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) : Prop :=
  IsRealizedFamily F → ∀ p : X → ℝ, (∀ v, 0 ≤ p v ∧ p v ≤ 1) →
    mu F p ≤ ∏ i, localGood F p i ∧
    (0 < mu F p → ∑ i, -Real.log (localGood F p i) ≤ -Real.log (mu F p))

/-! ## Full price Section 5: sharpness and the exact demand boundary -/

/-- One block with `a = 1`, `d = 2`, `n = 3`, no edges.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "Every proper subset is two-decomposable and the full triple is not."
Interface: none; a concrete data record.
Does not claim: anything. -/
def tripleBlock : RealizedData (Fin 3) where
  b := 1
  blk := fun _ => 0
  a := fun _ => 1
  d := fun _ => 2
  H := ∅

/-- Sharpness of `rho_star` even among alternative covers: for the triple block at `K = 2`, the
obstruction is exactly `{X}`; at `p_i = p_star`, `c_i = 1` (admissible since `phi(p_star) = 1`)
every generator has price one, every cover of the obstruction costs at least one and the
one-generator full-block cover `{X}` attains it (so the exact cover cost is one), the hazard is
`h_star`, and replacing `rho_star` by any smaller constant makes the right side of (F2) strictly
less than one.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "Replacing rho_star by any smaller constant in (F2) would make its right side strictly less than one."
Interface: none; everything is concrete.
Does not claim: the statements (statement only; the finite parts are decidable in principle); a necessity claim for multi-block families with some demand one (the source: "other constraints may change its budget"). -/
def SharpnessOfRhoStar : Prop :=
  IsRealizedFamily tripleBlock ∧
  paletteNumber 1 ∅ (fun _ => 2) = 2 ∧
  (∀ U : Finset (Fin 3), U ∈ obstruction tripleBlock 2 ↔ U = Finset.univ) ∧
  phi pStar = 1 ∧
  (∀ g : Finset (Fin 3), genPrice (fun _ => (1 : ℝ)) g = 1) ∧
  (∀ G : Finset (Finset (Fin 3)), IsGeneratorCover tripleBlock 2 G →
    1 ≤ famPrice (fun _ => (1 : ℝ)) G) ∧
  IsGeneratorCover tripleBlock 2 (fullBlockFamily tripleBlock) ∧
  famPrice (fun _ => (1 : ℝ)) (fullBlockFamily tripleBlock) = 1 ∧
  coverCost tripleBlock 2 (fun _ => (1 : ℝ)) = 1 ∧
  -Real.log (mu tripleBlock (fun _ => pStar)) = hStar ∧
  ∀ ρ : ℝ, ρ < rhoStar → cappedHazard ρ (mu tripleBlock (fun _ => pStar)) < 1

/-- One block with capacity `cap`, demand `1`, `n = cap + 1`, no edges.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "If d=1, n=a+1 and K=1, the sole obstruction is again the full block."
Interface: none; a concrete data record for each `cap`.
Does not claim: anything. -/
def demandOneBlock (cap : ℕ) : RealizedData (Fin (cap + 1)) where
  b := 1
  blk := fun _ => 0
  a := fun _ => cap
  d := fun _ => 1
  H := ∅

/-- The exact demand boundary: for every `a >= 1`, in the demand-one block at `K = 1` the sole
obstruction is the full block; at `p_i = p_star`, `c_i = 1` every cover costs at least one, but
`mu_p(D) = 1 - p_star^(a+1) > 1 - p_star = exp(-1)`, so the hazard is less than one and no
constant `rho <= 1` makes the same-palette transformed-price guarantee hold.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "    mu_p(D)=1-p_star^(a+1)>1-p_star=exp(-1),"
Interface: none; concrete for each `cap`.
Does not claim: the statements (statement only); any retraction of original P15-B ("whose local-budget premise is missing in that case"). -/
def DemandOneBoundary : Prop :=
  ∀ cap : ℕ, 1 ≤ cap →
    IsRealizedFamily (demandOneBlock cap) ∧
    (∀ U : Finset (Fin (cap + 1)), U ∈ obstruction (demandOneBlock cap) 1 ↔ U = Finset.univ) ∧
    1 ≤ coverCost (demandOneBlock cap) 1 (fun _ => (1 : ℝ)) ∧
    mu (demandOneBlock cap) (fun _ => pStar) = 1 - pStar ^ (cap + 1) ∧
    Real.exp (-1) < 1 - pStar ^ (cap + 1) ∧
    -Real.log (mu (demandOneBlock cap) (fun _ => pStar)) < 1 ∧
    ∀ ρ : ℝ, ρ ≤ 1 →
      cappedHazard ρ (mu (demandOneBlock cap) (fun _ => pStar)) <
        coverCost (demandOneBlock cap) 1 (fun _ => (1 : ℝ))

/-! ## Full price Section 6: rational certificates for `rho_star < 6/7` -/

/-- The short rational certificate chain of Section 6: `e < 31967/11760 < 87/32`,
`87/32 - 31967/11760 = 11/23520`, `sum_{j=0}^5 (11/6)^j/j! - 197/32 = 26081/933120 > 0`,
hence `3e - 2 < 197/32 < exp(11/6)`, `h_star > 7/6` and `rho_star < 6/7`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path frontiers/full_price_20260924/PROOF.md sha256 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9
Anchor: "Therefore 3e-2<197/32<exp(11/6), giving h_star>7/6 and rho_star<6/7."
Interface: none.
Does not claim: the statements (statement only); the two exact rational identities are exact arithmetic not re-proved here; the series and geometric-tail bounds behind `e < 31967/11760` are not specified. -/
def SixSeventhsCertificate : Prop :=
  Real.exp 1 < 31967 / 11760 ∧ (31967 / 11760 : ℝ) < 87 / 32 ∧
  (87 / 32 - 31967 / 11760 : ℝ) = 11 / 23520 ∧
  (∑ j ∈ Finset.range 6, (11 / 6 : ℝ) ^ j / (Nat.factorial j : ℝ)) - 197 / 32 = 26081 / 933120 ∧
  (0 : ℝ) < 26081 / 933120 ∧
  3 * Real.exp 1 - 2 < 197 / 32 ∧ (197 / 32 : ℝ) < Real.exp (11 / 6) ∧
  7 / 6 < hStar ∧ rhoStar < 6 / 7

end UniversalLaw.Spec.P15
