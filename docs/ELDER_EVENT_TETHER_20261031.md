# Tethering the pair-law probability to the actual elder event

**Object:** ELDER-EVENT-TETHER-01CZ-20261031. A statement-level specification proposal and a
dependency inventory. It contains no proof of any theorem, no compiled Lean, and no status
change. Credit 0 · scientific effect NONE · OBL OPEN.

**Why this note exists.** The owner's 31 October assessment puts first: define the actual
measurable elder-rule pair event, and connect every abstract estimate back to it. The Lean
statement package already shows where that connection is missing. This note states the missing
connection precisely, lists what in the repositories already supplies each part, and names the
one part that nothing supplies.

All Math- references are at Math- `main` `19a55750451b8016b50d1d20be6a18c021807ecb`, with Git
blob ids. All main references are at main `4712fc13`.

## 1. The gap, as the sources themselves record it

`specifications/lean_statements_20261010/Specifications/Lifetime.lean` (blob `f2508f7f`)
defines

```lean
structure PairLawInterface (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) where
  p : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → ℝ
  p_nonneg : ∀ r b k R, 0 ≤ p r b k R
  p_le_one : ∀ r b k R, p r b k R ≤ 1
```

Its own docstring says: "no field ties `p` to them" (the field `G` and the elder relation `I`),
and it does not claim "that `p` is the probability of any event under any measure declared
here". The selection statement `ParentUniformSelectionA` (Theorem A shape, `0 ≤ 1-p ≤ C r³`) is
a predicate on this free `p`.

Math-#543 (`cbz/lifetime_pair_law/CounterexampleE4.lean`, blob `3b7712e4`, head `9e89980b`)
makes that consequence formal: the constant `p ≡ 1` inhabits `PairLawInterface`, satisfies the
upper predicate, and contradicts the positive two-sided E4 lower rate at a positive radius.
GPT-6's formal review there (review 5707632836) reads it correctly: it is not a counterexample to
the geometric elder probability. It shows the interface does not mention that probability.

By contrast, the density statements `ParentCompactWindowDensityB` and
`ParentUnrestrictedLeadingDensityC` already consume the actual relation, through
`IsWindowElderPair … I` and `IsFiniteBar … I`. So the formal chain has an actual-event end
(Theorems B and C) and a free-number end (Theorem A). **Nothing formal links them.** In the
informal sources, the link is the selected-pair measure of `PERSISTENCE_BRIDGE.md` §4
(main, blob `509adbcc`).

## 2. The actual event (transcribed, not redefined)

From `PERSISTENCE_BRIDGE.md` §1, for fixed separation `r>0`, birth `b`, mark `k>0` and
frame `R` (direction `u` its first vector), with `M = pinM r u`, `S = pinS r u`:

```text
Y(f)  = (f(M), f(S), ∇f(M), ∇f(S))           pin data, in ℝ^(2+2d)
y*    = (b, b − k r³, 0, 0)                  prescribed pins
Q     = the canonical Gaussian regression law of f given Y = y*
W(f)  = |det H_f(M) det H_f(S)| · 1{H_f(M) < 0, index H_f(S) = d − 1}
Z     = E_Q W
e(f)  = 1{S is the actual ordinary elder death of M}
p     = E_Q[W e] / Z
```

Two features matter below. `Q` is a law on whole fields, because `e` depends on the whole
field, not on finitely many jets. And `Q` is evaluated at the single point `y*`, which has
probability zero, so it needs a canonical version, not just an almost-everywhere one.
`PERSISTENCE_BRIDGE.md` §4.1 supplies one, display (12): the law of
`f − C Σ⁻¹ G + C Σ⁻¹ z`, defined for every `z`.

## 3. The proposed tether: five conditions

A `PairLawInterface` inhabitant is *tethered* when, for every admissible `(r,b,k,R)`:

| | Condition | Why it is indispensable |
|---|---|---|
| T1 | `p r b k R = (∫ W·e dQ) / (∫ W dQ)`, with `e` the indicator of `I.ElderPartner f M S` | Without it `p` is free; #543 applies. |
| T2 | `Q` is the regression law of `G`'s own field given `Y = y*`, in the explicit form of (12), so the canonical version is pinned at the prescribed point | See §4: T1 alone does not exclude the constant-one model. |
| T3 | `e` is measurable for the σ-algebra on which `Q` lives | Otherwise `∫ W e dQ` is undefined. The source argument is the countable cover in `PERSISTENCE_BRIDGE.md` §3.2 and `AXIAL_FIRST_MOMENT.md` (A7)–(A8). |
| T4 | `0 < ∫ W dQ < ∞` | The quotient must be defined. The quantitative form `Z ≥ z_* r²` (Appendix B) is a separate, stronger input used only by the rate. |
| T5 | `I` is the actual ordinary superlevel elder relation on the Morse distinct-value locus | Otherwise `e` is the indicator of some other relation. |

The UNCOMPILED Lean shape, for discussion only, is:

```lean
-- UNCOMPILED PROPOSAL. Not part of any package. Names are placeholders.
structure TetheredPairLaw (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) extends PairLawInterface d L G I where
  Q        : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Measure (E d → ℝ)
  Q_is_regression : ∀ r b k R, IsCanonicalPinnedRegression G (Q r b k R) r b k R   -- T2
  e_meas   : ∀ r b k R, Measurable (elderIndicator I r R)                          -- T3
  Z_pos    : ∀ r b k R, 0 < ∫ f, typedWeight d r R f ∂(Q r b k R)                  -- T4
  p_eq     : ∀ r b k R, p r b k R =
               (∫ f, typedWeight d r R f * elderIndicator I r R f ∂(Q r b k R)) /
               (∫ f, typedWeight d r R f ∂(Q r b k R))                             -- T1
```

`IsCanonicalPinnedRegression`, `typedWeight` and `elderIndicator` do not exist. Writing them is
the work. T5 is a property of `I`, not of this structure.

## 4. Why T2 cannot be dropped (elementary remark)

Suppose T1, T3 and T4 hold, but `Q` is any probability law on fields with the prescribed pins.
Fix `(r,b,k,R)`, and suppose some smooth periodic `g` has the prescribed pin values, `W(g) > 0`
and `e(g) = 1`. The cap construction of `PERSISTENCE_BRIDGE.md` §2 is designed to produce such
functions; this remark only needs one to exist. Put `Q = δ_g`. Then `Z = W(g) > 0` and
`p = W(g)·1 / W(g) = 1`. Choosing such a `g` for each parameter gives `p ≡ 1` on the admissible
set again, which is exactly the #543 inhabitant. So **T1 without T2 still admits the #543
countermodel**. T2 is the weakest condition in the table that rules it out.

This is the answer, at the interface level, to "find the weakest indispensable lemma": it is
**the identification of `Q` with the canonical whole-field regression law of the actual field**.
The Kac–Rice and selection estimates are stated relative to it.

## 5. The negative control that a tethered interface must pass

Under T1–T5, `PERSISTENCE_BRIDGE.md` §5 display (16) states `Q^W{e = 0 and N_r = 1} > 0` for
each fixed pinned law in its example. If that proof holds, then `p < 1` strictly at those
parameters, so the constant-one inhabitant is **impossible** for a tethered interface. Any
future formal tether should carry a control showing that a `p ≡ 1` inhabitant fails, and that
dropping `Q_is_regression` makes it succeed again. That would be the #543 proof re-run against
the weakened structure.

(16) also gives no rate: "the neighborhood, determinant bound and probability can all shrink
with r". So the positive lower rate in E4 is **not** supplied by (16). It is a separate
analytic statement that must consume the actual `e` under the actual `Q`.

## 6. Dependency inventory: what already exists for each condition

| Needed | Existing artifact (Math- `19a55750` unless noted) | What it covers | What it does not |
|---|---|---|---|
| Field `G` | `frontiers/source_lifetime_realization_20261017/` (C291) `actualGaussianFieldInterface` | Realizes `GaussianFieldInterface d L` from `smoothLaw d L`, for `L > 0` | Morse genericity, critical-point measures, elder pairing (its README says so) |
| Relation `I` (T5) | `companions/torus_filtration_20261011/SuperlevelBarField.lean` (blob `f0300a6b`) | Inhabits the **Field** spec's `SuperlevelBarInterface` from the C263 definitions | Explicitly not the **Lifetime** spec's interface; under the C263/C264 FORMAL HOLD |
| Relation `I`, geometry | `frontiers/compact_morse_elder_adapter_20261018/` (C292), conventional proof | One ordinary H₀ interval per supplied nonessential maximum | No Lean; unique partner, measurability and exhaustive once-counting are separate |
| Relation `I`, attainment | Math-#487 (open draft), `frontiers/critical_level_attainment_20261020/` | Conditional attainment at the actual death height | Local path connectivity of the closed death superlevel is an explicit hypothesis (OPEN) |
| `Q`, finite-dimensional | `frontiers/determinant_moment_20261025/ResearchFormalCoreR1/GaussianRegressionKernel.lean` (blob `face5de8`) `gaussianRegression_hasCondDistrib`, `gaussianRegressionKernel_spec` | Canonical Gaussian conditional law, with the literal pushforward at **every** observation, for `Fin k → ℝ` residuals | Only finite-dimensional residuals |
| `Q`, source endpoints | `frontiers/source_derivative_gaussian_20261025/SourceEndpointEstimates.lean` (blob `5beba8ed`) `sourcePhysicalEndpointKernel_eventually_hasCondDistrib`, `sourcePhysicalEndpointWeightedProbability` | For `d = 2`, the conditional law of the 6-vector endpoint data given the confluent pins, eventually in small `r`, and its `W`-weighted version | The law of endpoint jets, not of the whole field, so it cannot evaluate `e` |
| `Q`, whole field (T2) | **Update, 1 November 2026:** main `experiments/whole_field_regression_kernel_20261101/` (Lean 4.34.1, Mathlib `d13f23b7`, standard axioms only). As of 31 October there was none in Lean. Informal source: `PERSISTENCE_BRIDGE.md` §4.1 display (12) | The field regression kernel, defined at every pin value, is a conditional law under pin/residual independence, and finite-restriction independence implies whole-field independence (product σ-algebra) | The actual Gaussian field's independence (only `Fin k` in Math-), a smooth-function Borel σ-algebra, continuity in the pins |
| `e` measurable (T3) | **None in Lean.** Informal: `PERSISTENCE_BRIDGE.md` §3.2, §4.2; `AXIAL_FIRST_MOMENT.md` (A7)–(A8) (blob `4cc3ef7d`) | — | — |
| `Z > 0` (T4) | Informal: Appendix B (`Z ≥ z_* r²` floor); the Lean weighted measure `prwMeasure` (`PalmReweighting.lean`) | — | Whether `prwMeasure` requires or proves a positive normalizer was not audited for this note |
| A ↔ B link | Informal: `PERSISTENCE_BRIDGE.md` §4.3, the selected-pair measure with one determinant product | — | No formal statement exists in either direction |

## 7. The shortest honest path, in order

1. **The whole-field regression kernel in Lean.** Generalize the `regressionReconstruction`
   pattern from `Fin k → ℝ` residuals to a field-valued residual, the infinite-dimensional
   form of (12), with the literal pushforward at every pin value. This is the only node in §6
   with neither a formal nor a source-verified Lean predecessor. It is also exactly T2.
   *Update, 1 November 2026:* the structural part is now formalized (see the §6 row). Two
   things remain open: the actual field's pin/residual independence beyond `Fin k`
   coordinates, and a smooth-function σ-algebra.
2. **Measurability of `e` on the Morse distinct-value locus** (T3), following the countable
   cover in §3.2 of the bridge.
3. **State `TetheredPairLaw`** and re-run the #543 construction against it and against its
   T2-weakened variant as the negative-control pair of §5.
4. Only then, **state the A ↔ B link** (§4.3 of the bridge) as a Prop shape relating tethered
   `p` to `IsWindowElderPair` densities.

Steps 1–3 are formalization of statements whose informal proofs exist in the sources. Step 4 is
the place where a proof of the actual theorem would start. None of the four is done.

## What this note does not establish

- It proves no theorem. The §4 remark is an elementary observation about interfaces. It is
  conditional on the existence of one cap-type function, which is not proved here.
- It compiles nothing. The Lean block in §3 is an UNCOMPILED proposal with placeholder names.
- It does not evaluate, review or accept `PERSISTENCE_BRIDGE.md`, `AXIAL_FIRST_MOMENT.md`, (12),
  (16), C291, C292, #487, #543 or any Lean module. Where it relies on them it says so.
- It does not establish a positive lower rate, an upper rate, a coefficient or any parameter
  range. The regional shrinking-witness node stays open.
- It assigns no status. Words such as "missing" or "none in Lean" describe search results at
  the stated commits. The search was a `git grep` over `*.lean` outside `inputs/` for
  regression, Palm, `condDistrib` and disintegration, and it may have missed differently-named
  work.
- It is Anthropic/Claude work for Dylan Roy at zero organizational-independence credit, with no
  human review.

Credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN.
