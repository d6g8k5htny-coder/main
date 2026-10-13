## Compile fixes at the pin (Lean v4.34.1, Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`)

The C254 handoff delivered both modules as NOT_COMPILED. At the pin they did not elaborate; the successor bytes in
this companion differ from the handoff bytes only as listed here. No theorem or lemma statement changed (all 41
statement lines are byte-identical before `:=`); the 21 + 20 theorem names and the 11 + 6 definition names keep their
order; no `sorry`, `axiom`, `native_decide` or `set_option` was added; every declaration's axioms are within
`propext`, `Classical.choice`, `Quot.sound`.

`TorusTopology.lean` (handoff sha256 `efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4`):
- `def torus_periodBasis`: declared type `Basis (Fin d) ℝ (torus_E d)` → `Module.Basis (Fin d) ℝ (torus_E d)`. At the
  pin the structure is `Module.Basis` (`Mathlib/LinearAlgebra/Basis/Defs.lean`, `namespace Module` line 76, `structure
  Basis` line 90); bare `Basis` was an unknown identifier that `autoImplicit` turned into a variable, the root of the
  first-run cascade. Same object; the body is unchanged.
- `theorem torus_basis_sum_eq_latticeVec`, proof: `Basis.unitsSMul_apply` → `Module.Basis.unitsSMul_apply` in both
  `simp` calls (`Mathlib/LinearAlgebra/Basis/SMul.lean` line 107); three unused simp arguments removed
  (`PiLp.single_apply` twice, `hij`) so the file elaborates with warnings as errors.
- `theorem torus_proj_isCoveringMap`, proof term: the discreteness witness `DiscreteTopology.isDiscrete` (which needs an
  instance on the set-coercion subtype that instance search did not bridge) → `SetLike.isDiscrete_iff_discreteTopology.mpr
  (torus_lattice_addSubgroup_discrete d L hL)` (`Mathlib/Topology/Constructions.lean` line 268), passed to the same
  lemma `AddSubgroup.isAddQuotientCoveringMap_of_comm`.
- `theorem torus_pathConnectedSpace`, proof: `infer_instance` → `exact inferInstanceAs <| PathConnectedSpace (Quotient _)`
  (the pattern Mathlib uses for `AddCircle`; `infer_instance` does not unfold `HasQuotient.Quotient` to reach
  `Quotient.instPathConnectedSpace`).
- Module docstring: two lines recording the above; `namespace ResearchFormalCoreR1` → `namespace TorusFiltration`
  (companion namespace; no qualified self-reference existed).

`ClosedSuperlevel.lean` (handoff sha256 `1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e`):
- `def superlevel_componentMap`: the second argument of `ZerothHomotopy.lift` is `∀ ⦃x y : X⦄, Path x y → f x = f y`
  (`Mathlib/Topology/Connected/PathConnected.lean` lines 153-157); strict-implicit binders are not introduced by `fun`,
  so `fun p => …` bound `p` to a point. Fixed to `fun _ _ p => ZerothHomotopy.sound (p.map f.continuous)`; the witness
  and the defined map are the same.
- Module docstring: three lines recording the above; `namespace ResearchFormalCoreR1` → `namespace TorusFiltration`.

Evidence of the fixes themselves is local (`lean -DwarningAsError=true --root=<companion>` inside the pinned `formal/`
Lake environment, exit 0 for both modules, `#print axioms` of all 58 declarations within the three allowed axioms): a
record of a computation on an unverified worktree, not a receipt. The receipt is the hosted companion run.
