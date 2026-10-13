import ResearchFormalCoreR1
import Lean
open Lean

def _inv.kind : ConstantInfo → String
  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque"
  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"

structure _inv.St where
  visited : NameSet := {}
  axioms : NameSet := {}

partial def _inv.collect (env : Environment) (c : Name) : StateM _inv.St Unit := do
  unless (← get).visited.contains c do
    modify fun s => { s with visited := s.visited.insert c }
    let collectExpr (e : Expr) : StateM _inv.St Unit := e.getUsedConstants.forM (_inv.collect env)
    match env.find? c with
    | some (.axiomInfo v)  => modify (fun s => { s with axioms := s.axioms.insert c }) *> collectExpr v.type
    | some (.defnInfo v)   => collectExpr v.type *> collectExpr v.value
    | some (.thmInfo v)    => collectExpr v.type *> collectExpr v.value
    | some (.opaqueInfo v) => collectExpr v.type *> collectExpr v.value
    | some (.quotInfo _)   => pure ()
    | some (.ctorInfo v)   => collectExpr v.type
    | some (.recInfo v)    => collectExpr v.type
    | some (.inductInfo v) => collectExpr v.type *> v.ctors.forM (_inv.collect env)
    | none                 => pure ()

def _inv.main2 (env : Environment) (names : List Name) : IO Unit := do
  let (_, s) := (names.forM (fun x => _inv.collect env x)).run {}
  let axs := (s.axioms.toArray.qsort (fun a b => a.toString < b.toString)).toList.map toString
  IO.println s!"AXIOMS|[{", ".intercalate axs}]"
  IO.println s!"CLOSURE|{s.visited.size}"

#eval show CoreM Unit from do _inv.main2 (← getEnv) [`ResearchFormalCoreR1.ec005_fold_gap, `ResearchFormalCoreR1.ec008_factor_expansion, `ResearchFormalCoreR1.ec010_generic, `ResearchFormalCoreR1.ec010_transverse, `ResearchFormalCoreR1.ec011_scalar_cancellation, `ResearchFormalCoreR1.ec014_contact_power, `ResearchFormalCoreR1.p02_lm008_r2_cancel, `ResearchFormalCoreR1.p02_lm008_cross_multiplied, `ResearchFormalCoreR1.p02_lm008_quotient_bound, `ResearchFormalCoreR1.p02_lm009_bad_event_threshold, `ResearchFormalCoreR1.p02_lm009_power40_identity, `ResearchFormalCoreR1.p02_lm009_r4_le_r3, `ResearchFormalCoreR1.p02_lm009_palm_r4_to_r3, `ResearchFormalCoreR1.p02_lm008_integral_cs, `ResearchFormalCoreR1.p02_lm008_event_cs, `ResearchFormalCoreR1.p02_lm008_event_numerator, `ResearchFormalCoreR1.p02_lm008_measure_transfer, `ResearchFormalCoreR1.p02_lm009_measure_transfer_r8_to_r3, `ResearchFormalCoreR1.p02_lm008_sqrt_counterexample, `ResearchFormalCoreR1.p02_lm008_upper_normalizer_counterexample, `ResearchFormalCoreR1.weightedLaw_apply, `ResearchFormalCoreR1.weightedLaw_isProbabilityMeasure, `ResearchFormalCoreR1.weightedLaw_event_real, `ResearchFormalCoreR1.weightedLaw_absolutelyContinuous, `ResearchFormalCoreR1.weightedLaw_congr_ae, `ResearchFormalCoreR1.weightedLaw_zero, `ResearchFormalCoreR1.weightedLaw_one, `ResearchFormalCoreR1.p02_lm008_probability_transfer, `ResearchFormalCoreR1.p02_lm009_probability_transfer_r8_to_r3, `ResearchFormalCoreR1.p02_lm009_markov_event, `ResearchFormalCoreR1.p02_lm009_badJet_measurable, `ResearchFormalCoreR1.p02_lm009_badJet_subset, `ResearchFormalCoreR1.p02_lm009_moment40_tail, `ResearchFormalCoreR1.p02_lm009_moment40_weighted_r4, `ResearchFormalCoreR1.p02_lm009_moment40_weighted_r3, `ResearchFormalCoreR1.p02_lm009_moment40_family_r3, `ResearchFormalCoreR1.p02_lm009_moment40_tail_finite, `ResearchFormalCoreR1.p02_lm009_moment40_tail_of_finite, `ResearchFormalCoreR1.p02_lm009_moment40_family_r3_interval, `ResearchFormalCoreR1.p02_lm009_moment40_family_r3_of_interval, `ResearchFormalCoreR1.weightPerturbation_integral, `ResearchFormalCoreR1.weightPerturbation_setIntegral, `ResearchFormalCoreR1.weightPerturbation_normalizer_lower, `ResearchFormalCoreR1.weightPerturbation_event_integral_bounds, `ResearchFormalCoreR1.weightPerturbation_quotient_bound, `ResearchFormalCoreR1.weightedLaw_event_perturbation, `ResearchFormalCoreR1.weightedLaw_event_perturbation_r2]
