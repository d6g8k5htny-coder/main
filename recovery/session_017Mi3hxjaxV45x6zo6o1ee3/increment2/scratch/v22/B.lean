import CapFirstExit

open Set unitInterval Filter Topology

namespace CapFirstExit

section Slice

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

example (f : ℝ × E → ℝ) (p : ℝ × E) (a b c : ℝ × E) :
    (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]).comp
      (fderiv ℝ (iteratedFDeriv ℝ 2 f) p) c = iteratedFDeriv ℝ 3 f p ![c, a, b] := rfl

example (f : ℝ × E → ℝ) (p : ℝ × E) (a b c d : ℝ × E) :
    (iteratedFDeriv ℝ 4 f p).curryLeft d ![c, a, b] = iteratedFDeriv ℝ 4 f p ![d, c, a, b] := rfl

example (f : ℝ × E → ℝ) (U : Set (ℝ × E)) (hU : IsOpen U) (hf : ContDiffOn ℝ 4 f U) (p : ℝ × E)
    (hp : p ∈ U) : DifferentiableAt ℝ (iteratedFDeriv ℝ 3 f) p :=
  (hf.contDiffAt (hU.mem_nhds hp)).differentiableAt_iteratedFDeriv (by norm_num)

example (f : ℝ × E → ℝ) (U : Set (ℝ × E)) (hU : IsOpen U) (hf : ContDiffOn ℝ 4 f U) :
    ContDiffOn ℝ 3 (fderiv ℝ f) U := hf.fderiv_of_isOpen hU (by norm_num)

example (f : ℝ × E → ℝ) (U : Set (ℝ × E)) (hU : IsOpen U) (hf : ContDiffOn ℝ 4 f U) :
    ContinuousOn (iteratedFDeriv ℝ 4 f) U :=
  (hf.continuousOn_iteratedFDerivWithin le_rfl hU.uniqueDiffOn).congr
    (fun x hx => (iteratedFDerivWithin_of_isOpen 4 hU hx).symm)

example : ((ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] ℝ) :=
  ContinuousLinearMap.precomp ℝ (ContinuousLinearMap.inr ℝ ℝ E)

example : ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] E →L[ℝ] ℝ) :=
  (ContinuousLinearMap.postcomp E (ContinuousLinearMap.precomp ℝ (ContinuousLinearMap.inr ℝ ℝ E))).comp
    (ContinuousLinearMap.precomp ((ℝ × E) →L[ℝ] ℝ) (ContinuousLinearMap.inr ℝ ℝ E))

end Slice

end CapFirstExit
