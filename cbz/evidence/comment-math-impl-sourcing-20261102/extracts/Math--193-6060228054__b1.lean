import GlobalJointFlow
import Mathlib.Analysis.InnerProductSpace.PiL2

noncomputable section
open ActualJointFlow Set
open scoped NNReal
namespace JointFlowTests

private theorem constant_exact {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] (c : E) : ∃ Φ : E × ℝ → E, Continuous Φ ∧
      (∀ x t, Φ (x,t) = x + t • c) ∧ (∀ x s t, Φ (x,s+t) = Φ (Φ (x,s),t)) := by
  obtain ⟨Φ,hc,h0,hd,ha⟩ := exists_global_joint_flow (fun _ : E => c) 0 ‖c‖₊
    (LipschitzWith.const c) (fun _ => le_rfl)
  refine ⟨Φ,hc,?_,ha⟩
  intro x t
  have hlin : ∀ r : ℝ, HasDerivAt (fun s : ℝ => x + s • c) c r := by
    intro r
    simpa only [id_eq,one_smul] using ((hasDerivAt_id r).smul_const c).const_add x
  have he := global_solution_unique (fun _ : E => c) 0 (LipschitzWith.const c)
    (fun r => Φ (x,r)) (fun r => x+r•c) (hd x) hlin (by simpa using h0 x)
  exact congrFun he t

-- Actual constructor outputs are identified by ODE uniqueness, including NEGATIVE time.
example : ∃ Φ : ℝ × ℝ → ℝ, Continuous Φ ∧ ∀ x t, Φ (x,t) = x := by
  obtain ⟨Φ,hc,he,ha⟩ := constant_exact (0:ℝ)
  exact ⟨Φ,hc,by simpa using he⟩
example : ∃ Φ : ℝ × ℝ → ℝ, Continuous Φ ∧ Φ (3,-2) = 1 := by
  obtain ⟨Φ,hc,he,ha⟩ := constant_exact (1:ℝ)
  exact ⟨Φ,hc,by have h := he 3 (-2); norm_num at h; exact h⟩
example : ∃ Φ : ℝ × ℝ → ℝ, Continuous Φ ∧ Φ (3,2) = 5 := by
  obtain ⟨Φ,hc,he,ha⟩ := constant_exact (1:ℝ)
  exact ⟨Φ,hc,by have h := he 3 2; norm_num at h; exact h⟩
example (c x : EuclideanSpace ℝ (Fin 2)) : ∃ Φ : EuclideanSpace ℝ (Fin 2) × ℝ → EuclideanSpace ℝ (Fin 2), Continuous Φ ∧ Φ (x,-1) = x-c := by
  obtain ⟨Φ,hc,he,ha⟩ := constant_exact c
  exact ⟨Φ,hc,by simpa [sub_eq_add_neg] using he x (-1)⟩
example (c : ℝ) : ∃ Φ : ℝ × ℝ → ℝ, Continuous Φ ∧
    ∀ x s t, Φ (x,s+t) = Φ (Φ (x,s),t) := by
  obtain ⟨Φ,hc,he,ha⟩ := constant_exact c
  exact ⟨Φ,hc,ha⟩
example (c : ℝ) : ∃ Φ : ℝ × ℝ → ℝ, Continuous Φ ∧
    ∀ x t, Φ (Φ (x,t),-t) = x := by
  obtain ⟨Φ,hc,he,ha⟩ := constant_exact c
  refine ⟨Φ,hc,?_⟩
  intro x t
  rw [he,he]
  simp
example {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (V : E → E) (K B : ℝ≥0) (hL : LipschitzWith K V) (hB : ∀ x, ‖V x‖ ≤ B)
    (x : E) (hx : V x = 0) : ∃ Φ : E × ℝ → E, Continuous Φ ∧ ∀ t, Φ (x,t) = x := by
  obtain ⟨Φ,hc,h0,hd,ha⟩ := exists_global_joint_flow V K B hL hB
  refine ⟨Φ,hc,?_⟩
  have he := global_solution_unique V K hL (fun t => Φ (x,t)) (fun _ => x)
    (hd x) (fun t => by simpa only [hx] using hasDerivAt_const t x) (h0 x)
  exact fun t => congrFun he t
example (f g : ℝ → ℝ) (hf : ∀ t, HasDerivAt f 0 t)
    (hg : ∀ t, HasDerivAt g 0 t) (h : f 0 = g 0) : f = g :=
  global_solution_unique (fun _ => 0) 0 (LipschitzWith.const 0) f g hf hg h

-- W0 local model: z'=-z, f=-z². These are explicit scalar controls, not a
-- claim that the unbounded linear field satisfies the constructor's globalB hypothesis.
example (y t : ℝ) : HasDerivAt (fun s => y * Real.exp (-s))
    (-(y * Real.exp (-t))) t := by
  simpa using ((Real.hasDerivAt_exp (-t)).comp t (hasDerivAt_neg t)).const_mul y
example (y : ℝ) (hy : y≠0) : ∀ t : ℝ, -(y * Real.exp (-t))^2 < 0 := by
  intro t
  exact neg_neg_of_pos (sq_pos_of_ne_zero (mul_ne_zero hy (Real.exp_ne_zero _)))
example (y : ℝ) (hy : y≠0) : ¬ ∃ t : ℝ, -(y * Real.exp (-t))^2 = 0 := by
  rintro ⟨t,ht⟩
  have hp := sq_pos_of_ne_zero (mul_ne_zero hy (Real.exp_ne_zero (-t)))
  linarith

-- W1 local model: y'=y, f=y². For every proposed nonnegative horizonT,
-- an actual positive initial point remains below target1 for ALL0≤t≤T.
example (y t : ℝ) : HasDerivAt (fun s => y * Real.exp s) (y * Real.exp t) t :=
  (Real.hasDerivAt_exp t).const_mul y
example (T : ℝ) (hT : 0≤T) : ∃ y : ℝ, 0<y ∧ y<1 ∧
    ∀ t ∈ Icc 0 T, (y * Real.exp t)^2 < 1 := by
  let y := Real.exp (-T)/2
  have hy : 0<y := by dsimp [y]; positivity
  have hb : Real.exp (-T) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
  have hmul : y * Real.exp T = 1/2 := by
    dsimp [y]
    rw [div_mul_eq_mul_div,← Real.exp_add]
    simp
  refine ⟨y,hy,by dsimp [y]; linarith,?_⟩
  intro t ht
  have hn : 0 < y * Real.exp t := mul_pos hy (Real.exp_pos _)
  have hl : y * Real.exp t ≤ 1/2 := by
    rw [← hmul]
    exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ht.2) hy.le
  nlinarith
end JointFlowTests

