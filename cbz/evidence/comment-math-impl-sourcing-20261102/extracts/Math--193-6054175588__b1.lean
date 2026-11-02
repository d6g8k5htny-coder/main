import SourceContactRank

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Jets I4ContactRank
  MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace ContactRankTests

-- A real shear: transverse vector has an axial component, so the frame is not orthogonal.
private def shear : (Fin 2 → ℝ) ≃L[ℝ] (Fin 2 → ℝ) where
  toFun x := ![x 0 + x 1, x 1]
  invFun x := ![x 0 - x 1, x 1]
  left_inv x := by ext i; fin_cases i <;> simp
  right_inv x := by ext i; fin_cases i <;> simp
  map_add' x y := by ext i; fin_cases i <;> simp; ring
  map_smul' c x := by ext i; fin_cases i <;> simp; ring
  continuous_toFun := by fun_prop
  continuous_invFun := by fun_prop

-- The nonorthogonal geometry is computed, not assumed.
example : frameVector 1 shear 0 = ![1,0] := by ext i; fin_cases i <;> simp [frameVector, shear]
example : frameVector 1 shear 1 = ![1,1] := by ext i; fin_cases i <;> simp [frameVector, shear]

-- Every real-cover location and every actual invertible frame; weights mix every contact order.
example {L : ℝ} (hL : 0 < L) (x : Fin 3 → ℝ)
    (T : (Fin 3 → ℝ) ≃L[ℝ] (Fin 3 → ℝ)) :
    0 < Var[contactObservable 2 L T x 1 ![2,-3,4] 5 (-6) ![7,-8]; sourceLaw 3] := by
  exact contactObservable_variance_pos 2 (by norm_num) hL T x _ _ _ _ _ (Or.inl (by norm_num))

-- Physical nonorthogonal frame, mixed second derivatives, arbitrary location.
example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[contactObservable 1 L shear x 0 0 0 0 (fun _ => 1); sourceLaw 2] := by
  apply contactObservable_variance_pos 1 (by norm_num) hL
  exact Or.inr (Or.inr (Or.inr (Or.inr (by intro h; have := congrFun h 0; norm_num at this))))

-- Pure original field observation, including the positive constant source mode.
example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[fun ω => sourceField 2 L ω x; sourceLaw 2] := by
  have h := contactObservable_variance_pos 1 (by norm_num) hL
    (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 1 0 0 0 0 (Or.inl (by norm_num))
  have he : contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 1 0 0 0 0 =
      (fun ω => sourceField 2 L ω x) := by funext ω; simp [contactObservable]
  rwa [he] at h

-- Axial and transverse gradients are both literal Fréchet observations.
example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[fun ω => sourceJet 2 L 1 ω x (fun _ => Pi.single 0 1); sourceLaw 2] := by
  have hw : (Pi.single (0 : Fin 2) (1 : ℝ) : Fin 2 → ℝ) ≠ 0 := by intro h; have := congrFun h 0; norm_num at this
  have h := contactObservable_variance_pos 1 (by norm_num) hL
    (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 (Pi.single 0 1) 0 0 0 (Or.inr (Or.inl hw))
  have he : contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 (Pi.single 0 1) 0 0 0 =
      (fun ω => sourceJet 2 L 1 ω x (fun _ => Pi.single 0 1)) := by
    funext ω; simp [contactObservable, frameVector]
  rwa [he] at h

example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[fun ω => sourceJet 2 L 1 ω x (fun _ => Pi.single 1 1); sourceLaw 2] := by
  have hw : (Pi.single (1 : Fin 2) (1 : ℝ) : Fin 2 → ℝ) ≠ 0 := by intro h; have := congrFun h 1; norm_num at this
  have h := contactObservable_variance_pos 1 (by norm_num) hL
    (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 (Pi.single 1 1) 0 0 0 (Or.inr (Or.inl hw))
  have he : contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 (Pi.single 1 1) 0 0 0 =
      (fun ω => sourceJet 2 L 1 ω x (fun _ => Pi.single 1 1)) := by
    funext ω; simp [contactObservable, frameVector]
  rwa [he] at h

-- Pure second, third and mixed-second contact entries.
example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[fun ω => sourceJet 2 L 2 ω x (fun _ => Pi.single 0 1); sourceLaw 2] := by
  have h := contactObservable_variance_pos 1 (by norm_num) hL
    (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 0 1 0 0
    (Or.inr (Or.inr (Or.inl (by norm_num))))
  have he : contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 0 1 0 0 =
      (fun ω => sourceJet 2 L 2 ω x (fun _ => Pi.single 0 1)) := by
    funext ω; simp [contactObservable, frameVector]
  rwa [he] at h

example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[fun ω => sourceJet 2 L 3 ω x (fun _ => Pi.single 0 1); sourceLaw 2] := by
  have h := contactObservable_variance_pos 1 (by norm_num) hL
    (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 0 0 1 0
    (Or.inr (Or.inr (Or.inr (Or.inl (by norm_num)))))
  have he : contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ)) x 0 0 0 1 0 =
      (fun ω => sourceJet 2 L 3 ω x (fun _ => Pi.single 0 1)) := by
    funext ω; simp [contactObservable, frameVector]
  rwa [he] at h

example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ))
      x 0 0 0 0 (fun _ => 1); sourceLaw 2] := by
  apply contactObservable_variance_pos 1 (by norm_num) hL
  exact Or.inr (Or.inr (Or.inr (Or.inr (by intro h; have := congrFun h 0; norm_num at this))))

-- An odd contrast that cancels at the unit axial frequency still has positive actual variance.
example {L : ℝ} (hL : 0 < L) (x : Fin 2 → ℝ) :
    0 < Var[contactObservable 1 L (ContinuousLinearEquiv.refl ℝ (Fin 2 → ℝ))
      x 0 (sourceC L ^ 2 • Pi.single 0 1) 0 1 0; sourceLaw 2] := by
  apply contactObservable_variance_pos 1 (by norm_num) hL
  exact Or.inr (Or.inr (Or.inr (Or.inl (by norm_num))))

-- Zero weights are a genuine degeneracy: the positive-rank premise cannot be deleted.
example (d : ℕ) (L : ℝ) (x : Fin (d+1) → ℝ)
    (T : (Fin (d+1) → ℝ) ≃L[ℝ] (Fin (d+1) → ℝ)) :
    Var[contactObservable d L T x 0 0 0 0 0; sourceLaw (d+1)] = 0 := by
  have he : contactObservable d L T x 0 0 0 0 0 = 0 := by funext ω; simp [contactObservable]
  rw [he, variance_zero]

-- Exact third derivative sine/cosine signs at phase zero are independently evaluated.
example : iteratedDeriv 3 Real.cos 0 = 0 := by norm_num
example : iteratedDeriv 3 Real.sin 0 = -1 := by norm_num

-- Finite-witness controls: basis frequencies miss linear/cubic and cross-quadratic contrasts.
example (c : ℝ) (hc : c ≠ 0) :
    c * c^2 - c^3 = 0 ∧ 2*c*c^2 - (2*c)^3 ≠ 0 := by
  constructor
  · ring
  · have hc3 : c^3 ≠ 0 := pow_ne_zero 3 hc
    intro h
    have he : 2*c*c^2 - (2*c)^3 = -6*c^3 := by ring
    rw [he] at h
    exact hc3 ((mul_eq_zero.mp h).resolve_left (by norm_num))

example (c : ℝ) (hc : c ≠ 0) :
    (c*1)*(c*0) = 0 ∧ (c*0)*(c*1) = 0 ∧ (c*(1+0))*(c*(0+1)) ≠ 0 := by
  simp [hc]

end ContactRankTests

