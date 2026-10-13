import Specifications.Side24

/-! Verify probe 2 (Side24-covariance-density_verify). Probe only; `sorry` allowed here, never
elsewhere. ReferenceCovarianceLowerBound route: (1) odd-order contractions of `Kinf` at `0` vanish
(by the symmetry `Kinf (-y) = Kinf y`, no level-four formula needed), (2) `Cref` entries reduce to
the literal direction lists, (3) the `SymIndex` index set enumerates by `decide`, (4) the odd-block
quadratic form is nonnegative. -/

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

section OddOrders

variable {d : ℕ}

theorem verify_kinf_contDiff : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf
  exact ContDiff.exp (ContDiff.div_const (ContDiff.neg (contDiff_norm_sq ℝ)) 2)

/-- Odd-order iterated derivatives of the even function `Kinf` vanish at the origin. -/
theorem verify_odd_order_zero (n : ℕ) (hn : Odd n) (m : Fin n → E d) :
    iteratedFDeriv ℝ n (Kinf d) 0 m = 0 := by
  set g : E d →L[ℝ] E d := -(ContinuousLinearMap.id ℝ (E d)) with hg
  have hcomp : Kinf d ∘ g = Kinf d := by
    funext y; simp [Kinf, hg]
  have h := ContinuousLinearMap.iteratedFDeriv_comp_right g verify_kinf_contDiff 0
    (i := n) (mod_cast le_top)
  rw [hcomp] at h
  have h2 := congrArg (fun T => T m) h
  simp only [ContinuousMultilinearMap.compContinuousLinearMap_apply, map_zero, hg,
    ContinuousLinearMap.neg_apply, ContinuousLinearMap.id_apply] at h2
  have h3 : iteratedFDeriv ℝ n (Kinf d) 0 (fun i => -(m i)) =
      (-1 : ℝ) ^ n • iteratedFDeriv ℝ n (Kinf d) 0 m := by
    have := (iteratedFDeriv ℝ n (Kinf d) 0).map_smul_univ (fun _ => (-1 : ℝ)) m
    simpa [Finset.prod_const] using this
  rw [h3, hn.neg_one_pow, neg_one_smul] at h2
  linarith

theorem verify_order_three (a b c : E d) : derivContraction d (Kinf d) 0 [a, b, c] = 0 :=
  verify_odd_order_zero 3 (by decide) _

theorem verify_order_five (a b c e f : E d) :
    derivContraction d (Kinf d) 0 [a, b, c, e, f] = 0 :=
  verify_odd_order_zero 5 (by decide) _

/-! `Cref` entries reduce to literal direction lists (the `++` of `jetDirs` computes). -/

theorem verify_cref_tH (u : E d) (R : Fin d → E d) (p : SymIndex d) :
    Cref d u R (Sum.inr (Sum.inl ())) (Sum.inr (Sum.inr p)) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [verify_order_five]
  simp

theorem verify_cref_Ht (u : E d) (R : Fin d → E d) (p : SymIndex d) :
    Cref d u R (Sum.inr (Sum.inr p)) (Sum.inr (Sum.inl ())) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [verify_order_five]
  simp

theorem verify_cref_GH (u : E d) (R : Fin d → E d) (i : Fin d) (p : SymIndex d) :
    Cref d u R (Sum.inl i) (Sum.inr (Sum.inr p)) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [verify_order_three]
  simp

theorem verify_cref_HG (u : E d) (R : Fin d → E d) (i : Fin d) (p : SymIndex d) :
    Cref d u R (Sum.inr (Sum.inr p)) (Sum.inl i) = 0 := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  rw [verify_order_three]
  simp

-- The `t`–`G` entry is `-(D^4 K(0)[u,u,u,R i])`, the shape the order-four pairing lemma serves.
theorem verify_cref_tG_shape (u : E d) (R : Fin d → E d) (i : Fin d) :
    Cref d u R (Sum.inr (Sum.inl ())) (Sum.inl i) = -derivContraction d (Kinf d) 0 [u, u, u, R i] := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append, svecScale]
  simp

-- The `H`–`H` entry carries the `svec` scales and an order-four contraction.
theorem verify_cref_HH_shape (u : E d) (R : Fin d → E d) (p q : SymIndex d) :
    Cref d u R (Sum.inr (Sum.inr p)) (Sum.inr (Sum.inr q)) =
      svecScale d (Sum.inr (Sum.inr p)) * svecScale d (Sum.inr (Sum.inr q)) *
        derivContraction d (Kinf d) 0 [R p.1.1, R p.1.2, R q.1.1, R q.1.2] := by
  simp only [Cref, jointCov, Matrix.of_apply, jetDirs, List.cons_append, List.nil_append]
  simp

end OddOrders

/-! ### Index bookkeeping -/

example : (Finset.univ : Finset (SymIndex 2)) =
    {⟨(0, 0), le_rfl⟩, ⟨(0, 1), by decide⟩, ⟨(1, 1), le_rfl⟩} := by decide

example : Fintype.card (SymIndex 3) = 6 := by decide

example (f : JointIndex 2 → ℝ) :
    ∑ a, f a = ∑ i : Fin 2, f (Sum.inl i) + f (Sum.inr (Sum.inl ())) +
      ∑ p : SymIndex 2, f (Sum.inr (Sum.inr p)) := by
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type]
  simp [add_assoc]

/-! ### The odd block `[[1,-3],[-3,15]] - I/3` has a nonnegative quadratic form (det `7/9`). -/
example (x y : ℝ) : 0 ≤ (2 / 3 : ℝ) * x ^ 2 - 6 * x * y + (44 / 3) * y ^ 2 := by
  nlinarith [sq_nonneg (x - (9 / 2) * y), sq_nonneg y]

/-! ### PosSemidef over a sum index type via the dotProduct characterisation elaborates. -/
example (M : Matrix (JointIndex 2) (JointIndex 2) ℝ) (h1 : M.IsHermitian)
    (h2 : ∀ x : JointIndex 2 → ℝ, 0 ≤ dotProduct x (M.mulVec x)) : M.PosSemidef :=
  Matrix.posSemidef_iff_dotProduct_mulVec.mpr ⟨h1, fun x => by simpa using h2 x⟩
