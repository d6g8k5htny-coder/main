import Mathlib.Analysis.ODE.ExistUnique
import Mathlib.Tactic

noncomputable section
open Set Metric
open scoped Topology NNReal
namespace ActualJointFlow
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

theorem global_solution_unique (V : E → E) (K : ℝ≥0) (hLip : LipschitzWith K V)
    (f g : ℝ → E) (hf : ∀ t, HasDerivAt f (V (f t)) t)
    (hg : ∀ t, HasDerivAt g (V (g t)) t) (heq : f 0 = g 0) : f = g := by
  exact ODE_solution_unique_univ (v := fun _ => V) (s := fun _ => univ)
    (fun _ => hLip.lipschitzOnWith) (fun t => ⟨hf t,mem_univ _⟩)
    (fun t => ⟨hg t,mem_univ _⟩) heq

section Construction
variable (V : E → E) (K B : ℝ≥0) (hLip : LipschitzWith K V) (hBound : ∀ x, ‖V x‖ ≤ B)
include hLip hBound

private theorem finite_picard (x : E) (T : ℝ≥0) :
    IsPicardLindelof (fun _ : ℝ => V)
      (⟨0,neg_nonpos.mpr T.coe_nonneg,T.coe_nonneg⟩ : Icc (-(T:ℝ)) T) x (B*T+1) 1 B K where
  lipschitzOnWith := fun _ _ => hLip.lipschitzOnWith
  continuousOn := fun _ _ => continuousOn_const
  norm_le := fun _ _ y _ => hBound y
  mul_max_le := by simp

private def family (x : E) (T : ℝ≥0) : E × ℝ → E :=
  Classical.choose ((finite_picard V K B hLip hBound x T).exists_forall_mem_closedBall_eq_hasDerivWithinAt_continuousOn)

private theorem family_spec (x : E) (T : ℝ≥0) :
    (∀ y ∈ closedBall x (1:ℝ), family V K B hLip hBound x T (y,0) = y ∧
      ∀ t ∈ Icc (-(T:ℝ)) T, HasDerivWithinAt
        (fun s => family V K B hLip hBound x T (y,s))
        (V (family V K B hLip hBound x T (y,t))) (Icc (-(T:ℝ)) T) t) ∧
    ContinuousOn (family V K B hLip hBound x T) (closedBall x 1 ×ˢ Icc (-(T:ℝ)) T) :=
  Classical.choose_spec ((finite_picard V K B hLip hBound x T).exists_forall_mem_closedBall_eq_hasDerivWithinAt_continuousOn)

private theorem family_deriv (x : E) (T : ℝ≥0) (y : E) (hy : y ∈ closedBall x 1)
    (t : ℝ) (ht : t ∈ Ioo (-(T:ℝ)) T) :
    HasDerivAt (fun s => family V K B hLip hBound x T (y,s))
      (V (family V K B hLip hBound x T (y,t))) t :=
  ((family_spec V K B hLip hBound x T).1 y hy).2 t (Ioo_subset_Icc_self ht) |>.hasDerivAt
    (Icc_mem_nhds ht.1 ht.2)

private theorem family_agree (x z : E) (T S : ℝ≥0) (hT : 0 < T) (hS : 0 < S)
    (y : E) (hyx : y ∈ closedBall x 1) (hyz : y ∈ closedBall z 1) (t : ℝ)
    (htT : t ∈ Ioo (-(T:ℝ)) T) (htS : t ∈ Ioo (-(S:ℝ)) S) :
    family V K B hLip hBound x T (y,t) = family V K B hLip hBound z S (y,t) := by
  let R : ℝ := min (T:ℝ) S
  have hR : 0 < R := lt_min hT hS
  have hxsub : Ioo (-R) R ⊆ Ioo (-(T:ℝ)) T :=
    Ioo_subset_Ioo (neg_le_neg (min_le_left _ _)) (min_le_left _ _)
  have hzsub : Ioo (-R) R ⊆ Ioo (-(S:ℝ)) S :=
    Ioo_subset_Ioo (neg_le_neg (min_le_right _ _)) (min_le_right _ _)
  have h0 : (0:ℝ) ∈ Ioo (-R) R := ⟨by linarith,hR⟩
  have ht : t ∈ Ioo (-R) R := by
    refine ⟨?_,lt_min htT.2 htS.2⟩
    dsimp [R]
    rcases le_total (T:ℝ) S with h | h
    · rw [min_eq_left h]; exact htT.1
    · rw [min_eq_right h]; exact htS.1
  have he := ODE_solution_unique_of_mem_Ioo (v := fun _ : ℝ => V) (s := fun _ => univ)
    (fun _ _ => hLip.lipschitzOnWith) h0
    (fun s hs => ⟨family_deriv V K B hLip hBound x T y hyx s (hxsub hs),mem_univ _⟩)
    (fun s hs => ⟨family_deriv V K B hLip hBound z S y hyz s (hzsub hs),mem_univ _⟩)
    (by rw [((family_spec V K B hLip hBound x T).1 y hyx).1,
      ((family_spec V K B hLip hBound z S).1 y hyz).1])
  exact he ht

private def horizon (t : ℝ) : ℝ≥0 := ⟨|t|+1,by positivity⟩
omit hLip hBound in
private theorem horizon_pos (t : ℝ) : 0 < horizon t := by
  change (0:ℝ)< |t|+1
  positivity
omit hLip hBound in
private theorem horizon_contains (t : ℝ) : t ∈ Ioo (-(horizon t:ℝ)) (horizon t) := by
  change -(|t|+1)<t ∧ t< |t|+1
  constructor <;> linarith [neg_abs_le t,le_abs_self t]

private def constructed : E × ℝ → E := fun p =>
  family V K B hLip hBound p.1 (horizon p.2) p

private theorem constructed_initial (x : E) : constructed V K B hLip hBound (x,0) = x :=
  ((family_spec V K B hLip hBound x (horizon 0)).1 x (mem_closedBall_self (by norm_num))).1

private theorem constructed_deriv (x : E) (t : ℝ) :
    HasDerivAt (fun s => constructed V K B hLip hBound (x,s))
      (V (constructed V K B hLip hBound (x,t))) t := by
  let T := horizon t
  have hEq : ∀ᶠ s in 𝓝 t,
      constructed V K B hLip hBound (x,s) = family V K B hLip hBound x T (x,s) := by
    filter_upwards [Ioo_mem_nhds (horizon_contains t).1 (horizon_contains t).2] with s hs
    exact family_agree V K B hLip hBound x x (horizon s) T (horizon_pos s) (horizon_pos t)
      x (mem_closedBall_self (by norm_num)) (mem_closedBall_self (by norm_num)) s
      (horizon_contains s) hs
  have hd := family_deriv V K B hLip hBound x T x (mem_closedBall_self (by norm_num)) t
    (horizon_contains t)
  exact hd.congr_of_eventuallyEq hEq

private theorem constructed_continuous : Continuous (constructed V K B hLip hBound) := by
  rw [continuous_iff_continuousAt]
  intro p
  let T := horizon p.2
  have hn : closedBall p.1 (1:ℝ) ×ˢ Ioo (-(T:ℝ)) T ∈ 𝓝 p := by
    rw [nhds_prod_eq]
    exact Filter.prod_mem_prod (closedBall_mem_nhds p.1 (by norm_num))
      (Ioo_mem_nhds (horizon_contains p.2).1 (horizon_contains p.2).2)
  have hn' : closedBall p.1 (1:ℝ) ×ˢ Icc (-(T:ℝ)) T ∈ 𝓝 p :=
    Filter.mem_of_superset hn (prod_mono (Subset.rfl) Ioo_subset_Icc_self)
  have hc := (family_spec V K B hLip hBound p.1 T).2.continuousAt hn'
  have heq : constructed V K B hLip hBound =ᶠ[𝓝 p] family V K B hLip hBound p.1 T := by
    filter_upwards [hn] with q hq
    exact family_agree V K B hLip hBound q.1 p.1 (horizon q.2) T
      (horizon_pos q.2) (horizon_pos p.2) q.1 (mem_closedBall_self (by norm_num)) hq.1 q.2
      (horizon_contains q.2) hq.2
  exact hc.congr_of_eventuallyEq heq

private theorem constructed_add (x : E) (s t : ℝ) :
    constructed V K B hLip hBound (x,s+t) =
      constructed V K B hLip hBound (constructed V K B hLip hBound (x,s),t) := by
  let Φ := constructed V K B hLip hBound
  have hshift : ∀ r, HasDerivAt (fun u => Φ (x,s+u)) (V (Φ (x,s+r))) r := by
    intro r
    simpa only [Function.comp_def,one_smul] using
      (constructed_deriv V K B hLip hBound x (s+r)).scomp r ((hasDerivAt_id r).const_add s)
  have he := global_solution_unique V K hLip (fun r => Φ (x,s+r))
    (fun r => Φ (Φ (x,s),r)) hshift
    (constructed_deriv V K B hLip hBound (Φ (x,s)))
    (by simp only [add_zero,Φ,constructed_initial])
  exact congrFun he t

end Construction

theorem exists_global_joint_flow (V : E → E) (K B : ℝ≥0)
    (hLip : LipschitzWith K V) (hBound : ∀ x, ‖V x‖ ≤ B) :
    ∃ Φ : E × ℝ → E, Continuous Φ ∧ (∀ x, Φ (x,0) = x) ∧
      (∀ x t, HasDerivAt (fun s => Φ (x,s)) (V (Φ (x,t))) t) ∧
      (∀ x s t, Φ (x,s+t) = Φ (Φ (x,s),t)) := by
  exact ⟨constructed V K B hLip hBound,constructed_continuous V K B hLip hBound,
    constructed_initial V K B hLip hBound,constructed_deriv V K B hLip hBound,
    constructed_add V K B hLip hBound⟩

end ActualJointFlow

