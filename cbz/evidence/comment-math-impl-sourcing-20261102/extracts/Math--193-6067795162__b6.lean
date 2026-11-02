import QuadraticCone
import OpenComponentGluing

noncomputable section
open Set CategoryTheory
namespace MorseCone
universe u
variable {X : Type u} [TopologicalSpace X] [T1Space X]
variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Genuine local function normal form; no component or attachment conclusions. -/
structure NormalForm (X E F : Type*) [TopologicalSpace X]
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] where
  f : X → ℝ
  p : X
  c : ℝ
  ρ : ℝ
  hρ : 0 < ρ
  chart : OpenPartialHomeomorph X (E × F)
  center_mem : p ∈ chart.source
  center_eq : chart p = 0
  ball_target : radiusBall ρ ⊆ chart.target
  equation : ∀ x ∈ chart.source, f x = c + quadratic (chart x)

def closedLevel (D : NormalForm X E F) := {x : X | D.c ≤ D.f x}
def oldSet (D : NormalForm X E F) : Set (closedLevel D) := {x | x.val ≠ D.p}
def patchSet (D : NormalForm X E F) : Set (closedLevel D) :=
  {x | x.val ∈ D.chart.source ∧ D.chart x.val ∈ radiusBall D.ρ}

private theorem radius_open (ρ : ℝ) : IsOpen (radiusBall (E := E) (F := F) ρ) :=
  isOpen_lt (by fun_prop) continuous_const

private theorem symm_zero (D : NormalForm X E F) : D.chart.symm 0 = D.p := by
  simpa only [D.center_eq] using D.chart.left_inv D.center_mem

private def liftCone (D : NormalForm X E F) (q : cone (E := E) (F := F) D.ρ) :
    closedLevel D := by
  refine ⟨D.chart.symm q.val, ?_⟩
  have hq := D.ball_target q.property.2
  change D.c ≤ D.f (D.chart.symm q.val)
  rw [D.equation _ (D.chart.map_target hq), D.chart.right_inv hq]
  have hs := (sq_le_sq₀ (norm_nonneg q.val.2) (norm_nonneg q.val.1)).2 q.property.1
  change D.c ≤ D.c + (‖q.val.1‖^2 - ‖q.val.2‖^2)
  linarith

private theorem liftCone_continuous (D : NormalForm X E F) : Continuous (liftCone D) := by
  apply Continuous.subtype_mk
  exact D.chart.continuousOn_symm.comp_continuous continuous_subtype_val
    (fun q => D.ball_target q.property.2)

private theorem patch_range (D : NormalForm X E F) : patchSet D = range (liftCone D) := by
  ext x
  constructor
  · intro hx
    have he := D.equation x.val hx.1
    have hf : D.c ≤ D.f x.val := x.property
    rw [he] at hf
    have hs : ‖(D.chart x.val).2‖^2 ≤ ‖(D.chart x.val).1‖^2 := by
      dsimp [quadratic] at hf
      linarith
    let q : cone (E := E) (F := F) D.ρ := ⟨D.chart x.val,
      ⟨(sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).1 hs, hx.2⟩⟩
    refine ⟨q, ?_⟩
    apply Subtype.ext
    exact D.chart.left_inv hx.1
  · rintro ⟨q, rfl⟩
    refine ⟨D.chart.map_target (D.ball_target q.property.2), ?_⟩
    change D.chart (D.chart.symm q.val) ∈ radiusBall D.ρ
    rw [D.chart.right_inv (D.ball_target q.property.2)]
    exact q.property.2

private def liftPuncture (D : NormalForm X E F)
    (q : puncture (E := E) (F := F) D.ρ) : oldSet D := by
  refine ⟨liftCone D ⟨q.val,q.property.1⟩, ?_⟩
  intro he
  have hh := congrArg D.chart he
  change D.chart (D.chart.symm q.val) = D.chart D.p at hh
  rw [D.chart.right_inv (D.ball_target q.property.1.2), D.center_eq] at hh
  exact q.property.2 (mem_singleton_iff.mpr hh)

private theorem liftPuncture_continuous (D : NormalForm X E F) : Continuous (liftPuncture D) := by
  apply Continuous.subtype_mk
  exact (liftCone_continuous D).comp (Continuous.subtype_mk continuous_subtype_val (fun q => q.property.1))

private theorem old_patch_range (D : NormalForm X E F) :
    {x : oldSet D | x.val ∈ patchSet D} = range (liftPuncture D) := by
  ext x
  constructor
  · intro hx
    have hx' : x.val ∈ range (liftCone D) := by rw [← patch_range D]; exact hx
    obtain ⟨q,hq⟩ := hx'
    have hn : q.val ≠ 0 := by
      intro hzero
      apply x.property
      calc
        x.val.val = D.chart.symm q.val := (congrArg Subtype.val hq).symm
        _ = D.chart.symm 0 := congrArg D.chart.symm hzero
        _ = D.p := symm_zero D
    refine ⟨⟨q.val, q.property, by simpa using hn⟩, ?_⟩
    exact Subtype.ext hq
  · rintro ⟨q,rfl⟩
    change liftCone D ⟨q.val,q.property.1⟩ ∈ patchSet D
    rw [patch_range D]
    exact ⟨⟨q.val,q.property.1⟩,rfl⟩

private theorem range_pc {A B : Type*} [TopologicalSpace A] [PathConnectedSpace A]
    [TopologicalSpace B] (g : A → B) (hg : Continuous g) : IsPathConnected (range g) := by
  simpa only [image_univ] using (pathConnectedSpace_iff_univ.mp inferInstance).image hg

private theorem old_open (D : NormalForm X E F) : IsOpen (oldSet D) :=
  isClosed_singleton.isOpen_compl.preimage continuous_subtype_val

private def positiveLift (D : NormalForm X ℝ F) (q : positiveLobe (F := F) D.ρ) : oldSet D :=
  liftPuncture D ⟨q.val, by rw [one_puncture_partition]; exact Or.inl q.property⟩

private def negativeLift (D : NormalForm X ℝ F) (q : negativeLobe (F := F) D.ρ) : oldSet D :=
  liftPuncture D ⟨q.val, by rw [one_puncture_partition]; exact Or.inr q.property⟩

private theorem positiveLift_continuous (D : NormalForm X ℝ F) : Continuous (positiveLift D) := by
  exact (liftPuncture_continuous D).comp (Continuous.subtype_mk continuous_subtype_val
    (fun q : positiveLobe (F := F) D.ρ => by
      rw [one_puncture_partition]; exact Or.inl q.property))

private theorem negativeLift_continuous (D : NormalForm X ℝ F) : Continuous (negativeLift D) := by
  exact (liftPuncture_continuous D).comp (Continuous.subtype_mk continuous_subtype_val
    (fun q : negativeLobe (F := F) D.ρ => by
      rw [one_puncture_partition]; exact Or.inr q.property))

private theorem old_patch_lobes (D : NormalForm X ℝ F) :
    {x : oldSet D | x.val ∈ patchSet D} = range (positiveLift D) ∪ range (negativeLift D) := by
  rw [old_patch_range D]
  ext x
  constructor
  · rintro ⟨q,rfl⟩
    have hq : q.val ∈ positiveLobe D.ρ ∪ negativeLobe D.ρ := by
      rw [← one_puncture_partition]
      exact q.property
    rcases hq with hp | hn
    · exact Or.inl ⟨⟨q.val,hp⟩,rfl⟩
    · exact Or.inr ⟨⟨q.val,hn⟩,rfl⟩
  · rintro (⟨q,rfl⟩ | ⟨q,rfl⟩)
    · exact ⟨⟨q.val,by rw [one_puncture_partition]; exact Or.inl q.property⟩,rfl⟩
    · exact ⟨⟨q.val,by rw [one_puncture_partition]; exact Or.inr q.property⟩,rfl⟩

theorem patch_open (D : NormalForm X E F) : IsOpen (patchSet D) := by
  change IsOpen (Subtype.val ⁻¹' (D.chart.source ∩ D.chart ⁻¹' radiusBall D.ρ))
  exact (D.chart.isOpen_inter_preimage (radius_open D.ρ)).preimage continuous_subtype_val

theorem critical_cover (D : NormalForm X E F) : oldSet D ∪ patchSet D = univ := by
  ext x
  constructor
  · intro _; trivial
  · intro _
    by_cases he : x.val = D.p
    · refine Or.inr ⟨he ▸ D.center_mem, ?_⟩
      change ‖(D.chart x.val).1‖^2+‖(D.chart x.val).2‖^2<D.ρ^2
      simp only [he,D.center_eq,Prod.fst_zero,Prod.snd_zero,norm_zero,zero_pow (by decide : 2≠0),zero_add]
      exact sq_pos_of_pos D.hρ
    · exact Or.inl he

theorem patch_pathConnected (D : NormalForm X E F) : IsPathConnected (patchSet D) := by
  letI : PathConnectedSpace (cone (E := E) (F := F) D.ρ) :=
    (isPathConnected_iff_pathConnectedSpace).mp (cone_pathConnected D.ρ D.hρ)
  rw [patch_range D]
  exact range_pc (liftCone D) (liftCone_continuous D)

theorem zero_foot (D : NormalForm X (EuclideanSpace ℝ (Fin 0)) F)
    (a : ZerothHomotopy (oldSet D)) : ¬ MorseOpenGluing.Foot (oldSet D) (patchSet D) a := by
  rintro ⟨x,hx,_⟩
  have hx' : x.val ∈ range (liftCone D) := by rw [← patch_range D]; exact hx
  obtain ⟨q,hq⟩ := hx'
  have hz : q.val=0 := by
    have hh := q.property
    have hh' : q.val ∈ ({0} : Set (EuclideanSpace ℝ (Fin 0) × F)) :=
      (congrArg (fun S : Set (EuclideanSpace ℝ (Fin 0) × F) => q.val ∈ S)
        (cone_zero_positive D.ρ D.hρ)).mp hh
    exact mem_singleton_iff.mp hh'
  apply x.property
  calc
    x.val.val = D.chart.symm q.val := (congrArg Subtype.val hq).symm
    _ = D.chart.symm 0 := congrArg D.chart.symm hz
    _ = D.p := symm_zero D

theorem higher_foot (D : NormalForm X E F) (hE : 1 < Module.rank ℝ E) :
    ∃ a : ZerothHomotopy (oldSet D), ∀ q,
      MorseOpenGluing.Foot (oldSet D) (patchSet D) q ↔ q = a := by
  letI : PathConnectedSpace (puncture (E := E) (F := F) D.ρ) :=
    isPathConnected_iff_pathConnectedSpace.mp (higher_puncture_pathConnected D.ρ D.hρ hE)
  have hs := range_pc (liftPuncture D) (liftPuncture_continuous D)
  obtain ⟨a,ha,hjoin⟩ := hs
  refine ⟨ZerothHomotopy.mk a, ?_⟩
  intro q
  constructor
  · rintro ⟨x,hx,hq⟩
    have hx' : x ∈ range (liftPuncture D) := by rw [← old_patch_range D]; exact hx
    exact hq.symm.trans (ZerothHomotopy.sound (hjoin hx').symm.somePath)
  · intro hq
    subst q
    exact ⟨a,by
      change a ∈ {x : oldSet D | x.val ∈ patchSet D}
      rw [old_patch_range D]; exact ha,rfl⟩

theorem one_foot (D : NormalForm X ℝ F) :
    ∃ a b : ZerothHomotopy (oldSet D), ∀ q,
      MorseOpenGluing.Foot (oldSet D) (patchSet D) q ↔ q = a ∨ q = b := by
  letI : PathConnectedSpace (positiveLobe (F := F) D.ρ) :=
    isPathConnected_iff_pathConnectedSpace.mp (one_positive_lobe D.ρ D.hρ)
  letI : PathConnectedSpace (negativeLobe (F := F) D.ρ) :=
    isPathConnected_iff_pathConnectedSpace.mp (one_negative_lobe D.ρ D.hρ)
  have hp := range_pc (positiveLift D) (positiveLift_continuous D)
  have hn := range_pc (negativeLift D) (negativeLift_continuous D)
  obtain ⟨a,ha,hpa⟩ := hp
  obtain ⟨b,hb,hnb⟩ := hn
  refine ⟨ZerothHomotopy.mk a,ZerothHomotopy.mk b,?_⟩
  intro q
  constructor
  · rintro ⟨x,hx,hq⟩
    have hx' : x ∈ range (positiveLift D) ∪ range (negativeLift D) := by
      rw [← old_patch_lobes D]; exact hx
    rcases hx' with hxpos | hxneg
    · exact Or.inl (hq.symm.trans (ZerothHomotopy.sound (hpa hxpos).symm.somePath))
    · exact Or.inr (hq.symm.trans (ZerothHomotopy.sound (hnb hxneg).symm.somePath))
  · rintro (hq | hq)
    · subst q
      exact ⟨a,by
        change a ∈ {x : oldSet D | x.val ∈ patchSet D}
        rw [old_patch_lobes D]; exact Or.inl ha,rfl⟩
    · subst q
      exact ⟨b,by
        change b ∈ {x : oldSet D | x.val ∈ patchSet D}
        rw [old_patch_lobes D]; exact Or.inr hb,rfl⟩

noncomputable def zero_event (D : NormalForm X (EuclideanSpace ℝ (Fin 0)) F) :
    ElementaryHistory.Event (MorseOpenGluing.inclusionMap (oldSet D)) := by
  exact MorseOpenGluing.birthEvent (oldSet D) (patchSet D) (old_open D) (patch_open D)
    (critical_cover D) (patch_pathConnected D) (zero_foot D)

noncomputable def higher_event (D : NormalForm X E F) (hE : 1 < Module.rank ℝ E) :
    ElementaryHistory.Event (MorseOpenGluing.inclusionMap (oldSet D)) := by
  let a := Classical.choose (higher_foot D hE)
  have ha := Classical.choose_spec (higher_foot D hE)
  exact MorseOpenGluing.neutralEvent (oldSet D) (patchSet D) (old_open D) (patch_open D)
    (critical_cover D) (patch_pathConnected D) a ha

noncomputable def one_event (D : NormalForm X ℝ F) :
    ElementaryHistory.Event (MorseOpenGluing.inclusionMap (oldSet D)) := by
  let a := Classical.choose (one_foot D)
  let b := Classical.choose (Classical.choose_spec (one_foot D))
  have hf : ∀ q, MorseOpenGluing.Foot (oldSet D) (patchSet D) q ↔ q=a ∨ q=b :=
    Classical.choose_spec (Classical.choose_spec (one_foot D))
  by_cases hab : a=b
  · exact MorseOpenGluing.neutralEvent (oldSet D) (patchSet D) (old_open D) (patch_open D)
      (critical_cover D) (patch_pathConnected D) a (by intro q; simpa only [← hab, or_self] using hf q)
  · exact MorseOpenGluing.mergeEvent (oldSet D) (patchSet D) (old_open D) (patch_open D)
      (critical_cover D) (patch_pathConnected D) a b hab hf

end MorseCone
