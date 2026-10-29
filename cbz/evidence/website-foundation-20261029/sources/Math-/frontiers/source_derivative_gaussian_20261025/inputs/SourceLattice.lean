import SourceFourierSeries
import Mathlib.Data.Finset.Max

noncomputable section
open I4Foundation I4Weights I4Source MeasureTheory
open scoped BigOperators
namespace I4Lattice

def latticeFrequency (d : ℕ) (m : SourceMode d) : Fin d → ℤ :=
  match m with
  | none => 0
  | some (k, b) => if b then -k.1 else k.1

theorem halfLattice_ne_zero (d : ℕ) {k : Fin d → ℤ} (hk : k ∈ halfLattice d) :
    k ≠ 0 := by
  rcases hk with ⟨i, hi, _⟩
  intro h
  have : k i = 0 := congrFun h i
  omega

theorem halfLattice_partition (d : ℕ) {k : Fin d → ℤ} (hk : k ≠ 0) :
    (k ∈ halfLattice d ∨ -k ∈ halfLattice d) ∧
      ¬ (k ∈ halfLattice d ∧ -k ∈ halfLattice d) := by
  classical
  have hex : ∃ i, k i ≠ 0 := by
    by_contra h
    push Not at h
    exact hk (funext h)
  let s : Finset (Fin d) := Finset.univ.filter (fun i => k i ≠ 0)
  have hs : s.Nonempty := by
    obtain ⟨i, hi⟩ := hex
    exact ⟨i, Finset.mem_filter.mpr ⟨Finset.mem_univ i, hi⟩⟩
  let i := s.min' hs
  have hi : k i ≠ 0 := (Finset.mem_filter.mp (Finset.min'_mem s hs)).2
  have hbefore (j : Fin d) (hj : j < i) : k j = 0 := by
    by_contra hn
    have hle := Finset.min'_le s j (Finset.mem_filter.mpr ⟨Finset.mem_univ j, hn⟩)
    exact (not_le_of_gt hj) hle
  constructor
  · rcases lt_or_gt_of_ne hi with hneg | hpos
    · right
      refine ⟨i, ?_, ?_⟩
      · change 0 < -k i
        omega
      · intro j hj
        change -k j = 0
        rw [hbefore j hj, neg_zero]
    · exact Or.inl ⟨i, hpos, hbefore⟩
  · rintro ⟨⟨i, hi, hib⟩, ⟨j, hj, hjb⟩⟩
    change 0 < -k j at hj
    rcases lt_trichotomy i j with hij | heq | hji
    · have h0 := hjb i hij
      change -k i = 0 at h0
      omega
    · subst j
      omega
    · have h0 := hib j hji
      omega

theorem sourceWeight_neg (d : ℕ) (L : ℝ) (k : Fin d → ℤ) :
    sourceWeight d L (-k) = sourceWeight d L k := by
  rw [sourceWeight_eq_physical, sourceWeight_eq_physical]
  simp only [Pi.neg_apply, Int.cast_neg, mul_neg, neg_sq]

theorem phase_lattice_shift (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k n : Fin d → ℤ) (x : Fin d → ℝ) :
    phase d L k (fun i => x i + L * (n i : ℝ)) =
      phase d L k x + ((∑ i, k i * n i : ℤ) : ℝ) * (2 * Real.pi) := by
  have he (i : Fin d) : sourceC L * (k i : ℝ) * (x i + L * (n i : ℝ)) =
      sourceC L * (k i : ℝ) * x i + ((k i * n i : ℤ) : ℝ) * (2 * Real.pi) := by
    simp only [Int.cast_mul]
    unfold sourceC
    field_simp
  unfold phase
  simp_rw [he]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul]
  simp only [Int.cast_sum]

theorem modeCoefficient_lattice_shift (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (m : SourceMode d) (n : Fin d → ℤ) (x : Fin d → ℝ) :
    modeCoefficient d L m (fun i => x i + L * (n i : ℝ)) =
      modeCoefficient d L m x := by
  cases m with
  | none => rfl
  | some p =>
    rcases p with ⟨k,b⟩
    cases b <;> simp only [modeCoefficient, Bool.false_eq_true, ite_false, ite_true]
    · rw [phase_lattice_shift d hL, Real.cos_add_int_mul_two_pi]
    · rw [phase_lattice_shift d hL, Real.sin_add_int_mul_two_pi]

theorem sourceField_lattice_shift (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SourceMode d → ℝ) (n : Fin d → ℤ) (x : Fin d → ℝ) :
    sourceField d L ω (fun i => x i + L * (n i : ℝ)) = sourceField d L ω x := by
  unfold sourceField scalarSeries
  apply tsum_congr
  intro m
  change modeCoefficient d L m (fun i => x i + L * (n i : ℝ)) * ω m =
    modeCoefficient d L m x * ω m
  rw [modeCoefficient_lattice_shift d hL]

theorem pairedCoefficient_kernel (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k : ↥(halfLattice d)) (x y : Fin d → ℝ) :
    (∑ b : Bool, modeCoefficient d L (some (k,b)) x *
      modeCoefficient d L (some (k,b)) y) =
      (2 * sourceWeight d L k / sourceZ d L) * Real.cos (phase d L k (x - y)) := by
  have hphase : phase d L k (x - y) = phase d L k x - phase d L k y := by
    simp [phase, mul_sub, Finset.sum_sub_distrib]
  have hz : 0 ≤ 2 * sourceWeight d L k / sourceZ d L := by
    rw [sourceWeight_eq_physical]
    exact div_nonneg (mul_nonneg (by norm_num) (Real.exp_pos _).le) (sourceZ_pos d hL).le
  rw [Fintype.sum_bool]
  simp only [modeCoefficient, ite_true, Bool.false_eq_true, ite_false]
  rw [hphase, Real.cos_sub]
  calc
    _ = (Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L)) *
      (Real.cos (phase d L k x) * Real.cos (phase d L k y) +
        Real.sin (phase d L k x) * Real.sin (phase d L k y)) := by ring
    _ = _ := by rw [Real.mul_self_sqrt hz]

theorem latticeFrequency_bijective (d : ℕ) : Function.Bijective (latticeFrequency d) := by
  classical
  constructor
  · intro m n he
    cases m with
    | none =>
      cases n with
      | none => rfl
      | some p =>
        rcases p with ⟨k,b⟩
        have hk := halfLattice_ne_zero d k.property
        cases b <;> simp [latticeFrequency] at he <;> exact (hk (by first | exact he | exact he.symm)).elim
    | some p =>
      rcases p with ⟨k,b⟩
      cases n with
      | none =>
        have hk := halfLattice_ne_zero d k.property
        cases b <;> simp [latticeFrequency] at he <;> exact (hk he).elim
      | some q =>
        rcases q with ⟨l,c⟩
        cases b <;> cases c <;> simp only [latticeFrequency, Bool.false_eq_true, ite_false, ite_true] at he
        · have hkl : k = l := Subtype.ext he
          subst l
          rfl
        · have hn : -k.1 ∈ halfLattice d := by simpa only [he, neg_neg] using l.property
          exact ((halfLattice_partition d (halfLattice_ne_zero d k.property)).2 ⟨k.property,hn⟩).elim
        · have hn : -k.1 ∈ halfLattice d := he.symm ▸ l.property
          exact ((halfLattice_partition d (halfLattice_ne_zero d k.property)).2 ⟨k.property,hn⟩).elim
        · have hkl : k = l := Subtype.ext (neg_inj.mp he)
          subst l
          rfl
  · intro k
    by_cases hk : k = 0
    · exact ⟨none, by simpa [latticeFrequency] using hk.symm⟩
    · rcases (halfLattice_partition d hk).1 with hp | hn
      · exact ⟨some (⟨k,hp⟩,false), by simp [latticeFrequency]⟩
      · exact ⟨some (⟨-k,hn⟩,true), by simp [latticeFrequency]⟩

end I4Lattice
