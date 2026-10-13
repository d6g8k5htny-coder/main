import Specifications.RN

/-! RN-B_verify probe 1 (adversarial verification of the triage verdict PROVABLE_NOW for
`UniversalLaw.Spec.RN.RareClusterPoissonObstructionO`).

Purpose: the triage agent elaborated the pieces (a)-(e) separately but never assembled them.
This probe attempts the FULL proof of the Prop as stated, with its own binders and no extra
hypothesis, no `sorry`, no `native_decide`. If it compiles, `#print axioms` at the end lists the
axioms used. Nothing here is a theorem of the packet; this file is a probe only. -/

open UniversalLaw.Spec.RN

namespace RNBVerify

/-- (e) Half the `l^1` distance dominates the mass difference of every tail event `{n ≥ k}`,
for two summable families of total mass one. -/
theorem tail_diff_le_pmfTV (μ ν : ℕ → ℝ) (hμ : Summable μ) (hν : Summable ν)
    (h1 : ∑' n, μ n = 1) (h1' : ∑' n, ν n = 1) (k : ℕ) :
    |∑' n : ℕ, (μ (n + k) - ν (n + k))| ≤ pmfTV μ ν := by
  unfold pmfTV
  have hfs : Summable (fun n => μ n - ν n) := hμ.sub hν
  have hfabs : Summable (fun n => |μ n - ν n|) := hfs.abs
  have htot : ∑' n, (μ n - ν n) = 0 := by rw [hμ.tsum_sub hν, h1, h1']; ring
  have hsplit := Summable.sum_add_tsum_nat_add k hfs
  rw [htot] at hsplit
  have hsplit_abs := Summable.sum_add_tsum_nat_add k hfabs
  have hhead : |∑ i ∈ Finset.range k, (μ i - ν i)| ≤ ∑ i ∈ Finset.range k, |μ i - ν i| :=
    Finset.abs_sum_le_sum_abs _ _
  have htail : |∑' i : ℕ, (μ (i + k) - ν (i + k))| ≤ ∑' i : ℕ, |μ (i + k) - ν (i + k)| := by
    have := norm_tsum_le_tsum_norm (f := fun i : ℕ => μ (i + k) - ν (i + k)) ?_
    · simpa only [Real.norm_eq_abs] using this
    · simpa only [Real.norm_eq_abs] using (summable_nat_add_iff k).mpr hfabs
  have heq : ∑ i ∈ Finset.range k, (μ i - ν i) = -∑' i : ℕ, (μ (i + k) - ν (i + k)) := by
    linarith
  rw [heq, abs_neg] at hhead
  linarith

/-- Difference of tails splits as the difference of the tails. -/
theorem tsum_tail_sub (μ ν : ℕ → ℝ) (hμ : Summable μ) (hν : Summable ν) (k : ℕ) :
    ∑' n : ℕ, (μ (n + k) - ν (n + k)) = ∑' n : ℕ, μ (n + k) - ∑' n : ℕ, ν (n + k) :=
  Summable.tsum_sub ((summable_nat_add_iff k).mpr hμ) ((summable_nat_add_iff k).mpr hν)

/-- Tail `{n ≥ 1}` of a family of total mass one. -/
theorem tail1 (μ : ℕ → ℝ) (hs : Summable μ) (h1 : ∑' n, μ n = 1) :
    ∑' n : ℕ, μ (n + 1) = 1 - μ 0 := by
  have h2 := Summable.sum_add_tsum_nat_add 1 hs
  rw [h1, Finset.sum_range_one] at h2
  linarith

/-- Tail `{n ≥ 2}` of a family of total mass one. -/
theorem tail2 (μ : ℕ → ℝ) (hs : Summable μ) (h1 : ∑' n, μ n = 1) :
    ∑' n : ℕ, μ (n + 2) = 1 - μ 0 - μ 1 := by
  have h2 := Summable.sum_add_tsum_nat_add 2 hs
  rw [h1] at h2
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at h2
  linarith

/-- The Poisson pmf has total mass one for `lam ≥ 0`. -/
theorem poisson_hasSum (lam : ℝ) (hl : 0 ≤ lam) : HasSum (poissonPMF lam) 1 := by
  have h : HasSum (fun n : ℕ => Real.exp (-lam) * lam ^ n / (Nat.factorial n : ℝ)) 1 :=
    ProbabilityTheory.hasSum_one_poissonMeasure ⟨lam, hl⟩
  exact h

theorem poisson0 (lam : ℝ) : poissonPMF lam 0 = Real.exp (-lam) := by simp [poissonPMF]

theorem poisson1 (lam : ℝ) : poissonPMF lam 1 = Real.exp (-lam) * lam := by simp [poissonPMF]

/-- (c) Markov for the mean: `∑_{n ≥ 1} μ n ≤ ∑ n μ n`. -/
theorem markov (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
    (hm : Summable (fun n : ℕ => (n : ℝ) * μ n)) :
    ∑' n : ℕ, μ (n + 1) ≤ ∑' n : ℕ, (n : ℝ) * μ n := by
  have hs1 : Summable (fun n : ℕ => μ (n + 1)) := (summable_nat_add_iff 1).mpr hs
  have hm1 : Summable (fun n : ℕ => ((n + 1 : ℕ) : ℝ) * μ (n + 1)) :=
    (summable_nat_add_iff 1).mpr hm
  calc ∑' n : ℕ, μ (n + 1) ≤ ∑' n : ℕ, ((n + 1 : ℕ) : ℝ) * μ (n + 1) := by
        refine hs1.tsum_le_tsum (fun n => ?_) hm1
        have : (1 : ℝ) ≤ ((n + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n)
        nlinarith [h0 (n + 1)]
    _ ≤ ∑' n : ℕ, (n : ℝ) * μ n := by
        have h2 := Summable.sum_add_tsum_nat_add 1 hm
        simp only [Finset.sum_range_one, Nat.cast_zero, zero_mul, zero_add] at h2
        exact le_of_eq h2

/-- (b) The upper half with witness `lam = 0`: `pmfTV μ (poissonPMF 0) = 1 - μ 0`. -/
theorem upper (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ) (h1 : ∑' n, μ n = 1) :
    pmfTV μ (poissonPMF 0) = 1 - μ 0 := by
  unfold pmfTV
  have hμ1 : μ 0 ≤ 1 := by
    rw [← h1]; exact hs.le_tsum 0 (fun j _ => h0 j)
  have hterm : ∀ n, |μ n - poissonPMF 0 n| = if n = 0 then 1 - μ 0 else μ n := by
    intro n
    unfold poissonPMF
    rcases n with _ | n
    · simp
      rw [abs_sub_comm, abs_of_nonneg (by linarith)]
    · simp [abs_of_nonneg (h0 _)]
  simp_rw [hterm]
  have hsum : Summable (fun n : ℕ => if n = 0 then 1 - μ 0 else μ n) := by
    rw [← summable_nat_add_iff 1]
    simpa using (summable_nat_add_iff 1).mpr hs
  rw [hsum.tsum_eq_zero_add]
  have htail : ∑' n : ℕ, (if n + 1 = 0 then 1 - μ 0 else μ (n + 1)) = ∑' n : ℕ, μ (n + 1) :=
    tsum_congr (fun n => by simp)
  rw [htail, if_pos rfl]
  have h2 := Summable.sum_add_tsum_nat_add 1 hs
  rw [h1, Finset.sum_range_one] at h2
  linarith

/-- (d1) `1 - exp(-x) ≥ x/2` for `0 ≤ x ≤ 1/2`. -/
theorem d1 (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1 / 2) : x / 2 ≤ 1 - Real.exp (-x) := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := -x) (by rw [abs_neg, abs_of_nonneg h0]; linarith)
  rw [abs_le] at h
  nlinarith [h.1, h.2]

/-- (d2) `exp(-x) (1 + x) ≥ 1 - x^2` for `x ≥ 0`. -/
theorem d2 (x : ℝ) (h0 : 0 ≤ x) : 1 - x ^ 2 ≤ Real.exp (-x) * (1 + x) := by
  have h := Real.add_one_le_exp (-x)
  nlinarith [h, Real.exp_pos (-x)]

/-- THE PROP, as stated, with exactly its own binders. -/
theorem main (μ : ℝ → ℕ → ℝ) (a A : ℝ) : RareClusterPoissonObstructionO μ a A := by
  intro ha hH
  obtain ⟨r0, hr0, H⟩ := hH
  -- `A ≥ 0` from the mean bound at `r = r0`.
  have hA : 0 ≤ A := by
    obtain ⟨h0, _hs, _h1, _hm, hmean, _hH2⟩ := H r0 hr0 le_rfl
    have hnn : 0 ≤ ∑' n : ℕ, (n : ℝ) * μ r0 n :=
      tsum_nonneg (fun n => mul_nonneg (Nat.cast_nonneg n) (h0 n))
    have hr3 : 0 < r0 ^ 3 := pow_pos hr0 3
    by_contra hneg
    push_neg at hneg
    have : A * r0 ^ 3 < 0 := mul_neg_of_neg_of_pos hneg hr3
    linarith
  have hden : 0 < a + 2 * A := by linarith
  refine ⟨min r0 (min (a / 2) (min (1 / (a + 2 * A)) (1 / 2))), ?_, ?_⟩
  · refine lt_min hr0 (lt_min ?_ (lt_min ?_ ?_))
    · linarith
    · exact one_div_pos.mpr hden
    · norm_num
  intro r hr hrle
  have hr_r0 : r ≤ r0 := le_trans hrle (min_le_left _ _)
  have hr_a : r ≤ a / 2 := le_trans hrle (le_trans (min_le_right _ _) (min_le_left _ _))
  have hr_den : r ≤ 1 / (a + 2 * A) :=
    le_trans hrle (le_trans (min_le_right _ _) (le_trans (min_le_right _ _) (min_le_left _ _)))
  have hr_half : r ≤ 1 / 2 :=
    le_trans hrle (le_trans (min_le_right _ _) (le_trans (min_le_right _ _) (min_le_right _ _)))
  obtain ⟨h0, hs, h1, hm, hmean, hH2⟩ := H r hr hr_r0
  have hr3 : 0 < r ^ 3 := pow_pos hr 3
  have hr2 : 0 < r ^ 2 := pow_pos hr 2
  -- Markov: `1 - μ r 0 ≤ A r^3`.
  have hmark : 1 - μ r 0 ≤ A * r ^ 3 := by
    have := markov (μ r) h0 hs hm
    rw [tail1 (μ r) hs h1] at this
    linarith
  constructor
  · intro lam hl
    have hν : HasSum (poissonPMF lam) 1 := poisson_hasSum lam hl
    have hνs : Summable (poissonPMF lam) := hν.summable
    have hν1 : ∑' n : ℕ, poissonPMF lam n = 1 := hν.tsum_eq
    have hexp_pos : 0 < Real.exp (-lam) := Real.exp_pos _
    by_cases hcase : lam ≤ r ^ 2
    · -- Test the event `{n ≥ 2}`.
      have hev := tail_diff_le_pmfTV (μ r) (poissonPMF lam) hs hνs h1 hν1 2
      rw [tsum_tail_sub (μ r) (poissonPMF lam) hs hνs 2, tail2 (μ r) hs h1,
        tail2 (poissonPMF lam) hνs hν1] at hev
      rw [poisson0 lam, poisson1 lam] at hev
      obtain ⟨_, hev2⟩ := abs_le.mp hev
      have hd2 := d2 lam hl
      have hpois : 1 - Real.exp (-lam) - Real.exp (-lam) * lam ≤ lam ^ 2 := by nlinarith
      have hlam2 : lam ^ 2 ≤ r ^ 4 := by
        have := mul_le_mul hcase hcase hl hr2.le
        nlinarith
      have hr4 : r ^ 4 ≤ (a / 2) * r ^ 3 := by
        have : r ^ 4 = r * r ^ 3 := by ring
        rw [this]
        exact mul_le_mul_of_nonneg_right hr_a hr3.le
      linarith
    · -- Test the event `{n ≥ 1}`.
      push_neg at hcase
      have hev := tail_diff_le_pmfTV (μ r) (poissonPMF lam) hs hνs h1 hν1 1
      rw [tsum_tail_sub (μ r) (poissonPMF lam) hs hνs 1, tail1 (μ r) hs h1,
        tail1 (poissonPMF lam) hνs hν1] at hev
      rw [poisson0 lam] at hev
      obtain ⟨hev1, _⟩ := abs_le.mp hev
      have hexp : Real.exp (-lam) ≤ Real.exp (-(r ^ 2)) := Real.exp_le_exp.mpr (by linarith)
      have hr2half : r ^ 2 ≤ 1 / 2 := by
        have := mul_le_mul_of_nonneg_left hr_half hr.le
        nlinarith
      have hd1 : r ^ 2 / 2 ≤ 1 - Real.exp (-(r ^ 2)) := d1 (r ^ 2) hr2.le hr2half
      have hr_den' : (a + 2 * A) * r ≤ 1 := by
        have := (le_div_iff₀ hden).mp hr_den
        linarith
      have key : (a / 2) * r ^ 3 ≤ r ^ 2 / 2 - A * r ^ 3 := by
        have := mul_le_mul_of_nonneg_left hr_den' hr2.le
        nlinarith
      linarith
  · refine ⟨0, le_rfl, ?_⟩
    rw [upper (μ r) h0 hs h1]
    exact hmark

end RNBVerify

#print axioms RNBVerify.main
