import Specifications.RN

/-!
# RN: kernel-checked proof of the abstract rare-cluster Poisson obstruction
# (`RareClusterPoissonObstructionO`, statement module `Specifications.RN`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/RN.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here), applied to its own parameters `μ a A`. A kernel-checked proof of such a
`Prop` is evidence about that `Prop` only: it does not make the informal Layer 0 note
(OA-C6-RARE-CLUSTER-20260929-v1, Theorem O of c6_rare_cluster_laws_20260929/PROOF.md) reviewed,
accepted or "verified", it does not identify the formal objects (`pmfTV`, `poissonPMF`, the
family `μ`) with the source's (that alignment is PENDING_INDEPENDENT_REVIEW), it says nothing
about whether (H1) or (H2) hold for any Gaussian window count, and it promotes, closes,
discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.RNCluster.Aux` (prefix `rc_`). The route is the
source's, with the thresholds rearranged so that no `A^2` appears:
* the antecedents at `r = r0` give `2 a r0^3 ≤ 2 (1 - μ 0 - μ 1) ≤ ∑ n μ n ≤ A r0^3`, hence
  `2a ≤ A` and in particular `A > 0`;
* `rStar := min r0 (min (a/2) (min (1/(a + 2A)) (1/2)))`;
* upper half, witness `lam = 0`: `poissonPMF 0 = δ_0`, so `pmfTV (μ r) (poissonPMF 0) = 1 - μ r 0
  = ∑_{n ≥ 1} μ r n ≤ ∑ n μ r n ≤ A r^3`;
* lower half, `lam ≥ 0`: half the `l^1` distance dominates the difference of the masses of every
  tail event `{n ≥ k}` (for two summable families of total mass one). For `lam ≤ r^2` the tail
  `{n ≥ 2}` gives `pmfTV ≥ (1 - μ 0 - μ 1) - (1 - e^{-lam} - lam e^{-lam}) ≥ a r^3 - lam^2 ≥
  a r^3 - r^4 ≥ (a/2) r^3`; for `lam > r^2` the tail `{n ≥ 1}` gives `pmfTV ≥ (1 - e^{-lam}) -
  (1 - μ 0) ≥ (1 - e^{-r^2}) - A r^3 ≥ r^2/2 - A r^3 ≥ (a/2) r^3`, using `x/2 ≤ 1 - e^{-x}` on
  `[0, 1/2]` and `r (a + 2A) ≤ 1`.
Total mass one of `poissonPMF lam` for `lam ≥ 0` is Mathlib's
`ProbabilityTheory.hasSum_one_poissonMeasure`.
-/

namespace UniversalLaw.Proofs.RNCluster

open UniversalLaw.Spec.RN

namespace Aux

/-! ### (a) Tails of a probability mass function on `ℕ` -/

/-- `∑_{n ≥ 1} μ n = 1 - μ 0` for a summable family of total mass one. -/
theorem rc_tail_one_eq (μ : ℕ → ℝ) (hs : Summable μ) (h1 : ∑' n, μ n = 1) :
    ∑' n : ℕ, μ (n + 1) = 1 - μ 0 := by
  have h := Summable.sum_add_tsum_nat_add 1 hs
  rw [h1, Finset.sum_range_one] at h
  linarith

/-- `∑_{n ≥ 2} μ n = 1 - μ 0 - μ 1` for a summable family of total mass one. -/
theorem rc_tail_two_eq (μ : ℕ → ℝ) (hs : Summable μ) (h1 : ∑' n, μ n = 1) :
    ∑' n : ℕ, μ (n + 2) = 1 - μ 0 - μ 1 := by
  have h := Summable.sum_add_tsum_nat_add 2 hs
  rw [h1] at h
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at h
  linarith

/-- Markov for the mean: `∑_{n ≥ 1} μ n ≤ ∑ n μ n` for a nonnegative family with summable
mean. -/
theorem rc_tail_one_le_mean (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
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

/-- `1 - μ 0 ≤ ∑ n μ n` for a probability mass function with summable mean. -/
theorem rc_one_sub_head_le_mean (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
    (h1 : ∑' n, μ n = 1) (hm : Summable (fun n : ℕ => (n : ℝ) * μ n)) :
    1 - μ 0 ≤ ∑' n : ℕ, (n : ℝ) * μ n := by
  have h := rc_tail_one_le_mean μ h0 hs hm
  rwa [rc_tail_one_eq μ hs h1] at h

/-- `2 (1 - μ 0 - μ 1) ≤ ∑ n μ n`: the mean dominates twice the mass of `{n ≥ 2}` (this is the
source's remark `2a ≤ A`, before dividing by `r^3`). -/
theorem rc_two_mul_tail_two_le_mean (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
    (h1 : ∑' n, μ n = 1) (hm : Summable (fun n : ℕ => (n : ℝ) * μ n)) :
    2 * (1 - μ 0 - μ 1) ≤ ∑' n : ℕ, (n : ℝ) * μ n := by
  have hs2 : Summable (fun n : ℕ => μ (n + 2)) := (summable_nat_add_iff 2).mpr hs
  have hm2 : Summable (fun n : ℕ => ((n + 2 : ℕ) : ℝ) * μ (n + 2)) :=
    (summable_nat_add_iff 2).mpr hm
  have htail : ∑' n : ℕ, μ (n + 2) = 1 - μ 0 - μ 1 := rc_tail_two_eq μ hs h1
  have hsplit := Summable.sum_add_tsum_nat_add 2 hm
  have hhead : ∑ i ∈ Finset.range 2, (i : ℝ) * μ i = μ 1 := by
    simp [Finset.sum_range_succ]
  rw [hhead] at hsplit
  have hle : ∑' n : ℕ, 2 * μ (n + 2) ≤ ∑' n : ℕ, ((n + 2 : ℕ) : ℝ) * μ (n + 2) := by
    refine (hs2.mul_left 2).tsum_le_tsum (fun n => ?_) hm2
    have : (2 : ℝ) ≤ ((n + 2 : ℕ) : ℝ) := by exact_mod_cast Nat.le_add_left 2 n
    exact mul_le_mul_of_nonneg_right this (h0 _)
  rw [tsum_mul_left, htail] at hle
  linarith [h0 1]

/-! ### (b) The Poisson family -/

/-- `poissonPMF lam` has total mass one for `lam ≥ 0` (Mathlib's
`ProbabilityTheory.hasSum_one_poissonMeasure`, whose summand is definitionally
`poissonPMF lam n`). -/
theorem rc_hasSum_poissonPMF (lam : ℝ) (hl : 0 ≤ lam) : HasSum (poissonPMF lam) 1 := by
  have h : HasSum (fun n : ℕ => Real.exp (-lam) * lam ^ n / (Nat.factorial n : ℝ)) 1 :=
    ProbabilityTheory.hasSum_one_poissonMeasure ⟨lam, hl⟩
  exact h

/-- `poissonPMF lam 0 = e^{-lam}`. -/
theorem rc_poissonPMF_zero (lam : ℝ) : poissonPMF lam 0 = Real.exp (-lam) := by
  simp [poissonPMF]

/-- `poissonPMF lam 1 = e^{-lam} lam`. -/
theorem rc_poissonPMF_one (lam : ℝ) : poissonPMF lam 1 = Real.exp (-lam) * lam := by
  simp [poissonPMF]

/-- The Poisson tail `∑_{n ≥ 1} Pois(lam)(n) = 1 - e^{-lam}` for `lam ≥ 0`. -/
theorem rc_poisson_tail_one (lam : ℝ) (hl : 0 ≤ lam) :
    ∑' n : ℕ, poissonPMF lam (n + 1) = 1 - Real.exp (-lam) := by
  have hsum := rc_hasSum_poissonPMF lam hl
  have h := Summable.sum_add_tsum_nat_add 1 hsum.summable
  rw [hsum.tsum_eq, Finset.sum_range_one, rc_poissonPMF_zero] at h
  linarith

/-- The Poisson tail `∑_{n ≥ 2} Pois(lam)(n) = 1 - e^{-lam} - lam e^{-lam}` for `lam ≥ 0`. -/
theorem rc_poisson_tail_two (lam : ℝ) (hl : 0 ≤ lam) :
    ∑' n : ℕ, poissonPMF lam (n + 2) = 1 - Real.exp (-lam) - lam * Real.exp (-lam) := by
  have hsum := rc_hasSum_poissonPMF lam hl
  have h := Summable.sum_add_tsum_nat_add 2 hsum.summable
  rw [hsum.tsum_eq] at h
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at h
  rw [rc_poissonPMF_zero, rc_poissonPMF_one] at h
  linarith

/-- `x/2 ≤ 1 - e^{-x}` for `0 ≤ x ≤ 1/2`, from `|e^y - 1 - y| ≤ y^2` on `|y| ≤ 1`. -/
theorem rc_half_le_one_sub_exp (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1 / 2) :
    x / 2 ≤ 1 - Real.exp (-x) := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := -x) (by rw [abs_neg, abs_of_nonneg h0]; linarith)
  rw [abs_le] at h
  nlinarith [h.1, h.2]

/-- `Pois(lam)(n ≥ 2) = 1 - e^{-lam} - lam e^{-lam} ≤ lam^2` for `lam ≥ 0`, from
`1 + y ≤ e^y` multiplied by `1 + lam ≥ 0`. -/
theorem rc_poisson_tail_two_le_sq (lam : ℝ) (h0 : 0 ≤ lam) :
    1 - Real.exp (-lam) - lam * Real.exp (-lam) ≤ lam ^ 2 := by
  have h := Real.add_one_le_exp (-lam)
  have hprod := mul_nonneg (sub_nonneg.mpr h) (show (0 : ℝ) ≤ 1 + lam by linarith)
  nlinarith [hprod, Real.exp_pos (-lam)]

/-! ### (c) Total variation against `δ_0` and against tail events -/

/-- The upper half's witness: `pmfTV μ (poissonPMF 0) = 1 - μ 0` for a probability mass
function `μ` (`poissonPMF 0 = δ_0`). -/
theorem rc_pmfTV_poissonPMF_zero (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
    (h1 : ∑' n, μ n = 1) : pmfTV μ (poissonPMF 0) = 1 - μ 0 := by
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
  rw [htail, ite_eq_left rfl, rc_tail_one_eq μ hs h1]
  ring

/-- Half the `l^1` distance dominates the difference of the masses of every tail event
`{n ≥ k}`, for two summable families of total mass one: the head and the tail of `μ - ν` sum to
zero, so the tail difference is bounded by the head's `l^1` mass and by its own. -/
theorem rc_abs_tail_sub_le_pmfTV (μ ν : ℕ → ℝ) (hμ : Summable μ) (hν : Summable ν)
    (h1 : ∑' n, μ n = 1) (h1' : ∑' n, ν n = 1) (k : ℕ) :
    |(∑' n : ℕ, μ (n + k)) - ∑' n : ℕ, ν (n + k)| ≤ pmfTV μ ν := by
  have hμk : Summable (fun n : ℕ => μ (n + k)) := (summable_nat_add_iff k).mpr hμ
  have hνk : Summable (fun n : ℕ => ν (n + k)) := (summable_nat_add_iff k).mpr hν
  rw [← hμk.tsum_sub hνk]
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

/-- The tail `{n ≥ 2}` against `Pois(lam)`:
`|(1 - μ 0 - μ 1) - (1 - e^{-lam} - lam e^{-lam})| ≤ pmfTV μ (poissonPMF lam)`. -/
theorem rc_tail_two_bound (μ : ℕ → ℝ) (hs : Summable μ) (h1 : ∑' n, μ n = 1) (lam : ℝ)
    (hl : 0 ≤ lam) :
    |(1 - μ 0 - μ 1) - (1 - Real.exp (-lam) - lam * Real.exp (-lam))| ≤
      pmfTV μ (poissonPMF lam) := by
  have hP := rc_hasSum_poissonPMF lam hl
  have h := rc_abs_tail_sub_le_pmfTV μ (poissonPMF lam) hs hP.summable h1 hP.tsum_eq 2
  rwa [rc_tail_two_eq μ hs h1, rc_poisson_tail_two lam hl] at h

/-- The tail `{n ≥ 1}` against `Pois(lam)`:
`|(1 - μ 0) - (1 - e^{-lam})| ≤ pmfTV μ (poissonPMF lam)`. -/
theorem rc_tail_one_bound (μ : ℕ → ℝ) (hs : Summable μ) (h1 : ∑' n, μ n = 1) (lam : ℝ)
    (hl : 0 ≤ lam) :
    |(1 - μ 0) - (1 - Real.exp (-lam))| ≤ pmfTV μ (poissonPMF lam) := by
  have hP := rc_hasSum_poissonPMF lam hl
  have h := rc_abs_tail_sub_le_pmfTV μ (poissonPMF lam) hs hP.summable h1 hP.tsum_eq 1
  rwa [rc_tail_one_eq μ hs h1, rc_poisson_tail_one lam hl] at h

/-! ### (d) The two halves at a fixed `r` -/

/-- The lower half at a fixed `r` with `0 < r ≤ min (a/2) (min (1/(a + 2A)) (1/2))`, for a
probability mass function `μ` with summable mean satisfying (H1) and (H2) at `r`: every
`lam ≥ 0` is at distance at least `(a/2) r^3`. -/
theorem rc_lower_half (μ : ℕ → ℝ) (a A r : ℝ) (hA : 0 < A) (hr : 0 < r)
    (hr_a : r ≤ a / 2) (hr_A : r ≤ 1 / (a + 2 * A)) (hr_half : r ≤ 1 / 2)
    (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ) (h1 : ∑' n, μ n = 1)
    (hm : Summable (fun n : ℕ => (n : ℝ) * μ n)) (hmean : ∑' n : ℕ, (n : ℝ) * μ n ≤ A * r ^ 3)
    (hH2 : a * r ^ 3 ≤ 1 - μ 0 - μ 1) (lam : ℝ) (hlam : 0 ≤ lam) :
    (a / 2) * r ^ 3 ≤ pmfTV μ (poissonPMF lam) := by
  have hr3 : 0 ≤ r ^ 3 := pow_nonneg hr.le 3
  have hr2 : r ^ 2 ≤ 1 / 2 := by
    have h := pow_le_pow_left₀ hr.le hr_half 2
    norm_num at h
    linarith
  rcases le_or_gt lam (r ^ 2) with hcase | hcase
  · -- `lam ≤ r^2`: test the tail `{n ≥ 2}`
    have hk2 := abs_le.mp (rc_tail_two_bound μ hs h1 lam hlam)
    have hsq := rc_poisson_tail_two_le_sq lam hlam
    have hlam2 : lam ^ 2 ≤ (a / 2) * r ^ 3 := by
      calc lam ^ 2 ≤ (r ^ 2) ^ 2 := pow_le_pow_left₀ hlam hcase 2
        _ = r * r ^ 3 := by ring
        _ ≤ (a / 2) * r ^ 3 := mul_le_mul_of_nonneg_right hr_a hr3
    linarith [hk2.2]
  · -- `lam > r^2`: test the tail `{n ≥ 1}`
    have hk1 := abs_le.mp (rc_tail_one_bound μ hs h1 lam hlam)
    have hexp : Real.exp (-lam) ≤ Real.exp (-(r ^ 2)) :=
      Real.exp_le_exp.mpr (neg_le_neg hcase.le)
    have hhalf := rc_half_le_one_sub_exp (r ^ 2) (sq_nonneg r) hr2
    have hμ0 : 1 - μ 0 ≤ A * r ^ 3 := (rc_one_sub_head_le_mean μ h0 hs h1 hm).trans hmean
    have hrA : r * (a + 2 * A) ≤ 1 := by
      have hpos : 0 < a + 2 * A := by linarith
      exact (le_div_iff₀ hpos).mp hr_A
    have hkey : (a / 2) * r ^ 3 ≤ r ^ 2 / 2 - A * r ^ 3 := by
      have hprod := mul_nonneg (div_nonneg (sq_nonneg r) (show (0 : ℝ) ≤ 2 by norm_num))
        (sub_nonneg.mpr hrA)
      nlinarith [hprod]
    linarith [hk1.1]

/-- The upper half at a fixed `r`, witness `lam = 0`: `pmfTV μ (poissonPMF 0) = 1 - μ 0 ≤
∑ n μ n ≤ A r^3`. -/
theorem rc_upper_half (μ : ℕ → ℝ) (A r : ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
    (h1 : ∑' n, μ n = 1) (hm : Summable (fun n : ℕ => (n : ℝ) * μ n))
    (hmean : ∑' n : ℕ, (n : ℝ) * μ n ≤ A * r ^ 3) :
    pmfTV μ (poissonPMF 0) ≤ A * r ^ 3 := by
  rw [rc_pmfTV_poissonPMF_zero μ h0 hs h1]
  exact (rc_one_sub_head_le_mean μ h0 hs h1 hm).trans hmean

end Aux

/-! ## Theorem O's shape: the abstract implication (H1) ∧ (H2) ⇒ (3.1) -/

/-- Proves: `UniversalLaw.Spec.RN.RareClusterPoissonObstructionO` (claim `OA-C6-RARE-CLUSTER-20260929-v1`).
Method: the antecedents at `r0` give `2a ≤ A` (`Aux.rc_two_mul_tail_two_le_mean`,
`le_of_mul_le_mul_right`), so `rStar := min r0 (min (a/2) (min (1/(a + 2A)) (1/2)))` is positive;
the upper half is the witness `lam = 0` with `pmfTV (μ r) (poissonPMF 0) = 1 - μ r 0 ≤ ∑ n μ r n`
(`Aux.rc_upper_half`, from `Aux.rc_pmfTV_poissonPMF_zero` and `Aux.rc_one_sub_head_le_mean`);
the lower half tests the tail `{n ≥ 2}` when `lam ≤ r^2` and the tail `{n ≥ 1}` when
`lam > r^2` (`Aux.rc_lower_half`, from `Aux.rc_abs_tail_sub_le_pmfTV`, the Poisson tails via
`ProbabilityTheory.hasSum_one_poissonMeasure`, `Real.add_one_le_exp` and
`Real.abs_exp_sub_one_sub_id_le`).
Does not establish: the informal Theorem O (3.1) of the source as the source's statement (the
alignment of `pmfTV`, `poissonPMF` and the family `μ` with the source's `dTV`, `Pois(lambda)` and
`Law N_r` is PENDING_INDEPENDENT_REVIEW); the result (Theorem O) or its proof as a reviewed fact;
`2a ≤ A` as a registered statement (it is derived internally from the antecedents at `r0`, as in
the source, and is neither a hypothesis nor a conclusion of the Prop); that (H1) or (H2) hold for
the Gaussian window count (the source takes them from [D5] Theorem G_d and [C6] Corollary
Theta); the mean-matched statement `lambda = mu`; Theorems C and S, the factorial-moment
matching or the independent-replica limit; the conditional-law liminf remark; that the upper half
is the source's infimum bound verbatim (the Prop's existential form with witness `lam = 0` is
what is proved); the statement module's own `Does not claim:` text (the statement packet is not
edited here); the disposition of `OA-C6-RARE-CLUSTER-20260929-v1`, for which the register row
transcribes none (`disposition_transcribed: null`; title: PROOF_INDEX reviewed scoped result,
conditional packet) and which is in any case transcribed there, not derived here; nonauthor
review or acceptance of OA-C6-RARE-CLUSTER-20260929-v1. -/
theorem rareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) :
    UniversalLaw.Spec.RN.RareClusterPoissonObstructionO μ a A := by
  intro ha hex
  obtain ⟨r0, hr0, H⟩ := hex
  -- the antecedents at `r0` force `2a ≤ A`, hence `A > 0`
  have hA2 : 2 * a ≤ A := by
    obtain ⟨h0, hs, h1, hm, hmean, hH2⟩ := H r0 hr0 le_rfl
    have hmid := Aux.rc_two_mul_tail_two_le_mean (μ r0) h0 hs h1 hm
    have hr3 : 0 < r0 ^ 3 := pow_pos hr0 3
    have hmul : 2 * a * r0 ^ 3 ≤ A * r0 ^ 3 := by linarith
    exact le_of_mul_le_mul_right hmul hr3
  have hA : 0 < A := by linarith
  have hpos : 0 < a + 2 * A := by linarith
  refine ⟨min r0 (min (a / 2) (min (1 / (a + 2 * A)) (1 / 2))), ?_, ?_⟩
  · exact lt_min hr0 (lt_min (by linarith) (lt_min (div_pos one_pos hpos) (by norm_num)))
  · intro r hr hrle
    rw [le_min_iff, le_min_iff, le_min_iff] at hrle
    obtain ⟨hr_r0, hr_a, hr_A, hr_half⟩ := hrle
    obtain ⟨h0, hs, h1, hm, hmean, hH2⟩ := H r hr hr_r0
    refine ⟨?_, ⟨0, le_rfl, ?_⟩⟩
    · intro lam hlam
      exact Aux.rc_lower_half (μ r) a A r hA hr hr_a hr_A hr_half h0 hs h1 hm hmean hH2 lam hlam
    · exact Aux.rc_upper_half (μ r) A r h0 hs h1 hm hmean

end UniversalLaw.Proofs.RNCluster
