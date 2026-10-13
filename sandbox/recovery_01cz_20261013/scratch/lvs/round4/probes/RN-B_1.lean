import Specifications.RN

/-! RN-B probe 1 (triage only; `sorry` permitted here and nowhere else).
Purpose: confirm, at the pinned Mathlib, the names a proof of
`RareClusterPoissonObstructionO` would use, and elaborate the elementary steps of the
source's Theorem O route (c6-rare-cluster PROOF.md lines 57-66):
(a) `poissonPMF 0` is the point mass at `0`;
(b) `pmfTV μ (poissonPMF 0) = 1 - μ 0` for a pmf `μ` (the upper half, witness `lam = 0`);
(c) `P(N ≥ 1) ≤ E N` (Markov for the mean, so `1 - μ 0 ≤ A r^3`);
(d) `Pois(lam)(≥ 2) ≤ lam^2` for `0 ≤ lam ≤ 1` and `1 - exp(-x) ≥ x/2` for `0 ≤ x ≤ 1/2`;
(e) the half-`l^1` distance dominates the difference of the masses of `{n ≥ 2}`.
Nothing here is a theorem of the packet. -/

open UniversalLaw.Spec.RN

#check @Summable.sum_add_tsum_nat_add'
#check @Summable.sum_add_tsum_nat_add
#check @sum_add_tsum_nat_add'
#check @Summable.tsum_eq_zero_add
#check @Summable.le_tsum
#check @Summable.tsum_le_tsum
#check @tsum_nonneg
#check @Summable.tsum_sub
#check @Summable.tsum_add
#check @Summable.sub
#check @Summable.add
#check @Summable.abs
#check @summable_abs_iff
#check @Summable.norm
#check @norm_tsum_le_tsum_norm
#check @Summable.of_nonneg_of_le
#check @Summable.tsum_mul_left
#check @Real.abs_exp_sub_one_sub_id_le
#check @Real.add_one_le_exp
#check @NormedSpace.expSeries_div_hasSum_exp
#check @Real.exp_eq_exp_ℝ
#check @ProbabilityTheory.hasSum_one_poissonMeasure
#check @Real.summable_pow_div_factorial
#check @HasSum.tsum_eq
#check @Summable.comp_injective
#check @summable_nat_add_iff

/-- (a) `Pois(0) = δ_0`. -/
example (n : ℕ) : poissonPMF 0 n = if n = 0 then 1 else 0 := by
  unfold poissonPMF
  rcases n with _ | n
  · simp
  · simp

/-- (b) The upper half with witness `lam = 0`: `pmfTV μ (poissonPMF 0) = 1 - μ 0`. -/
example (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ) (h1 : ∑' n, μ n = 1) :
    pmfTV μ (poissonPMF 0) = 1 - μ 0 := by
  unfold pmfTV
  have hμ1 : μ 0 ≤ 1 := by
    rw [← h1]; exact hs.le_tsum 0 (fun j _ => h0 j)
  have hterm : ∀ n, |μ n - poissonPMF 0 n| = if n = 0 then 1 - μ 0 else μ n := by
    intro n
    unfold poissonPMF
    rcases n with _ | n
    · simp [abs_of_nonneg (sub_nonneg.mpr hμ1), abs_sub_comm]
    · simp [abs_of_nonneg (h0 _)]
  simp_rw [hterm]
  have hsum : Summable (fun n : ℕ => if n = 0 then 1 - μ 0 else μ n) := by
    refine (summable_nat_add_iff 1).mp ?_
    simpa using (summable_nat_add_iff 1).mpr hs
  rw [hsum.tsum_eq_zero_add]
  have htail : ∑' n : ℕ, (if n + 1 = 0 then 1 - μ 0 else μ (n + 1)) = ∑' n : ℕ, μ (n + 1) := by
    congr 1; funext n; simp
  rw [htail]
  have h2 := hs.sum_add_tsum_nat_add' (k := 1) (by simpa using (summable_nat_add_iff 1).mpr hs)
  simp only [Finset.sum_range_one] at h2
  rw [h1] at h2
  simp only [if_true]
  linarith

/-- (c) Markov for the mean: `∑_{n ≥ 1} μ n ≤ ∑ n μ n`. -/
example (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
    (hm : Summable (fun n : ℕ => (n : ℝ) * μ n)) :
    ∑' n : ℕ, μ (n + 1) ≤ ∑' n : ℕ, (n : ℝ) * μ n := by
  have hs1 : Summable (fun n : ℕ => μ (n + 1)) := (summable_nat_add_iff 1).mpr hs
  have hm1 : Summable (fun n : ℕ => ((n + 1 : ℕ) : ℝ) * μ (n + 1)) := (summable_nat_add_iff 1).mpr hm
  calc ∑' n : ℕ, μ (n + 1) ≤ ∑' n : ℕ, ((n + 1 : ℕ) : ℝ) * μ (n + 1) := by
        refine hs1.tsum_le_tsum (fun n => ?_) hm1
        have : (1 : ℝ) ≤ ((n + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n)
        nlinarith [h0 (n + 1)]
    _ ≤ ∑' n : ℕ, (n : ℝ) * μ n := by
        rw [← hm.sum_add_tsum_nat_add' (k := 1) hm1]
        simp [h0]

/-- (d1) `1 - exp(-x) ≥ x/2` for `0 ≤ x ≤ 1/2`, from `|exp y - 1 - y| ≤ y^2` on `|y| ≤ 1`. -/
example (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1 / 2) : x / 2 ≤ 1 - Real.exp (-x) := by
  have h := Real.abs_exp_sub_one_sub_id_le (x := -x) (by rw [abs_neg, abs_of_nonneg h0]; linarith)
  rw [abs_le] at h
  nlinarith [h.1, h.2]

/-- (d2) `exp(-x) (1 + x) ≥ 1 - x^2` for `0 ≤ x ≤ 1`, from `1 + y ≤ exp y`. -/
example (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 1) : 1 - x ^ 2 ≤ Real.exp (-x) * (1 + x) := by
  have h := Real.add_one_le_exp (-x)
  nlinarith [h, Real.exp_pos (-x)]

/-- (d3) The Poisson tail `∑_{n ≥ 2} Pois(lam)(n) = 1 - exp(-lam) - lam exp(-lam)` for `lam ≥ 0`. -/
example (lam : ℝ) (hl : 0 ≤ lam) :
    ∑' n : ℕ, poissonPMF lam (n + 2) = 1 - Real.exp (-lam) - lam * Real.exp (-lam) := by
  have hsum : HasSum (fun n : ℕ => poissonPMF lam n) 1 := by
    have := ProbabilityTheory.hasSum_one_poissonMeasure ⟨lam, hl⟩
    simpa [poissonPMF] using this
  have h2 := hsum.summable.sum_add_tsum_nat_add' (k := 2) ((summable_nat_add_iff 2).mpr hsum.summable)
  rw [hsum.tsum_eq] at h2
  simp [Finset.sum_range_succ, poissonPMF] at h2
  linarith

/-- (e) Half the `l^1` distance dominates the difference of the masses of `{n ≥ 2}`, for two
summable families of total mass one. Skeleton only (`sorry`): the route is
`sum_add_tsum_nat_add'` with `k = 2` on `μ - ν` and on `|μ - ν|`, then `norm_tsum_le_tsum_norm`
on the tail and the triangle inequality on the two head terms. -/
example (μ ν : ℕ → ℝ) (hμ : Summable μ) (hν : Summable ν)
    (h1 : ∑' n, μ n = 1) (h1' : ∑' n, ν n = 1) :
    |∑' n : ℕ, (μ (n + 2) - ν (n + 2))| ≤ (1 / 2) * ∑' n : ℕ, |μ n - ν n| := by
  sorry
