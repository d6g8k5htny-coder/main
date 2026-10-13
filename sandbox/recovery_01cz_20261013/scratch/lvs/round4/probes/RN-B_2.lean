import Specifications.RN

/-! RN-B probe 2 (triage only; `sorry` permitted here and nowhere else).
Re-run of the three steps of probe 1 that failed on probe engineering (lemma argument shape,
simp normal form, NNReal coercion), plus a full proof of the event-difference step (e):
(b) `pmfTV μ (poissonPMF 0) = 1 - μ 0` for a pmf `μ`;
(c) `∑_{n ≥ 1} μ n ≤ ∑ n μ n` (Markov for the mean);
(d3) `∑_{n ≥ 2} Pois(lam)(n) = 1 - exp(-lam) - lam exp(-lam)` for `lam ≥ 0`;
(e) `|∑_{n ≥ k} (μ n - ν n)| ≤ (1/2) ∑ |μ n - ν n|` for summable `μ, ν` of total mass one.
Nothing here is a theorem of the packet. -/

open UniversalLaw.Spec.RN

#check @Finset.abs_sum_le_sum_abs
#check @abs_sum_le_sum_abs

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

/-- (c) Markov for the mean: `∑_{n ≥ 1} μ n ≤ ∑ n μ n`. -/
example (μ : ℕ → ℝ) (h0 : ∀ n, 0 ≤ μ n) (hs : Summable μ)
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

/-- (d3) The Poisson tail `∑_{n ≥ 2} Pois(lam)(n) = 1 - exp(-lam) - lam exp(-lam)` for `lam ≥ 0`. -/
example (lam : ℝ) (hl : 0 ≤ lam) :
    ∑' n : ℕ, poissonPMF lam (n + 2) = 1 - Real.exp (-lam) - lam * Real.exp (-lam) := by
  have hsum : HasSum (fun n : ℕ => poissonPMF lam n) 1 := by
    have h : HasSum (fun n : ℕ => Real.exp (-lam) * lam ^ n / (Nat.factorial n : ℝ)) 1 :=
      ProbabilityTheory.hasSum_one_poissonMeasure ⟨lam, hl⟩
    exact h
  have h2 := Summable.sum_add_tsum_nat_add 2 hsum.summable
  rw [hsum.tsum_eq] at h2
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add] at h2
  have e0 : poissonPMF lam 0 = Real.exp (-lam) := by simp [poissonPMF]
  have e1 : poissonPMF lam 1 = Real.exp (-lam) * lam := by simp [poissonPMF]
  rw [e0, e1] at h2
  linarith

/-- (e) Half the `l^1` distance dominates the mass difference of every tail event `{n ≥ k}`,
for two summable families of total mass one. -/
example (μ ν : ℕ → ℝ) (hμ : Summable μ) (hν : Summable ν)
    (h1 : ∑' n, μ n = 1) (h1' : ∑' n, ν n = 1) (k : ℕ) :
    |∑' n : ℕ, (μ (n + k) - ν (n + k))| ≤ (1 / 2) * ∑' n : ℕ, |μ n - ν n| := by
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
