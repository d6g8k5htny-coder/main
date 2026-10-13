import Specifications.Lifetime

/-! Probe (Lifetime-B_1). Probe file only; nothing here is a theorem of the proof packet.

Purpose: (1) exhibit the constant typed pair laws `p ≡ 0` and `p ≡ 1` as inhabitants of
`PairLawInterface d L G I` for ANY `G`, `I`; (2) show that `ParentUniformSelectionA` is false at
`p ≡ 0`, and that `PlanarElderLowerE2` and `PlanarTwoSidedRateE4` are false at `p ≡ 1`, whenever
their arithmetic hypotheses hold; (3) show `p ≡ 1` satisfies the hypothesis of `PlanarDensityGapE11`
(so that hypothesis cannot constrain the conclusion, which does not mention `PL`); (4) a cheap
witness that the universal closure of `IsUnrestrictedPair` is false (`M = S`). No `sorry`. -/

open UniversalLaw.Spec.Lifetime

noncomputable section

/-- The constant pair law `p ≡ 0`. -/
def plZero (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L) (I : SuperlevelBarInterface d L) :
    PairLawInterface d L G I :=
  { p := fun _ _ _ _ => 0
    p_nonneg := fun _ _ _ _ => le_rfl
    p_le_one := fun _ _ _ _ => zero_le_one }

/-- The constant pair law `p ≡ 1`. -/
def plOne (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L) (I : SuperlevelBarInterface d L) :
    PairLawInterface d L G I :=
  { p := fun _ _ _ _ => 1
    p_nonneg := fun _ _ _ _ => zero_le_one
    p_le_one := fun _ _ _ _ => le_rfl }

/-- For every `C` and `rStar > 0` there is `r ∈ (0, rStar]` with `C r^3 < 1`. -/
lemma exists_small (rStar C : ℝ) (h : 0 < rStar) :
    ∃ r : ℝ, 0 < r ∧ r ≤ rStar ∧ C * r ^ 3 < 1 := by
  have hpos : 0 < 1 / (|C| + 1) := by positivity
  refine ⟨min rStar (1 / (|C| + 1)), lt_min h hpos, min_le_left _ _, ?_⟩
  set r := min rStar (1 / (|C| + 1)) with hr
  have hr0 : 0 < r := lt_min h hpos
  have hr1 : r ≤ 1 / (|C| + 1) := min_le_right _ _
  have hle1 : r ≤ 1 := by
    refine hr1.trans ?_
    rw [div_le_one (by positivity)]
    linarith [abs_nonneg C]
  have h3 : r ^ 3 ≤ r := pow_le_of_le_one hr0.le hle1 (by norm_num)
  calc C * r ^ 3 ≤ |C| * r ^ 3 := mul_le_mul_of_nonneg_right (le_abs_self C) (by positivity)
    _ ≤ |C| * r := mul_le_mul_of_nonneg_left h3 (abs_nonneg C)
    _ ≤ |C| * (1 / (|C| + 1)) := mul_le_mul_of_nonneg_left hr1 (abs_nonneg C)
    _ < 1 := by
        rw [mul_one_div, div_lt_one (by positivity)]
        linarith

/-- (A) is false at `p ≡ 0` whenever its arithmetic hypotheses hold. -/
example (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L) (I : SuperlevelBarInterface d L)
    (bLo bHi kLo kHi : ℝ) (hd : 2 ≤ d) (hL : 0 < L) (hb : bLo ≤ bHi) (hk : 0 < kLo)
    (hkk : kLo ≤ kHi) :
    ¬ ParentUniformSelectionA d L G I (plZero d L G I) bLo bHi kLo kHi := by
  intro h
  obtain ⟨rStar, hr, C, hC⟩ := h hd hL hb hk hkk
  obtain ⟨r, hr0, hrle, hsmall⟩ := exists_small rStar C hr
  have h2 := (hC r hr0 hrle bLo (Set.left_mem_Icc.mpr hb) kLo (Set.left_mem_Icc.mpr hkk)
    (EuclideanSpace.basisFun (Fin d) ℝ)).2
  change (1 : ℝ) - 0 ≤ C * r ^ 3 at h2
  linarith

/-- (E2) is false at `p ≡ 1` whenever its arithmetic hypotheses hold. -/
example (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (bLo bHi kLo kHi : ℝ) (hL : 0 < L) (hb : bLo ≤ bHi) (hk : 0 < kLo) (hkk : kLo ≤ kHi) :
    ¬ PlanarElderLowerE2 L G I (plOne 2 L G I) bLo bHi kLo kHi := by
  intro h
  obtain ⟨c, hc, rStar, hr, hall⟩ := h hL hb hk hkk
  have h1 := hall rStar hr le_rfl bLo (Set.left_mem_Icc.mpr hb) kLo (Set.left_mem_Icc.mpr hkk)
    (EuclideanSpace.basisFun (Fin 2) ℝ)
  change c * rStar ^ 3 ≤ (1 : ℝ) - 1 at h1
  have : 0 < c * rStar ^ 3 := by positivity
  linarith

/-- `p ≡ 1` satisfies the shape (A) with `rStar = 1`, `C = 0` (the hypothesis of E4 and E11). -/
lemma plOne_satisfies_A (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L)
    (I : SuperlevelBarInterface d L) (bLo bHi kLo kHi : ℝ) :
    ParentUniformSelectionA d L G I (plOne d L G I) bLo bHi kLo kHi := by
  intro _ _ _ _ _
  refine ⟨1, one_pos, 0, fun r _ _ b _ k _ R => ?_⟩
  change (0 : ℝ) ≤ 1 - 1 ∧ (1 : ℝ) - 1 ≤ 0 * r ^ 3
  constructor <;> simp

/-- (E4) is false at `p ≡ 1` whenever its arithmetic hypotheses hold. -/
example (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (bLo bHi kLo kHi : ℝ) (hL : 0 < L) (hb : bLo ≤ bHi) (hk : 0 < kLo) (hkk : kLo ≤ kHi) :
    ¬ PlanarTwoSidedRateE4 L G I (plOne 2 L G I) bLo bHi kLo kHi := by
  intro h
  obtain ⟨c, hc, C, rStar, hr, hall⟩ :=
    h (plOne_satisfies_A 2 L G I bLo bHi kLo kHi) hL hb hk hkk
  have h1 := (hall rStar hr le_rfl bLo (Set.left_mem_Icc.mpr hb) kLo (Set.left_mem_Icc.mpr hkk)
    (EuclideanSpace.basisFun (Fin 2) ℝ)).1
  change c * rStar ^ 3 ≤ (1 : ℝ) - 1 at h1
  have : 0 < c * rStar ^ 3 := by positivity
  linarith

/-- The hypothesis of (E11) is satisfied by `p ≡ 1`; the conclusion does not mention `PL`. -/
example (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (bLo bHi kLo kHi : ℝ) :
    ParentUniformSelectionA 2 L G I (plOne 2 L G I) bLo bHi kLo kHi :=
  plOne_satisfies_A 2 L G I bLo bHi kLo kHi

/-- Cheap witness for the predicate vocabulary: the torus distance of a point to itself is
`≤ 0`, so no `(g, M, M)` is an unrestricted pair, for every `d`, `L`, `g`. -/
lemma torusDist_self_le_zero (d : ℕ) (L : ℝ) (M : E d) : torusDist d L M M ≤ 0 := by
  unfold torusDist
  have hbdd : BddBelow (Set.range fun n : Fin d → ℤ => ‖M - M + latticeVec d L n‖) := by
    refine ⟨0, ?_⟩
    rintro _ ⟨n, rfl⟩
    exact norm_nonneg _
  refine (ciInf_le hbdd (0 : Fin d → ℤ)).trans ?_
  have h0 : latticeVec d L (0 : Fin d → ℤ) = 0 := by
    unfold latticeVec
    simp
  simp [h0]

example (d : ℕ) (L : ℝ) (g : E d → ℝ) (M : E d) : ¬ IsUnrestrictedPair d L g M M := by
  intro h
  have := h.1.2.2
  linarith [torusDist_self_le_zero d L M]

#check @EuclideanSpace.basisFun
#check @ProbabilityTheory.gaussianReal
#check @Asymptotics.IsEquivalent
#check @MeasureTheory.ae_eq_of_forall_setIntegral_eq_of_sigmaFinite

end
