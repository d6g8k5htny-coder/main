import W3Scratch.D2SchurLinks
import Mathlib.NumberTheory.ModularForms.JacobiTheta.TwoVariable
import Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation
import Mathlib.RingTheory.Polynomial.Hermite.Gaussian

/-!
# W3Scratch.Side24Law — SCRATCH / ENGINEERING ONLY

Grok Bot agent 13, task W3f (Oct 7, 2026). Not reviewed, not aligned, not a discharge.
A Lean lemma ≠ alignment acceptance ≠ discharge.

Merged sources only, Math- main @ 9fd261135b41daf1e377f9ef2db193fee6ec36be:
* N = reviews/side24_d2_certified_small_L_claude_20261005/NOTE.md, blob 852f7403a8a06a1ef061e363df9502ae16f7a341
  - N L21: `θ_L(t) = Σ_n e^{−(t+Ln)²/2}`, `q_L = θ_L/θ_L(0)`.
  - N L22: one coordinate of the spectral law has raw moments m2, m4, m6.
  - N L53: `θ_L^{(j)}(0) = Σ_n He_j(Ln) e^{−(Ln)²/2}` (image sums).
  - N L54: Poisson-dual law `P(X = 2πk/L) ∝ e^{−(2πk/L)²/2}`.
* C = reviews/side24_d2_certified_small_L_claude_20261005/certified_d2.py, blob 6fd5b751cc4157136ae92225838c12d33dc997f3
  - C L8: same θ_L as N L21.  C L111-121: He_2, He_4, He_6 (probabilists' Hermite).
  - C L137-153 `moments_image`: returns `(-θ2/θ0, θ4/θ0, -θ6/θ0)`, i.e. m_j = (-1)^(j/2) θ^{(j)}(0)/θ(0).
  - C L156-177 `moments_dual`: weights 1 (k = 0) and 2 e^{−(hk)²/2} (k ≥ 1), h = 2π/L; m_j = Σ w ω^j / Σ w.

Readings (flagged): k ranges over ℤ (N L54 gives no range; C sums k = 0 plus 2× k ≥ 1, which is
the symmetric ℤ-sum).  The θ-route moment formula with the sign (-1)^(j/2) is stated in C L153
(`moments_image`) and in iba1 NOTE (reviews/iba1_periodic_jet_claude_20261005/NOTE.md, blob
e36e341f) L32 (m2 = −q2, m4 = q4, m6 = −q6), whose conventions N L24 and C L13 adopt; N does not
write it directly.  Reader 403/6051288028 item A1.
-/

noncomputable section
namespace W3Scratch.Side24
open ResearchFormalCoreR1 W3Scratch Real

/-! ## Generic countable discrete law -/

section Generic
variable {ι : Type*} (w a : ι → ℝ)

/-- Unnormalized moment sum Σ w_i a_i^n. -/
def rawSum (n : ℕ) : ℝ := ∑' i, w i * a i ^ n
/-- Normalized raw moment m_n = Σ w a^n / Σ w. -/
def lawMoment (n : ℕ) : ℝ := rawSum w a n / rawSum w a 0

variable {w a}

theorem residual4_tsum (hs : ∀ n : ℕ, Summable fun i => w i * a i ^ n) (m : ℝ) :
    Summable (fun i => w i * (a i ^ 2 - m) ^ 2) ∧
      ∑' i, w i * (a i ^ 2 - m) ^ 2 = rawSum w a 4 - 2 * m * rawSum w a 2 + m ^ 2 * rawSum w a 0 := by
  have e : (fun i => w i * (a i ^ 2 - m) ^ 2) =
      fun i => (w i * a i ^ 4 - 2 * m * (w i * a i ^ 2)) + m ^ 2 * (w i * a i ^ 0) := by
    funext i; ring
  have h1 : Summable fun i => w i * a i ^ 4 - 2 * m * (w i * a i ^ 2) :=
    (hs 4).sub ((hs 2).mul_left _)
  have h2 : Summable fun i => m ^ 2 * (w i * a i ^ 0) := (hs 0).mul_left _
  refine ⟨e ▸ h1.add h2, ?_⟩
  rw [e, h1.tsum_add h2, (hs 4).tsum_sub ((hs 2).mul_left _), tsum_mul_left, tsum_mul_left]
  rfl

theorem residual6_tsum (hs : ∀ n : ℕ, Summable fun i => w i * a i ^ n) (c : ℝ) :
    Summable (fun i => w i * (a i ^ 3 - c * a i) ^ 2) ∧
      ∑' i, w i * (a i ^ 3 - c * a i) ^ 2 =
        rawSum w a 6 - 2 * c * rawSum w a 4 + c ^ 2 * rawSum w a 2 := by
  have e : (fun i => w i * (a i ^ 3 - c * a i) ^ 2) =
      fun i => (w i * a i ^ 6 - 2 * c * (w i * a i ^ 4)) + c ^ 2 * (w i * a i ^ 2) := by
    funext i; ring
  have h1 : Summable fun i => w i * a i ^ 6 - 2 * c * (w i * a i ^ 4) :=
    (hs 6).sub ((hs 4).mul_left _)
  have h2 : Summable fun i => c ^ 2 * (w i * a i ^ 2) := (hs 2).mul_left _
  refine ⟨e ▸ h1.add h2, ?_⟩
  rw [e, h1.tsum_add h2, (hs 6).tsum_sub ((hs 4).mul_left _), tsum_mul_left, tsum_mul_left]
  rfl

theorem sq_pos_of_ne {x : ℝ} (hx : x ≠ 0) : 0 < x ^ 2 :=
  lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hx))

/-- Countable-support analogue of `finite_law_moment_hyps`: two nonzero atoms with distinct squares
give 0 < m2, m2² < m4 and Δ > 0. -/
theorem countable_law_moment_hyps (hw : ∀ i, 0 ≤ w i)
    (hs : ∀ n : ℕ, Summable fun i => w i * a i ^ n) (i j : ι) (hwi : 0 < w i) (hwj : 0 < w j)
    (hai : a i ≠ 0) (haj : a j ≠ 0) (hij : a i ^ 2 ≠ a j ^ 2) :
    0 < lawMoment w a 2 ∧ lawMoment w a 2 ^ 2 < lawMoment w a 4 ∧
      0 < d2Delta (lawMoment w a 2) (lawMoment w a 4) (lawMoment w a 6) := by
  have hZ : 0 < rawSum w a 0 :=
    (hs 0).tsum_pos (fun k => by rw [pow_zero, mul_one]; exact hw k) i
      (by rw [pow_zero, mul_one]; exact hwi)
  have hS2 : 0 < rawSum w a 2 :=
    (hs 2).tsum_pos (fun k => mul_nonneg (hw k) (sq_nonneg _)) i (mul_pos hwi (sq_pos_of_ne hai))
  -- pick an atom whose square differs from a given value
  have pick : ∀ v : ℝ, ∃ k, 0 < w k ∧ a k ≠ 0 ∧ a k ^ 2 ≠ v := by
    intro v
    by_cases h : a i ^ 2 = v
    · exact ⟨j, hwj, haj, fun h' => hij (h.trans h'.symm)⟩
    · exact ⟨i, hwi, hai, h⟩
  set Z := rawSum w a 0 with hZdef
  set S2 := rawSum w a 2
  set S4 := rawSum w a 4
  set S6 := rawSum w a 6
  -- m4 − m2² > 0
  obtain ⟨k, hwk, -, hk⟩ := pick (S2 / Z)
  obtain ⟨hsum4, heq4⟩ := residual4_tsum hs (S2 / Z)
  have hpos4 : 0 < ∑' i, w i * (a i ^ 2 - S2 / Z) ^ 2 :=
    hsum4.tsum_pos (fun i => mul_nonneg (hw i) (sq_nonneg _)) k
      (mul_pos hwk (sq_pos_of_ne (sub_ne_zero.mpr hk)))
  rw [heq4] at hpos4
  have h4 : 0 < S4 * Z - S2 ^ 2 := by
    have e : S4 - 2 * (S2 / Z) * S2 + (S2 / Z) ^ 2 * Z = (S4 * Z - S2 ^ 2) / Z := by
      field_simp; ring
    rw [e] at hpos4
    exact (div_pos_iff_of_pos_right hZ).mp hpos4
  -- m2 m6 − m4² > 0
  obtain ⟨k', hwk', hak', hk'⟩ := pick (S4 / S2)
  obtain ⟨hsum6, heq6⟩ := residual6_tsum hs (S4 / S2)
  have hpos6 : 0 < ∑' i, w i * (a i ^ 3 - S4 / S2 * a i) ^ 2 := by
    refine hsum6.tsum_pos (fun i => mul_nonneg (hw i) (sq_nonneg _)) k' (mul_pos hwk' ?_)
    have : a k' ^ 3 - S4 / S2 * a k' = a k' * (a k' ^ 2 - S4 / S2) := by ring
    rw [this]
    exact sq_pos_of_ne (mul_ne_zero hak' (sub_ne_zero.mpr hk'))
  rw [heq6] at hpos6
  have h6 : 0 < S6 * S2 - S4 ^ 2 := by
    have e : S6 - 2 * (S4 / S2) * S4 + (S4 / S2) ^ 2 * S2 = (S6 * S2 - S4 ^ 2) / S2 := by
      field_simp; ring
    rw [e] at hpos6
    exact (div_pos_iff_of_pos_right hS2).mp hpos6
  refine ⟨div_pos hS2 hZ, ?_, ?_⟩
  · unfold lawMoment
    rw [div_pow, div_lt_div_iff₀ (by positivity) hZ]
    nlinarith [mul_pos h4 hZ]
  · unfold lawMoment d2Delta
    have e : S6 / Z - (S4 / Z) ^ 2 / (S2 / Z) = (S6 * S2 - S4 ^ 2) / (Z * S2) := by
      field_simp
    rw [e]; exact div_pos h6 (mul_pos hZ hS2)

end Generic

/-! ## The SIDE24 Poisson-dual law (N L54; C L156-177) -/

/-- Atom 2πk/L (N L54). -/
def atom (L : ℝ) (k : ℤ) : ℝ := 2 * π * k / L
/-- Unnormalized weight e^{−(2πk/L)²/2} (N L54). -/
def weight (L : ℝ) (k : ℤ) : ℝ := Real.exp (-(atom L k) ^ 2 / 2)
/-- Normalizer Σ_k e^{−(2πk/L)²/2}. -/
def dualZ (L : ℝ) : ℝ := rawSum (weight L) (atom L) 0
/-- Raw moment m_n(L) of the dual law (C L176-177: w[j]/w[0]). -/
def dualMoment (L : ℝ) (n : ℕ) : ℝ := lawMoment (weight L) (atom L) n

theorem weight_pos (L : ℝ) (k : ℤ) : 0 < weight L k := Real.exp_pos _

/-- Summability of Σ_k w_k · atom_k^n for every n (in particular n = 6: X⁶ integrability). -/
theorem summable_weight_mul_atom_pow {L : ℝ} (hL : L ≠ 0) (n : ℕ) :
    Summable fun k : ℤ => weight L k * atom L k ^ n := by
  have hT : 0 < 2 * π / L ^ 2 := by positivity
  have hb := (summable_pow_mul_jacobiTheta₂_term_bound 0 hT n).mul_left ((2 * π / |L|) ^ n)
  refine Summable.of_norm_bounded hb (fun k => le_of_eq ?_)
  have hexp : -(atom L k) ^ 2 / 2 = -π * (2 * π / L ^ 2 * (k : ℝ) ^ 2 - 2 * 0 * |(k : ℝ)|) := by
    unfold atom; field_simp; ring
  have habs : |atom L k| = 2 * π / |L| * |(k : ℝ)| := by
    unfold atom
    rw [abs_div, abs_mul, abs_mul, abs_two, abs_of_pos pi_pos]
    ring
  rw [Real.norm_eq_abs, abs_mul, abs_pow, weight, abs_of_pos (Real.exp_pos _), habs, hexp,
    mul_pow]
  simp only [Int.cast_abs]
  ring

/-- X⁶ integrability of the dual law, stated as summability. -/
theorem summable_sixth {L : ℝ} (hL : L ≠ 0) : Summable fun k : ℤ => weight L k * atom L k ^ 6 :=
  summable_weight_mul_atom_pow hL 6

theorem dualZ_pos {L : ℝ} (hL : L ≠ 0) : 0 < dualZ L :=
  (summable_weight_mul_atom_pow hL 0).tsum_pos
    (fun k => by rw [pow_zero, mul_one]; exact (weight_pos L k).le) 0
    (by rw [pow_zero, mul_one]; exact weight_pos L 0)

/-- Normalization: the probabilities w_k / Z sum to 1. -/
theorem dual_normalized {L : ℝ} (hL : L ≠ 0) : ∑' k : ℤ, weight L k / dualZ L = 1 := by
  have h := dualZ_pos hL
  rw [tsum_div_const]
  have : ∑' k : ℤ, weight L k = dualZ L := by
    unfold dualZ rawSum; simp only [pow_zero, mul_one]
  rw [this, div_self (ne_of_gt h)]

/-- Symmetry: odd raw moments vanish (atom(−k) = −atom(k), weight even). -/
theorem dual_odd_moment_zero (L : ℝ) (m : ℕ) : rawSum (weight L) (atom L) (2 * m + 1) = 0 := by
  unfold rawSum
  have hneg : ∀ k : ℤ, weight L (-k) * atom L (-k) ^ (2 * m + 1) =
      -(weight L k * atom L k ^ (2 * m + 1)) := by
    intro k
    have ha : atom L (-k) = -atom L k := by unfold atom; push_cast; ring
    rw [weight, weight, ha, neg_sq, Odd.neg_pow ⟨m, rfl⟩]; ring
  have h := tsum_comp_neg (fun k : ℤ => weight L k * atom L k ^ (2 * m + 1))
  simp only [hneg, tsum_neg] at h
  linarith

/-- Moment hypotheses for the SIDE24 dual law at EVERY nonzero period L (atoms k = 1, 2). -/
theorem dual_moment_hyps {L : ℝ} (hL : L ≠ 0) :
    0 < dualMoment L 2 ∧ dualMoment L 2 ^ 2 < dualMoment L 4 ∧
      0 < d2Delta (dualMoment L 2) (dualMoment L 4) (dualMoment L 6) := by
  have h1 : atom L 1 ≠ 0 := by
    unfold atom; push_cast; exact div_ne_zero (by positivity) hL
  have h2 : atom L 2 ≠ 0 := by
    unfold atom; push_cast; exact div_ne_zero (by positivity) hL
  have h12 : atom L 1 ^ 2 ≠ atom L 2 ^ 2 := by
    unfold atom; push_cast
    intro h
    have hpi : (2 * π / L) ^ 2 ≠ 0 := pow_ne_zero 2 (div_ne_zero (by positivity) hL)
    apply hpi
    have : (2 * π * 2 / L) ^ 2 = 4 * (2 * π / L) ^ 2 := by ring
    have : (2 * π * 1 / L) ^ 2 = (2 * π / L) ^ 2 := by ring
    linarith
  exact countable_law_moment_hyps (fun k => (weight_pos L k).le) (summable_weight_mul_atom_pow hL)
    1 2 (weight_pos L 1) (weight_pos L 2) h1 h2 h12

/-- Every-direction positivity for the exact dual-law moments, at every nonzero L.  This is a new
all-L ≠ 0 statement.  N certifies "positive in every direction" (N L59, L78) only numerically, by
interval enclosures, at the six values L ∈ {24, 8, 2π, 4, 3, 2} (N L17, L65-73); this theorem is
not that certificate, and identifying the two needs analytic question (a) of the README. -/
theorem dual_positivity_every_direction {L : ℝ} (hL : L ≠ 0) (c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    0 < d2DetV (dualMoment L 2) (dualMoment L 4) (srcQ c s) ∧
      0 < d2Schur (dualMoment L 2) (dualMoment L 4) (srcQ c s) ∧
      0 < d2Tau (dualMoment L 2) (dualMoment L 4) (dualMoment L 6) (srcQ c s) := by
  obtain ⟨h2, h4, h6⟩ := dual_moment_hyps hL
  exact positivity_every_direction _ _ _ h2 h4 h6 c s hu

/-! ## The θ-series route (N L21, L53; C L137-153) -/

/-- θ_L(t) = Σ_n e^{−(t+Ln)²/2} (N L21). -/
def theta (L t : ℝ) : ℝ := ∑' n : ℤ, Real.exp (-(t + L * n) ^ 2 / 2)
/-- Image sum Σ_n He_j(Ln) e^{−(Ln)²/2} (N L53), probabilists' Hermite `Polynomial.hermite`. -/
def imageSum (L : ℝ) (j : ℕ) : ℝ :=
  ∑' n : ℤ, Polynomial.aeval (L * n) (Polynomial.hermite j) * Real.exp (-(L * n) ^ 2 / 2)
/-- θ-route moment m_j = (−1)^(j/2) θ^{(j)}(0)/θ(0), via image sums (C L153). -/
def imageMoment (L : ℝ) (j : ℕ) : ℝ := (-1) ^ (j / 2) * imageSum L j / imageSum L 0

theorem imageSum_zero (L : ℝ) : imageSum L 0 = theta L 0 := by
  unfold imageSum theta
  simp [Polynomial.hermite_zero]

/-- Poisson link, j = 0 (closed): θ_L(0) = (L²/(2π))^{−1/2} · Σ_k e^{−(2πk/L)²/2}, from
Mathlib's `Real.tsum_exp_neg_mul_int_sq`. -/
theorem theta_zero_eq_dual {L : ℝ} (hL : L ≠ 0) :
    theta L 0 = 1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) * dualZ L := by
  have ha : 0 < L ^ 2 / (2 * π) := by positivity
  have h := Real.tsum_exp_neg_mul_int_sq ha
  have e1 : ∀ n : ℤ, Real.exp (-π * (L ^ 2 / (2 * π)) * (n : ℝ) ^ 2) =
      Real.exp (-((0 : ℝ) + L * n) ^ 2 / 2) := by
    intro n; congr 1; field_simp; ring
  have e2 : ∀ n : ℤ, Real.exp (-π / (L ^ 2 / (2 * π)) * (n : ℝ) ^ 2) = weight L n * atom L n ^ 0 := by
    intro n; rw [pow_zero, mul_one, weight, atom]; congr 1; field_simp
  simp only [e1, e2] at h
  exact h

/-- Step (iii) of the Poisson link, closed: the j-th t-derivative at 0 of one θ-term is
(−1)^j He_j(y) e^{−y²/2} (Mathlib `deriv_gaussian_eq_hermite_mul_gaussian`). -/
theorem theta_term_iteratedDeriv (y : ℝ) (j : ℕ) :
    iteratedDeriv j (fun t => Real.exp (-(t + y) ^ 2 / 2)) 0 =
      (-1 : ℝ) ^ j * Polynomial.aeval y (Polynomial.hermite j) * Real.exp (-y ^ 2 / 2) := by
  have h := congrFun (iteratedDeriv_comp_add_const j (fun z : ℝ => Real.exp (-(z ^ 2 / 2))) y) 0
  simp only [zero_add] at h
  have hf : (fun t : ℝ => Real.exp (-(t + y) ^ 2 / 2)) = fun z => Real.exp (-((z + y) ^ 2 / 2)) := by
    funext t; congr 1; ring
  rw [hf, h, iteratedDeriv_eq_iterate, Polynomial.deriv_gaussian_eq_hermite_mul_gaussian]
  congr 2; ring

/-- N L53 holds termwise: the image sum of even order is the sum of the termwise 2m-th
derivatives of θ_L at 0.  (The exchange of Σ and d/dt, for L ≠ 0, is in `Side24Poisson`:
`iteratedDeriv_theta`, `iteratedDeriv_theta_zero_even`.) -/
theorem imageSum_even_eq_tsum_termwise (L : ℝ) (m : ℕ) :
    imageSum L (2 * m) =
      ∑' n : ℤ, iteratedDeriv (2 * m) (fun t => Real.exp (-(t + L * n) ^ 2 / 2)) 0 := by
  unfold imageSum
  congr 1; funext n
  rw [theta_term_iteratedDeriv, pow_mul, neg_one_sq, one_pow, one_mul]

end W3Scratch.Side24
