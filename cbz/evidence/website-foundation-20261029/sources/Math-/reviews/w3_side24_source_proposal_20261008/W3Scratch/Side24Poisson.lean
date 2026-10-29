import W3Scratch.Side24Law

/-!
# W3Scratch.Side24Poisson — SCRATCH / ENGINEERING ONLY

Grok Bot agent 13, task W3g (Oct 7, 2026). Not reviewed, not aligned, not a discharge.
A Lean lemma ≠ alignment acceptance ≠ discharge.

Extends `W3Scratch.Side24Law` (W3f). Sources as there (Math- main @ 9fd26113, unchanged at
8d70baa7): N = NOTE.md blob 852f7403 (L21, L53, L54), C = certified_d2.py blob 6fd5b751
(L137-153, L156-177).  Reading R1 (k ∈ ℤ) as in W3f.  The sign (-1)^(j/2) is stated in C L153 and
iba1 NOTE (blob e36e341f) L32; c²+s²=1 is supported by C L12 (see W3f and reader 403/6051288028).
N L53 writes θ_L^{(j)}(0) = Σ_n He_j(Ln) e^{−(Ln)²/2} with no sign.  Here it is formalized for even j
(`iteratedDeriv_theta_zero_even`).  For odd j it also holds as written (not formalized here):
the termwise derivative carries (−1)^j (Mathlib `deriv_gaussian_eq_hermite_mul_gaussian`), and
reindexing n → −n with He_j(−x) = (−1)^j He_j(x) removes it; both sides vanish because θ_L is even.
So this is not a source discrepancy (reader 403/6051288028 item A3).  N and C use only
j = 0, 2, 4, 6 (C L144; iba1 NOTE L29).
The W3f commented target `imageMoment_eq_dualMoment` (0 < L) is proved below under L ≠ 0.  (That
stale W3f comment block was removed from Side24Law.lean in the Math-#403 amend.)
-/

noncomputable section
namespace W3Scratch.Side24
open ResearchFormalCoreR1 W3Scratch Real
open scoped ContDiff

/-! ## (a) General-t real θ cosine identity (Poisson summation) -/

/-- θ_L(t) = (L²/(2π))^{-1/2} Σ_k w_k cos(2πkt/L) for every real t, L ≠ 0. -/
theorem theta_eq_cos_series {L : ℝ} (hL : L ≠ 0) (t : ℝ) :
    theta L t = 1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) *
      ∑' k : ℤ, weight L k * Real.cos (atom L k * t) := by
  have ha0 : 0 < L ^ 2 / (2 * π) := by positivity
  set a : ℂ := ((L ^ 2 / (2 * π) : ℝ) : ℂ) with ha_def
  set b : ℂ := ((-(t * L / (2 * π)) : ℝ) : ℂ) with hb_def
  have ha : 0 < a.re := by rw [ha_def, Complex.ofReal_re]; exact ha0
  have h := Complex.tsum_exp_neg_quadratic ha b
  have hL1 : ∀ n : ℤ, Complex.exp (-((π : ℝ) : ℂ) * a * (n : ℂ) ^ 2 + 2 * ((π : ℝ) : ℂ) * b * (n : ℂ))
      = ((Real.exp (t ^ 2 / 2) * Real.exp (-(t + L * n) ^ 2 / 2) : ℝ) : ℂ) := by
    intro n
    rw [Complex.ofReal_mul, Complex.ofReal_exp, Complex.ofReal_exp, ← Complex.exp_add]
    congr 1
    rw [ha_def, hb_def]; push_cast
    field_simp
    ring
  have hR1 : ∀ n : ℤ, Complex.exp (-((π : ℝ) : ℂ) / a * ((n : ℂ) + Complex.I * b) ^ 2)
      = ((Real.exp (t ^ 2 / 2) : ℝ) : ℂ) *
          (((weight L n : ℝ) : ℂ) * Complex.exp (((atom L n * t : ℝ) : ℂ) * Complex.I)) := by
    intro n
    rw [weight, Complex.ofReal_exp, Complex.ofReal_exp, ← Complex.exp_add, ← Complex.exp_add]
    congr 1
    have hLc : (L : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr hL
    have hπc : ((π : ℝ) : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_ne_zero
    rw [ha_def, hb_def, atom]; push_cast
    field_simp
    linear_combination (-(L : ℂ) ^ 2 * (t : ℂ) ^ 2) * Complex.I_sq
  have hS : Summable fun n : ℤ =>
      ((weight L n : ℝ) : ℂ) * Complex.exp (((atom L n * t : ℝ) : ℂ) * Complex.I) := by
    refine Summable.of_norm ?_
    have : (fun n : ℤ => ‖((weight L n : ℝ) : ℂ) * Complex.exp (((atom L n * t : ℝ) : ℂ) * Complex.I)‖)
        = fun n : ℤ => weight L n * atom L n ^ 0 := by
      funext n
      rw [norm_mul, Complex.norm_exp_ofReal_mul_I, Complex.norm_real, Real.norm_eq_abs,
        abs_of_pos (weight_pos L n), pow_zero]
    rw [this]; exact summable_weight_mul_atom_pow hL 0
  have hre : (∑' n : ℤ, ((weight L n : ℝ) : ℂ) *
      Complex.exp (((atom L n * t : ℝ) : ℂ) * Complex.I)).re =
        ∑' k : ℤ, weight L k * Real.cos (atom L k * t) := by
    rw [Complex.re_tsum hS]
    congr 1; funext n
    rw [Complex.re_ofReal_mul, Complex.exp_ofReal_mul_I_re]
  have hcp : (1 : ℂ) / a ^ (1 / 2 : ℂ) = (((1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ)) : ℝ) : ℂ) := by
    rw [Complex.ofReal_div, Complex.ofReal_one, Complex.ofReal_cpow ha0.le, ha_def]
    push_cast; rfl
  have hl : ∑' n : ℤ, Complex.exp (-((π : ℝ) : ℂ) * a * (n : ℂ) ^ 2 + 2 * ((π : ℝ) : ℂ) * b * (n : ℂ))
      = ((Real.exp (t ^ 2 / 2) * theta L t : ℝ) : ℂ) := by
    rw [tsum_congr hL1, ← Complex.ofReal_tsum, tsum_mul_left]; rfl
  have hr : ∑' n : ℤ, Complex.exp (-((π : ℝ) : ℂ) / a * ((n : ℂ) + Complex.I * b) ^ 2)
      = ((Real.exp (t ^ 2 / 2) : ℝ) : ℂ) * ∑' n : ℤ,
          (((weight L n : ℝ) : ℂ) * Complex.exp (((atom L n * t : ℝ) : ℂ) * Complex.I)) := by
    rw [tsum_congr hR1, tsum_mul_left]
  rw [hl, hr, hcp] at h
  have h2 := congrArg Complex.re h
  rw [Complex.ofReal_re, Complex.re_ofReal_mul, Complex.re_ofReal_mul, hre] at h2
  have hE := Real.exp_pos (t ^ 2 / 2)
  exact mul_left_cancel₀ hE.ne' (by rw [h2]; ring)

/-! ## (b) Term-by-term differentiation of both series -/

/-- Standard Gaussian e^{-x²/2}, in the form used by Mathlib's Hermite lemma. -/
def gauss : ℝ → ℝ := fun y => Real.exp (-(y ^ 2 / 2))

theorem contDiff_gauss : ContDiff ℝ ∞ gauss := by unfold gauss; fun_prop

theorem hasDerivAt_iterate_deriv_gauss (j : ℕ) (x : ℝ) :
    HasDerivAt (deriv^[j] gauss) (deriv^[j + 1] gauss x) x := by
  rw [Function.iterate_succ_apply']
  exact ((contDiff_gauss.iterate_deriv j).differentiable (by simp)).differentiableAt.hasDerivAt

/-- Every real polynomial is bounded by B e^{|x|}. -/
theorem poly_exp_bound (p : Polynomial ℤ) :
    ∃ B : ℝ, 0 ≤ B ∧ ∀ x : ℝ, |Polynomial.aeval x p| ≤ B * Real.exp |x| := by
  induction p using Polynomial.induction_on' with
  | add p q hp hq =>
    obtain ⟨B1, hB1, h1⟩ := hp
    obtain ⟨B2, hB2, h2⟩ := hq
    refine ⟨B1 + B2, add_nonneg hB1 hB2, fun x => ?_⟩
    rw [map_add]
    calc |Polynomial.aeval x p + Polynomial.aeval x q|
        ≤ |Polynomial.aeval x p| + |Polynomial.aeval x q| := abs_add_le _ _
      _ ≤ B1 * Real.exp |x| + B2 * Real.exp |x| := add_le_add (h1 x) (h2 x)
      _ = (B1 + B2) * Real.exp |x| := by ring
  | monomial n c =>
    refine ⟨|(c : ℝ)| * n.factorial, by positivity, fun x => ?_⟩
    simp only [Polynomial.aeval_monomial, algebraMap_int_eq, Int.coe_castRingHom]
    rw [abs_mul, abs_pow, mul_assoc]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    have h := Real.pow_div_factorial_le_exp (x := |x|) (abs_nonneg x) n
    rw [div_le_iff₀ (by positivity)] at h
    linarith [mul_comm (Real.exp |x|) (n.factorial : ℝ)]

theorem abs_sub_sq_le {L R t : ℝ} (ht : |t| ≤ R) (n : ℤ) :
    |t + L * n| - (t + L * n) ^ 2 / 2 ≤
      R + -π * (L ^ 2 / (2 * π) * (n : ℝ) ^ 2 - 2 * ((R + 1) * |L| / (2 * π)) * |(n : ℝ)|) := by
  have e : -π * (L ^ 2 / (2 * π) * (n : ℝ) ^ 2 - 2 * ((R + 1) * |L| / (2 * π)) * |(n : ℝ)|)
      = -(L * n) ^ 2 / 2 + (R + 1) * |L * n| := by
    rw [abs_mul]; field_simp; ring
  rw [e]
  set m := L * (n : ℝ)
  have h1 : |t + m| ≤ |t| + |m| := abs_add_le t m
  have h2 : -(t * m) ≤ |t| * |m| := by rw [← abs_mul]; exact neg_le_abs _
  have h3 : |t| * |m| ≤ R * |m| := mul_le_mul_of_nonneg_right ht (abs_nonneg m)
  nlinarith [sq_nonneg t, abs_nonneg m]

/-- Uniform majorant for g^{(j)}(t + Ln) on |t| ≤ R. -/
theorem iterate_deriv_gauss_bound (j : ℕ) : ∃ B : ℝ, 0 ≤ B ∧ ∀ (L R t : ℝ), |t| ≤ R → ∀ n : ℤ,
    ‖deriv^[j] gauss (t + L * n)‖ ≤ B * (Real.exp R *
      Real.exp (-π * (L ^ 2 / (2 * π) * (n : ℝ) ^ 2 - 2 * ((R + 1) * |L| / (2 * π)) * |(n : ℝ)|))) := by
  obtain ⟨B, hB, hb⟩ := poly_exp_bound (Polynomial.hermite j)
  refine ⟨B, hB, fun L R t ht n => ?_⟩
  unfold gauss
  rw [Polynomial.deriv_gaussian_eq_hermite_mul_gaussian, Real.norm_eq_abs, abs_mul, abs_mul,
    abs_pow, abs_neg, abs_one, one_pow, one_mul, abs_of_pos (Real.exp_pos _)]
  calc |Polynomial.aeval (t + L * n) (Polynomial.hermite j)| * Real.exp (-((t + L * n) ^ 2 / 2))
      ≤ B * Real.exp |t + L * n| * Real.exp (-((t + L * n) ^ 2 / 2)) :=
        mul_le_mul_of_nonneg_right (hb _) (Real.exp_pos _).le
    _ = B * Real.exp (|t + L * n| - (t + L * n) ^ 2 / 2) := by
        rw [sub_eq_add_neg, Real.exp_add]; ring
    _ ≤ B * Real.exp (R + -π * (L ^ 2 / (2 * π) * (n : ℝ) ^ 2 -
          2 * ((R + 1) * |L| / (2 * π)) * |(n : ℝ)|)) :=
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (abs_sub_sq_le ht n)) hB
    _ = _ := by rw [Real.exp_add]

theorem summable_majorant {L : ℝ} (hL : L ≠ 0) (B R : ℝ) :
    Summable fun n : ℤ => B * (Real.exp R *
      Real.exp (-π * (L ^ 2 / (2 * π) * (n : ℝ) ^ 2 - 2 * ((R + 1) * |L| / (2 * π)) * |(n : ℝ)|))) := by
  have hT : 0 < L ^ 2 / (2 * π) := by positivity
  have h := summable_pow_mul_jacobiTheta₂_term_bound ((R + 1) * |L| / (2 * π)) hT 0
  simp only [pow_zero, one_mul, Int.cast_abs] at h
  exact (h.mul_left _).mul_left _

/-- Termwise j-th derivative series of θ_L: Σ_n g^{(j)}(t + Ln). -/
def thetaD (L : ℝ) (j : ℕ) (t : ℝ) : ℝ := ∑' n : ℤ, deriv^[j] gauss (t + L * n)

theorem summable_thetaD_terms {L : ℝ} (hL : L ≠ 0) (j : ℕ) (t : ℝ) :
    Summable fun n : ℤ => deriv^[j] gauss (t + L * n) := by
  obtain ⟨B, -, hb⟩ := iterate_deriv_gauss_bound j
  exact Summable.of_norm_bounded (summable_majorant hL B |t|) (fun n => hb L |t| t le_rfl n)

theorem hasDerivAt_thetaD {L : ℝ} (hL : L ≠ 0) (j : ℕ) (y : ℝ) :
    HasDerivAt (thetaD L j) (thetaD L (j + 1) y) y := by
  obtain ⟨B, -, hb⟩ := iterate_deriv_gauss_bound (j + 1)
  show HasDerivAt (fun t => ∑' n : ℤ, deriv^[j] gauss (t + L * n))
    (∑' n : ℤ, deriv^[j + 1] gauss (y + L * n)) y
  refine hasDerivAt_tsum_of_isPreconnected (g := fun (n : ℤ) t => deriv^[j] gauss (t + L * n))
    (g' := fun (n : ℤ) t => deriv^[j + 1] gauss (t + L * n))
    (summable_majorant hL B (|y| + 1)) Metric.isOpen_ball
    (convex_ball y 1).isPreconnected (fun n z _ => ?_) (fun n z hz => ?_)
    (Metric.mem_ball_self one_pos) (summable_thetaD_terms hL j y) (Metric.mem_ball_self one_pos)
  · have h2 := (hasDerivAt_iterate_deriv_gauss j (z + L * n)).comp z
      ((hasDerivAt_id' z).add_const (L * n))
    rw [mul_one] at h2
    exact h2
  · refine hb L (|y| + 1) z ?_ n
    rw [Metric.mem_ball, Real.dist_eq] at hz
    have := abs_sub_abs_le_abs_sub z y
    linarith

/-- Termwise j-th derivative series of the cosine side: Σ_k w_k ω_k^j cos(ω_k t + jπ/2). -/
def cosD (L : ℝ) (j : ℕ) (t : ℝ) : ℝ :=
  ∑' k : ℤ, weight L k * (atom L k ^ j * Real.cos (atom L k * t + j * (π / 2)))

theorem summable_abs_weight_mul_atom_pow {L : ℝ} (hL : L ≠ 0) (j : ℕ) :
    Summable fun k : ℤ => weight L k * |atom L k| ^ j := by
  refine (summable_weight_mul_atom_pow hL j).abs.congr (fun k => ?_)
  rw [abs_mul, abs_of_pos (weight_pos L k), abs_pow]

theorem cosD_term_bound (L : ℝ) (j : ℕ) (k : ℤ) (t : ℝ) :
    ‖weight L k * (atom L k ^ j * Real.cos (atom L k * t + j * (π / 2)))‖ ≤
      weight L k * |atom L k| ^ j := by
  rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_of_pos (weight_pos L k), abs_pow]
  exact mul_le_mul_of_nonneg_left
    (mul_le_of_le_one_right (by positivity) (Real.abs_cos_le_one _)) (weight_pos L k).le

theorem hasDerivAt_cosD {L : ℝ} (hL : L ≠ 0) (j : ℕ) (y : ℝ) :
    HasDerivAt (cosD L j) (cosD L (j + 1) y) y := by
  show HasDerivAt
    (fun t => ∑' k : ℤ, weight L k * (atom L k ^ j * Real.cos (atom L k * t + j * (π / 2))))
    (∑' k : ℤ, weight L k *
      (atom L k ^ (j + 1) * Real.cos (atom L k * y + ((j + 1 : ℕ) : ℝ) * (π / 2)))) y
  refine hasDerivAt_tsum
    (g := fun k t => weight L k * (atom L k ^ j * Real.cos (atom L k * t + j * (π / 2))))
    (g' := fun k t => weight L k *
      (atom L k ^ (j + 1) * Real.cos (atom L k * t + ((j + 1 : ℕ) : ℝ) * (π / 2))))
    (summable_abs_weight_mul_atom_pow hL (j + 1)) (fun k z => ?_)
    (fun k z => cosD_term_bound L (j + 1) k z)
    (Summable.of_norm_bounded (summable_abs_weight_mul_atom_pow hL j)
      (fun k => cosD_term_bound L j k y)) y
  have h := ((((hasDerivAt_id' z).const_mul (atom L k)).add_const ((j : ℝ) * (π / 2))).cos.const_mul
    (atom L k ^ j)).const_mul (weight L k)
  convert h using 1
  have e : ((j + 1 : ℕ) : ℝ) * (π / 2) = (j : ℝ) * (π / 2) + π / 2 := by push_cast; ring
  rw [e, ← add_assoc, Real.cos_add_pi_div_two]
  ring

theorem thetaD_zero (L t : ℝ) : thetaD L 0 t = theta L t := by
  unfold thetaD theta gauss
  congr 1; funext n
  simp only [Function.iterate_zero, id]
  congr 1; ring

theorem cosD_zero (L t : ℝ) : cosD L 0 t = ∑' k : ℤ, weight L k * Real.cos (atom L k * t) := by
  unfold cosD; simp

/-- Term-by-term differentiation, θ side: iteratedDeriv j θ_L = Σ_n g^{(j)}(· + Ln). -/
theorem iteratedDeriv_theta {L : ℝ} (hL : L ≠ 0) (j : ℕ) : iteratedDeriv j (theta L) = thetaD L j := by
  induction j with
  | zero => funext t; rw [iteratedDeriv_zero, thetaD_zero]
  | succ j ih => funext t; rw [iteratedDeriv_succ, ih]; exact (hasDerivAt_thetaD hL j t).deriv

/-- Same, stated as "derivative of the sum = sum of the derivatives". -/
theorem iteratedDeriv_theta_termwise {L : ℝ} (hL : L ≠ 0) (j : ℕ) (t : ℝ) :
    iteratedDeriv j (theta L) t =
      ∑' n : ℤ, iteratedDeriv j (fun s => Real.exp (-(s + L * n) ^ 2 / 2)) t := by
  rw [iteratedDeriv_theta hL]
  unfold thetaD
  congr 1; funext n
  have h := congrFun (iteratedDeriv_comp_add_const j gauss (L * n)) t
  have hf : (fun s : ℝ => Real.exp (-(s + L * n) ^ 2 / 2)) = fun s => gauss (s + L * n) := by
    funext s; unfold gauss; congr 1; ring
  rw [hf, h, iteratedDeriv_eq_iterate]

/-- Term-by-term differentiation, cosine side. -/
theorem iteratedDeriv_cos_series {L : ℝ} (hL : L ≠ 0) (j : ℕ) :
    iteratedDeriv j (fun t => ∑' k : ℤ, weight L k * Real.cos (atom L k * t)) = cosD L j := by
  induction j with
  | zero => funext t; rw [iteratedDeriv_zero, cosD_zero]
  | succ j ih => funext t; rw [iteratedDeriv_succ, ih]; exact (hasDerivAt_cosD hL j t).deriv

/-- Poisson identity for every termwise derivative (all j, all t). -/
theorem thetaD_eq_cosD {L : ℝ} (hL : L ≠ 0) (j : ℕ) (t : ℝ) :
    thetaD L j t = 1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) * cosD L j t := by
  induction j generalizing t with
  | zero => rw [thetaD_zero, cosD_zero]; exact theta_eq_cos_series hL t
  | succ j ih =>
    have h1 := hasDerivAt_thetaD hL j t
    have hf : thetaD L j = fun s => 1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) * cosD L j s := funext ih
    rw [hf] at h1
    exact h1.unique ((hasDerivAt_cosD hL j t).const_mul _)

/-! ## (c) θ-route moments = dual-law moments -/

theorem imageSum_even_eq_thetaD (L : ℝ) (m : ℕ) : imageSum L (2 * m) = thetaD L (2 * m) 0 := by
  unfold imageSum thetaD gauss
  congr 1; funext n
  rw [Polynomial.deriv_gaussian_eq_hermite_mul_gaussian, zero_add, pow_mul, neg_one_sq, one_pow,
    one_mul, neg_div]

/-- N L53 as written, for even order: θ_L^{(2m)}(0) = Σ_n He_{2m}(Ln) e^{−(Ln)²/2}. -/
theorem iteratedDeriv_theta_zero_even {L : ℝ} (hL : L ≠ 0) (m : ℕ) :
    iteratedDeriv (2 * m) (theta L) 0 = imageSum L (2 * m) := by
  rw [iteratedDeriv_theta hL, imageSum_even_eq_thetaD]

theorem cosD_even_zero (L : ℝ) (m : ℕ) :
    cosD L (2 * m) 0 = (-1) ^ m * rawSum (weight L) (atom L) (2 * m) := by
  unfold cosD rawSum
  have hc : Real.cos (((2 * m : ℕ) : ℝ) * (π / 2)) = (-1) ^ m := by
    rw [show ((2 * m : ℕ) : ℝ) * (π / 2) = (m : ℝ) * π by push_cast; ring]
    exact Real.cos_nat_mul_pi m
  simp only [mul_zero, zero_add, hc]
  rw [← tsum_mul_left]; congr 1; funext k; ring

theorem imageSum_even_eq {L : ℝ} (hL : L ≠ 0) (m : ℕ) :
    imageSum L (2 * m) = 1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) *
      ((-1) ^ m * rawSum (weight L) (atom L) (2 * m)) := by
  rw [imageSum_even_eq_thetaD, thetaD_eq_cosD hL, cosD_even_zero]

/-- The link: θ-route moment (N L53 / C L153) = dual-law moment (N L54 / C L156-177), every even
order, every L ≠ 0. -/
theorem imageMoment_even_eq_dualMoment {L : ℝ} (hL : L ≠ 0) (m : ℕ) :
    imageMoment L (2 * m) = dualMoment L (2 * m) := by
  have hA0 : (1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) : ℝ) ≠ 0 := by
    have : 0 < (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) := Real.rpow_pos_of_pos (by positivity) _
    positivity
  have h2m := imageSum_even_eq hL m
  have h0 := imageSum_even_eq hL 0
  rw [mul_zero, pow_zero, one_mul] at h0
  have hs : ((-1 : ℝ) ^ m) ^ 2 = 1 := by rw [← pow_mul, mul_comm, pow_mul, neg_one_sq, one_pow]
  unfold imageMoment dualMoment lawMoment
  rw [show 2 * m / 2 = m by omega, h2m, h0]
  rw [show (-1 : ℝ) ^ m * (1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) *
      ((-1) ^ m * rawSum (weight L) (atom L) (2 * m))) =
      ((-1 : ℝ) ^ m) ^ 2 * (1 / (L ^ 2 / (2 * π)) ^ (1 / 2 : ℝ) *
        rawSum (weight L) (atom L) (2 * m)) by ring, hs, one_mul, mul_div_mul_left _ _ hA0]

/-- j = 2, 4, 6 (the W3f commented target, now with L ≠ 0 instead of 0 < L). -/
theorem imageMoment_eq_dualMoment {L : ℝ} (hL : L ≠ 0) (j : ℕ) (hj : j = 2 ∨ j = 4 ∨ j = 6) :
    imageMoment L j = dualMoment L j := by
  rcases hj with rfl | rfl | rfl
  · exact imageMoment_even_eq_dualMoment hL 1
  · exact imageMoment_even_eq_dualMoment hL 2
  · exact imageMoment_even_eq_dualMoment hL 3

/-- Corollary: the θ-route moments satisfy 0 < m2, m2² < m4, Δ > 0 for every L ≠ 0. -/
theorem image_moment_hyps {L : ℝ} (hL : L ≠ 0) :
    0 < imageMoment L 2 ∧ imageMoment L 2 ^ 2 < imageMoment L 4 ∧
      0 < d2Delta (imageMoment L 2) (imageMoment L 4) (imageMoment L 6) := by
  rw [imageMoment_eq_dualMoment hL 2 (Or.inl rfl), imageMoment_eq_dualMoment hL 4 (Or.inr (Or.inl rfl)),
    imageMoment_eq_dualMoment hL 6 (Or.inr (Or.inr rfl))]
  exact dual_moment_hyps hL

/-- Corollary: every-direction positivity for the θ-route moments, every L ≠ 0 (R3: c²+s²=1). -/
theorem image_positivity_every_direction {L : ℝ} (hL : L ≠ 0) (c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    0 < d2DetV (imageMoment L 2) (imageMoment L 4) (srcQ c s) ∧
      0 < d2Schur (imageMoment L 2) (imageMoment L 4) (srcQ c s) ∧
      0 < d2Tau (imageMoment L 2) (imageMoment L 4) (imageMoment L 6) (srcQ c s) := by
  rw [imageMoment_eq_dualMoment hL 2 (Or.inl rfl), imageMoment_eq_dualMoment hL 4 (Or.inr (Or.inl rfl)),
    imageMoment_eq_dualMoment hL 6 (Or.inr (Or.inr rfl))]
  exact dual_positivity_every_direction hL c s hu

end W3Scratch.Side24
