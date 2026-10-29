import ResearchFormalCoreR1.D2Schur
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

/-!
# W3Scratch.D2SchurLinks — SCRATCH / ENGINEERING ONLY

Grok Bot agent 13, tasks W3d + W3e (Oct 7, 2026). Not reviewed, not aligned, not a discharge.
A Lean lemma ≠ alignment acceptance ≠ discharge.

Links between `ResearchFormalCoreR1.D2Schur` (raw moments m2, m4, m6; parameter q) and
the source objects of reviews/side24_d2_certified_small_L_claude_20261005/NOTE.md
(blob 852f7403, cumulants k2, k4, k6; direction u = (c, s)) and of Math-#302 comment
6005199919 (body sha256 28f5d504...).

Readings used (stated, not claimed as the author's intent):
* NOTE L23 cumulant formulas are those of a symmetric coordinate law (odd moments zero).
* NOTE L26: u = (c, s) is a unit direction, c² + s² = 1. The NOTE does not write it at L26, but the
  sources support it: C (certified_d2.py, blob 6fd5b751) L12 states u = (cos t, sin t), NOTE L43
  parametrizes u by an angle, and NOTE L27's Σu⁴ = 1 − 2q holds iff c² + s² = 1 (the residual is
  (c² + s² − 1)(c² + s² + 1)). Reader 403/6051288028 item A2.
-/

noncomputable section
namespace W3Scratch
open ResearchFormalCoreR1

/-! ## Source objects, written literally -/

/-- NOTE L23. -/
def k2 (m2 : ℝ) : ℝ := m2
/-- NOTE L23. -/
def k4 (m2 m4 : ℝ) : ℝ := m4 - 3 * m2 ^ 2
/-- NOTE L23. -/
def k6 (m2 m4 m6 : ℝ) : ℝ := m6 - 15 * m4 * m2 + 30 * m2 ^ 3
/-- NOTE L30, in cumulants. -/
def srcAlpha (m2 m4 q : ℝ) : ℝ := 3 * k2 m2 ^ 2 + k4 m2 m4 - 2 * k4 m2 m4 * q
/-- NOTE L30, in cumulants. -/
def srcGamma (m2 m4 q : ℝ) : ℝ := k2 m2 ^ 2 + 2 * k4 m2 m4 * q
/-- NOTE L26. -/
def srcQ (c s : ℝ) : ℝ := c ^ 2 * s ^ 2
/-- NOTE L26. -/
def srcE (c s : ℝ) : ℝ := c * s * (c ^ 2 - s ^ 2)
/-- NOTE L34, first form, with e² = q(1 − 4q) (L28). -/
def srcDet (m2 m4 q : ℝ) : ℝ :=
  srcAlpha m2 m4 q * srcGamma m2 m4 q - k4 m2 m4 ^ 2 * (q * (1 - 4 * q))
/-- NOTE L35, with e² = q(1 − 4q). -/
def srcSchur (m2 m4 q : ℝ) : ℝ :=
  srcAlpha m2 m4 q -
    (srcGamma m2 m4 q ^ 3 + (2 * srcGamma m2 m4 q + srcAlpha m2 m4 q) * k4 m2 m4 ^ 2 *
      (q * (1 - 4 * q))) / srcDet m2 m4 q
/-- NOTE L36, in cumulants (division is Lean's total division). -/
def srcTau (m2 m4 m6 q : ℝ) : ℝ :=
  6 * k2 m2 ^ 3 + 9 * k2 m2 * k4 m2 m4 * (1 - 2 * q) +
    (k6 m2 m4 m6 - k4 m2 m4 ^ 2 / k2 m2) * (1 - 3 * q)
/-- NOTE L43: q(θ) = (1 − cos 4θ)/8. -/
def srcQTheta (θ : ℝ) : ℝ := (1 - Real.cos (4 * θ)) / 8

/-! ## W3d (b): the missing link det V = αγ − k4² e² -/

theorem d2Alpha_eq_src (m2 m4 q : ℝ) : d2Alpha m2 m4 q = srcAlpha m2 m4 q := by
  unfold d2Alpha srcAlpha k2 k4; ring

theorem d2Gamma_eq_src (m2 m4 q : ℝ) : d2Gamma m2 m4 q = srcGamma m2 m4 q := by
  unfold d2Gamma srcGamma k2 k4; ring

theorem srcE_sq (c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    srcE c s ^ 2 = srcQ c s * (1 - 4 * srcQ c s) := by
  unfold srcE srcQ
  have h : (c ^ 2 - s ^ 2) ^ 2 = (c ^ 2 + s ^ 2) ^ 2 - 4 * (c ^ 2 * s ^ 2) := by ring
  rw [mul_pow, h, hu]; ring

theorem d2DetV_eq_alpha_gamma_q (m2 m4 q : ℝ) :
    d2DetV m2 m4 q = d2Alpha m2 m4 q * d2Gamma m2 m4 q - k4 m2 m4 ^ 2 * (q * (1 - 4 * q)) := by
  unfold d2DetV d2Alpha d2Gamma k4; ring

theorem d2DetV_eq_srcDet (m2 m4 q : ℝ) : d2DetV m2 m4 q = srcDet m2 m4 q := by
  unfold srcDet; rw [d2DetV_eq_alpha_gamma_q, d2Alpha_eq_src, d2Gamma_eq_src]

theorem d2DetV_eq_alpha_gamma_e (m2 m4 c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    d2DetV m2 m4 (srcQ c s) =
      srcAlpha m2 m4 (srcQ c s) * srcGamma m2 m4 (srcQ c s) - k4 m2 m4 ^ 2 * srcE c s ^ 2 := by
  rw [srcE_sq c s hu, ← d2Alpha_eq_src, ← d2Gamma_eq_src]
  exact d2DetV_eq_alpha_gamma_q m2 m4 (srcQ c s)

theorem d2DetV_eq_det_varV (m2 m4 c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    d2DetV m2 m4 (srcQ c s) =
      Matrix.det !![srcAlpha m2 m4 (srcQ c s), -(k4 m2 m4 * srcE c s);
                    -(k4 m2 m4 * srcE c s), srcGamma m2 m4 (srcQ c s)] := by
  rw [Matrix.det_fin_two_of, d2DetV_eq_alpha_gamma_e m2 m4 c s hu]; ring

theorem d2DetV_eq_note_linear (m2 m4 q : ℝ) :
    d2DetV m2 m4 q =
      k2 m2 ^ 2 * (3 * k2 m2 ^ 2 + k4 m2 m4) + k4 m2 m4 * (4 * k2 m2 ^ 2 + k4 m2 m4) * q := by
  unfold d2DetV k2 k4; ring

/-- The unit-norm hypothesis cannot be dropped (c = s = 1, m2 = 1, m4 = 2). -/
theorem d2DetV_eq_alpha_gamma_e_needs_unit :
    d2DetV 1 2 (srcQ 1 1) ≠
      srcAlpha 1 2 (srcQ 1 1) * srcGamma 1 2 (srcQ 1 1) - k4 1 2 ^ 2 * srcE 1 1 ^ 2 := by
  unfold d2DetV srcAlpha srcGamma srcQ srcE k2 k4; norm_num

/-! ## W3e (a): sweep lemma — q = c²s² over the unit circle is exactly [0, 1/4] -/

theorem srcQ_mem_Icc (c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) : srcQ c s ∈ Set.Icc (0 : ℝ) (1 / 4) := by
  unfold srcQ
  constructor
  · positivity
  · have h : (c ^ 2 + s ^ 2) ^ 2 - 4 * (c ^ 2 * s ^ 2) = (c ^ 2 - s ^ 2) ^ 2 := by ring
    rw [hu] at h
    nlinarith [sq_nonneg (c ^ 2 - s ^ 2)]

theorem exists_unit_of_mem_Icc (q : ℝ) (hq : q ∈ Set.Icc (0 : ℝ) (1 / 4)) :
    ∃ c s : ℝ, c ^ 2 + s ^ 2 = 1 ∧ srcQ c s = q := by
  obtain ⟨hq0, hq1⟩ := hq
  set r := Real.sqrt (1 - 4 * q) with hr
  have hr0 : 0 ≤ r := Real.sqrt_nonneg _
  have hr2 : r ^ 2 = 1 - 4 * q := Real.sq_sqrt (by linarith)
  have hr1 : r ≤ 1 := by nlinarith
  refine ⟨Real.sqrt ((1 + r) / 2), Real.sqrt ((1 - r) / 2), ?_, ?_⟩
  · rw [Real.sq_sqrt (by linarith), Real.sq_sqrt (by linarith)]; ring
  · unfold srcQ
    rw [Real.sq_sqrt (by linarith), Real.sq_sqrt (by linarith)]
    nlinarith

/-- (a) Sweep lemma as a set equality. -/
theorem unit_q_range :
    {q : ℝ | ∃ c s : ℝ, c ^ 2 + s ^ 2 = 1 ∧ q = srcQ c s} = Set.Icc (0 : ℝ) (1 / 4) := by
  ext q
  constructor
  · rintro ⟨c, s, hu, rfl⟩; exact srcQ_mem_Icc c s hu
  · intro hq
    obtain ⟨c, s, hu, h⟩ := exists_unit_of_mem_Icc q hq
    exact ⟨c, s, hu, h.symm⟩

/-- (a) Same, as an image of the unit circle. -/
theorem unit_q_image :
    (fun p : ℝ × ℝ => srcQ p.1 p.2) '' {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1} =
      Set.Icc (0 : ℝ) (1 / 4) := by
  rw [← unit_q_range]
  ext q
  constructor
  · rintro ⟨⟨c, s⟩, hu, rfl⟩; exact ⟨c, s, hu, rfl⟩
  · rintro ⟨c, s, hu, rfl⟩; exact ⟨⟨c, s⟩, hu, rfl⟩

/-- NOTE L43's angle parameterization agrees with q = c²s² at (c, s) = (cos θ, sin θ). -/
theorem srcQTheta_eq (θ : ℝ) : srcQTheta θ = srcQ (Real.cos θ) (Real.sin θ) := by
  unfold srcQTheta srcQ
  have h4 : Real.cos (4 * θ) = 2 * Real.cos (2 * θ) ^ 2 - 1 := by
    rw [show 4 * θ = 2 * (2 * θ) by ring, Real.cos_two_mul]
  have h2 : Real.cos (2 * θ) = 2 * Real.cos θ ^ 2 - 1 := Real.cos_two_mul θ
  have hs : Real.sin θ ^ 2 = 1 - Real.cos θ ^ 2 := by
    have := Real.sin_sq_add_cos_sq θ; linarith
  rw [h4, h2, hs]; ring

/-- (a) Angle form: the range of q(θ) = (1 − cos 4θ)/8 is exactly [0, 1/4]. -/
theorem srcQTheta_range : Set.range srcQTheta = Set.Icc (0 : ℝ) (1 / 4) := by
  ext q
  constructor
  · rintro ⟨θ, rfl⟩
    rw [srcQTheta_eq]
    exact srcQ_mem_Icc _ _ (by rw [add_comm]; exact Real.sin_sq_add_cos_sq θ)
  · rintro ⟨hq0, hq1⟩
    refine ⟨Real.arccos (1 - 8 * q) / 4, ?_⟩
    unfold srcQTheta
    rw [show 4 * (Real.arccos (1 - 8 * q) / 4) = Real.arccos (1 - 8 * q) by ring,
      Real.cos_arccos (by linarith) (by linarith)]
    ring

/-! ## W3d (c) + W3e (a): the endpoint minimum, over q and over unit directions -/

theorem d2_schur_endpoint_isLeast (m2 m4 : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4) :
    IsLeast (d2Schur m2 m4 '' Set.Icc (0 : ℝ) (1 / 4))
      (d2Numerator m2 m4 / max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4)) := by
  constructor
  · rcases le_total ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) (m2 ^ 2 * m4) with h | h
    · refine ⟨0, ⟨le_refl _, by norm_num⟩, ?_⟩
      have hD := d2_det_pos m2 m4 0 hm2 hm4 (le_refl _) (by norm_num)
      rw [d2_schur_ratio m2 m4 0 (ne_of_gt hD), d2_det_axis, max_eq_left h]
    · refine ⟨1 / 4, ⟨by norm_num, le_refl _⟩, ?_⟩
      have hD := d2_det_pos m2 m4 (1 / 4) hm2 hm4 (by norm_num) (le_refl _)
      rw [d2_schur_ratio m2 m4 (1 / 4) (ne_of_gt hD), d2_det_diagonal, max_eq_right h]
  · rintro _ ⟨q, ⟨hq0, hq1⟩, rfl⟩
    exact d2_schur_endpoint_lower m2 m4 q hm2 hm4 hq0 hq1

theorem d2_schur_endpoint_sInf (m2 m4 : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4) :
    sInf (d2Schur m2 m4 '' Set.Icc (0 : ℝ) (1 / 4)) =
      d2Numerator m2 m4 / max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4) :=
  (d2_schur_endpoint_isLeast m2 m4 hm2 hm4).csInf_eq

/-- (a) The comment's L33 minimum, stated directly over unit directions u = (c, s). -/
theorem d2_schur_min_over_unit_directions (m2 m4 : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4) :
    IsLeast {σ : ℝ | ∃ c s : ℝ, c ^ 2 + s ^ 2 = 1 ∧ σ = d2Schur m2 m4 (srcQ c s)}
      (d2Numerator m2 m4 / max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4)) := by
  have h : {σ : ℝ | ∃ c s : ℝ, c ^ 2 + s ^ 2 = 1 ∧ σ = d2Schur m2 m4 (srcQ c s)} =
      d2Schur m2 m4 '' Set.Icc (0 : ℝ) (1 / 4) := by
    rw [← unit_q_range]
    ext σ
    constructor
    · rintro ⟨c, s, hu, rfl⟩; exact ⟨srcQ c s, ⟨c, s, hu, rfl⟩, rfl⟩
    · rintro ⟨q, ⟨c, s, hu, rfl⟩, rfl⟩; exact ⟨c, s, hu, rfl⟩
  rw [h]; exact d2_schur_endpoint_isLeast m2 m4 hm2 hm4

/-- (a) The same minimum over the NOTE's angle θ ∈ ℝ (q(θ) = (1 − cos 4θ)/8). -/
theorem d2_schur_min_over_angles (m2 m4 : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4) :
    IsLeast (Set.range fun θ : ℝ => d2Schur m2 m4 (srcQTheta θ))
      (d2Numerator m2 m4 / max (m2 ^ 2 * m4) ((m4 - m2 ^ 2) * (m4 + 3 * m2 ^ 2) / 4)) := by
  have h : (Set.range fun θ : ℝ => d2Schur m2 m4 (srcQTheta θ)) =
      d2Schur m2 m4 '' Set.Icc (0 : ℝ) (1 / 4) := by
    rw [← srcQTheta_range, ← Set.range_comp]; rfl
  rw [h]; exact d2_schur_endpoint_isLeast m2 m4 hm2 hm4

/-! ## W3e (b)1: cumulants ↔ raw moments -/

/-- Inverse map (symmetric law): m4 = k4 + 3k2², m6 = k6 + 15 k2 k4 + 15 k2³. -/
theorem moments_of_cumulants (m2 m4 m6 : ℝ) :
    m2 = k2 m2 ∧ m4 = k4 m2 m4 + 3 * k2 m2 ^ 2 ∧
      m6 = k6 m2 m4 m6 + 15 * k2 m2 * k4 m2 m4 + 15 * k2 m2 ^ 3 := by
  unfold k2 k4 k6; refine ⟨rfl, by ring, by ring⟩

/-- Round trip from cumulants: any (K2, K4, K6) is hit by exactly the moments above. -/
theorem cumulants_of_moments (K2 K4 K6 : ℝ) :
    k2 K2 = K2 ∧ k4 K2 (K4 + 3 * K2 ^ 2) = K4 ∧
      k6 K2 (K4 + 3 * K2 ^ 2) (K6 + 15 * K2 * K4 + 15 * K2 ^ 3) = K6 := by
  unfold k2 k4 k6; refine ⟨rfl, by ring, by ring⟩

/-- NOTE L48 Gaussian reference: (k2, k4, k6) = (1, 0, 0) iff (m2, m4, m6) = (1, 3, 15). -/
theorem gaussian_cumulants_iff (m2 m4 m6 : ℝ) :
    (k2 m2 = 1 ∧ k4 m2 m4 = 0 ∧ k6 m2 m4 m6 = 0) ↔ (m2 = 1 ∧ m4 = 3 ∧ m6 = 15) := by
  unfold k2 k4 k6
  constructor
  · rintro ⟨h2, h4, h6⟩
    subst h2
    have h4' : m4 = 3 := by linarith
    subst h4'
    exact ⟨rfl, rfl, by linarith⟩
  · rintro ⟨rfl, rfl, rfl⟩; norm_num

/-- Helper: (m4 − 3m2²)²/m2 = m4²/m2 − 6 m4 m2 + 9 m2³ for ALL real m2 (both sides 0 at m2 = 0). -/
theorem k4_sq_div (m2 m4 : ℝ) :
    k4 m2 m4 ^ 2 / k2 m2 = m4 ^ 2 / m2 - 6 * m4 * m2 + 9 * m2 ^ 3 := by
  unfold k2 k4
  rcases eq_or_ne m2 0 with h | h
  · subst h; simp
  · field_simp; ring

/-- The NOTE's τ² in cumulants equals the Lean `d2Tau` with NO hypothesis on m2
(the existing `d2_tau_cumulant_eq` assumes m2 ≠ 0; it is not needed). -/
theorem d2Tau_eq_srcTau (m2 m4 m6 q : ℝ) : d2Tau m2 m4 m6 q = srcTau m2 m4 m6 q := by
  unfold srcTau
  rw [k4_sq_div]
  unfold d2Tau d2Delta k2 k4 k6
  ring

/-- The NOTE's Σ_{A|V} in cumulants equals the Lean `d2Schur`, all real q, no hypotheses. -/
theorem d2Schur_eq_srcSchur (m2 m4 q : ℝ) : d2Schur m2 m4 q = srcSchur m2 m4 q := by
  unfold d2Schur srcSchur
  rw [← d2DetV_eq_srcDet, ← d2Alpha_eq_src, ← d2Gamma_eq_src]
  unfold k4; ring

/-- Comment 6005199919 L15/L17 right-hand side: x(2x+b)(4x+b), x = m2², b = k4. -/
theorem d2Numerator_eq_comment (m2 m4 : ℝ) :
    d2Numerator m2 m4 = m2 ^ 2 * (2 * m2 ^ 2 + k4 m2 m4) * (4 * m2 ^ 2 + k4 m2 m4) := by
  unfold d2Numerator k4; ring

/-- Comment L29: Δ = m6 − m4²/m2 is literally `d2Delta`. NOTE's k6 − k4²/k2 in moments. -/
theorem src_delta_cumulant (m2 m4 m6 : ℝ) :
    k6 m2 m4 m6 - k4 m2 m4 ^ 2 / k2 m2 = d2Delta m2 m4 m6 - 9 * m4 * m2 + 21 * m2 ^ 3 := by
  rw [k4_sq_div]; unfold k6 d2Delta; ring

/-! ## W3e (b)3: all-q identities specialize to the source's restricted statements -/

/-- Generic: an all-q statement restricts to [0, 1/4] and to every unit direction. -/
theorem restrict_all_q {P : ℝ → Prop} (h : ∀ q, P q) :
    (∀ q ∈ Set.Icc (0 : ℝ) (1 / 4), P q) ∧ (∀ c s : ℝ, c ^ 2 + s ^ 2 = 1 → P (srcQ c s)) :=
  ⟨fun q _ => h q, fun c s _ => h (srcQ c s)⟩

/-- NOTE §2 L34–36 in every unit direction, obtained from the all-q identities by specialization. -/
theorem note_section2_every_direction (m2 m4 m6 c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    srcQ c s ∈ Set.Icc (0 : ℝ) (1 / 4) ∧
    d2DetV m2 m4 (srcQ c s) =
      srcAlpha m2 m4 (srcQ c s) * srcGamma m2 m4 (srcQ c s) - k4 m2 m4 ^ 2 * srcE c s ^ 2 ∧
    d2DetV m2 m4 (srcQ c s) = k2 m2 ^ 2 * (3 * k2 m2 ^ 2 + k4 m2 m4) +
      k4 m2 m4 * (4 * k2 m2 ^ 2 + k4 m2 m4) * srcQ c s ∧
    d2Schur m2 m4 (srcQ c s) = srcSchur m2 m4 (srcQ c s) ∧
    d2Tau m2 m4 m6 (srcQ c s) = srcTau m2 m4 m6 (srcQ c s) ∧
    d2Alpha m2 m4 (srcQ c s) * d2DetV m2 m4 (srcQ c s) - d2Gamma m2 m4 (srcQ c s) ^ 3 -
      (2 * d2Gamma m2 m4 (srcQ c s) + d2Alpha m2 m4 (srcQ c s)) *
        (m4 - 3 * m2 ^ 2) ^ 2 * srcQ c s * (1 - 4 * srcQ c s) = d2Numerator m2 m4 :=
  ⟨srcQ_mem_Icc c s hu, d2DetV_eq_alpha_gamma_e m2 m4 c s hu, d2DetV_eq_note_linear _ _ _,
    d2Schur_eq_srcSchur _ _ _, d2Tau_eq_srcTau _ _ _ _, d2_schur_numerator _ _ _⟩

/-- NOTE L48 Gaussian controls hold on [0, 1/4] (specialization of `d2_gaussian_control`). -/
theorem gaussian_control_on_Icc :
    ∀ q ∈ Set.Icc (0 : ℝ) (1 / 4), d2DetV 1 3 q = 3 ∧ d2Schur 1 3 q = 8 / 3 ∧ d2Tau 1 3 15 q = 6 :=
  (restrict_all_q d2_gaussian_control).1

/-! ## W3e (b)2: positivity in every direction, general form -/

/-- NOTE L59/L78 "positive in every direction", as a general real-variable statement:
any moments with 0 < m2, m2² < m4, 0 < Δ give τ², det V, Σ_{A|V} > 0 for every unit u. -/
theorem positivity_every_direction (m2 m4 m6 : ℝ) (hm2 : 0 < m2) (hm4 : m2 ^ 2 < m4)
    (hΔ : 0 < d2Delta m2 m4 m6) (c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    0 < d2DetV m2 m4 (srcQ c s) ∧ 0 < d2Schur m2 m4 (srcQ c s) ∧
      0 < d2Tau m2 m4 m6 (srcQ c s) := by
  obtain ⟨h0, h1⟩ := srcQ_mem_Icc c s hu
  exact ⟨d2_det_pos _ _ _ hm2 hm4 h0 h1, d2_schur_pos _ _ _ hm2 hm4 h0 h1,
    d2_tau_pos _ _ _ _ hm2 hm4 hΔ h0 h1⟩

/-- Weighted Lagrange identity (finite sums). -/
theorem lagrange_weighted {ι : Type*} [Fintype ι] (p f g : ι → ℝ) :
    2 * ((∑ i, p i * f i ^ 2) * (∑ i, p i * g i ^ 2) - (∑ i, p i * (f i * g i)) ^ 2) =
      ∑ i, ∑ j, p i * p j * (f i * g j - f j * g i) ^ 2 := by
  have e : ∀ i j, p i * p j * (f i * g j - f j * g i) ^ 2 =
      (p i * f i ^ 2) * (p j * g j ^ 2) + (p i * g i ^ 2) * (p j * f j ^ 2) -
        (2 * (p i * (f i * g i))) * (p j * (f j * g j)) := by intro i j; ring
  simp only [e, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
    ← Finset.sum_mul]
  ring

/-- Raw moment of a finite weighted law. -/
def finMoment {ι : Type*} [Fintype ι] (p x : ι → ℝ) (n : ℕ) : ℝ := ∑ i, p i * x i ^ n

theorem double_sum_pos {ι : Type*} [Fintype ι] (F : ι → ι → ℝ) (hF : ∀ i j, 0 ≤ F i j)
    (a b : ι) (hab : 0 < F a b) : 0 < ∑ i, ∑ j, F i j := by
  apply Finset.sum_pos' (fun i _ => Finset.sum_nonneg (fun j _ => hF i j))
  exact ⟨a, Finset.mem_univ _, Finset.sum_pos' (fun j _ => hF a j) ⟨b, Finset.mem_univ _, hab⟩⟩

/-- Moment hypotheses for any finite probability law with two nonzero atoms of distinct squares
(a finite-support analogue of D2MomentBridge's two-atom lemmas; self-contained). -/
theorem finite_law_moment_hyps {ι : Type*} [Fintype ι] (p x : ι → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1) (a b : ι) (hpa : 0 < p a) (hpb : 0 < p b) (ha0 : x a ≠ 0)
    (hb0 : x b ≠ 0) (hab : x a ^ 2 ≠ x b ^ 2) :
    0 < finMoment p x 2 ∧ finMoment p x 2 ^ 2 < finMoment p x 4 ∧
      0 < d2Delta (finMoment p x 2) (finMoment p x 4) (finMoment p x 6) := by
  have hm2 : 0 < finMoment p x 2 := by
    unfold finMoment
    exact Finset.sum_pos' (fun i _ => mul_nonneg (hp i) (sq_nonneg _))
      ⟨a, Finset.mem_univ _, mul_pos hpa (by positivity)⟩
  -- m4 − m2² via f = 1, g = x²
  have L1 := lagrange_weighted p (fun _ => (1 : ℝ)) (fun i => x i ^ 2)
  have hpos1 : 0 < ∑ i, ∑ j, p i * p j * ((fun _ => (1 : ℝ)) i * (fun i => x i ^ 2) j -
      (fun _ => (1 : ℝ)) j * (fun i => x i ^ 2) i) ^ 2 := by
    apply double_sum_pos _ (fun i j => mul_nonneg (mul_nonneg (hp i) (hp j)) (sq_nonneg _)) a b
    have : x b ^ 2 - x a ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
    simp only [one_mul]
    exact mul_pos (mul_pos hpa hpb) (by positivity)
  have hm4 : finMoment p x 2 ^ 2 < finMoment p x 4 := by
    have e1 : (∑ i, p i * (fun _ => (1 : ℝ)) i ^ 2) = 1 := by simpa using hsum
    have e2 : (∑ i, p i * (fun i => x i ^ 2) i ^ 2) = finMoment p x 4 := by
      unfold finMoment; refine Finset.sum_congr rfl (fun i _ => ?_); ring
    have e3 : (∑ i, p i * ((fun _ => (1 : ℝ)) i * (fun i => x i ^ 2) i)) = finMoment p x 2 := by
      unfold finMoment; refine Finset.sum_congr rfl (fun i _ => ?_); ring
    rw [e1, e2, e3] at L1
    linarith
  -- m2 m6 − m4² via f = x, g = x³
  have L2 := lagrange_weighted p x (fun i => x i ^ 3)
  have hpos2 : 0 < ∑ i, ∑ j, p i * p j * (x i * (fun i => x i ^ 3) j -
      x j * (fun i => x i ^ 3) i) ^ 2 := by
    apply double_sum_pos _ (fun i j => mul_nonneg (mul_nonneg (hp i) (hp j)) (sq_nonneg _)) a b
    have h1 : x a * x b ^ 3 - x b * x a ^ 3 = x a * x b * (x b ^ 2 - x a ^ 2) := by ring
    have : x b ^ 2 - x a ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
    simp only []
    rw [h1]
    exact mul_pos (mul_pos hpa hpb) (by positivity)
  have hm6 : finMoment p x 4 ^ 2 < finMoment p x 2 * finMoment p x 6 := by
    have e1 : (∑ i, p i * x i ^ 2) = finMoment p x 2 := rfl
    have e2 : (∑ i, p i * (fun i => x i ^ 3) i ^ 2) = finMoment p x 6 := by
      unfold finMoment; refine Finset.sum_congr rfl (fun i _ => ?_); ring
    have e3 : (∑ i, p i * (x i * (fun i => x i ^ 3) i)) = finMoment p x 4 := by
      unfold finMoment; refine Finset.sum_congr rfl (fun i _ => ?_); ring
    rw [e1, e2, e3] at L2
    linarith
  refine ⟨hm2, hm4, ?_⟩
  unfold d2Delta
  rw [sub_pos, div_lt_iff₀ hm2]
  linarith

/-- (b)2 general statement for finite laws: positivity of τ², det V, Σ_{A|V} in every direction. -/
theorem finite_law_positivity_every_direction {ι : Type*} [Fintype ι] (p x : ι → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i = 1) (a b : ι) (hpa : 0 < p a) (hpb : 0 < p b)
    (ha0 : x a ≠ 0) (hb0 : x b ≠ 0) (hab : x a ^ 2 ≠ x b ^ 2) (c s : ℝ) (hu : c ^ 2 + s ^ 2 = 1) :
    0 < d2DetV (finMoment p x 2) (finMoment p x 4) (srcQ c s) ∧
      0 < d2Schur (finMoment p x 2) (finMoment p x 4) (srcQ c s) ∧
      0 < d2Tau (finMoment p x 2) (finMoment p x 4) (finMoment p x 6) (srcQ c s) := by
  obtain ⟨h2, h4, h6⟩ := finite_law_moment_hyps p x hp hsum a b hpa hpb ha0 hb0 hab
  exact positivity_every_direction _ _ _ h2 h4 h6 c s hu

end W3Scratch
