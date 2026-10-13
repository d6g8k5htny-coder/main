import CapFirstExit
open Set CapFirstExit

/-- A convex function with equal values at `a < c` has derivative `≥ 0` to the right of `c` and
`≤ 0` to the left of `a`. -/
theorem convex_deriv_sign {φ : ℝ → ℝ} {D : Set ℝ} {a c x d : ℝ} (hφ : ConvexOn ℝ D φ)
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (heq : φ a = φ c) (hx : x ∈ D)
    (hd : HasDerivAt φ d x) : (c ≤ x → 0 ≤ d) ∧ (x ≤ a → d ≤ 0) := by
  have h0 : (φ c - φ a) / (c - a) = 0 := by rw [heq, sub_self, zero_div]
  constructor
  · intro hcx
    have hax : a < x := hac.trans_le hcx
    have h1 : (φ c - φ a) / (c - a) ≤ (φ x - φ a) / (x - a) :=
      hφ.secant_mono haD hcD hx hac.ne' hax.ne' hcx
    have h2 := hφ.slope_le_of_hasDerivAt haD hx hax hd
    rw [slope_def_field] at h2
    linarith
  · intro hxa
    rcases hxa.eq_or_lt with h | h
    · subst h
      have h2 := hφ.le_slope_of_hasDerivAt hx hcD hac hd
      rw [slope_def_field] at h2
      linarith
    · have h1 : (φ a - φ x) / (a - x) ≤ (φ c - φ a) / (c - a) :=
        hφ.slope_mono_adjacent hx hcD h hac
      have h2 := hφ.le_slope_of_hasDerivAt hx haD h hd
      rw [slope_def_field] at h2
      linarith

/-- **L6 (scalar).** If `u + (m/2)t²` is convex and `u - (m/2)t²` concave on `D` (weakly
`|u''| ≤ m`), `u(∓r/2) = 0` and `u` has derivative `d` at `x ∈ D`, then `|d| ≤ m · max(|x|, r/2)`.
Outside the pins the chord through them has slope `0`; between them the chord to the nearer pin and
L5's bound on `u(x)` give `mr/2`. -/
theorem two_node_deriv_bound {u : ℝ → ℝ} {D : Set ℝ} {r m x d : ℝ} (hr : 0 < r) (hm : 0 ≤ m)
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ D (fun t => u t - m / 2 * t ^ 2))
    (haD : -r / 2 ∈ D) (hcD : r / 2 ∈ D) (ha : u (-r / 2) = 0) (hc : u (r / 2) = 0)
    (hx : x ∈ D) (hd : HasDerivAt u d x) : |d| ≤ m * max |x| (r / 2) := by
  have hac : -r / 2 < r / 2 := by linarith
  have hq : HasDerivAt (fun t : ℝ => m / 2 * t ^ 2) (m * x) x := by
    convert (hasDerivAt_pow 2 x).const_mul (m / 2) using 1
    push_cast
    ring
  have hg : HasDerivAt (fun t => u t + m / 2 * t ^ 2) (d + m * x) x := hd.add hq
  have hcv' : ConvexOn ℝ D (fun t => -u t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hk : HasDerivAt (fun t => -u t + m / 2 * t ^ 2) (-d + m * x) x := hd.neg.add hq
  have hgeq : (fun t => u t + m / 2 * t ^ 2) (-r / 2) = (fun t => u t + m / 2 * t ^ 2) (r / 2) := by
    simp only [ha, hc]; ring
  have hkeq : (fun t => -u t + m / 2 * t ^ 2) (-r / 2) =
      (fun t => -u t + m / 2 * t ^ 2) (r / 2) := by
    simp only [ha, hc]; ring
  obtain ⟨g1, g2⟩ := convex_deriv_sign hcv haD hcD hac hgeq hx hg
  obtain ⟨k1, k2⟩ := convex_deriv_sign hcv' haD hcD hac hkeq hx hk
  rcases le_or_gt (r / 2) x with hcx | hxc
  · have h1 := g1 hcx
    have h2 := k1 hcx
    have hxpos : 0 ≤ x := by linarith
    rw [abs_of_nonneg hxpos]
    calc |d| ≤ m * x := abs_le.mpr ⟨by linarith, by linarith⟩
      _ ≤ m * max x (r / 2) := mul_le_mul_of_nonneg_left (le_max_left _ _) hm
  rcases le_or_gt x (-r / 2) with hxa | hax
  · have h1 := g2 hxa
    have h2 := k2 hxa
    have hxneg : x ≤ 0 := by linarith
    rw [abs_of_nonpos hxneg]
    calc |d| ≤ m * -x := abs_le.mpr ⟨by linarith, by linarith⟩
      _ ≤ m * max (-x) (r / 2) := mul_le_mul_of_nonneg_left (le_max_left _ _) hm
  · -- between the pins
    have hu := two_node_bound hcv hcc haD hcD hac ha hc hx
    have hP : (x - -r / 2) * (x - r / 2) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonpos hP] at hu
    have hcx : 0 < r / 2 - x := by linarith
    have hs1 := hcv.le_slope_of_hasDerivAt hx hcD hxc hg
    have hs2 := hcv'.le_slope_of_hasDerivAt hx hcD hxc hk
    rw [slope_def_field, le_div_iff₀ hcx] at hs1 hs2
    simp only [hc] at hs1 hs2
    have hu1 := (abs_le.mp hu).1
    have hu2 := (abs_le.mp hu).2
    have key1 : d * (r / 2 - x) ≤ m / 2 * r * (r / 2 - x) := by nlinarith
    have key2 : -d * (r / 2 - x) ≤ m / 2 * r * (r / 2 - x) := by nlinarith
    have hd1 := le_of_mul_le_mul_right key1 hcx
    have hd2 := le_of_mul_le_mul_right key2 hcx
    calc |d| ≤ m / 2 * r := abs_le.mpr ⟨by linarith, hd1⟩
      _ = m * (r / 2) := by ring
      _ ≤ m * max |x| (r / 2) := mul_le_mul_of_nonneg_left (le_max_right _ _) hm

section V
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
open Filter Topology

/-- **L6 (CAP (7)).** For the slice derivatives `w t` at `y = 0` with derivative `w' x` at `x`, under
L5's hypotheses: `‖w' x‖ ≤ m · max(|x|, r/2)`; on `[-2r, 2r]` this is SP's `‖w'‖ ≤ 2mr`. -/
theorem slice_gradient_deriv_bound {w : ℝ → E →L[ℝ] ℝ} {w' : E →L[ℝ] ℝ} {D : Set ℝ} {r m x : ℝ}
    (hr : 0 < r) (hm : 0 ≤ m)
    (hcv : ∀ v : E, ConvexOn ℝ D (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hcc : ∀ v : E, ConcaveOn ℝ D (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (haD : -r / 2 ∈ D) (hcD : r / 2 ∈ D) (ha : w (-r / 2) = 0) (hc : w (r / 2) = 0)
    (hx : x ∈ D) (hd : HasDerivAt w w' x) : ‖w'‖ ≤ m * max |x| (r / 2) := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have hdv : HasDerivAt (fun t => w t v) (w' v) x := by
    simpa using hd.clm_apply (hasDerivAt_const x v)
  have h := two_node_deriv_bound (u := fun t => w t v) (m := m * ‖v‖) hr (by positivity) (hcv v)
    (hcc v) haD hcD (by simp [ha]) (by simp [hc]) hx hdv
  rw [Real.norm_eq_abs]
  calc |w' v| ≤ m * ‖v‖ * max |x| (r / 2) := h
    _ = m * max |x| (r / 2) * ‖v‖ := by ring

/-- **L9 and the Schur maximum.** Let `Q` be a bilinear form on `ℝ × E` with `δ`-negative definite
transverse part, and let `η` satisfy the differentiated ridge equation `Q (1, η) (0, ·) = 0`. Then
`δ ‖η‖ ≤ ‖Q (1, 0) (0, ·)‖` (L9: `‖h'‖ ≤ ‖∂_x ∇_y f‖ / δ`), `Q (1, η) (1, η) = Q (1, η) (1, 0)`, and,
if `Q` is symmetric, `Q (1, ζ) (1, ζ) ≤ Q (1, η) (1, η)` for every `ζ`: the Schur complement
`F' = f_xx + f_xy[h']` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]`. -/
theorem schur_max {Q : ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ} {δ : ℝ} {η : E} (hδ : 0 < δ)
    (hneg : ∀ v : E, Q (0, v) (0, v) ≤ -δ * ‖v‖ ^ 2)
    (hridge : (Q (1, η)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    δ * ‖η‖ ≤ ‖(Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ∧
      Q (1, η) (1, η) = Q (1, η) (1, 0) ∧
      ((∀ v w, Q v w = Q w v) → ∀ ζ : E, Q (1, ζ) (1, ζ) ≤ Q (1, η) (1, η)) := by
  have h0 : ∀ v : E, Q (1, η) (0, v) = 0 := fun v => by
    simpa using congrArg (fun L : E →L[ℝ] ℝ => L v) hridge
  have hsplit : ∀ ζ : E, ((1 : ℝ), ζ) = ((1 : ℝ), (0 : E)) + ((0 : ℝ), ζ) := fun ζ => by simp
  refine ⟨?_, ?_, ?_⟩
  · have h1 : Q (1, 0) (0, η) + Q (0, η) (0, η) = 0 := by
      have := h0 η
      rw [hsplit η, map_add, add_apply] at this
      exact this
    have h2 : Q (1, 0) (0, η) ≤ ‖(Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ * ‖η‖ := by
      have := ((Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)).le_opNorm η
      have h3 : ((Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) η = Q (1, 0) (0, η) := by simp
      rw [h3] at this
      exact (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using this)
    have h4 := hneg η
    have h5 : δ * ‖η‖ * ‖η‖ ≤ ‖(Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ * ‖η‖ := by
      nlinarith
    rcases (norm_nonneg η).eq_or_lt with h | h
    · rw [← h, mul_zero]; exact norm_nonneg _
    · exact le_of_mul_le_mul_right h5 h
  · have e : Q (1, η) (1, η) = Q (1, η) ((1, 0) + (0, η)) := by rw [← hsplit η]
    rw [e, map_add, h0 η, add_zero]
  · intro hsym ζ
    have hz : ((1 : ℝ), ζ) = ((1 : ℝ), η) + ((0 : ℝ), ζ - η) := by simp
    have h1 := h0 (ζ - η)
    have h2 : Q (0, ζ - η) (1, η) = 0 := by rw [hsym]; exact h1
    have h3 := hneg (ζ - η)
    rw [hz, map_add, map_add, add_apply, add_apply, h1, h2]
    nlinarith [sq_nonneg ‖ζ - η‖]

/-- **The correction terms of L12.** Let `T a b v` be the third derivative of `f` (the derivative of
`D²f[a, b]` in direction `v`). If every block other than `f_xxx` has norm at most `m`, in the combined
form below, then `T (1, b) (1, b) (1, a) ≥ f_xxx - m((1 + ‖b‖)²(1 + ‖a‖) - 1)`. With `‖a‖, ‖b‖ ≤ u`
this is SP's adverse bound `m u (3 + 3u + u²)`. -/
theorem trilinear_lower {T : ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {m : ℝ}
    (hT : ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (a b : E) :
    T (1, 0) (1, 0) (1, 0) - m * ((1 + ‖b‖) * (1 + ‖b‖) * (1 + ‖a‖) - 1) ≤ T (1, b) (1, b) (1, a) := by
  have h := (abs_le.mp (hT 1 1 1 b b a)).1
  simp only [abs_one, one_mul, mul_one] at h
  linarith

/-- **SP's constants close L12.** With `m ≥ 2`, `δ > rm(8m - 5)` and `u = 2s + 2s²`, `s = mr/δ`
(L9's bound on `‖h'‖`): `7/4 - m((1 + u)³ - 1) ≥ 1/4`. SP's exact value at `k = 11` is
`7/4 - 2554128/1771561 = 0.308… > 1/4`. -/
theorem l12_constants {r m δ : ℝ} (hr : 0 < r) (hm : 2 ≤ m) (hδ : r * m * (8 * m - 5) < δ) :
    1 / 4 ≤ 7 / 4 - m * ((1 + (2 * (m * r / δ) + 2 * (m * r / δ) ^ 2)) ^ 3 - 1) := by
  have hrm : 0 < r * m := mul_pos hr (by linarith)
  have h11 : 11 * (r * m) ≤ r * m * (8 * m - 5) := by nlinarith
  have hδ0 : 0 < δ := by linarith
  set s := m * r / δ with hs
  have hs0 : 0 ≤ s := by positivity
  have hs1 : s ≤ 1 / 11 := by
    rw [hs, div_le_iff₀ hδ0]; nlinarith
  have hms : m * s ≤ 2 / 11 := by
    have : m * s = m * m * r / δ := by rw [hs]; ring
    rw [this, div_le_iff₀ hδ0]
    nlinarith [mul_nonneg (mul_nonneg hr.le (by linarith : (0 : ℝ) ≤ m)) (by linarith : (0 : ℝ) ≤ m - 2)]
  set u := 2 * s + 2 * s ^ 2 with hu
  have hu0 : 0 ≤ u := by positivity
  have hu1 : u ≤ 24 / 121 := by rw [hu]; nlinarith
  have hmu : m * u ≤ 48 / 121 := by
    have : m * u = 2 * (m * s) * (1 + s) := by rw [hu]; ring
    rw [this]
    have h1 : 1 + s ≤ 12 / 11 := by linarith
    have h2 : 0 ≤ m * s := by positivity
    nlinarith
  have hexp : m * ((1 + u) ^ 3 - 1) = (m * u) * (3 + 3 * u + u ^ 2) := by ring
  rw [hexp]
  have h3 : 3 + 3 * u + u ^ 2 ≤ 3 + 3 * (24 / 121) + (24 / 121) ^ 2 := by nlinarith
  have h4 : (m * u) * (3 + 3 * u + u ^ 2) ≤ (48 / 121) * (3 + 3 * (24 / 121) + (24 / 121) ^ 2) :=
    mul_le_mul hmu h3 (by positivity) (by norm_num)
  norm_num at h4 ⊢
  linarith

/-- **The transverse Hessian is the transverse block of `D²f`.** If `y ↦ ∂_y f (x, y)` has
derivative `H` at `y₀` and `Df` has derivative `D2f` at `(x, y₀)`, then
`H v w = D²f (x, y₀) [(0, v), (0, w)]`. -/
theorem transverse_hessian_eq {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {H : E →L[ℝ] E →L[ℝ] ℝ} {x : ℝ} {y₀ : E}
    (hH : HasFDerivAt (fun y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) H y₀)
    (hD2 : HasFDerivAt Df D2f (x, y₀)) (v w : E) : H v w = D2f (0, v) (0, w) := by
  have h1 := hH.clm_apply (hasFDerivAt_const w y₀)
  have h2 := (hD2.comp y₀ (hasFDerivAt_prodMk_right x y₀)).clm_apply
    (hasFDerivAt_const ((0 : ℝ), w) y₀)
  have h3 := h1.unique h2
  have := congrArg (fun L : E →L[ℝ] ℝ => L v) h3
  simpa using this

/-- **The differentiated ridge equation.** If `∂_y f (t, h t) = 0` for `t` near `x`, `h` has
derivative `h'` at `x` and `Df` has derivative `D2f` at `(x, h x)`, then
`D²f (x, h x) [(1, h'), (0, ·)] = 0`, i.e. `∂_x ∇_y f + D_y² f · h' = 0` (L11's first identity). -/
theorem ridge_second_identity {Df : ℝ × E → ℝ × E →L[ℝ] ℝ}
    {D2f : ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ} {h : ℝ → E} {h' : E} {x : ℝ}
    (hridge : ∀ᶠ t in 𝓝 x, (Df (t, h t)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0)
    (hh : HasDerivAt h h' x) (hD2 : HasFDerivAt Df D2f (x, h x)) :
    (D2f (1, h')).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by
  have hp : HasDerivAt (fun t => (t, h t)) ((1 : ℝ), h') x := (hasDerivAt_id x).prodMk hh
  have h1 : HasDerivAt (fun t => Df (t, h t)) (D2f (1, h')) x := hD2.comp_hasDerivAt x hp
  have h2 : HasDerivAt (fun t => (Df (t, h t)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      ((D2f (1, h')).comp (ContinuousLinearMap.inr ℝ ℝ E)) x := by
    have := h1.clm_comp (hasDerivAt_const x (ContinuousLinearMap.inr ℝ ℝ E))
    simpa using this
  have h3 : HasDerivAt (fun t => (Df (t, h t)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 x :=
    (hasDerivAt_const x (0 : E →L[ℝ] ℝ)).congr_of_eventuallyEq hridge
  exact h2.unique h3

/-- **L12 (CAP (11)–(12)): the convexity input `hconv`.** Along the ridge, let `Df`, `D2f` be the
first two derivatives of `f`, with `D2f` symmetric and its transverse part `δ`-negative definite,
`‖h'‖ ≤ u`, and for every `‖b‖ ≤ u` let `s ↦ D²f(s, h s)[(1, b), (1, b)]` have derivative `≥ 1/4`
(SP: `f_xxx ≥ 7/4` minus the adverse part `m u (3 + 3u + u²)`; see `trilinear_lower`). Then
`F - (x + r/2)(x - r/2)/8`, `F x = ∂_x f (x, h x)`, is convex on `[-2r, 2r]`. No second derivative of
`h` is used: `F' = D²f[(1, h'), (1, 0)]` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]` (`schur_max`),
so `F'(x₂) - F'(x₁) ≥ D²f(x₂)[(1, η), (1, η)] - D²f(x₁)[(1, η), (1, η)] ≥ (x₂ - x₁)/4` with
`η = h'(x₁)`. -/
theorem ridge_hconv {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {Φ : ℝ → E → ℝ} {h h' : ℝ → E} {U : Set ℝ} {r δ u : ℝ} (hδ : 0 < δ) (hU : IsOpen U)
    (hIU : Icc (-2 * r) (2 * r) ⊆ U)
    (hridge : ∀ x ∈ U, (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0)
    (hh : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x)
    (hD2 : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt Df (D2f (x, h x)) (x, h x))
    (hsym : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v w, D2f (x, h x) v w = D2f (x, h x) w v)
    (hneg : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E, D2f (x, h x) (0, v) (0, v) ≤ -δ * ‖v‖ ^ 2)
    (hu : ∀ x ∈ Icc (-2 * r) (2 * r), ‖h' x‖ ≤ u)
    (hΦ : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ b : E,
      HasDerivAt (fun s => D2f (s, h s) (1, b) (1, b)) (Φ x b) x)
    (hΦb : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ b : E, ‖b‖ ≤ u → 1 / 4 ≤ Φ x b) :
    ConvexOn ℝ (Icc (-2 * r) (2 * r))
      (fun x => Df (x, h x) (1, 0) - (x + r / 2) * (x - r / 2) / 8) := by
  have hid : ∀ x ∈ Icc (-2 * r) (2 * r),
      (D2f (x, h x) (1, h' x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := fun x hx =>
    ridge_second_identity (Filter.eventually_of_mem (hU.mem_nhds (hIU hx)) hridge) (hh x hx)
      (hD2 x hx)
  have hF : ∀ x ∈ Icc (-2 * r) (2 * r),
      HasDerivAt (fun t => Df (t, h t) (1, 0) - (t + r / 2) * (t - r / 2) / 8)
        (D2f (x, h x) (1, h' x) (1, 0) - x / 4) x := by
    intro x hx
    have hp : HasDerivAt (fun t => (t, h t)) ((1 : ℝ), h' x) x := (hasDerivAt_id x).prodMk (hh x hx)
    have h1 : HasDerivAt (fun t => Df (t, h t)) (D2f (x, h x) (1, h' x)) x :=
      (hD2 x hx).comp_hasDerivAt x hp
    have h2 : HasDerivAt (fun t => Df (t, h t) (1, 0)) (D2f (x, h x) (1, h' x) (1, 0)) x := by
      simpa using h1.clm_apply (hasDerivAt_const x ((1 : ℝ), (0 : E)))
    have h3 : HasDerivAt (fun t : ℝ => (t + r / 2) * (t - r / 2) / 8) (x / 4) x := by
      have := ((hasDerivAt_pow 2 x).div_const 8).sub_const (r ^ 2 / 32)
      convert this using 1
      · funext t
        ring
      · push_cast
        ring
    exact h2.sub h3
  have hmono : MonotoneOn (fun x => D2f (x, h x) (1, h' x) (1, 0) - x / 4)
      (Icc (-2 * r) (2 * r)) := by
    intro x1 hx1 x2 hx2 hle
    have hφ : ∀ t ∈ Icc (-2 * r) (2 * r),
        HasDerivAt (fun s => D2f (s, h s) (1, h' x1) (1, h' x1) - s / 4)
          (Φ t (h' x1) - 1 / 4) t := fun t ht =>
      (hΦ t ht (h' x1)).sub ((hasDerivAt_id t).div_const 4)
    have hφmono : MonotoneOn (fun s => D2f (s, h s) (1, h' x1) (1, h' x1) - s / 4)
        (Icc (-2 * r) (2 * r)) := by
      refine monotoneOn_of_deriv_nonneg (convex_Icc _ _)
        (fun t ht => (hφ t ht).continuousAt.continuousWithinAt)
        (fun t ht => (hφ t (interior_subset ht)).differentiableAt.differentiableWithinAt) ?_
      intro t ht
      rw [(hφ t (interior_subset ht)).deriv]
      have := hΦb t (interior_subset ht) (h' x1) (hu x1 hx1)
      linarith
    have h12 := hφmono hx1 hx2 hle
    obtain ⟨-, he1, -⟩ := schur_max hδ (hneg x1 hx1) (hid x1 hx1)
    obtain ⟨-, he2, hmax2⟩ := schur_max hδ (hneg x2 hx2) (hid x2 hx2)
    have hm2 := hmax2 (hsym x2 hx2) (h' x1)
    simp only at h12 ⊢
    rw [← he1, ← he2]
    linarith
  refine MonotoneOn.convexOn_of_deriv (convex_Icc _ _)
    (fun x hx => (hF x hx).continuousAt.continuousWithinAt)
    (fun x hx => (hF x (interior_subset hx)).differentiableAt.differentiableWithinAt) ?_
  intro x1 hx1 x2 hx2 hle
  rw [(hF x1 (interior_subset hx1)).deriv, (hF x2 (interior_subset hx2)).deriv]
  exact hmono (interior_subset hx1) (interior_subset hx2) hle

end V
