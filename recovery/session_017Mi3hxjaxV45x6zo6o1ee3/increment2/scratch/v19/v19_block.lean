
/-! ### L6, L9 and L12 (v1.9)

The last analytic input of the cap, L12's convexity `hconv`, is derived here. L6 bounds `w'` by
chord slopes of L5's convexity pair (no integral), L9 bounds `h'` through the differentiated ridge
equation, and L12 uses that `F' = D²f[(1, h'), (1, 0)]` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]`,
so that no second derivative of `h` is needed. `ridge_capHyp_C3` composes them with
`ridge_capHyp_source`. -/

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


/-- **L3 in `y` for `∂_x ∇_y f`.** If along the segment `s ↦ (x, s • y)` the third derivatives
satisfy `|T (1, 0) (0, v) (0, y)| ≤ m ‖v‖ ‖y‖` (the block `f_xyy`), then
`‖∂_x ∇_y f (x, y) - ∂_x ∇_y f (x, 0)‖ ≤ m ‖y‖`. -/
theorem dx_grad_increment {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {x m : ℝ} {y : E} (hm : 0 ≤ m)
    (hD3 : ∀ s ∈ Icc (0 : ℝ) 1, ∀ a b : ℝ × E,
      HasFDerivAt (fun q => D2f q a b) (T3 (x, s • y) a b) (x, s • y))
    (hT : ∀ s ∈ Icc (0 : ℝ) 1, ∀ v : E, |T3 (x, s • y) (1, 0) (0, v) (0, y)| ≤ m * ‖v‖ * ‖y‖) :
    ‖(D2f (x, y) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) -
      (D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ m * ‖y‖ := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have hφ : ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => D2f (x, s • y) (1, 0) (0, v))
      (T3 (x, s • y) (1, 0) (0, v) (0, y)) s := by
    intro s hs
    have hp : HasDerivAt (fun s : ℝ => (x, s • y)) ((0 : ℝ), y) s := by
      simpa using (hasDerivAt_const s x).prodMk ((hasDerivAt_id s).smul_const y)
    exact (hD3 s hs (1, 0) (0, v)).comp_hasDerivAt s hp
  have hφb : ∀ s ∈ Icc (0 : ℝ) 1, ‖T3 (x, s • y) (1, 0) (0, v) (0, y)‖ ≤ m * ‖v‖ * ‖y‖ :=
    fun s hs => by rw [Real.norm_eq_abs]; exact hT s hs v
  have := (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
    (fun s hs => (hφ s hs).hasDerivWithinAt) hφb (left_mem_Icc.mpr zero_le_one)
    (right_mem_Icc.mpr zero_le_one)
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at this
  simp only [sub_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.inr_apply]
  calc ‖(D2f (x, y)) (1, 0) (0, v) - (D2f (x, 0)) (1, 0) (0, v)‖ ≤ m * ‖v‖ * ‖y‖ := this
    _ = m * ‖y‖ * ‖v‖ := by ring

/-- **L9's arithmetic.** From `δ ‖h'‖ ≤ ‖∂_x∇_y f(x, h)‖ ≤ 2mr + m‖h‖` and `δ ‖h‖ ≤ 2mr²`:
`‖h'‖ ≤ 2s + 2s²` with `s = mr/δ` (SP: `u = 2/k + 2/k²`). -/
theorem l9_arith {δ m r a b c : ℝ} (hδ : 0 < δ) (hm : 0 ≤ m) (ha : δ * a ≤ c)
    (hc : c ≤ 2 * m * r + m * b) (hb : δ * b ≤ 2 * m * r ^ 2) :
    a ≤ 2 * (m * r / δ) + 2 * (m * r / δ) ^ 2 := by
  have key : δ * δ * a ≤ 2 * m * r * δ + 2 * m * m * r ^ 2 := by nlinarith
  have hrhs : 2 * (m * r / δ) + 2 * (m * r / δ) ^ 2 =
      (2 * m * r * δ + 2 * m * m * r ^ 2) / (δ * δ) := by
    field_simp
  rw [hrhs, le_div_iff₀ (by positivity)]
  linarith

/-- **L12's bound at one point.** With the combined block bound, `f_xxx ≥ 7/4`, `‖a‖, ‖b‖ ≤ u`
and `7/4 - m((1 + u)³ - 1) ≥ 1/4` (`l12_constants`): `T (1, b) (1, b) (1, a) ≥ 1/4`. -/
theorem l12_lower {T : ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {m u : ℝ} {a b : E} (hm : 0 ≤ m)
    (hT : ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hxxx : 7 / 4 ≤ T (1, 0) (1, 0) (1, 0)) (hc : 1 / 4 ≤ 7 / 4 - m * ((1 + u) ^ 3 - 1))
    (ha : ‖a‖ ≤ u) (hb : ‖b‖ ≤ u) : 1 / 4 ≤ T (1, b) (1, b) (1, a) := by
  have h1 := trilinear_lower hT a b
  have hnb := norm_nonneg b
  have hna := norm_nonneg a
  have hb1 : 1 + ‖b‖ ≤ 1 + u := by linarith
  have ha1 : 1 + ‖a‖ ≤ 1 + u := by linarith
  have h4 : (1 + ‖b‖) * (1 + ‖b‖) * (1 + ‖a‖) ≤ (1 + u) ^ 3 := by
    have e : (1 + u) ^ 3 = (1 + u) * (1 + u) * (1 + u) := by ring
    rw [e]
    exact mul_le_mul (mul_le_mul hb1 hb1 (by linarith) (by linarith)) ha1 (by linarith)
      (mul_nonneg (by linarith) (by linarith))
  have h5 := mul_le_mul_of_nonneg_left (sub_le_sub_right h4 1) hm
  linarith

/-- **The ridge facts from SP's source data.** Under the hypotheses of `ridge_capHyp_source`, for a
ridge `h` on the collar: `m ≥ 2`, `δ = λ - 5rm > rm(8m - 5)`, `h x` lies in the open ball and
`δ ‖h x‖ ≤ 2mr²` on `[-2r, 2r]` (L5 and L8), `h` is differentiable there (L7), and the transverse
Hessian along the ridge is `δ`-negative definite (L4). -/
theorem ridge_source_facts [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) {h : ℝ → E}
    (hhU : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) :
    2 ≤ m ∧ r * m * (8 * m - 5) < lam - 5 * r * m ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r)) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), (lam - 5 * r * m) * ‖h x‖ ≤ 2 * m * r ^ 2) ∧
      (∃ h' : ℝ → E, ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E, H (x, h x) v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2) := by
  have hUX : Ioo (-2 * r - ε) (2 * r + ε) ⊆ Icc (-2 * r - ε) (2 * r + ε) := Ioo_subset_Icc_self
  have hcapX : Icc (-2 * r) (r / 2) ⊆ Icc (-2 * r - ε) (2 * r + ε) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  -- the slice derivatives are the transverse parts of `Df`
  have hD : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) ((Df p).comp (ContinuousLinearMap.inr ℝ ℝ E)) p.2 := by
    rintro ⟨x, y⟩ hp
    exact slice_hasFDerivAt (hDf (x, y) hp)
  -- L1–L2: `m ≥ 2`
  have hg0 : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    have hx' : (x, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
      ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, h0Y⟩
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    exact (hDf _ hx').comp_hasDerivAt x hp
  have hG0a : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hG0c : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hm : 2 ≤ m := hermite_m_ge_two (g := fun x => f (x, 0)) hr hg0 hG0cv hG0cc hG0a hG0c hnorm
  have hm0 : 0 ≤ m := by linarith
  -- L3–L4 on the full domain
  have hlip := hessian_increment_D hr hε hHx hHxb hHr hHrb
  have hneg := transverse_hessian_le_D hm0 hε hεr hlip hlam
  have hconcX := slices_strongConcave_D hm0 hε hεr hD hH hlip hlam
  have h11 : 11 * (r * m) < lam - 5 * r * m := by
    nlinarith [mul_nonneg (mul_nonneg hr.le hm0) (sub_nonneg.2 hm)]
  have hrm : 0 < r * m := mul_pos hr (by linarith)
  have hδ0 : 0 < lam - 5 * r * m := by linarith
  -- L5 on the full domain: `‖w x‖ ≤ 3mr² < 2rδ`
  have hwv : ∀ (t : ℝ) (v : E),
      ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v = Df (t, 0) (0, v) := fun t v => by simp
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp only [hwv]
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp only [hwv]
  have hwa : (Df (-r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hM]
  have hwc : (Df (r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hS]
  have hwlt : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ < (lam - 5 * r * m) * (2 * r) := by
    intro x hx
    have h1 := slice_gradient_bound (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      hm0 hwcv' hwcc' (show -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩) (by linarith)
      hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 6 * r ^ 2 := by
      have hxx : 0 ≤ (x - (-2 * r - ε)) * (2 * r + ε - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      have hee : 0 ≤ (r / 2 - ε) * (9 * r / 2 + ε) :=
        mul_nonneg (by linarith) (by linarith)
      rw [abs_le]; constructor <;> nlinarith [sq_nonneg x, sq_nonneg r]
    have h3 : ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 3 * m * r ^ 2 :=
      calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
          ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
        _ ≤ m / 2 * (6 * r ^ 2) := by gcongr
        _ = 3 * m * r ^ 2 := by ring
    have h4 : 22 * r * (r * m) < (lam - 5 * r * m) * (2 * r) := by nlinarith
    nlinarith
  have hslice0 : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) ((Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 :=
    fun x hx => hD (x, 0) ⟨hx, h0Y⟩
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hcrit : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := fun x hx => (hhU x hx).2
  -- L8: the ridge is in the open ball on `[-2r, 2r]`
  have hint : ∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r) := by
    intro x hx
    have hxU := hIU hx
    have h8 := slice_ridge_bound (hconcX x (hUX hxU)) h0Y (hball x hxU) (hcrit x hxU)
      (hslice0 x (hUX hxU))
    have h9 := hwlt x (hUX hxU)
    rw [Metric.mem_ball, dist_zero_right]
    have : (lam - 5 * r * m) * ‖h x‖ < (lam - 5 * r * m) * (2 * r) := by linarith
    exact lt_of_mul_lt_mul_left this hδ0.le
  have hridgeX : ∀ x ∈ Icc (-2 * r) (2 * r),
      (x, h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => ⟨hUX (hIU hx), hball x (hIU hx)⟩
  -- L7 regularity: the transverse part of `DG` is `H`, negative definite
  have hDGH : ∀ x ∈ Icc (-2 * r) (2 * r),
      DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E = H (x, h x) := by
    intro x hx
    have h1 : HasFDerivAt (fun y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E))
        ((DG (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (h x) :=
      (hGs _ (hridgeX x hx)).hasFDerivAt.comp (h x) (hasFDerivAt_prodMk_right x (h x))
    exact h1.unique (hH _ (hridgeX x hx))
  have hnegR : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      (DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E) v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [hDGH x hx]
    exact hneg _ (hridgeX x hx) v
  have hG : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ᶠ v in 𝓝 (x, h x),
      HasFDerivAt (fun y => f (v.1, y)) ((Df v).comp (ContinuousLinearMap.inr ℝ ℝ E)) v.2 := by
    intro x hx
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with v hv
    exact hD v ⟨hUX hv.1, Metric.ball_subset_closedBall hv.2⟩
  have hstrict := ridge_differentiable (f := f)
    (G := fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG := fun x => DG (x, h x))
    (h := h) (U := Ioo (-2 * r - ε) (2 * r + ε)) hδ0 isOpen_Ioo hIU
    (fun x hx => hconcX x (hUX hx)) hball hcrit hint hG
    (fun x hx => hGs _ (hridgeX x hx)) hnegR
  have hmemA : -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hmemC : r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hw2 : ∀ x ∈ Icc (-2 * r) (2 * r),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 2 * m * r ^ 2 := by
    intro x hx
    have h1 := slice_gradient_bound
      (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) hm0 hwcv' hwcc' hmemA hmemC
      (by linarith) hwa hwc (hUX (hIU hx))
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 4 * r ^ 2 := by
      have hxx : 0 ≤ (x - -2 * r) * (2 * r - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      rw [abs_le]; constructor <;> nlinarith [sq_nonneg x, sq_nonneg r]
    calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
        ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
      _ ≤ m / 2 * (4 * r ^ 2) := by gcongr
      _ = 2 * m * r ^ 2 := by ring
  refine ⟨hm, by nlinarith, hint, fun x hx =>
    (slice_ridge_bound (hconcX x (hUX (hIU hx))) h0Y (hball x (hIU hx)) (hcrit x (hIU hx))
      (hslice0 x (hUX (hIU hx)))).trans (hw2 x hx),
    ⟨_, fun x hx => (hstrict x hx).hasFDerivAt.hasDerivAt⟩,
    fun x hx v => hneg _ (hridgeX x hx) v⟩


/-- **The cap from SP's `C³` source data (L1–L20 composed; L12 proved).** As
`ridge_capHyp_source`, with L12's `hconv` derived instead of assumed. Additionally assume, on the
collar: a second derivative `D2f` of `f`, third derivatives `T3 p a b` of `p ↦ D²f p [a, b]`, every
third-derivative block other than `f_xxx` of norm at most `m` (combined form), and `f_xxx ≥ 7/4`
(SP's L4, from L2 and (H2)). Then the ridge exists, and every ridge has `h(∓r/2) = 0` and yields
(C1)–(C3) and (C2⁺) on the cap. Inside: L5 gives `‖w‖ ≤ 2mr²` on `[-2r, 2r]`, L8 `‖h‖ ≤ 2mr²/δ`, L6
`‖w'‖ ≤ 2mr`, L3 in `y` `‖∂_x∇_y f(x, h)‖ ≤ ‖w'‖ + m‖h‖`, L9 `‖h'‖ ≤ 2s + 2s²` with `s = mr/δ`,
`trilinear_lower` and `l12_constants` give `F'' ≥ 1/4` in the form of `ridge_hconv`. -/
theorem ridge_capHyp_C3 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hT : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hxxx : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), 7 / 4 ≤ T3 p (1, 0) (1, 0) (1, 0))
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  obtain ⟨hex, hcapS⟩ := ridge_capHyp_source hr hε hεr hDf hGs hH hHx hHxb hHr hHrb hlam hH1 hM hS
    hG0cv hG0cc hwcv hwcc hnorm
  refine ⟨hex, fun h hhU => hcapS h hhU ?_⟩
  obtain ⟨hm, hδ, hint, hh8, ⟨h', hhd⟩, hnegH⟩ := ridge_source_facts hr hε hεr hDf hGs hH hHx hHxb
    hHr hHrb hlam hH1 hM hS hG0cv hG0cc hwcv hwcc hnorm hhU
  have hm0 : 0 ≤ m := by linarith
  have hδ0 : 0 < lam - 5 * r * m := by
    have : 0 < r * m * (8 * m - 5) := mul_pos (mul_pos hr (by linarith)) (by linarith)
    linarith
  have hUX : Ioo (-2 * r - ε) (2 * r + ε) ⊆ Icc (-2 * r - ε) (2 * r + ε) := Ioo_subset_Icc_self
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hridgeX : ∀ x ∈ Icc (-2 * r) (2 * r),
      (x, h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => ⟨hUX (hIU hx), hball x (hIU hx)⟩
  have hridgeU : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := fun x hx =>
    (slice_hasFDerivAt (hDf _ ⟨hUX hx, hball x hx⟩)).unique (hhU x hx).2
  have hD2r : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt Df (D2f (x, h x)) (x, h x) :=
    fun x hx => hD2 _ (hridgeX x hx)
  have hsym : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v w, D2f (x, h x) v w = D2f (x, h x) w v := by
    intro x hx v w
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    have hev : ∀ᶠ q in 𝓝 (x, h x), HasFDerivAt f (Df q) q := by
      filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with q hq
      exact hDf q ⟨hUX hq.1, Metric.ball_subset_closedBall hq.2⟩
    exact second_derivative_symmetric_of_eventually hev (hD2r x hx) v w
  have hnegD : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      D2f (x, h x) (0, v) (0, v) ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [← transverse_hessian_eq (hH _ (hridgeX x hx)) (hD2r x hx) v v]
    exact hnegH x hx v
  -- L6: `‖w' x‖ ≤ 2mr`
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp
  have hw6 : ∀ x ∈ Icc (-2 * r) (2 * r),
      ‖(D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 2 * m * r := by
    intro x hx
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    have h1 : HasDerivAt (fun t => Df (t, 0)) (D2f (x, 0) (1, 0)) x :=
      (hD2 _ ⟨hUX (hIU hx), h0Y⟩).comp_hasDerivAt x hp
    have hd : HasDerivAt (fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
        ((D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) x := by
      simpa using h1.clm_comp (hasDerivAt_const x (ContinuousLinearMap.inr ℝ ℝ E))
    have h2 := slice_gradient_deriv_bound hr hm0 hwcv' hwcc'
      (show -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (by simp [hM]) (by simp [hS]) (hUX (hIU hx)) hd
    have h3 : max |x| (r / 2) ≤ 2 * r :=
      max_le (abs_le.mpr ⟨by linarith [hx.1], by linarith [hx.2]⟩) (by linarith)
    calc ‖(D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ m * max |x| (r / 2) := h2
      _ ≤ m * (2 * r) := mul_le_mul_of_nonneg_left h3 hm0
      _ = 2 * m * r := by ring
  -- L3 in `y`, then L9
  have hu : ∀ x ∈ Icc (-2 * r) (2 * r), ‖h' x‖ ≤
      2 * (m * r / (lam - 5 * r * m)) + 2 * (m * r / (lam - 5 * r * m)) ^ 2 := by
    intro x hx
    have hseg : ∀ s ∈ Icc (0 : ℝ) 1,
        (x, s • h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
      intro s hs
      have hhY : ‖h x‖ ≤ 2 * r := by
        simpa [Metric.mem_closedBall, dist_zero_right] using hball x (hIU hx)
      refine ⟨hUX (hIU hx), ?_⟩
      rw [Metric.mem_closedBall, dist_zero_right, norm_smul, Real.norm_eq_abs, abs_of_nonneg hs.1]
      nlinarith [hs.2, norm_nonneg (h x)]
    have hinc := dx_grad_increment (D2f := D2f) (T3 := T3) (x := x) (y := h x) hm0
      (fun s hs => hD3 _ (hseg s hs))
      (fun s hs v => by
        have := hT _ (hseg s hs) 1 0 0 0 v (h x)
        simpa [mul_assoc] using this)
    have hBn : ‖(D2f (x, h x) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤
        2 * m * r + m * ‖h x‖ :=
      (norm_le_norm_add_norm_sub' _ _).trans (add_le_add (hw6 x hx) hinc)
    have hid := ridge_second_identity
      (Filter.eventually_of_mem (isOpen_Ioo.mem_nhds (hIU hx)) hridgeU) (hhd x hx) (hD2r x hx)
    obtain ⟨h9, -, -⟩ := schur_max hδ0 (hnegD x hx) hid
    exact l9_arith hδ0 hm0 h9 hBn (hh8 x hx)
  -- L12
  have hc := l12_constants hr hm hδ
  have hΦ : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ b : E,
      HasDerivAt (fun s => D2f (s, h s) (1, b) (1, b)) (T3 (x, h x) (1, b) (1, b) (1, h' x)) x := by
    intro x hx b
    have hp : HasDerivAt (fun s => (s, h s)) ((1 : ℝ), h' x) x := (hasDerivAt_id x).prodMk (hhd x hx)
    exact (hD3 _ (hridgeX x hx) (1, b) (1, b)).comp_hasDerivAt x hp
  exact ridge_hconv (Φ := fun x b => T3 (x, h x) (1, b) (1, b) (1, h' x)) hδ0 isOpen_Ioo hIU
    hridgeU hhd hD2r hsym hnegD hu hΦ
    (fun x hx b hb => l12_lower hm0 (hT _ (hridgeX x hx)) (hxxx _ (hridgeX x hx)) hc (hu x hx) hb)
