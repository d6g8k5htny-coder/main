import SourceJets

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Jets MeasureTheory
open scoped BigOperators

-- The zero order is the actual total sourceField, even without L != 0.
example (d : ℕ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ)
    (v : Fin 0 → (Fin d → ℝ)) :
    sourceJet d 0 0 ω x v = sourceField d 0 ω x :=
  sourceJet_zero d 0 ω x v

-- The empty direction product must preserve the coefficient value.
example (d : ℕ) (L : ℝ) (m : SourceMode d) (x : Fin d → ℝ)
    (v : Fin 0 → (Fin d → ℝ)) :
    modeJetCoefficient d L m 0 x v = modeCoefficient d L m x :=
  iteratedFDeriv_zero_apply v

-- The constant Fourier mode contributes no positive-order derivative.
example (d : ℕ) (L : ℝ) (j : ℕ) (x : Fin d → ℝ)
    (v : Fin (j + 1) → (Fin d → ℝ)) :
    modeJetCoefficient d L none (j + 1) x v = 0 := by
  unfold modeJetCoefficient
  change (iteratedFDeriv ℝ (j + 1)
    (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L)) x) v = 0
  simp only [iteratedFDeriv_succ_const, Pi.zero_apply, zero_apply]

-- Raw zero-direction observations are singular, for every actual sample.
example (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ) :
    sourceJet d L 1 ω x (fun _ => 0) = 0 := by
  exact (iteratedFDeriv ℝ 1 (sourceField d L ω) x).map_coord_zero 0 rfl

example (d : ℕ) (L : ℝ) (m : SourceMode d) (x : Fin d → ℝ)
    (v : Fin 4 → (Fin d → ℝ)) (hv : v 2 = 0) :
    modeJetCoefficient d L m 4 x v = 0 := by
  exact (iteratedFDeriv ℝ 4 (modeCoefficient d L m) x).map_coord_zero 2 hv

-- Fourth-order, nonunit directions use the full product of norms.
example (m : SourceMode 3) (x : Fin 3 → ℝ) (v : Fin 4 → (Fin 3 → ℝ)) :
    |modeJetCoefficient 3 24 m 4 x v| ≤ modeMajorant 3 24 m * ∏ i, ‖v i‖ :=
  modeJetCoefficient_bound 3 (by norm_num) m 4 (by norm_num) x v

-- One event supports different points, orders and direction tuples at once.
example : ∀ᵐ ω ∂sourceLaw 3, ∀ (x y : Fin 3 → ℝ)
    (v : Fin 1 → (Fin 3 → ℝ)) (w : Fin 4 → (Fin 3 → ℝ)),
    sourceJet 3 24 1 ω x v = scalarSeries (fun m => modeJetCoefficient 3 24 m 1 x v) ω ∧
    sourceJet 3 24 4 ω y w = scalarSeries (fun m => modeJetCoefficient 3 24 m 4 y w) ω := by
  filter_upwards [sourceJet_eq_series_ae 3 (L := 24) (by norm_num)] with ω hω
  intro x y v w
  exact ⟨hω 1 (by norm_num) x v, hω 4 (by norm_num) y w⟩

-- Literal dimension-zero and negative-period cases remain admitted.
example (x : Fin 0 → ℝ) (v : Fin 4 → (Fin 0 → ℝ)) :
    Summable (fun m => modeJetCoefficient 0 (-24) m 4 x v) :=
  modeJetCoefficient_summable 0 (by norm_num) 4 (by norm_num) x v

