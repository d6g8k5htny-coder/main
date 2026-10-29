import ElderClearance

noncomputable section
namespace TorusFiltration

variable {n : ℕ} (g : superlevelBar_E n → ℝ)
  (D : Set (superlevelBar_E n)) (d : ℝ)
  (htrace : IsClopen (elderTrap_strictTrace g D d))
  (M : superlevelBar_E n) (hM : M ∈ D)
  (hheight : ∀ x ∈ D, g x ≤ g M)
  (P : superlevelBar_E n) (hP : g M < g P) (γ : Path M P)
  (hγ : ∀ t : unitInterval, d ≤ g (γ t))

include D htrace hM hheight P hP γ hγ

/-- A strict subthreshold is genuinely attained as strict clearance. -/
theorem elder_control_below : d - 1 ∈ superlevelBar_clearanceSet g M := by
  rw [elderTrap_clearance_eq_Iio g D d htrace M hM hheight P hP γ hγ]
  change d - 1 < d
  linarith

/-- The attained closed death is excluded from the strict clearance set. -/
theorem elder_control_death_excluded : d ∉ superlevelBar_clearanceSet g M := by
  rw [elderTrap_clearance_eq_Iio g D d htrace M hM hheight P hP γ hγ]
  exact lt_irrefl d

/-- A strict geometric height gap gives positive lifetime in the literal data. -/
theorem elder_control_positive_lifetime (hg : Continuous g) (hdm : d < g M) :
    0 < g M - superlevelBar_deathHeight g M := by
  rw [elderTrap_death_eq g hg D d htrace M hM hheight P hP γ hγ]
  linarith

end TorusFiltration
