import Mathlib

open Set Filter Topology

namespace CapFirstExit

section PartialBlock

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem block_of_partial {B : ContinuousMultilinearMap ℝ (fun _ : Fin 3 => ℝ × E) ℝ} {m : ℝ}
    (hs₁₂ : ∀ a b c : ℝ × E, B ![a, b, c] = B ![b, a, c])
    (hs₂₃ : ∀ a b c : ℝ × E, B ![a, b, c] = B ![a, c, b])
    (k₁ : ∀ y : E, |B ![((0 : ℝ), y), (1, 0), (1, 0)]| ≤ m * ‖y‖)
    (k₂ : ∀ y₁ y₂ : E, |B ![((0 : ℝ), y₁), (0, y₂), (1, 0)]| ≤ m * ‖y₁‖ * ‖y₂‖)
    (k₃ : ∀ y₁ y₂ y₃ : E, |B ![((0 : ℝ), y₁), (0, y₂), (0, y₃)]| ≤ m * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E) :
    |B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![(1, 0), (1, 0), (1, 0)]| ≤
      m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  have hu0 : ∀ x b c z : ℝ × E, Function.update ![x, b, c] 0 z = ![z, b, c] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have hu1 : ∀ a x c z : ℝ × E, Function.update ![a, x, c] 1 z = ![a, z, c] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have hu2 : ∀ a b x z : ℝ × E, Function.update ![a, b, x] 2 z = ![a, b, z] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have a₁ : ∀ x x' b c : ℝ × E, B ![x + x', b, c] = B ![x, b, c] + B ![x', b, c] := by
    intro x x' b c
    have := B.map_update_add ![x, b, c] 0 x x'
    rwa [hu0, hu0, hu0] at this
  have a₂ : ∀ a x x' c : ℝ × E, B ![a, x + x', c] = B ![a, x, c] + B ![a, x', c] := by
    intro a x x' c
    have := B.map_update_add ![a, x, c] 1 x x'
    rwa [hu1, hu1, hu1] at this
  have a₃ : ∀ a b x x' : ℝ × E, B ![a, b, x + x'] = B ![a, b, x] + B ![a, b, x'] := by
    intro a b x x'
    have := B.map_update_add ![a, b, x] 2 x x'
    rwa [hu2, hu2, hu2] at this
  have m₁ : ∀ (t : ℝ) (x b c : ℝ × E), B ![t • x, b, c] = t * B ![x, b, c] := by
    intro t x b c
    have := B.map_update_smul ![x, b, c] 0 t x
    rwa [hu0, hu0, smul_eq_mul] at this
  have m₂ : ∀ (t : ℝ) (a x c : ℝ × E), B ![a, t • x, c] = t * B ![a, x, c] := by
    intro t a x c
    have := B.map_update_smul ![a, x, c] 1 t x
    rwa [hu1, hu1, smul_eq_mul] at this
  have m₃ : ∀ (t : ℝ) (a b x : ℝ × E), B ![a, b, t • x] = t * B ![a, b, x] := by
    intro t a b x
    have := B.map_update_smul ![a, b, x] 2 t x
    rwa [hu2, hu2, smul_eq_mul] at this
  set e : ℝ × E := ((1 : ℝ), (0 : E)) with he_def
  have hsp : ∀ (s : ℝ) (y : E), ((s, y) : ℝ × E) = s • e + ((0 : ℝ), y) := by
    intro s y; ext <;> simp [he_def]
  set P₁ := |s₁| + ‖y₁‖
  set P₂ := |s₂| + ‖y₂‖
  set P₃ := |s₃| + ‖y₃‖
  -- the multilinear expansion
  have expand : B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![e, e, e] =
      (s₂ * s₃ * B ![((0 : ℝ), y₁), e, e] + s₂ * B ![((0 : ℝ), y₁), e, (0, y₃)] +
        s₃ * B ![((0 : ℝ), y₁), (0, y₂), e] + B ![((0 : ℝ), y₁), (0, y₂), (0, y₃)]) +
      (s₁ * s₃ * B ![e, ((0 : ℝ), y₂), e] + s₁ * B ![e, ((0 : ℝ), y₂), (0, y₃)]) +
      s₁ * s₂ * B ![e, e, ((0 : ℝ), y₃)] := by
    rw [hsp s₁ y₁, hsp s₂ y₂, hsp s₃ y₃]
    simp only [a₁, a₂, a₃, m₁, m₂, m₃]
    ring
  rw [he_def] at expand ⊢
  rw [expand]
  -- the seven partial blocks, brought to the canonical order by symmetry
  have b₁ := k₁ y₁
  have b₂ : |B ![((0 : ℝ), y₁), (1, 0), (0, y₃)]| ≤ m * ‖y₁‖ * ‖y₃‖ := by
    rw [hs₂₃]; exact k₂ y₁ y₃
  have b₃ := k₂ y₁ y₂
  have b₄ := k₃ y₁ y₂ y₃
  have b₅ : |B ![(1, 0), ((0 : ℝ), y₂), (1, 0)]| ≤ m * ‖y₂‖ := by
    rw [hs₁₂]; exact k₁ y₂
  have b₆ : |B ![(1, 0), ((0 : ℝ), y₂), (0, y₃)]| ≤ m * ‖y₂‖ * ‖y₃‖ := by
    rw [hs₁₂, hs₂₃]; exact k₂ y₂ y₃
  have b₇ : |B ![(1, 0), (1, 0), ((0 : ℝ), y₃)]| ≤ m * ‖y₃‖ := by
    rw [hs₂₃, hs₁₂]; exact k₁ y₃
  have hP : m * (P₁ * P₂ * P₃ - |s₁| * |s₂| * |s₃|) =
      (|s₂| * |s₃| * (m * ‖y₁‖) + |s₂| * (m * ‖y₁‖ * ‖y₃‖) + |s₃| * (m * ‖y₁‖ * ‖y₂‖) +
        m * ‖y₁‖ * ‖y₂‖ * ‖y₃‖) +
      (|s₁| * |s₃| * (m * ‖y₂‖) + |s₁| * (m * ‖y₂‖ * ‖y₃‖)) + |s₁| * |s₂| * (m * ‖y₃‖) := by
    simp only [P₁, P₂, P₃]; ring
  rw [hP]
  refine (abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
    ((abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
      ((abs_add_le _ _).trans (add_le_add ?_ ?_)) ?_)) ?_)) ((abs_add_le _ _).trans
      (add_le_add ?_ ?_)))) ?_)
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₁ (by positivity)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₂ (abs_nonneg _)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₃ (abs_nonneg _)
  · exact b₄
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₅ (by positivity)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₆ (abs_nonneg _)
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₇ (by positivity)

end PartialBlock

#print axioms block_of_partial
#print axioms iteratedFDeriv_three_swap₂₃
end CapFirstExit
