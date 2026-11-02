import SourceSpatialKernel
import SourceTorus

noncomputable section
open I4Foundation I4Weights I4Source I4Lattice I4Kernel I4SourceCovariance I4Spatial I4Torus
open MeasureTheory ProbabilityTheory
open scoped BigOperators
local instance : Fact (0 < (24 : ℝ)) := ⟨by norm_num⟩

-- The physical exponent must use Euclidean coordinate squares, not the Pi norm.
example : (∑ _i : Fin 2, (1 : ℝ) ^ 2) ≠ ‖(fun _ : Fin 2 => (1 : ℝ))‖ ^ 2 := by
  have hn : ‖(fun _ : Fin 2 => (1 : ℝ))‖ = 1 := by simp
  rw [hn]
  norm_num

-- Full Fourier cosines cannot be factored term by term into coordinate cosines.
example : Real.cos (Real.pi / 2 + Real.pi / 2) ≠
    Real.cos (Real.pi / 2) * Real.cos (Real.pi / 2) := by
  norm_num [Real.cos_add, Real.cos_pi_div_two, Real.sin_pi_div_two]

-- Empty tuples give a singleton lattice even at zero period.
example (z : Fin 0 → ℝ) : sourceImageSum 0 0 z = 1 := by
  simp [sourceImageSum, tsum_fintype]

example (L : ℝ) (z : Fin 0 → ℝ) : sourceSpatialKernel 0 L z = 1 := by
  simp [sourceSpatialKernel, sourceImageSum, tsum_fintype]

example (z : Fin 2 → ℝ) : Summable (fun n : Fin 2 → ℤ =>
    Real.exp (-(∑ i, (z i + 24 * (n i : ℝ)) ^ 2) / 2)) :=
  image_summable 2 (by norm_num) z

example : 0 < sourceImageSum 3 (-24) 0 := image_zero_pos 3 (by norm_num)

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) : sourceSpatialKernel d L 0 = 1 := by
  rw [← spectralKernel_eq_spatialKernel d hL, spectralKernel_zero d hL]

example (z : Fin 3 → ℝ) : sourceSpectralKernel 3 24 z =
    (∑' n : Fin 3 → ℤ, Real.exp (-(∑ i, (z i + 24 * (n i : ℝ)) ^ 2) / 2)) /
      (∑' n : Fin 3 → ℤ, Real.exp (-(∑ i, (24 * (n i : ℝ)) ^ 2) / 2)) := by
  simpa [sourceSpatialKernel, sourceImageSum] using
    spectralKernel_eq_spatialKernel 3 (L := 24) (by norm_num) z

example (x y : Fin 3 → ℝ) :
    cov[fun ω => sourceField 3 24 ω x, fun ω => sourceField 3 24 ω y; sourceLaw 3] =
      (∑' n : Fin 3 → ℤ, Real.exp (-(∑ i, ((x i - y i) + 24 * (n i : ℝ)) ^ 2) / 2)) /
        (∑' n : Fin 3 → ℤ, Real.exp (-(∑ i, (24 * (n i : ℝ)) ^ 2) / 2)) := by
  simpa [sourceSpatialKernel, sourceImageSum] using
    sourceField_covariance_spatial 3 (L := 24) (by norm_num) x y

example (x y : Fin 2 → ℝ) :
    cov[fun ω => sourceField 2 (-24) ω x, fun ω => sourceField 2 (-24) ω y; sourceLaw 2] =
      sourceSpatialKernel 2 (-24) (x-y) := sourceField_covariance_spatial 2 (by norm_num) x y

example (x y t : Fin 3 → ℝ) :
    cov[fun ω => sourceField 3 24 ω (x+t), fun ω => sourceField 3 24 ω (y+t); sourceLaw 3] =
      sourceSpatialKernel 3 24 (x-y) := by
  simpa only [add_sub_add_right_eq_sub] using
    sourceField_covariance_spatial 3 (L := 24) (by norm_num) (x+t) (y+t)

example (q p : SourceTorus 3 24) :
    cov[fun ω => torusSourceField 3 24 ω q, fun ω => torusSourceField 3 24 ω p; sourceLaw 3] =
      sourceSpatialKernel 3 24 (torusRepresentative 3 24 q - torusRepresentative 3 24 p) := by
  rw [torusSourceField_covariance, spectralKernel_eq_spatialKernel 3 (by norm_num)]

example (x y : Fin 3 → ℝ) :
    cov[fun ω => torusSourceField 3 24 ω (torusProjection 3 24 x),
      fun ω => torusSourceField 3 24 ω (torusProjection 3 24 y); sourceLaw 3] =
      sourceSpatialKernel 3 24 (x-y) := by
  rw [torusSourceField_covariance_pullback, spectralKernel_eq_spatialKernel 3 (by norm_num)]

