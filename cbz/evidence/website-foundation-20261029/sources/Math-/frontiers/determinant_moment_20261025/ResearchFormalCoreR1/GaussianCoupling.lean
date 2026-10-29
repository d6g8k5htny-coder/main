import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.HasLaw
import Mathlib.Analysis.SpecialFunctions.ContinuousFunctionalCalculus.Rpow.Isometric

/-!
# One continuous standard-Gaussian coupling of a finite Gaussian family

PSD square-root continuity allows singular covariance and arbitrary varying
means. The affine maps have the literal pinned multivariateGaussian laws.
Actual source endpoint means/covariances and their limits remain application
joins; see ../SOURCE_GAUSSIAN_COUPLING.md for the frozen source contract.
-/

namespace ResearchFormalCoreR1
open MeasureTheory ProbabilityTheory Filter Matrix WithLp
open scoped Topology ENNReal NNReal MatrixOrder Matrix.Norms.L2Operator
set_option autoImplicit false

private theorem gc_matrix_eval_tendsto {ι R : Type*} [Fintype ι] [DecidableEq ι]
    {u : Filter R} {S : R → Matrix ι ι ℝ} {S0 : Matrix ι ι ℝ}
    (hS : Tendsto S u (𝓝 S0)) (z : EuclideanSpace ℝ ι) :
    Tendsto (fun r => Matrix.toEuclideanCLM (𝕜 := ℝ) (S r) z) u
      (𝓝 (Matrix.toEuclideanCLM (𝕜 := ℝ) S0 z)) := by
  have hcoord (i j : ι) : Tendsto (fun r => S r i j) u (𝓝 (S0 i j)) :=
    ((continuous_apply j).tendsto (S0 i)).comp
      (((continuous_apply i).tendsto S0).comp hS)
  have hvec : Tendsto (fun r => (S r) *ᵥ ofLp z) u (𝓝 (S0 *ᵥ ofLp z)) := by
    apply tendsto_pi_nhds.mpr
    intro i
    exact tendsto_finsetSum Finset.univ (fun j _ =>
      (hcoord i j).mul tendsto_const_nhds)
  change Tendsto (fun r => toLp 2 ((S r) *ᵥ ofLp z)) u
    (𝓝 (toLp 2 (S0 *ᵥ ofLp z)))
  exact ((PiLp.continuous_toLp 2 (fun _ : ι => ℝ)).tendsto _).comp hvec

/-- The canonical affine vector on one common standard Gaussian space. -/
noncomputable def gc_coupledVector {ι : Type*} [Fintype ι] [DecidableEq ι]
    (m : EuclideanSpace ℝ ι) (S : Matrix ι ι ℝ) (z : EuclideanSpace ℝ ι) :
    EuclideanSpace ℝ ι := m + Matrix.toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S) z

/-- Exact marginal law; this is the pinned multivariateGaussian definition. -/
theorem gc_coupledVector_hasLaw {ι : Type*} [Fintype ι] [DecidableEq ι]
    (m : EuclideanSpace ℝ ι) (S : Matrix ι ι ℝ) :
    HasLaw (gc_coupledVector m S) (multivariateGaussian m S)
      (stdGaussian (EuclideanSpace ℝ ι)) where
  aemeasurable := by unfold gc_coupledVector; fun_prop
  map_eq := rfl

/-- Every common Gaussian coordinate converges when means and PSD covariances
converge. No invertibility, fixed rank, or square-root Lipschitz estimate. -/
theorem gc_coupledVector_tendsto {ι κ : Type*} [Fintype ι] [DecidableEq ι]
    (u : Filter κ) (m : κ → EuclideanSpace ℝ ι) (S : κ → Matrix ι ι ℝ)
    (m0 : EuclideanSpace ℝ ι) (S0 : Matrix ι ι ℝ)
    (hm : Tendsto m u (𝓝 m0)) (hS : Tendsto S u (𝓝 S0))
    (hPSD : ∀ᶠ t in u, (S t).PosSemidef) (hPSD0 : S0.PosSemidef)
    (z : EuclideanSpace ℝ ι) :
    Tendsto (fun t => gc_coupledVector (m t) (S t) z) u
      (𝓝 (gc_coupledVector m0 S0 z)) := by
  have hwithin : Tendsto S u (𝓝[{B : Matrix ι ι ℝ | 0 ≤ B}] S0) :=
    tendsto_nhdsWithin_iff.mpr ⟨hS, hPSD.mono (fun t ht => ht.nonneg)⟩
  have hsqrt : Tendsto (fun t => CFC.sqrt (S t)) u (𝓝 (CFC.sqrt S0)) :=
    (CFC.continuousOn_sqrt S0 hPSD0.nonneg).tendsto.comp hwithin
  have heval := gc_matrix_eval_tendsto hsqrt z
  exact hm.add heval
end ResearchFormalCoreR1
