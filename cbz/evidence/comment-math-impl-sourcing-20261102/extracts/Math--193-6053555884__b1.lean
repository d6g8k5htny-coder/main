import SourceJetLaw

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Jets I4JetLaw I4Kernel
  I4SourceCovariance MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal

-- Zeroth-order observations are the unchanged field, for every coefficient sample.
example (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ) :
    sourceJet d L 0 ω x (fun i => Fin.elim0 i) = sourceField d L ω x :=
  sourceJet_zero d L ω x _

example (d : ℕ) (L : ℝ) (m : SourceMode d) (x : Fin d → ℝ) :
    modeJetCoefficient d L m 0 x (fun i => Fin.elim0 i) = modeCoefficient d L m x := rfl

-- The actual constant Fourier coefficient has no positive-order derivatives.
example (d : ℕ) (L : ℝ) (j : ℕ) (x : Fin d → ℝ)
    (v : Fin (j+1) → (Fin d → ℝ)) :
    modeJetCoefficient d L none (j+1) x v = 0 := by
  unfold modeJetCoefficient
  change (iteratedFDeriv ℝ (j+1) (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L)) x) v = 0
  rw [iteratedFDeriv_succ_const]
  simp

-- Zero-direction observations are identically zero, including the order-four boundary.
example (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ) :
    sourceJet d L 1 ω x (fun _ => 0) = 0 := by
  unfold sourceJet
  exact ContinuousMultilinearMap.map_zero _

example (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ) :
    sourceJet d L 4 ω x (fun _ => 0) = 0 := by
  unfold sourceJet
  exact ContinuousMultilinearMap.map_zero _

-- Empty physical dimension forces every positive-order direction to be zero.
example (L : ℝ) (j : ℕ) (ω : SourceMode 0 → ℝ) (x : Fin 0 → ℝ)
    (v : Fin (j+1) → (Fin 0 → ℝ)) : sourceJet 0 L (j+1) ω x v = 0 := by
  unfold sourceJet
  rw [show v = 0 from Subsingleton.elim _ _]
  exact ContinuousMultilinearMap.map_zero _

-- The empty observation vector is a legitimate degenerate Gaussian vector.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (r : Fin 0 → ℕ)
    (x : Fin 0 → (Fin d → ℝ)) (v : (i : Fin 0) → Fin (r i) → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω i => sourceJet d L (r i) ω (x i) (v i)) (sourceLaw d) :=
  sourceJet_finiteVector_hasGaussianLaw d hL 0 r (fun i => Fin.elim0 i) x v

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (r : Fin 0 → ℕ)
    (x : Fin 0 → (Fin d → ℝ)) (v : (i : Fin 0) → Fin (r i) → (Fin d → ℝ))
    (c : Fin 0 → ℝ) :
    HasGaussianLaw (fun ω => ∑ i : Fin 0, c i * sourceJet d L (r i) ω (x i) (v i))
      (sourceLaw d) :=
  sourceJet_linearCombination_hasGaussianLaw d hL 0 r (fun i => Fin.elim0 i) x v c

-- One true joint vector mixes orders0,1,2,3,4, points and direction tuples.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin 5 → (Fin d → ℝ))
    (v : (i : Fin 5) → Fin i.val → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω (i : Fin 5) => sourceJet d L i.val ω (x i) (v i)) (sourceLaw d) := by
  exact sourceJet_finiteVector_hasGaussianLaw d hL 5 (fun i => i.val)
    (fun i => by omega) x v

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin 5 → (Fin d → ℝ))
    (v : (i : Fin 5) → Fin i.val → (Fin d → ℝ)) (c : Fin 5 → ℝ) :
    HasGaussianLaw (fun ω => ∑ i : Fin 5, c i * sourceJet d L i.val ω (x i) (v i))
      (sourceLaw d) := by
  exact sourceJet_linearCombination_hasGaussianLaw d hL 5 (fun i => i.val)
    (fun i => by omega) x v c

-- Duplicate observations are allowed; their contrast is exactly zero.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) (v : Fin 1 → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω (_ : Fin 2) => sourceJet d L 1 ω x v) (sourceLaw d) := by
  exact sourceJet_finiteVector_hasGaussianLaw d hL 2 (fun _ => 1)
    (fun _ => by norm_num) (fun _ => x) (fun _ => v)

example (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) (x : Fin d → ℝ)
    (v : Fin 1 → (Fin d → ℝ)) :
    sourceJet d L 1 ω x v - sourceJet d L 1 ω x v = 0 := sub_self _

example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (v : Fin 1 → (Fin d → ℝ)) :
    Var[fun ω => sourceJet d L 1 ω x v - sourceJet d L 1 ω x v; sourceLaw d] = 0 := by
  simp only [sub_self]
  exact variance_zero _

-- Degenerate derivative laws are N(0,0), rather than a falsely asserted positive-rank law.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    (sourceLaw d).map (fun ω => sourceJet d L 1 ω x (fun _ => 0)) = gaussianReal 0 0 := by
  simpa [modeJetCoefficient] using sourceJet_map_gaussian d hL 1 (by norm_num) x (fun _ => 0)

example {L : ℝ} (hL : L ≠ 0) (x : Fin 0 → ℝ) :
    Var[fun ω => sourceJet 0 L 4 ω x (fun _ => 0); sourceLaw 0] = 0 := by
  have hc (m : SourceMode 0) : modeJetCoefficient 0 L m 4 x (fun _ => 0) = 0 := by
    unfold modeJetCoefficient
    exact ContinuousMultilinearMap.map_zero _
  simpa only [hc, zero_pow (by norm_num : (2 : ℕ) ≠ 0), tsum_zero]
    using sourceJet_variance 0 hL 4 (by norm_num) x (fun _ => 0)

-- Order0 recovers the already proved centered, unit-variance physical field law.
example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    ∫ ω, sourceField d L ω x ∂sourceLaw d = 0 := by
  simpa only [sourceJet_zero] using sourceJet_integral_zero d hL 0 (by norm_num) x
    (fun i => Fin.elim0 i)

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    Var[fun ω => sourceJet d L 0 ω x (fun i => Fin.elim0 i); sourceLaw d] = 1 := by
  simp_rw [sourceJet_zero]
  exact sourceField_variance_one d hL x

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x y : Fin d → ℝ) :
    cov[fun ω => sourceJet d L 0 ω x (fun i => Fin.elim0 i),
      fun ω => sourceJet d L 0 ω y (fun i => Fin.elim0 i); sourceLaw d] =
      sourceSpectralKernel d L (x-y) := by
  simp_rw [sourceJet_zero]
  exact sourceField_covariance d hL x y

-- Both covariance orders may be4; this needs no sample-C8 assumption.
example (x y : Fin 3 → ℝ) (v w : Fin 4 → (Fin 3 → ℝ)) :
    cov[fun ω => sourceJet 3 24 4 ω x v,
      fun ω => sourceJet 3 24 4 ω y w; sourceLaw 3] =
      ∑' m, modeJetCoefficient 3 24 m 4 x v * modeJetCoefficient 3 24 m 4 y w :=
  sourceJet_covariance 3 (by norm_num) 4 4 (by norm_num) (by norm_num) x y v w

-- Graded cross covariance preserves distinct derivative orders and points.
example (x y : Fin 2 → ℝ) (v : Fin 1 → (Fin 2 → ℝ)) (w : Fin 3 → (Fin 2 → ℝ)) :
    cov[fun ω => sourceJet 2 24 1 ω x v,
      fun ω => sourceJet 2 24 3 ω y w; sourceLaw 2] =
      ∑' m, modeJetCoefficient 2 24 m 1 x v * modeJetCoefficient 2 24 m 3 y w :=
  sourceJet_covariance 2 (by norm_num) 1 3 (by norm_num) (by norm_num) x y v w

