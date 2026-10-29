import Mathlib.MeasureTheory.Order.Group.Lattice
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.HasLaw
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order
import Mathlib.Tactic.FunProp
import Mathlib.MeasureTheory.Function.LpSpace.InfiniteSum
import Mathlib.MeasureTheory.Function.LpSeminorm.SMul
import Mathlib.MeasureTheory.Function.L1Space.Integrable

/-! Weighted absolute Gaussian series. The coordinates need only standard
Gaussian marginals; no independence is assumed. See SOURCE_GAUSSIAN_MAJORANTS.md. -/

noncomputable section
open MeasureTheory ProbabilityTheory Filter
open scoped BigOperators Topology ENNReal NNReal
set_option autoImplicit false

namespace I4GaussianMajorant

def weightedAbsSeries {I Ω : Type*} (w : I → ℝ) (U : I → Ω → ℝ) (ω : Ω) : ℝ :=
  ∑' i, w i * |U i ω|

def standardAbsMoment (n : ℕ) : ℝ := ∫ z : ℝ, |z| ^ n ∂gaussianReal 0 1

theorem standardAbsMoment_integrable (n : ℕ) :
    Integrable (fun z : ℝ => |z| ^ n) (gaussianReal 0 1) := by
  simpa only [Real.norm_eq_abs, id_eq] using
    (memLp_id_gaussianReal (μ := 0) (v := 1) (n : ℝ≥0)).integrable_norm_pow'

theorem weightedAbsSeries_nonneg {I Ω : Type*} (w : I → ℝ) (U : I → Ω → ℝ)
    (hw0 : ∀ i, 0 ≤ w i) (ω : Ω) : 0 ≤ weightedAbsSeries w U ω :=
  tsum_nonneg (fun i => mul_nonneg (hw0 i) (abs_nonneg _))

theorem weightedAbsSeries_measurable {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (w : I → ℝ) (U : I → Ω → ℝ) (hU : ∀ i, Measurable (U i)) :
    Measurable (weightedAbsSeries w U) :=
  Measurable.tsum (fun i => measurable_const.mul (continuous_abs.measurable.comp (hU i)))

private theorem coordinate_eLpNorm {I Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (w : I → ℝ) (U : I → Ω → ℝ) (hw0 : ∀ i, 0 ≤ w i)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P) (p : ℝ≥0∞) (i : I) :
    eLpNorm (fun ω => w i * |U i ω|) p P =
      ENNReal.ofReal (w i) * eLpNorm id p (gaussianReal 0 1) := by
  have hmap := eLpNorm_map_measure (p := p) (f := U i) (g := fun z : ℝ => ‖z‖)
    (by fun_prop) (hlaw i).aemeasurable
  have hn : eLpNorm (fun z : ℝ => ‖z‖) p (gaussianReal 0 1) =
      eLpNorm id p (gaussianReal 0 1) := by
    simpa only [id_eq] using eLpNorm_norm (μ := gaussianReal 0 1) (p := p) id aestronglyMeasurable_id
  rw [(hlaw i).map_eq, hn] at hmap
  have hu : eLpNorm (fun ω => |U i ω|) p P = eLpNorm id p (gaussianReal 0 1) := by
    simpa only [Function.comp_def, Real.norm_eq_abs] using hmap.symm
  calc
    _ = ‖w i‖ₑ * eLpNorm (fun ω => |U i ω|) p P := by
      have heq : (w i • (fun ω => |U i ω|)) = (fun ω => w i * |U i ω|) := by
        funext ω
        rfl
      rw [← heq]
      exact eLpNorm_const_smul (w i) (fun ω => |U i ω|) p P
    _ = _ := by rw [hu]; simp [Real.enorm_eq_ofReal_abs, abs_of_nonneg (hw0 i)]

theorem weightedAbsSeries_memLp_bound {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (w : I → ℝ) (U : I → Ω → ℝ)
    (hU : ∀ i, Measurable (U i)) (hw0 : ∀ i, 0 ≤ w i) (hw : Summable w)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P)
    (p : ℝ≥0) (hp : 1 ≤ p) :
    MemLp (weightedAbsSeries w U) p P ∧
      eLpNorm (weightedAbsSeries w U) p P ≤
        ENNReal.ofReal (∑' i, w i) * eLpNorm id p (gaussianReal 0 1) := by
  classical
  let F : I → Ω → ℝ := fun i ω => w i * |U i ω|
  let C := ENNReal.ofReal (∑' i, w i) * eLpNorm id p (gaussianReal 0 1)
  have hG := memLp_id_gaussianReal (μ := 0) (v := 1) p
  have hC : C ≠ ∞ := ENNReal.mul_ne_top ENNReal.ofReal_ne_top hG.eLpNorm_ne_top
  have hsum : (∑' i, eLpNorm (F i) p P) = C := by
    simp_rw [F, coordinate_eLpNorm P w U hw0 hlaw]
    rw [ENNReal.tsum_mul_right, ← ENNReal.ofReal_tsum_of_nonneg hw0 hw]
  have hae : ∀ᵐ ω ∂P, Summable (fun i => F i ω) := by
    have ht := summable_norm_of_tsum_eLpNorm_ne_top (show (1 : ℝ≥0∞) ≤ p by exact_mod_cast hp)
      (by rw [hsum]; exact hC)
    filter_upwards [ht] with ω hω
    simpa only [F, Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (hw0 _) (abs_nonneg _))] using hω
  have hbound (s : Finset I) : eLpNorm (fun ω => ∑ i ∈ s, F i ω) p P ≤ C := by
    calc
      _ ≤ ∑ i ∈ s, eLpNorm (F i) p P :=
        by
          have heq : (∑ i ∈ s, F i) = (fun ω => ∑ i ∈ s, F i ω) := by
            funext ω
            simp
          rw [← heq]
          exact
            (eLpNorm_sum_le (μ := P) (f := F) (s := s)
              (show (1 : ℝ≥0∞) ≤ p by exact_mod_cast hp))
      _ ≤ ∑' i, eLpNorm (F i) p P := ENNReal.sum_le_tsum s
      _ = C := hsum
  have hnorm : eLpNorm (weightedAbsSeries w U) p P ≤ C := by
    apply Lp.eLpNorm_le_of_ae_tendsto (u := (atTop : Filter (Finset I)))
      (f := fun s ω => ∑ i ∈ s, F i ω) (Filter.Eventually.of_forall hbound)
      (fun s => (s.measurable_fun_sum (fun i _ => measurable_const.mul (continuous_abs.measurable.comp (hU i)))).aestronglyMeasurable)
      (weightedAbsSeries_measurable w U hU).aestronglyMeasurable
    filter_upwards [hae] with ω hω
    exact hω.hasSum
  exact ⟨lt_of_le_of_lt hnorm (lt_top_iff_ne_top.mpr hC), hnorm⟩

theorem weightedAbsSeries_ae_summable {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (w : I → ℝ) (U : I → Ω → ℝ)
    (_hU : ∀ i, Measurable (U i)) (hw0 : ∀ i, 0 ≤ w i) (hw : Summable w)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P) :
    ∀ᵐ ω ∂P, Summable (fun i => w i * |U i ω|) := by
  have hG := memLp_id_gaussianReal (μ := 0) (v := 1) 1
  have hs : (∑' i, eLpNorm (fun ω => w i * |U i ω|) 1 P) ≠ ∞ := by
    simp_rw [coordinate_eLpNorm P w U hw0 hlaw]
    rw [ENNReal.tsum_mul_right, ← ENNReal.ofReal_tsum_of_nonneg hw0 hw]
    exact ENNReal.mul_ne_top ENNReal.ofReal_ne_top hG.eLpNorm_ne_top
  have ht := summable_norm_of_tsum_eLpNorm_ne_top (le_refl (1 : ℝ≥0∞)) hs
  filter_upwards [ht] with ω hω
  simpa only [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (hw0 _) (abs_nonneg _))] using hω

private theorem eLpNorm_nat_pow_integral {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (f : Ω → ℝ) (hf : AEStronglyMeasurable f P)
    (hnonneg : ∀ ω, 0 ≤ f ω) (n : ℕ) (hn : n ≠ 0)
    (hint : Integrable (fun ω => f ω ^ n) P) :
    eLpNorm f n P ^ n = ENNReal.ofReal (∫ ω, f ω ^ n ∂P) := by
  have he := eLpNorm_nnreal_pow_eq_lintegral (p := (n : ℝ≥0))
    (by exact_mod_cast hn) hf
  have he' : eLpNorm f n P ^ n = ∫⁻ ω, ENNReal.ofReal (f ω ^ n) ∂P := by
    simpa only [NNReal.coe_natCast, ENNReal.coe_natCast, ENNReal.rpow_natCast,
      Real.enorm_eq_ofReal_abs, abs_of_nonneg (hnonneg _),
      ← ENNReal.ofReal_pow (hnonneg _)] using he
  rw [he', ofReal_integral_eq_lintegral_ofReal hint
    (Filter.Eventually.of_forall (fun ω => pow_nonneg (hnonneg ω) n))]

theorem weightedAbsSeries_moment {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (w : I → ℝ) (U : I → Ω → ℝ)
    (hU : ∀ i, Measurable (U i)) (hw0 : ∀ i, 0 ≤ w i) (hw : Summable w)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P) (n : ℕ) :
    Integrable (fun ω => weightedAbsSeries w U ω ^ n) P ∧
      (∫ ω, weightedAbsSeries w U ω ^ n ∂P) ≤
        (∑' i, w i) ^ n * standardAbsMoment n := by
  rcases eq_or_ne n 0 with rfl | hn
  · simp [standardAbsMoment]
  have hp : (1 : ℝ≥0) ≤ n := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hn
  have hb := weightedAbsSeries_memLp_bound P w U hU hw0 hw hlaw (n : ℝ≥0) hp
  have hnonneg := weightedAbsSeries_nonneg w U hw0
  have hi : Integrable (fun ω => weightedAbsSeries w U ω ^ n) P := by
    simpa only [Real.norm_eq_abs, abs_of_nonneg (hnonneg _)] using hb.1.integrable_norm_pow'
  refine ⟨hi, ?_⟩
  have hstd := standardAbsMoment_integrable n
  have hpow := pow_le_pow_left' hb.2 n
  have href : eLpNorm id (n : ℝ≥0∞) (gaussianReal 0 1) ^ n =
      ENNReal.ofReal (standardAbsMoment n) := by
    have he := eLpNorm_nat_pow_integral (gaussianReal 0 1) (fun z : ℝ => |z|)
      (by fun_prop) (fun z => abs_nonneg z) n hn hstd
    have hnorm : eLpNorm (fun z : ℝ => |z|) (n : ℝ≥0∞) (gaussianReal 0 1) =
        eLpNorm id (n : ℝ≥0∞) (gaussianReal 0 1) := by
      simpa only [Real.norm_eq_abs, id_eq] using
        eLpNorm_norm (μ := gaussianReal 0 1) (p := (n : ℝ≥0∞)) id aestronglyMeasurable_id
    rw [hnorm] at he
    exact he
  change eLpNorm (weightedAbsSeries w U) (n : ℝ≥0∞) P ^ n ≤
    (ENNReal.ofReal (∑' i, w i) * eLpNorm id (n : ℝ≥0∞) (gaussianReal 0 1)) ^ n at hpow
  rw [mul_pow, eLpNorm_nat_pow_integral P (weightedAbsSeries w U)
    (weightedAbsSeries_measurable w U hU).aestronglyMeasurable hnonneg n hn hi, href,
    ← ENNReal.ofReal_pow (tsum_nonneg hw0), ← ENNReal.ofReal_mul (pow_nonneg (tsum_nonneg hw0) n)] at hpow
  exact (ENNReal.ofReal_le_ofReal_iff (mul_nonneg (pow_nonneg (tsum_nonneg hw0) n)
    (integral_nonneg (fun z => pow_nonneg (abs_nonneg z) n)))).mp hpow

/- Exponential support coauthored by
/root/normalizer_palm_research/gaussian_reference_probe. -/

private lemma exp_mul_abs_le (s x : ℝ) :
    Real.exp (s * |x|) ≤ Real.exp (s*x) + Real.exp ((-s)*x) := by
  rcases le_total 0 x with hx | hx
  · rw [abs_of_nonneg hx]
    linarith [Real.exp_pos ((-s)*x)]
  · rw [abs_of_nonpos hx]
    have h : s * (-x) = (-s)*x := by ring
    rw [h]
    linarith [Real.exp_pos (s*x)]

private theorem foldedGaussian_exp (s : ℝ) :
    Integrable (fun z : ℝ => Real.exp (s * |z|)) (gaussianReal 0 1) ∧
      (∫ z : ℝ, Real.exp (s * |z|) ∂gaussianReal 0 1) ≤ 2*Real.exp (s^2/2) := by
  have hp := integrable_exp_mul_gaussianReal (μ := 0) (v := 1) s
  have hn := integrable_exp_mul_gaussianReal (μ := 0) (v := 1) (-s)
  have hi : Integrable (fun z : ℝ => Real.exp (s * |z|)) (gaussianReal 0 1) := by
    apply (hp.add hn).mono' (by fun_prop)
    filter_upwards [] with z
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact exp_mul_abs_le s z
  refine ⟨hi, ?_⟩
  calc
    _ ≤ ∫ z : ℝ, (Real.exp (s*z) + Real.exp ((-s)*z)) ∂gaussianReal 0 1 :=
      integral_mono hi (hp.add hn) (fun z => exp_mul_abs_le s z)
    _ = 2*Real.exp (s^2/2) := by
      rw [integral_add hp hn]
      have hplus := congrFun (mgf_fun_id_gaussianReal (μ := 0) (v := 1)) s
      have hminus := congrFun (mgf_fun_id_gaussianReal (μ := 0) (v := 1)) (-s)
      simp only [mgf, zero_mul, zero_add, NNReal.coe_one, one_mul, neg_sq] at hplus hminus
      rw [hplus, hminus]
      ring

private lemma finite_exp_jensen {I : Type*} (s : Finset I) (w x : I → ℝ)
    (W t : ℝ) (hW : 0 < W) (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i ∈ s, w i ≤ W) :
    Real.exp (t * ∑ i ∈ s, w i * |x i|) ≤
      (1-∑ i ∈ s, w i/W) + ∑ i ∈ s, (w i/W)*Real.exp (t*W*|x i|) := by
  have hmass : ∑ i ∈ s, w i/W ≤ 1 := by
    rw [← Finset.sum_div]
    exact (div_le_one hW).mpr hsum
  have hj := convexOn_exp.map_add_sum_le
    (t := s) (w := fun i => w i/W) (p := fun i => t*W*|x i|)
    (v := 1-∑ i ∈ s, w i/W) (q := 0)
    (fun i _ => div_nonneg (hw i) hW.le) (by ring)
    (fun _ _ => Set.mem_univ _) (by linarith) (Set.mem_univ _)
  simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at hj
  have heq : (∑ i ∈ s, (w i/W)*(t*W*|x i|)) = t * ∑ i ∈ s, w i*|x i| := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    field_simp
  simpa only [heq] using hj

private lemma finite_exp_integrable_bound {I Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (s : Finset I) (w : I → ℝ)
    (U : I → Ω → ℝ) (hmeas : ∀ i, Measurable (U i))
    (W t K : ℝ) (hW : 0 < W) (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i ∈ s, w i ≤ W) (hK : 1 ≤ K)
    (hUi : ∀ i, Integrable (fun ω => Real.exp (t*W*|U i ω|)) P)
    (hUb : ∀ i, (∫ ω, Real.exp (t*W*|U i ω|) ∂P) ≤ K) :
    Integrable (fun ω => Real.exp (t*∑ i ∈ s, w i*|U i ω|)) P ∧
      (∫ ω, Real.exp (t*∑ i ∈ s, w i*|U i ω|) ∂P) ≤ K := by
  let a := ∑ i ∈ s, w i/W
  have ha0 : 0 ≤ a := Finset.sum_nonneg (fun i _ => div_nonneg (hw i) hW.le)
  have ha1 : a ≤ 1 := by
    dsimp [a]
    rw [← Finset.sum_div]
    exact (div_le_one hW).mpr hsum
  have hsumint : Integrable (fun ω => ∑ i ∈ s, (w i/W)*Real.exp (t*W*|U i ω|)) P :=
    integrable_finsetSum _ (fun i _ => (hUi i).const_mul _)
  have hrhs : Integrable (fun ω => (1-a)+∑ i ∈ s, (w i/W)*Real.exp (t*W*|U i ω|)) P :=
    (integrable_const _).add hsumint
  have hmeasf : AEStronglyMeasurable (fun ω => Real.exp (t*∑ i ∈ s, w i*|U i ω|)) P := by
    fun_prop
  have hint : Integrable (fun ω => Real.exp (t*∑ i ∈ s, w i*|U i ω|)) P := by
    apply hrhs.mono' hmeasf
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact finite_exp_jensen s w (fun i => U i ω) W t hW hw hsum
  refine ⟨hint, ?_⟩
  calc
    _ ≤ ∫ ω, ((1-a)+∑ i ∈ s, (w i/W)*Real.exp (t*W*|U i ω|)) ∂P :=
      integral_mono hint hrhs (fun ω => finite_exp_jensen s w (fun i => U i ω) W t hW hw hsum)
    _ = (1-a)+∑ i ∈ s, (w i/W)*(∫ ω, Real.exp (t*W*|U i ω|) ∂P) := by
      rw [integral_add (integrable_const _) hsumint, integral_finsetSum _
        (fun i _ => (hUi i).const_mul _)]
      simp only [integral_const_mul, integral_const, probReal_univ, smul_eq_mul, one_mul]
    _ ≤ (1-a)+∑ i ∈ s, (w i/W)*K := by
      exact add_le_add le_rfl (Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (hUb i)
        (div_nonneg (hw i) hW.le)))
    _ = (1-a)+a*K := by rw [← Finset.sum_mul]
    _ ≤ K := by nlinarith


private theorem weightedAbsSeries_exp_posWeight {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (w : I → ℝ) (U : I → Ω → ℝ)
    (hmeas : ∀ i, Measurable (U i)) (hw : ∀ i, 0 ≤ w i) (hws : Summable w)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P)
    (haesum : ∀ᵐ ω ∂P, Summable (fun i => w i*|U i ω|))
    (hW : 0 < ∑' i, w i) (t : ℝ) :
    Integrable (fun ω => Real.exp (t*weightedAbsSeries w U ω)) P ∧
      (∫ ω, Real.exp (t*weightedAbsSeries w U ω) ∂P) ≤
        2*Real.exp (t^2*(∑' i, w i)^2/2) := by
  let W := ∑' i, w i
  let K := 2*Real.exp ((t*W)^2/2)
  have hK : 1 ≤ K := by
    have hexp : 1 ≤ Real.exp ((t*W)^2/2) := Real.one_le_exp_iff.mpr (by positivity)
    dsimp [K]
    linarith
  have hUi (i : I) : Integrable (fun ω => Real.exp (t*W*|U i ω|)) P :=
    (hlaw i).integrable_comp (foldedGaussian_exp (t*W)).1
  have hUb (i : I) : (∫ ω, Real.exp (t*W*|U i ω|) ∂P) ≤ K := by
    have heq := (hlaw i).integral_comp
      (show AEStronglyMeasurable (fun z : ℝ => Real.exp (t*W*|z|)) (gaussianReal 0 1) by fun_prop)
    exact heq.trans_le (foldedGaussian_exp (t*W)).2
  have hfinite (s : Finset I) :
      Integrable (fun ω => Real.exp (t*∑ i ∈ s, w i*|U i ω|)) P ∧
        (∫ ω, Real.exp (t*∑ i ∈ s, w i*|U i ω|) ∂P) ≤ K :=
    finite_exp_integrable_bound P s w U hmeas W t K hW hw
      (hws.sum_le_tsum s (fun i _ => hw i)) hK hUi hUb
  have hlim : (fun ω => liminf (fun s : Finset I =>
      ENNReal.ofReal (Real.exp (t*∑ i ∈ s, w i*|U i ω|))) atTop) =ᵐ[P]
      (fun ω => ENNReal.ofReal (Real.exp (t*weightedAbsSeries w U ω))) := by
    filter_upwards [haesum] with ω hω
    have hc : Tendsto (fun s : Finset I => Real.exp (t*∑ i ∈ s, w i*|U i ω|)) atTop
        (𝓝 (Real.exp (t*weightedAbsSeries w U ω))) :=
      (Real.continuous_exp.tendsto _).comp (tendsto_const_nhds.mul hω.hasSum)
    exact ((ENNReal.continuous_ofReal.tendsto _).comp hc).liminf_eq
  have hfinbound (s : Finset I) :
      (∫⁻ ω, ENNReal.ofReal (Real.exp (t*∑ i ∈ s, w i*|U i ω|)) ∂P) ≤ ENNReal.ofReal K := by
    rw [← ofReal_integral_eq_lintegral_ofReal (hfinite s).1
      (Eventually.of_forall (fun ω => (Real.exp_pos _).le))]
    exact ENNReal.ofReal_le_ofReal (hfinite s).2
  have hlin : (∫⁻ ω, ENNReal.ofReal (Real.exp (t*weightedAbsSeries w U ω)) ∂P) ≤
      ENNReal.ofReal K := by
    rw [← lintegral_congr_ae hlim]
    calc
      _ ≤ liminf (fun s : Finset I => ∫⁻ ω,
        ENNReal.ofReal (Real.exp (t*∑ i ∈ s, w i*|U i ω|)) ∂P) atTop :=
        lintegral_liminf_le (fun s => by fun_prop)
      _ ≤ liminf (fun _s : Finset I => ENNReal.ofReal K) atTop :=
        liminf_le_liminf (Eventually.of_forall hfinbound)
      _ = ENNReal.ofReal K := liminf_const _
  have hmeasGamma : Measurable (weightedAbsSeries w U) := by
    apply Measurable.tsum
    intro i
    exact measurable_const.mul (hmeas i).abs
  have hint : Integrable (fun ω => Real.exp (t*weightedAbsSeries w U ω)) P := by
    refine ⟨by fun_prop, ?_⟩
    apply (hasFiniteIntegral_iff_ofReal (Eventually.of_forall (fun ω => (Real.exp_pos _).le))).mpr
    exact hlin.trans_lt ENNReal.ofReal_lt_top
  refine ⟨hint, ?_⟩
  have hb : (∫ ω, Real.exp (t*weightedAbsSeries w U ω) ∂P) ≤ K := by
    rw [← ofReal_integral_eq_lintegral_ofReal hint
      (Eventually.of_forall (fun ω => (Real.exp_pos _).le))] at hlin
    exact (ENNReal.ofReal_le_ofReal_iff (by linarith : 0 ≤ K)).mp hlin
  convert hb using 1; simp only [K, mul_pow, W]


theorem weightedAbsSeries_exp_bound {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (w : I → ℝ) (U : I → Ω → ℝ)
    (hmeas : ∀ i, Measurable (U i)) (hw : ∀ i, 0 ≤ w i) (hws : Summable w)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P)
    (t : ℝ) :
    Integrable (fun ω => Real.exp (t*weightedAbsSeries w U ω)) P ∧
      (∫ ω, Real.exp (t*weightedAbsSeries w U ω) ∂P) ≤
        2*Real.exp (t^2*(∑' i, w i)^2/2) := by
  have haesum := weightedAbsSeries_ae_summable P w U hmeas hw hws hlaw
  classical
  by_cases hzero : (∑' i, w i) = 0
  · have hwi (i : I) : w i = 0 := by
      have hi : w i ≤ ∑' i, w i := by
        simpa only [Finset.sum_singleton] using
          hws.sum_le_tsum ({i} : Finset I) (fun j _ => hw j)
      exact le_antisymm (hzero ▸ hi) (hw i)
    constructor
    · simp [weightedAbsSeries, hwi]
    · simp [weightedAbsSeries, hwi]
  · exact weightedAbsSeries_exp_posWeight P w U hmeas hw hws hlaw haesum
      (lt_of_le_of_ne (tsum_nonneg hw) (Ne.symm hzero)) t


/-- Exponential Markov with t=R/W² gives the Gaussian upper tail without independence. -/
theorem weightedAbsSeries_upper_tail {I Ω : Type*} [Countable I] [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] (w : I → ℝ) (U : I → Ω → ℝ)
    (hU : ∀ i, Measurable (U i)) (hw0 : ∀ i, 0 ≤ w i) (hw : Summable w)
    (hlaw : ∀ i, HasLaw (U i) (gaussianReal 0 1) P)
    (hW : 0 < ∑' i, w i) (R : ℝ) (hR : 0 ≤ R) :
    P.real {ω | R ≤ weightedAbsSeries w U ω} ≤
      2 * Real.exp (-(R ^ 2) / (2 * (∑' i, w i) ^ 2)) := by
  let W := ∑' i, w i
  let t := R / W ^ 2
  have hW0 : W ≠ 0 := ne_of_gt hW
  have ht : 0 ≤ t := div_nonneg hR (sq_nonneg W)
  have hb := weightedAbsSeries_exp_bound P w U hU hw0 hw hlaw t
  have hm := measure_ge_le_exp_mul_mgf (μ := P) (X := weightedAbsSeries w U) R ht hb.1
  calc
    _ ≤ Real.exp (-t * R) * mgf (weightedAbsSeries w U) P t := hm
    _ ≤ Real.exp (-t * R) * (2 * Real.exp (t ^ 2 * W ^ 2 / 2)) :=
      mul_le_mul_of_nonneg_left hb.2 (Real.exp_pos _).le
    _ = 2 * Real.exp (-t * R + t ^ 2 * W ^ 2 / 2) := by
      rw [Real.exp_add]
      ring
    _ = _ := by
      congr 2
      dsimp [t]
      field_simp [hW0]
      ring

end I4GaussianMajorant
