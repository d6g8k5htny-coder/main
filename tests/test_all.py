"""
Unit tests for the complexity physics code.

Besides arithmetic checks, these tests pin the findings recorded in
results/validation_report.md, so reintroducing a defect fails a test.

Run with: pytest tests/test_all.py -v
"""

import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from src.complexity import (
    TotalComplexity,
    RetrodictionComplexity,
    BarrierComplexity,
    ComplexityParameters,
    compute_generation_complexity,
    demonstrate_three_generations
)
from src.gauge_theory import (
    GaugeGroupComplexity,
    RepresentationComplexity,
    StandardModelAnalysis,
    compare_gauge_groups
)
from src.fermion_sector import (
    Cl6MassHierarchy,
    FermionGeneration,
    ThreeGenerationTheorem,
    CliffordAlgebra,
    PDG_MASSES
)
from src.helicity_barrier import (
    HelicityBarrier,
    TurbulenceModel,
    PSPDataAnalysis,
    PSPEncounter,
    SolarWindParameters,
    generate_synthetic_data
)
from src.riemann_zeta import (
    RiemannZeta,
    SpectralAnalysis,
    RHComplexityConnection,
    ZetaZero,
    FIRST_ZEROS,
    zero_count
)
from src.validation import (
    ValidationSuite,
    EmpiricalData,
    ValidationStatus,
    Measurement
)


@pytest.fixture(scope="module")
def zeros_100():
    with open(os.path.join(ROOT, "data", "riemann_zeros", "first_100_zeros.json")) as fh:
        return json.load(fh)["zeros"]


# ============================================================
# Complexity module
# ============================================================

class TestComplexityParameters:

    def test_default_values(self):
        params = ComplexityParameters()
        assert params.alpha == 1.0
        assert params.gamma_1 == 1.0
        assert params.gamma_2 == -2.0

    def test_stelle_ratio(self):
        assert ComplexityParameters().stelle_ratio() == pytest.approx(-0.5)

    def test_custom_stelle_ratio(self):
        assert ComplexityParameters(gamma_1=2.0, gamma_2=-4.0).stelle_ratio() == -0.5


class TestRetrodictionComplexity:

    def test_flat_space(self):
        assert RetrodictionComplexity().compute(ricci_scalar=0.0, volume=1.0) == 0.0

    def test_curved_space(self):
        assert RetrodictionComplexity().compute(ricci_scalar=1.0, volume=1.0) != 0.0

    def test_claimed_stelle_ratio_keeps_spin2_ghost(self):
        pred = RetrodictionComplexity().stelle_prediction()
        assert pred["claimed_ratio"] == -0.5
        assert pred["stelle_ratio"] == pytest.approx(-0.5)
        assert pred["weyl_squared_coefficient"] == -1.0
        assert pred["spin2_ghost_free"] is False

    def test_only_vanishing_ricci_squared_term_is_ghost_free(self):
        rc = RetrodictionComplexity(ComplexityParameters(gamma_1=1.0, gamma_2=0.0))
        assert rc.stelle_prediction()["spin2_ghost_free"] is True


class TestBarrierComplexity:

    def test_generation_minimum_at_default_alpha(self):
        result = BarrierComplexity().generation_minimum()
        assert result["minimum_at"] == 3
        assert result["minimum_at_3"] is True
        assert result["alpha_threshold"] == pytest.approx(math.log(2))

    def test_generation_minimum_moves_below_threshold(self):
        bc = BarrierComplexity(ComplexityParameters(alpha_gen=0.5))
        assert bc.generation_minimum()["minimum_at"] == 2

    def test_generation_barrier_values(self):
        bc = BarrierComplexity()
        b3 = bc.compute(n_generations=3)
        assert b3 < bc.compute(n_generations=2)
        assert b3 < bc.compute(n_generations=4)

    def test_helicity_term_uses_compressibility(self):
        bc = BarrierComplexity()
        base = bc.compute(n_generations=3)
        active = bc.compute(n_generations=3, cross_helicity=0.6, plasma_beta=0.3,
                            magnetic_compressibility=0.0)
        assert active - base == pytest.approx(0.1843)
        inactive = bc.compute(n_generations=3, cross_helicity=0.6, plasma_beta=0.7,
                              magnetic_compressibility=0.0)
        assert inactive == base

    def test_helicity_barrier_law(self):
        result = BarrierComplexity().helicity_barrier_law(np.array([0.0, 0.5, 1.0]))
        assert len(result) == 3
        assert np.all(result >= 0)


class TestTotalComplexity:

    def test_compute_sm(self):
        result = TotalComplexity().compute(gauge_group="SU3xSU2xU1", n_generations=3)
        assert result["C"] == result["R"] + result["K"] + result["B"]

    def test_validate_predictions_reports_inputs(self):
        validation = TotalComplexity().validate_predictions()
        assert validation["three_generations"]["minimum_at_3"] is True
        assert validation["stelle_ratio"]["spin2_ghost_free"] is False
        assert validation["gauge_group"]["gauge_groups_below_SM"] == ["SU2", "SU3", "U1"]


class TestGenerationComplexity:

    def test_compute_generation_complexity(self):
        c3 = compute_generation_complexity(3)
        for n in (1, 2, 4):
            assert c3 < compute_generation_complexity(n)

    def test_readme_claim_has_counterexample(self):
        # The README asserts the minimum is n = 3 for all α > 0.
        assert compute_generation_complexity(1, alpha=0.1) < compute_generation_complexity(3, alpha=0.1)

    def test_demonstrate_three_generations(self):
        result = demonstrate_three_generations()
        assert result["all_equal_3"] is False
        assert result["matches_threshold_rule"] is True
        assert result["alpha_threshold"] == pytest.approx(math.log(2))

    @pytest.mark.parametrize("K", [0.5, 1.0, 2.0, 5.0])
    def test_threshold_rule_for_other_weights(self, K):
        result = demonstrate_three_generations(alpha_range=(0.01, 10.0), n_points=400, K_1gen=K)
        assert result["matches_threshold_rule"] is True


# ============================================================
# Gauge theory module
# ============================================================

class TestGaugeGroupComplexity:

    def test_sm_complexity(self):
        assert GaugeGroupComplexity("SU3xSU2xU1").compute() == 8.0

    def test_su5_complexity(self):
        assert GaugeGroupComplexity("SU5").compute() == 20.0

    def test_sm_below_su5(self):
        assert GaugeGroupComplexity("SU3xSU2xU1").compute() < GaugeGroupComplexity("SU5").compute()

    def test_breakdown(self):
        breakdown = GaugeGroupComplexity("SU3xSU2xU1").breakdown()
        assert len(breakdown["factors"]) == 3

    def test_rank_computation(self):
        assert GaugeGroupComplexity("SU3xSU2xU1").rank == 4


class TestRepresentationComplexity:

    def test_sm_representation(self):
        assert RepresentationComplexity().compute("SU3xSU2xU1", 3) == pytest.approx(86.0)

    def test_so10_spinor_casimir(self):
        # K_gauge = 5·8; matter = 3 generations × 16 × C₂(16) = 45/8
        assert RepresentationComplexity().compute("SO10", 3) == pytest.approx(40 + 3 * 16 * 45 / 8)

    def test_anomaly_free(self):
        rep = RepresentationComplexity()
        for group in ("SU3xSU2xU1", "SU5", "SO10"):
            assert rep.anomaly_coefficient(group, 3) == 0.0

    def test_undefined_matter_content_raises(self):
        rep = RepresentationComplexity()
        with pytest.raises(ValueError):
            rep.compute("E6", 3)
        with pytest.raises(ValueError):
            rep.anomaly_coefficient("E6", 3)


class TestStandardModelAnalysis:

    def test_anomaly_cancellation(self):
        result = StandardModelAnalysis().fermion_representation_check()
        assert result["all_anomalies_cancel"] is True
        for key in ("anomaly_U1_cubed", "anomaly_SU2_U1", "anomaly_SU3_U1", "anomaly_gravitational"):
            assert result[key] == 0.0

    def test_higgs_mechanism(self):
        result = StandardModelAnalysis().higgs_mechanism_complexity()
        assert "K_before_EWSB" in result
        assert "K_after_EWSB" in result

    def test_coupling_normalisation(self):
        inv = StandardModelAnalysis().gauge_coupling_unification()["inverse_couplings"]
        assert inv["1/alpha_1"] == pytest.approx(59.0, abs=0.1)
        assert inv["1/alpha_2"] == pytest.approx(29.6, abs=0.1)
        assert inv["1/alpha_3"] == pytest.approx(8.48, abs=0.01)


class TestCompareGaugeGroups:

    def test_compare_all(self):
        result = compare_gauge_groups()
        assert "SU3xSU2xU1" in result and "SU5" in result
        assert all(v >= 0 for v in result.values())

    def test_subgroups_score_below_sm(self):
        result = compare_gauge_groups()
        for group in ("U1", "SU2", "SU3"):
            assert result[group] < result["SU3xSU2xU1"]


# ============================================================
# Fermion sector module
# ============================================================

class TestCliffordAlgebra:

    def test_cl6_dimension(self):
        assert CliffordAlgebra(6).dimension == 64

    def test_cl6_grades(self):
        assert CliffordAlgebra(6).grade_dimensions == [1, 6, 15, 20, 15, 6, 1]


class TestCl6MassHierarchy:

    def test_suppression_parameter(self):
        assert Cl6MassHierarchy().suppression_parameter == 6 / 64

    def test_grade_assignment_matches_manuscript(self):
        cl6 = Cl6MassHierarchy()
        assert cl6.mass_suppression(3) == 1.0
        assert cl6.mass_suppression(2) == cl6.epsilon
        assert cl6.mass_suppression(1) == cl6.epsilon ** 2

    def test_mass_ratios(self):
        ratios = Cl6MassHierarchy().mass_ratios()
        assert len(ratios) == 6
        assert all(r == pytest.approx(64 / 6) for r in ratios.values())

    def test_mass_ratio_misses(self):
        # The manuscript's own tables quote 12.6, 55, 4.2, 1.9, 1.6 and ~20
        factors = {k: v["factor"] for k, v in Cl6MassHierarchy().validate_mass_ratios().items()}
        expected = {"top_charm": 12.75, "charm_up": 55.1, "bottom_strange": 4.20,
                    "strange_down": 1.88, "tau_muon": 1.58, "muon_electron": 19.4}
        for name, value in expected.items():
            assert factors[name] == pytest.approx(value, rel=0.01)

    def test_cabibbo_exponent(self):
        result = Cl6MassHierarchy().cabibbo_exponent()
        assert result["exponent"] == pytest.approx(0.6315, abs=1e-3)
        assert result["epsilon_to_manuscript_exponent"] == pytest.approx(0.2097, abs=1e-3)

    def test_fit_epsilon(self):
        result = Cl6MassHierarchy().fit_epsilon()
        assert result["epsilon_geometric_mean"] == pytest.approx(0.0126, abs=1e-4)
        assert result["theoretical_over_fit"] > 7


class TestThreeGenerationTheorem:

    def test_find_minimum(self):
        assert ThreeGenerationTheorem().find_minimum()["minimum_at"] == 3

    def test_sensitivity_analysis(self):
        theorem = ThreeGenerationTheorem()
        result = theorem.sensitivity_analysis(n_points=200)
        assert result["all_equal_3"] is False
        assert result["matches_threshold_rule"] is True
        assert theorem.alpha == 1.0

    def test_above_threshold_minimum_is_3(self):
        result = ThreeGenerationTheorem().sensitivity_analysis(alpha_range=(0.7, 100.0), n_points=200)
        assert result["all_equal_3"] is True

    def test_experimental_evidence(self):
        evidence = ThreeGenerationTheorem().experimental_evidence()
        assert evidence["LEP_neutrinos"]["sigma_from_3"] == pytest.approx(2.0)
        assert evidence["LEP_neutrinos_reanalysis"]["sigma_from_3"] == pytest.approx(0.5, abs=0.01)


class TestFermionGeneration:

    def test_from_pdg(self):
        assert "up" in FermionGeneration.from_pdg(1).quarks
        assert "tau" in FermionGeneration.from_pdg(3).leptons

    def test_total_mass(self):
        assert FermionGeneration.from_pdg(3).total_mass() > 100

    def test_neutrino_limits_in_gev(self):
        assert PDG_MASSES["nu_mu"]["upper_limit"] == pytest.approx(0.19e-3)
        assert PDG_MASSES["nu_tau"]["upper_limit"] == pytest.approx(18.2e-3)


# ============================================================
# Helicity barrier module
# ============================================================

class TestHelicityBarrier:

    def test_barrier_activation(self):
        barrier = HelicityBarrier()
        assert barrier.is_active(0.3, 0.5) is True
        assert barrier.is_active(0.7, 0.5) is False
        assert barrier.is_active(0.3, 0.2) is False

    def test_constitutive_law(self):
        assert HelicityBarrier().delta_zeta4(0.0) == pytest.approx(0.1843)

    def test_manuscript_standard_errors(self):
        fit = HelicityBarrier().fit
        assert (fit.a0_error, fit.a1_error, fit.tau_error) == (0.012, 0.031, 0.008)

    def test_no_comparison_without_new_fit(self):
        result = HelicityBarrier().validate_prediction()
        assert result["fitted_here"] is False
        assert result["consistent"] is None

    def test_fitter_recovers_synthetic_coefficients(self):
        barrier = HelicityBarrier()
        compressibility, delta_zeta4 = generate_synthetic_data(500, noise_level=0.001, seed=0)
        fit = barrier.fit_to_data(compressibility, delta_zeta4)
        assert fit.a0 == pytest.approx(0.1843, abs=0.002)
        assert fit.a1 == pytest.approx(-0.2051, abs=0.01)
        assert fit.tau == pytest.approx(0.022, abs=0.01)
        assert barrier.validate_prediction()["consistent"] is True

    def test_polynomial_negative_beyond_fitted_range(self):
        fit = HelicityBarrier().fit
        assert fit.a0 + fit.a1 * 2.0 + fit.a2 * 4.0 < 0


class TestSolarWindParameters:

    def test_plasma_beta_and_alfven_speed(self):
        params = SolarWindParameters(density=5.0, temperature=1e5, magnetic_field=5.0, velocity=400.0)
        assert params.plasma_beta == pytest.approx(0.694, rel=0.01)
        assert params.alfven_speed == pytest.approx(48.8, rel=0.01)


class TestPSPDataAnalysis:

    def test_requires_supplied_observations(self):
        with pytest.raises(ValueError):
            PSPDataAnalysis([])

    def test_threshold_rule_on_synthetic_encounters(self):
        encounters = [
            PSPEncounter(1, 20.0, "synthetic", (0.3, 1.0), (0.5, 0.8), True),
            PSPEncounter(2, 30.0, "synthetic", (0.6, 2.0), (0.2, 0.4), False),
        ]
        analysis = PSPDataAnalysis(encounters)
        assert analysis.barrier_detection_summary()["barrier_detected"] == 1
        assert analysis.validate_beta_critical()["prediction_accuracy"] == 1.0


class TestTurbulenceModel:

    def test_cascade_spectrum(self):
        k = np.logspace(-6, -3, 100)
        result = TurbulenceModel().cascade_spectrum(k, plasma_beta=0.3, cross_helicity=0.6)
        assert result["barrier_active"] is True
        assert np.all(result["E_with_barrier"] <= result["E_kolmogorov"])

    def test_no_heating_when_inactive(self):
        params = SolarWindParameters(density=5.0, temperature=1e5, magnetic_field=5.0, velocity=400.0)
        assert TurbulenceModel().heating_rate(params, cross_helicity=0.6)["ion_heating_rate_W_m3"] == 0.0


class TestSyntheticData:

    def test_generate_synthetic_data(self):
        compressibility, delta_zeta4 = generate_synthetic_data(100, noise_level=0.001, seed=1)
        assert len(compressibility) == len(delta_zeta4) == 100
        assert np.all((compressibility >= 0) & (compressibility <= 1))


# ============================================================
# Riemann zeta module
# ============================================================

class TestZetaZero:

    def test_complex_value(self):
        z = ZetaZero(14.134725).complex_value
        assert z.real == 0.5
        assert z.imag == pytest.approx(14.134725)


class TestRiemannZeta:

    def test_first_zeros_loaded(self):
        assert len(RiemannZeta().zeros) == 30

    def test_zero_density(self):
        rz = RiemannZeta()
        assert rz.zero_density(1000) > rz.zero_density(100)

    def test_zero_count_at_verified_height(self):
        assert zero_count(3e12) == pytest.approx(1.236e13, rel=1e-3)

    def test_local_density_formula(self):
        assert RiemannZeta().local_density(2 * math.pi * math.e) == pytest.approx(1 / (2 * math.pi))

    def test_verify_rh_local(self):
        result = RiemannZeta().verify_rh_local()
        assert result["all_vanish"] is True
        assert result["strictly_increasing"] is True
        assert result["unique"] is True
        assert result["bracket_count_mismatches"] == []
        assert result["z_sign_changes"] is True
        assert FIRST_ZEROS[-1] < result["final_bracket"] < 103.7255
        assert result["complete_and_ordered"] is True
        assert result["independent_comparison"].startswith("not run here")
        assert "Odlyzko's zeros6 table" in result["independent_comparison"]

    def test_verify_rh_local_rejects_wrong_zero(self):
        perturbed = list(FIRST_ZEROS)
        perturbed[10] += 5e-7
        result = RiemannZeta(perturbed).verify_rh_local()
        assert result["all_vanish"] is False
        assert result["complete_and_ordered"] is False

    def test_verify_rh_local_rejects_duplicate(self):
        duplicated = FIRST_ZEROS[:10] + [FIRST_ZEROS[9]] + FIRST_ZEROS[10:]
        result = RiemannZeta(duplicated).verify_rh_local()
        assert result["unique"] is False
        assert result["complete_and_ordered"] is False

    def test_verify_rh_local_rejects_swapped_pair(self):
        swapped = list(FIRST_ZEROS)
        swapped[4], swapped[5] = swapped[5], swapped[4]
        result = RiemannZeta(swapped).verify_rh_local()
        assert result["strictly_increasing"] is False
        assert result["complete_and_ordered"] is False

    def test_verify_rh_local_rejects_missing_zero(self):
        missing = FIRST_ZEROS[:12] + FIRST_ZEROS[13:]
        result = RiemannZeta(missing).verify_rh_local()
        assert result["bracket_count_mismatches"] != []
        assert result["complete_and_ordered"] is False

    def test_data_file_zeros(self, zeros_100):
        assert len(zeros_100) == 100
        assert np.allclose(zeros_100[:30], FIRST_ZEROS, atol=1e-12, rtol=0)
        result = RiemannZeta(zeros_100).verify_rh_local()
        assert result["bracket_count_mismatches"] == []
        assert zeros_100[-1] < result["final_bracket"] < 237.7699
        assert result["complete_and_ordered"] is True


class TestZeroCrossCheck:
    """The recorded FLINT cross-check must describe the current data file."""

    def record(self):
        with open(os.path.join(ROOT, "results", "zero_cross_check_arb.json")) as fh:
            return json.load(fh)

    def test_record_matches_data_file(self):
        record = self.record()
        with open(os.path.join(ROOT, "data", "riemann_zeros", "first_100_zeros.json"), "rb") as fh:
            assert record["data_sha256"] == hashlib.sha256(fh.read()).hexdigest()
        summary = record["summary"]
        assert summary["zeros_checked"] == 100
        assert summary["all_within_half_unit"] is True
        assert summary["all_real_parts_half"] is True
        assert summary["bracket_count_mismatches"] == []
        assert [b["nzeros"] for b in record["brackets"]] == list(range(101))

    def test_record_reproduces(self):
        pytest.importorskip("flint")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import cross_check_zeros_arb
        assert cross_check_zeros_arb.cross_check() == self.record()


class TestZeroCrossCheckPari:
    """The recorded PARI cross-check must describe the current data file and agree with FLINT."""

    def record(self, name="zero_cross_check_pari.json"):
        with open(os.path.join(ROOT, "results", name)) as fh:
            return json.load(fh)

    def test_record_matches_data_file(self):
        record = self.record()
        with open(os.path.join(ROOT, "data", "riemann_zeros", "first_100_zeros.json"), "rb") as fh:
            assert record["data_sha256"] == hashlib.sha256(fh.read()).hexdigest()
        summary = record["summary"]
        assert summary["zeros_checked"] == 100
        assert summary["all_within_half_unit"] is True
        assert summary["counts_all_equal_n"] is True
        assert summary["gap_exceeds_mesh_spacing"] is True

    def test_pari_and_flint_records_agree_to_30_digits(self):
        from decimal import Decimal, getcontext
        getcontext().prec = 90
        pari = self.record()["zeros"]
        flint = self.record("zero_cross_check_arb.json")["zeros"]
        assert len(pari) == len(flint) == 100
        for p, f in zip(pari, flint):
            digits_before_point = len(f["flint_imag"].split(".")[0])
            half_unit = Decimal(10) ** (digits_before_point - 30) / 2
            gap = abs(Decimal(p["pari_60"]) - Decimal(f["flint_imag"]))
            assert gap <= half_unit + Decimal(repr(f["flint_radius"])), f["n"]

    def test_record_reproduces(self):
        pytest.importorskip("cypari2")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import cross_check_zeros_pari
        assert cross_check_zeros_pari.cross_check() == self.record()


class TestZeroCrossCheckOeis:
    """The recorded OEIS comparison must describe the current data file; no network needed."""

    def record(self):
        with open(os.path.join(ROOT, "results", "zero_cross_check_oeis.json")) as fh:
            return json.load(fh)

    def test_record_matches_data_file(self, zeros_100):
        from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
        record = self.record()
        with open(os.path.join(ROOT, "data", "riemann_zeros", "first_100_zeros.json"), "rb") as fh:
            raw = fh.read()
        assert record["data_sha256"] == hashlib.sha256(raw).hexdigest()
        summary = record["summary"]
        assert summary["all_expansions_within_half_unit"] is True and summary["expansions_checked"] == 10
        assert summary["floor_mismatches"] == [] and summary["nearest_integer_mismatches"] == []
        stored = [Decimal(row["stored"]) for row in record["expansions"]]
        assert [float(v) for v in stored] == zeros_100[:10]
        values = [Decimal(repr(z)) for z in zeros_100]
        checks = record["integer_checks"]
        assert checks["A013629"]["published_first_n"] == [int(v.to_integral_value(rounding=ROUND_FLOOR)) for v in values]
        assert checks["A002410"]["published_first_n"] == [int(v.to_integral_value(rounding=ROUND_HALF_UP)) for v in values]

    def test_odlyzko_table_record_matches_data_file(self, zeros_100):
        from decimal import Decimal, ROUND_HALF_EVEN
        with open(os.path.join(ROOT, "results", "zero_cross_check_odlyzko.json")) as fh:
            record = json.load(fh)
        with open(os.path.join(ROOT, "data", "riemann_zeros", "first_100_zeros.json"), "rb") as fh:
            assert record["data_sha256"] == hashlib.sha256(fh.read()).hexdigest()
        assert record["source"]["tarball_sha256"] == (
            "8919f01992718b9bf5c0602dbf16dd9d6f58b141b25f67f5cfd59f6cd0f9a0d4")
        rows = record["zeros"]
        assert [float(r["stored"]) for r in rows] == zeros_100
        for r in rows:
            rounded = Decimal(r["stored"]).quantize(Decimal("1e-9"), rounding=ROUND_HALF_EVEN)
            assert rounded == Decimal(r["published"]), r["n"]
        summary = record["summary"]
        assert summary["all_within_tolerance"] is True
        assert summary["published_strictly_increasing"] is True
        assert summary["zero_after_last_above_final_bracket"] is True

    def test_published_digits_agree_to_20_significant_digits(self):
        from decimal import Decimal
        for row in self.record()["expansions"]:
            half_unit = Decimal(10) ** (len(row["stored"].split(".")[0]) - 20) / 2
            assert abs(Decimal(row["published_first_25_significant_digits"]) - Decimal(row["stored"])) <= half_unit, row["n"]


class TestSpectralAnalysis:

    def test_normalized_spacings(self):
        spacings = SpectralAnalysis().normalized_spacings()
        assert len(spacings) == 29
        assert 0.9 < np.mean(spacings) < 1.1

    def test_nearest_neighbor_distribution(self):
        result = SpectralAnalysis().nearest_neighbor_distribution(bins=20)
        assert len(result["gue_prediction"]) == 20

    def test_level_repulsion(self, zeros_100):
        result = SpectralAnalysis(zeros_100).level_repulsion()
        assert result["gue_wigner_surmise"] == pytest.approx(0.112, abs=0.002)
        assert result["poisson"] == pytest.approx(0.393, abs=0.001)
        assert result["fraction_below_half"] < 0.2

    def test_number_variance_gue_formula(self):
        result = SpectralAnalysis().number_variance(L_range=(1.0, 1.0), n_points=1)
        assert result["gue_prediction"][0] == pytest.approx(
            (math.log(2 * math.pi) + np.euler_gamma + 1) / math.pi**2
        )


class TestRHComplexityConnection:

    def test_complexity_argument_is_labelled_hypothesis(self):
        assert "Hypothesis" in RHComplexityConnection().complexity_argument()["status"]

    def test_de_bruijn_newman(self):
        result = RHComplexityConnection().de_bruijn_newman()
        assert (result["lower_bound"], result["upper_bound"]) == (0.0, 0.2)
        assert "Rodgers" in result["lower_bound_source"]
        assert "Platt" in result["upper_bound_source"]

    def test_explicit_formula(self, zeros_100):
        result = RHComplexityConnection().explicit_formula_test(100.5, zeros=zeros_100)
        assert result["psi_exact"] == pytest.approx(94.0453, abs=1e-3)
        assert result["relative_error"] < 0.02


# ============================================================
# Validation module
# ============================================================

class TestMeasurement:

    def test_is_consistent(self):
        m = Measurement("test", 1.0, 0.1)
        assert m.is_consistent(1.0)
        assert m.is_consistent(1.1)
        assert not m.is_consistent(2.0)

    def test_sigma_deviation(self):
        m = Measurement("test", 1.0, 0.1)
        assert m.sigma_deviation(1.0) == 0.0
        assert m.sigma_deviation(1.2) == pytest.approx(2.0)


class TestEmpiricalData:

    def test_pdg_data_loaded(self):
        data = EmpiricalData()
        assert "N_nu" in data.pdg and "m_top" in data.pdg

    def test_planck_data_loaded(self):
        data = EmpiricalData()
        assert "n_s" in data.planck and "f_NL_local" in data.planck

    def test_manuscript_values_are_not_measurements(self):
        data = EmpiricalData()
        assert "tau_helicity" in data.manuscript_reported
        assert not hasattr(data, "helio")


class TestValidationSuite:

    def test_three_generations_is_consistency(self):
        result = ValidationSuite().validate_three_generations()
        assert result.status == ValidationStatus.CONSISTENT
        assert result.sigma_deviation == pytest.approx(2.0)

    def test_self_sourced_claims_are_not_independent(self):
        suite = ValidationSuite()
        for result in (suite.validate_helicity_barrier(), suite.validate_critical_beta()):
            assert result.status == ValidationStatus.NOT_INDEPENDENT
            assert result.sigma_deviation is None

    def test_stelle_ratio_unsupported(self):
        assert ValidationSuite().validate_stelle_ratio().status == ValidationStatus.UNSUPPORTED

    def test_mass_ratios_falsified(self):
        result = ValidationSuite().validate_mass_ratios()
        assert result.status == ValidationStatus.FALSIFIED
        assert result.sigma_deviation > 50

    def test_nothing_confirmed(self):
        results = ValidationSuite().validate_all()
        assert all(r.status != ValidationStatus.CONFIRMED for r in results.values())

    def test_summary(self):
        summary = ValidationSuite().summary()
        assert summary["total_claims"] == 7
        assert sum(summary["status_counts"].values()) == 7

    def test_generate_report(self):
        report = ValidationSuite().generate_report()
        assert report.startswith("# Empirical Validation Report")
        assert "| Cl(6) Mass Ratios | falsified |" in report


# ============================================================
# Integration
# ============================================================

class TestIntegration:

    def test_full_framework_runs(self):
        assert TotalComplexity().compute(gauge_group="SU3xSU2xU1", n_generations=3)["C"] > 0

    def test_fermion_gauge_consistency(self):
        assert ThreeGenerationTheorem().find_minimum()["minimum_at"] == 3
        assert StandardModelAnalysis().fermion_representation_check()["all_anomalies_cancel"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
