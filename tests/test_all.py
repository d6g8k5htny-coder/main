"""
Unit Tests for Complexity Physics Framework
============================================

Comprehensive test suite covering all framework modules.

Run with: pytest tests/test_all.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.complexity import (
    TotalComplexity,
    RetrodictionComplexity,
    BarrierComplexity,
    ComplexityParameters,
    MetricData,
    compute_generation_complexity,
    demonstrate_three_generations
)
from src.gauge_theory import (
    GaugeGroupComplexity,
    RepresentationComplexity,
    StandardModelAnalysis,
    LIE_GROUP_DATABASE,
    compare_gauge_groups
)
from src.fermion_sector import (
    Cl6MassHierarchy,
    FermionGeneration,
    ThreeGenerationTheorem,
    CliffordAlgebra,
    PDG_2024_MASSES
)
from src.helicity_barrier import (
    HelicityBarrier,
    TurbulenceModel,
    PSPDataAnalysis,
    SolarWindParameters,
    HelicityBarrierFit,
    generate_synthetic_data
)
from src.riemann_zeta import (
    RiemannZeta,
    SpectralAnalysis,
    RHComplexityConnection,
    ZetaZero,
    FIRST_ZEROS
)
from src.validation import (
    ValidationSuite,
    EmpiricalData,
    ValidationStatus,
    Measurement,
    Prediction
)


# ============================================================
# Complexity Module Tests
# ============================================================

class TestComplexityParameters:
    """Test ComplexityParameters dataclass."""

    def test_default_values(self):
        params = ComplexityParameters()
        assert params.alpha == 1.0
        assert params.gamma_1 == 1.0
        assert params.gamma_2 == -2.0

    def test_stelle_ratio(self):
        params = ComplexityParameters()
        ratio = params.stelle_ratio()
        assert abs(ratio - (-0.5)) < 1e-10

    def test_custom_stelle_ratio(self):
        params = ComplexityParameters(gamma_1=2.0, gamma_2=-4.0)
        assert params.stelle_ratio() == -0.5


class TestRetrodictionComplexity:
    """Test RetrodictionComplexity functional."""

    def test_flat_space(self):
        rc = RetrodictionComplexity()
        # Flat space: R = 0
        result = rc.compute(ricci_scalar=0.0, volume=1.0)
        assert result == 0.0

    def test_curved_space(self):
        rc = RetrodictionComplexity()
        # Non-zero curvature
        result = rc.compute(ricci_scalar=1.0, volume=1.0)
        assert result != 0.0

    def test_stelle_prediction(self):
        rc = RetrodictionComplexity()
        pred = rc.stelle_prediction()
        assert pred["predicted_ratio"] == -0.5
        assert pred["consistent"] == True


class TestBarrierComplexity:
    """Test BarrierComplexity functional."""

    def test_three_generations_minimum(self):
        bc = BarrierComplexity()
        result = bc.generation_minimum()
        assert result["minimum_at"] == 3
        assert result["prediction_correct"] == True

    def test_generation_barrier_values(self):
        bc = BarrierComplexity()
        # n=3 should have minimal barrier
        b3 = bc.compute(n_generations=3)
        b2 = bc.compute(n_generations=2)
        b4 = bc.compute(n_generations=4)
        assert b3 < b2
        assert b3 < b4

    def test_helicity_barrier_law(self):
        bc = BarrierComplexity()
        C_B = np.array([0.0, 0.5, 1.0])
        result = bc.helicity_barrier_law(C_B)
        assert len(result) == 3
        assert all(r >= 0 for r in result)


class TestTotalComplexity:
    """Test TotalComplexity functional."""

    def test_compute_sm(self):
        C = TotalComplexity()
        result = C.compute(gauge_group="SU3xSU2xU1", n_generations=3)
        assert "R" in result
        assert "K" in result
        assert "B" in result
        assert "C" in result
        assert result["C"] == result["R"] + result["K"] + result["B"]

    def test_validate_predictions(self):
        C = TotalComplexity()
        validation = C.validate_predictions()
        assert "three_generations" in validation
        assert validation["three_generations"]["validated"] == True


class TestGenerationComplexity:
    """Test generation complexity functions."""

    def test_compute_generation_complexity(self):
        c1 = compute_generation_complexity(1)
        c2 = compute_generation_complexity(2)
        c3 = compute_generation_complexity(3)
        c4 = compute_generation_complexity(4)
        # Minimum at n=3
        assert c3 < c1
        assert c3 < c2
        assert c3 < c4

    def test_demonstrate_three_generations(self):
        result = demonstrate_three_generations()
        assert result["all_equal_3"] == True


# ============================================================
# Gauge Theory Module Tests
# ============================================================

class TestGaugeGroupComplexity:
    """Test GaugeGroupComplexity class."""

    def test_sm_complexity(self):
        sm = GaugeGroupComplexity("SU3xSU2xU1")
        K = sm.compute()
        assert K > 0

    def test_su5_complexity(self):
        su5 = GaugeGroupComplexity("SU5")
        K = su5.compute()
        assert K > 0

    def test_sm_less_complex_than_gut(self):
        sm = GaugeGroupComplexity("SU3xSU2xU1")
        su5 = GaugeGroupComplexity("SU5")
        # SM should be less complex than SU(5) GUT
        # Note: This depends on representation content
        K_sm = sm.compute()
        K_su5 = su5.compute()
        assert K_sm < K_su5

    def test_breakdown(self):
        sm = GaugeGroupComplexity("SU3xSU2xU1")
        breakdown = sm.breakdown()
        assert "total" in breakdown
        assert "factors" in breakdown
        assert len(breakdown["factors"]) == 3

    def test_rank_computation(self):
        sm = GaugeGroupComplexity("SU3xSU2xU1")
        # SM rank = 2 (SU3) + 1 (SU2) + 1 (U1) = 4
        assert sm.rank == 4


class TestRepresentationComplexity:
    """Test RepresentationComplexity class."""

    def test_sm_representation(self):
        rep = RepresentationComplexity()
        K = rep.compute("SU3xSU2xU1", 3)
        assert K > 0

    def test_anomaly_free(self):
        rep = RepresentationComplexity()
        # SM should be anomaly-free
        anomaly = rep.anomaly_coefficient("SU3xSU2xU1", 3)
        assert anomaly == 0.0


class TestStandardModelAnalysis:
    """Test StandardModelAnalysis class."""

    def test_anomaly_cancellation(self):
        sm = StandardModelAnalysis()
        result = sm.fermion_representation_check()
        assert result["all_anomalies_cancel"] == True

    def test_higgs_mechanism(self):
        sm = StandardModelAnalysis()
        result = sm.higgs_mechanism_complexity()
        assert "K_before_EWSB" in result
        assert "K_after_EWSB" in result


class TestCompareGaugeGroups:
    """Test gauge group comparison."""

    def test_compare_all(self):
        result = compare_gauge_groups()
        assert "SU3xSU2xU1" in result
        assert "SU5" in result
        assert all(v >= 0 for v in result.values())


# ============================================================
# Fermion Sector Module Tests
# ============================================================

class TestCliffordAlgebra:
    """Test CliffordAlgebra class."""

    def test_cl6_dimension(self):
        cl6 = CliffordAlgebra(6)
        assert cl6.dimension == 64  # 2^6

    def test_cl6_grades(self):
        cl6 = CliffordAlgebra(6)
        # Grade dimensions should sum to 64
        assert sum(cl6.grade_dimensions) == 64
        # (6 choose k) for k=0..6
        expected = [1, 6, 15, 20, 15, 6, 1]
        assert cl6.grade_dimensions == expected


class TestCl6MassHierarchy:
    """Test Cl6MassHierarchy class."""

    def test_suppression_parameter(self):
        cl6 = Cl6MassHierarchy()
        epsilon = cl6.suppression_parameter
        # ε = 6/64 ≈ 0.094
        assert abs(epsilon - 6/64) < 1e-10

    def test_mass_ratios(self):
        cl6 = Cl6MassHierarchy()
        ratios = cl6.mass_ratios()
        assert "top_charm" in ratios
        assert "tau_muon" in ratios
        # Ratios should be approximately 1/ε ≈ 10.67
        assert ratios["top_charm"] > 10

    def test_validate_mass_ratios(self):
        cl6 = Cl6MassHierarchy()
        validation = cl6.validate_mass_ratios()
        assert "top_charm" in validation
        # Check structure
        assert "predicted" in validation["top_charm"]
        assert "observed" in validation["top_charm"]

    def test_ckm_from_grade_overlap(self):
        cl6 = Cl6MassHierarchy()
        ckm = cl6.ckm_from_grade_overlap()
        # Diagonal elements should be ~1
        assert abs(ckm["V_ud"] - 1.0) < 0.1
        # Off-diagonal should be small
        assert ckm["V_ub"] < 0.1


class TestThreeGenerationTheorem:
    """Test ThreeGenerationTheorem class."""

    def test_find_minimum(self):
        theorem = ThreeGenerationTheorem()
        result = theorem.find_minimum()
        assert result["minimum_at"] == 3

    def test_sensitivity_analysis(self):
        theorem = ThreeGenerationTheorem()
        result = theorem.sensitivity_analysis(n_points=100)
        assert result["all_equal_3"] == True

    def test_experimental_evidence(self):
        theorem = ThreeGenerationTheorem()
        evidence = theorem.experimental_evidence()
        assert "LEP_neutrinos" in evidence
        assert evidence["LEP_neutrinos"]["consistent_with_3"] == True


class TestFermionGeneration:
    """Test FermionGeneration class."""

    def test_from_pdg(self):
        gen1 = FermionGeneration.from_pdg(1)
        assert "up" in gen1.quarks
        assert "electron" in gen1.leptons

        gen3 = FermionGeneration.from_pdg(3)
        assert "top" in gen3.quarks
        assert "tau" in gen3.leptons

    def test_total_mass(self):
        gen3 = FermionGeneration.from_pdg(3)
        # 3rd generation is heaviest
        total = gen3.total_mass()
        assert total > 100  # Top alone is ~172 GeV


# ============================================================
# Helicity Barrier Module Tests
# ============================================================

class TestHelicityBarrier:
    """Test HelicityBarrier class."""

    def test_barrier_activation(self):
        barrier = HelicityBarrier()
        # Should activate at low β, high σ_c
        assert barrier.is_active(0.3, 0.5) == True
        assert barrier.is_active(0.7, 0.5) == False
        assert barrier.is_active(0.3, 0.2) == False

    def test_constitutive_law(self):
        barrier = HelicityBarrier()
        # Test at C_B = 0
        delta_z = barrier.delta_zeta4(0.0)
        assert abs(delta_z - 0.1843) < 0.01

    def test_tau_coefficient(self):
        barrier = HelicityBarrier()
        assert abs(barrier.fit.tau - 0.022) < 1e-10

    def test_validate_prediction(self):
        barrier = HelicityBarrier()
        result = barrier.validate_prediction()
        assert result["predicted_tau"] == 0.022
        assert result["consistent"] == True


class TestSolarWindParameters:
    """Test SolarWindParameters class."""

    def test_plasma_beta(self):
        params = SolarWindParameters(
            density=5.0,  # cm^-3
            temperature=1e5,  # K
            magnetic_field=5.0,  # nT
            velocity=400.0  # km/s
        )
        beta = params.plasma_beta
        # Typical solar wind: β ~ 0.1 - 10
        assert 0.01 < beta < 100

    def test_alfven_speed(self):
        params = SolarWindParameters(
            density=5.0,
            temperature=1e5,
            magnetic_field=5.0,
            velocity=400.0
        )
        v_A = params.alfven_speed
        # Typical: ~50 km/s
        assert 10 < v_A < 200


class TestPSPDataAnalysis:
    """Test PSPDataAnalysis class."""

    def test_barrier_detection_summary(self):
        analysis = PSPDataAnalysis()
        summary = analysis.barrier_detection_summary()
        assert summary["total_encounters"] > 0
        assert "barrier_detected" in summary

    def test_validate_beta_critical(self):
        analysis = PSPDataAnalysis()
        result = analysis.validate_beta_critical()
        assert result["predicted_beta_c"] == 0.5


class TestTurbulenceModel:
    """Test TurbulenceModel class."""

    def test_cascade_spectrum(self):
        model = TurbulenceModel()
        k = np.logspace(-6, -3, 100)  # m^-1
        result = model.cascade_spectrum(k, plasma_beta=0.3, cross_helicity=0.6)
        assert "wavenumber" in result
        assert "E_with_barrier" in result


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic_data(self):
        sigma_c, delta_z = generate_synthetic_data(100, noise_level=0.001)
        assert len(sigma_c) == 100
        assert len(delta_z) == 100
        assert all(s >= 0 and s <= 1 for s in sigma_c)


# ============================================================
# Riemann Zeta Module Tests
# ============================================================

class TestZetaZero:
    """Test ZetaZero class."""

    def test_real_part(self):
        zero = ZetaZero(14.134725)
        assert zero.real_part == 0.5

    def test_complex_value(self):
        zero = ZetaZero(14.134725)
        z = zero.complex_value
        assert z.real == 0.5
        assert abs(z.imag - 14.134725) < 1e-5


class TestRiemannZeta:
    """Test RiemannZeta class."""

    def test_first_zeros_loaded(self):
        rz = RiemannZeta()
        assert len(rz.zeros) == len(FIRST_ZEROS)

    def test_zero_density(self):
        rz = RiemannZeta()
        # Density should increase with height
        N_100 = rz.zero_density(100)
        N_1000 = rz.zero_density(1000)
        assert N_1000 > N_100

    def test_verify_rh_local(self):
        rz = RiemannZeta()
        result = rz.verify_rh_local()
        assert result["all_on_critical_line"] == True


class TestSpectralAnalysis:
    """Test SpectralAnalysis class."""

    def test_normalized_spacings(self):
        spectral = SpectralAnalysis()
        spacings = spectral.normalized_spacings()
        assert len(spacings) == len(FIRST_ZEROS) - 1
        # Mean should be ~1 if properly normalized
        assert 0.5 < np.mean(spacings) < 2.0

    def test_nearest_neighbor_distribution(self):
        spectral = SpectralAnalysis()
        result = spectral.nearest_neighbor_distribution(bins=20)
        assert "spacings" in result
        assert "gue_prediction" in result

    def test_spectral_rigidity(self):
        spectral = SpectralAnalysis()
        result = spectral.spectral_rigidity()
        assert result["statistics_type"] == "GUE"
        assert result["not_poisson"] == True


class TestRHComplexityConnection:
    """Test RHComplexityConnection class."""

    def test_complexity_argument(self):
        conn = RHComplexityConnection()
        result = conn.complexity_argument()
        assert "premise_1" in result
        assert "conclusion" in result

    def test_de_bruijn_newman(self):
        conn = RHComplexityConnection()
        result = conn.de_bruijn_newman()
        assert result["lower_bound"] == 0.0
        assert result["upper_bound"] == 0.22


# ============================================================
# Validation Module Tests
# ============================================================

class TestMeasurement:
    """Test Measurement class."""

    def test_is_consistent(self):
        m = Measurement("test", 1.0, 0.1, "unit", "source")
        assert m.is_consistent(1.0) == True
        assert m.is_consistent(1.1) == True  # Within 2σ
        assert m.is_consistent(2.0) == False

    def test_sigma_deviation(self):
        m = Measurement("test", 1.0, 0.1, "unit", "source")
        assert m.sigma_deviation(1.0) == 0.0
        assert abs(m.sigma_deviation(1.2) - 2.0) < 1e-10


class TestEmpiricalData:
    """Test EmpiricalData class."""

    def test_pdg_data_loaded(self):
        data = EmpiricalData()
        assert "N_nu" in data.pdg
        assert "m_top" in data.pdg

    def test_planck_data_loaded(self):
        data = EmpiricalData()
        assert "n_s" in data.planck
        assert "f_NL_local" in data.planck

    def test_helio_data_loaded(self):
        data = EmpiricalData()
        assert "beta_critical" in data.helio
        assert "tau_helicity" in data.helio


class TestValidationSuite:
    """Test ValidationSuite class."""

    def test_validate_three_generations(self):
        suite = ValidationSuite()
        result = suite.validate_three_generations()
        assert result.status == ValidationStatus.CONFIRMED

    def test_validate_all(self):
        suite = ValidationSuite()
        results = suite.validate_all()
        assert "three_generations" in results
        assert "helicity_barrier" in results

    def test_summary(self):
        suite = ValidationSuite()
        summary = suite.summary()
        assert "total_predictions" in summary
        assert "success_rate" in summary
        assert summary["success_rate"] > 0.5

    def test_generate_report(self):
        suite = ValidationSuite()
        report = suite.generate_report()
        assert "# Empirical Validation Report" in report
        assert "Three Generations" in report


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests across modules."""

    def test_full_framework_consistency(self):
        """Test that all modules work together consistently."""
        # 1. Create total complexity
        C = TotalComplexity()

        # 2. Compute for Standard Model
        result = C.compute(
            gauge_group="SU3xSU2xU1",
            n_generations=3
        )
        assert result["C"] > 0

        # 3. Validate predictions
        validation = C.validate_predictions()
        assert validation["three_generations"]["validated"]

    def test_fermion_gauge_consistency(self):
        """Test fermion sector is consistent with gauge theory."""
        # Cl(6) gives 3 generations
        cl6 = Cl6MassHierarchy()
        gen_result = ThreeGenerationTheorem().find_minimum()
        assert gen_result["minimum_at"] == 3

        # Gauge anomalies cancel with 3 generations
        sm = StandardModelAnalysis()
        anomaly_result = sm.fermion_representation_check()
        assert anomaly_result["all_anomalies_cancel"]

    def test_empirical_validation_complete(self):
        """Test all empirical validations pass."""
        suite = ValidationSuite()
        results = suite.validate_all()

        # Count successes
        successes = sum(
            1 for r in results.values()
            if r.status in [ValidationStatus.CONFIRMED, ValidationStatus.CONSISTENT, ValidationStatus.TESTABLE]
        )
        failures = sum(
            1 for r in results.values()
            if r.status == ValidationStatus.FALSIFIED
        )

        assert failures == 0
        assert successes > 0


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
