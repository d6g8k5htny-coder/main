"""
Helicity Barrier and Solar Wind Turbulence Model
================================================

Code for the helicity-barrier material of the retired 2025 manuscript (body
§12 and Appendix C). The manuscript reports a regression of the intermittency
deviation Δζ₄ = ζ₄ - 2ζ₂ on magnetic compressibility C_B, fitted to ACE
magnetic-field data (1998-2011):

    |Δζ₄| = a₀ + a₁ C_B + τ C_B²,   τ = 0.022 ± 0.008

Neither those data nor that analysis are in this repository, so the
coefficients are quoted, not reproduced. The activation thresholds β_c ≈ 0.5
and σ_c ≳ 0.4 are as stated in the 2025 README. This module contains no ACE or
Parker Solar Probe measurements.

The helicity barrier itself is the mechanism of Squire, Meyrand & Schekochihin
(2022, Nature Astronomy 6, 715); the README also cites McIntyre et al. (2025,
Phys. Rev. X 15, 031008). Neither paper was checked here as a source for the
numbers above.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


PROTON_MASS = 1.6726219e-27  # kg
BOLTZMANN = 1.380649e-23  # J/K
MU_0 = 4 * np.pi * 1e-7  # H/m
SOLAR_RADIUS = 6.96e8  # m


@dataclass
class SolarWindParameters:
    """Bulk plasma parameters (n in cm⁻³, T in K, B in nT, v in km/s)."""
    density: float
    temperature: float
    magnetic_field: float
    velocity: float
    distance_AU: float = 1.0

    @property
    def plasma_beta(self) -> float:
        """Proton β = n k T / (B² / 2μ₀)."""
        n_si = self.density * 1e6
        B_si = self.magnetic_field * 1e-9
        return (n_si * BOLTZMANN * self.temperature) / (B_si**2 / (2 * MU_0))

    @property
    def alfven_speed(self) -> float:
        """Alfvén speed B / √(μ₀ρ) in km/s."""
        rho = self.density * 1e6 * PROTON_MASS
        return self.magnetic_field * 1e-9 / np.sqrt(MU_0 * rho) / 1000


@dataclass
class HelicityBarrierFit:
    """
    Coefficients of |Δζ₄| = a₀ + a₁ C_B + a₂ C_B².

    Defaults are the manuscript's regression (Table 12.4 / C.4), including its
    standard errors. The manuscript's reported R² = 0.82, F = 247.3 and
    df = 156 are mutually inconsistent for a two-regressor fit (they imply
    F ≈ 355), and the polynomial is negative for 1.007 < C_B < 8.32.
    """
    a0: float = 0.1843
    a1: float = -0.2051
    a2: float = 0.022
    a0_error: float = 0.012
    a1_error: float = 0.031
    a2_error: float = 0.008

    @property
    def tau(self) -> float:
        """Return curvature coefficient τ = a₂."""
        return self.a2

    @property
    def tau_error(self) -> float:
        """Return uncertainty in τ."""
        return self.a2_error


class HelicityBarrier:
    """
    Threshold activation plus the manuscript's constitutive law.

    Active when β < β_c and |σ_c| > threshold; the law relates |Δζ₄| to the
    magnetic compressibility C_B (not to σ_c).
    """

    def __init__(self, fit_params: Optional[HelicityBarrierFit] = None):
        self.fit = fit_params or HelicityBarrierFit()
        self.fitted_here = False
        self.beta_critical = 0.5
        self.sigma_c_threshold = 0.4

    def is_active(self, plasma_beta: float, cross_helicity: float) -> bool:
        """Check whether the stated activation thresholds are met."""
        return (
            plasma_beta < self.beta_critical and
            abs(cross_helicity) > self.sigma_c_threshold
        )

    def delta_zeta4(self, compressibility: float) -> float:
        """Evaluate |a₀ + a₁ C_B + a₂ C_B²|."""
        return float(self.delta_zeta4_array(np.asarray(compressibility)))

    def delta_zeta4_array(self, compressibility: np.ndarray) -> np.ndarray:
        """Vectorized version of delta_zeta4."""
        C_B = np.abs(compressibility)
        return np.abs(self.fit.a0 + self.fit.a1 * C_B + self.fit.a2 * C_B**2)

    def barrier_strength(
        self,
        plasma_beta: float,
        cross_helicity: float,
        compressibility: float
    ) -> float:
        """Return |Δζ₄| when the barrier is active, otherwise 0."""
        if not self.is_active(plasma_beta, cross_helicity):
            return 0.0
        return self.delta_zeta4(compressibility)

    def fit_to_data(
        self,
        compressibility: np.ndarray,
        delta_zeta4_obs: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> HelicityBarrierFit:
        """Least-squares fit of the quadratic law to user-supplied data."""
        def model(C_B, a0, a1, a2):
            return a0 + a1 * C_B + a2 * C_B**2

        popt, pcov = curve_fit(
            model, np.abs(compressibility), delta_zeta4_obs,
            p0=[0.18, -0.2, 0.02],
            sigma=weights,
            absolute_sigma=weights is not None
        )
        errors = np.sqrt(np.diag(pcov))
        self.fit = HelicityBarrierFit(
            a0=popt[0], a1=popt[1], a2=popt[2],
            a0_error=errors[0], a1_error=errors[1], a2_error=errors[2]
        )
        self.fitted_here = True
        return self.fit

    def validate_prediction(self) -> Dict[str, Any]:
        """
        Compare the current τ with the manuscript's τ = 0.022 ± 0.008.

        Before fit_to_data runs, the current τ is the manuscript value itself,
        so no comparison is reported. Whether supplied data are independent of
        the manuscript's is for the caller to establish.
        """
        reference_tau, reference_error = 0.022, 0.008
        result = {
            "reference_tau": reference_tau,
            "reference_error": reference_error,
            "fitted_tau": self.fit.tau,
            "fitted_error": self.fit.tau_error,
            "fitted_here": self.fitted_here,
            "consistent": None,
            "sigma_deviation": None,
        }
        if self.fitted_here:
            deviation = abs(self.fit.tau - reference_tau) / np.hypot(reference_error, self.fit.tau_error)
            result.update(consistent=bool(deviation < 2), sigma_deviation=float(deviation))
        return result


class TurbulenceModel:
    """
    Illustrative toy cascade, written for this code base rather than taken
    from the manuscript. Parameters are not calibrated to any data.
    """

    def __init__(self):
        self.barrier = HelicityBarrier()

    def cascade_spectrum(
        self,
        wavenumber: np.ndarray,
        injection_rate: float = 1.0,
        plasma_beta: float = 1.0,
        cross_helicity: float = 0.0,
        compressibility: float = 0.0,
        ion_scale: float = 1e5  # m, roughly the ion inertial length at 1 AU
    ) -> Dict[str, np.ndarray]:
        """Kolmogorov k^(-5/3) spectrum with an exponential cut beyond the ion scale."""
        k = wavenumber
        E_kolmogorov = injection_rate**(2/3) * k**(-5/3)

        active = self.barrier.is_active(plasma_beta, cross_helicity)
        if active:
            k_ion = 2 * np.pi / ion_scale
            strength = self.barrier.barrier_strength(plasma_beta, cross_helicity, compressibility)
            E_cascade = E_kolmogorov * np.where(k > k_ion, np.exp(-strength * (k - k_ion) / k_ion), 1.0)
        else:
            E_cascade = E_kolmogorov

        return {
            "wavenumber": k,
            "E_kolmogorov": E_kolmogorov,
            "E_with_barrier": E_cascade,
            "barrier_active": active
        }

    def heating_rate(
        self,
        plasma_params: SolarWindParameters,
        cross_helicity: float,
        compressibility: float = 0.0,
        delta_v: float = 30.0,  # km/s, assumed fluctuation amplitude
        correlation_length: float = 1e9  # m, assumed
    ) -> Dict[str, float]:
        """Toy estimate: cascade rate δv³/L, times |Δζ₄| as the diverted fraction."""
        beta = plasma_params.plasma_beta
        energy_density = 0.5 * plasma_params.density * 1e6 * PROTON_MASS * (delta_v * 1e3)**2
        cascade_rate = energy_density / (correlation_length / (delta_v * 1e3))

        active = self.barrier.is_active(beta, cross_helicity)
        heating_fraction = self.barrier.barrier_strength(beta, cross_helicity, compressibility) if active else 0.0

        return {
            "plasma_beta": beta,
            "barrier_active": active,
            "energy_density_J_m3": energy_density,
            "cascade_rate_W_m3": cascade_rate,
            "heating_fraction": heating_fraction,
            "ion_heating_rate_W_m3": cascade_rate * heating_fraction
        }


@dataclass
class PSPEncounter:
    """
    Observations from one Parker Solar Probe encounter, supplied by the caller.

    The repository ships no encounter data; the 2025-12-16 version contained
    invented β ranges and detection flags, which were removed.
    """
    encounter_number: int
    perihelion_Rs: float
    date: str
    plasma_beta_range: Tuple[float, float]
    sigma_c_range: Tuple[float, float]
    barrier_observed: bool

    @property
    def perihelion_AU(self) -> float:
        """Perihelion in AU."""
        return self.perihelion_Rs * SOLAR_RADIUS / 1.496e11


class PSPDataAnalysis:
    """Threshold checks over caller-supplied encounter observations."""

    def __init__(self, encounters: List[PSPEncounter]):
        if not encounters:
            raise ValueError("Supply encounter observations; none are bundled")
        self.encounters = encounters

    def barrier_detection_summary(self) -> Dict[str, Any]:
        """Count encounters with and without a reported barrier."""
        detected = [e for e in self.encounters if e.barrier_observed]
        return {
            "total_encounters": len(self.encounters),
            "barrier_detected": len(detected),
            "detection_rate": len(detected) / len(self.encounters),
            "detected_beta_ranges": [e.plasma_beta_range for e in detected],
            "not_detected_beta_ranges": [
                e.plasma_beta_range for e in self.encounters if not e.barrier_observed
            ],
        }

    def validate_beta_critical(self, beta_c: float = 0.5) -> Dict[str, Any]:
        """Accuracy of the rule 'barrier observed iff min β < β_c'."""
        correct = sum(
            (e.plasma_beta_range[0] < beta_c) == e.barrier_observed
            for e in self.encounters
        )
        return {
            "predicted_beta_c": beta_c,
            "prediction_accuracy": correct / len(self.encounters),
            "encounters_analyzed": len(self.encounters)
        }

    def sigma_c_threshold_analysis(self, threshold: float = 0.4) -> Dict[str, Any]:
        """Mean σ_c range endpoints for encounters with and without a barrier."""
        detected = [v for e in self.encounters if e.barrier_observed for v in e.sigma_c_range]
        missing = [v for e in self.encounters if not e.barrier_observed for v in e.sigma_c_range]
        return {
            "threshold": threshold,
            "detected_sigma_c_mean": float(np.mean(detected)) if detected else None,
            "not_detected_sigma_c_mean": float(np.mean(missing)) if missing else None,
        }


def generate_synthetic_data(
    n_points: int = 100,
    noise_level: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic (C_B, |Δζ₄|) pairs drawn from the manuscript's own coefficients.

    Fitting these recovers the coefficients; that tests the fitter, not the law.
    """
    rng = np.random.default_rng(seed)
    compressibility = np.linspace(0.0, 1.0, n_points)
    fit = HelicityBarrierFit()
    truth = np.abs(fit.a0 + fit.a1 * compressibility + fit.a2 * compressibility**2)
    return compressibility, truth + rng.normal(0, noise_level, n_points)


if __name__ == "__main__":
    barrier = HelicityBarrier()
    print("Quoted constitutive-law coefficients (2025 manuscript, not reproduced):")
    print(f"  a₀ = {barrier.fit.a0:.4f} ± {barrier.fit.a0_error:.3f}")
    print(f"  a₁ = {barrier.fit.a1:.4f} ± {barrier.fit.a1_error:.3f}")
    print(f"  τ  = {barrier.fit.a2:.4f} ± {barrier.fit.a2_error:.3f}")
    print(f"Comparison available without new data: {barrier.validate_prediction()['consistent']}")
