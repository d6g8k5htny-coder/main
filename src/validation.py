"""
Empirical Validation Tools
==========================

Compares claims of the retired 2025 framework with measurements, and says so
plainly when no real comparison exists.

Statuses:
    CONSISTENT       a measurement agrees with the claim
    TESTABLE         no measurement exists yet
    NOT_INDEPENDENT  the only available number comes from the claim's own source
    UNSUPPORTED      the cited derivation does not hold as written
    TENSION          a measurement disagrees moderately
    FALSIFIED        measurements exclude the claim as implemented
    CONFIRMED        an independent test agrees; nothing here qualifies

Measured values were transcribed from memory by the 2025-12-16 session and
are not re-verified; sources below say which edition they appear to match.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .fermion_sector import Cl6MassHierarchy


class ValidationStatus(Enum):
    """Status of a validation test."""
    CONFIRMED = "confirmed"
    CONSISTENT = "consistent"
    TESTABLE = "testable"
    NOT_INDEPENDENT = "not_independent"
    UNSUPPORTED = "unsupported"
    TENSION = "tension"
    FALSIFIED = "falsified"


@dataclass
class Measurement:
    """An experimental or observational measurement."""
    name: str
    value: float
    error: float
    unit: str = ""
    source: str = ""
    year: int = 2024

    def is_consistent(self, prediction: float, n_sigma: float = 2.0) -> bool:
        """Check if measurement is consistent with prediction."""
        return abs(self.value - prediction) <= n_sigma * self.error

    def sigma_deviation(self, prediction: float) -> float:
        """Return deviation from prediction in sigmas."""
        if self.error == 0:
            return float('inf') if self.value != prediction else 0.0
        return abs(self.value - prediction) / self.error


@dataclass
class Prediction:
    """A framework claim."""
    name: str
    value: float
    uncertainty: float = 0.0
    description: str = ""
    testable: bool = True


@dataclass
class ValidationResult:
    """Result of comparing a claim with data; sigma is None when nothing was compared."""
    prediction: Prediction
    measurement: Optional[Measurement]
    status: ValidationStatus
    sigma_deviation: Optional[float]
    notes: str = ""


class EmpiricalData:
    """Measurements used for comparison, plus values the manuscript reports about itself."""

    def __init__(self):
        self._load_pdg_data()
        self._load_planck_data()
        self._load_gw_data()
        self._load_manuscript_reported()
        self._load_rh_data()

    def _load_pdg_data(self):
        pdg = "PDG (values appear to match RPP 2022; transcribed from memory, not re-verified)"
        self.pdg = {
            "N_nu": Measurement("N_nu", 2.984, 0.008, "", "LEP/SLD Z-pole combination", 2006),
            "N_nu_reanalysis": Measurement(
                "N_nu", 2.9963, 0.0074, "", "Janot & Jadach, Phys. Lett. B 803 (2020) 135319", 2020
            ),
            "m_electron": Measurement("m_e", 0.51099895e-3, 0.00000015e-3, "GeV", pdg, 2022),
            "m_muon": Measurement("m_mu", 0.1056583755, 0.0000000023, "GeV", pdg, 2022),
            "m_tau": Measurement("m_tau", 1.77686, 0.00012, "GeV", pdg, 2022),
            "m_up": Measurement("m_u", 2.16e-3, 0.49e-3, "GeV", pdg + "; MS-bar at 2 GeV", 2022),
            "m_down": Measurement("m_d", 4.67e-3, 0.48e-3, "GeV", pdg + "; MS-bar at 2 GeV", 2022),
            "m_strange": Measurement("m_s", 93.4e-3, 8.6e-3, "GeV", pdg + "; MS-bar at 2 GeV", 2022),
            "m_charm": Measurement("m_c", 1.27, 0.02, "GeV", pdg + "; MS-bar at m_c", 2022),
            "m_bottom": Measurement("m_b", 4.18, 0.03, "GeV", pdg + "; MS-bar at m_b", 2022),
            "m_top": Measurement("m_t", 172.69, 0.30, "GeV", pdg + "; pole mass", 2022),
            "alpha_s": Measurement("alpha_s(M_Z)", 0.1179, 0.0009, "", pdg, 2022),
            "sin2_theta_W": Measurement("sin^2(theta_W)", 0.23121, 0.00004, "", pdg + "; MS-bar", 2022),
            "V_us": Measurement("|V_us|", 0.2243, 0.0008, "", pdg, 2022),
            "V_cb": Measurement("|V_cb|", 0.0408, 0.0014, "", pdg, 2022),
            "V_ub": Measurement("|V_ub|", 0.00382, 0.00020, "", pdg, 2022),
            "m_tprime_limit": Measurement(
                "m_t'", 656, 0, "GeV", "Lower limit as stated in the 2025 README; not re-verified", 2025
            ),
        }

    def _load_planck_data(self):
        self.planck = {
            "n_s": Measurement("n_s", 0.9649, 0.0042, "", "Planck 2018 VI", 2020),
            "r": Measurement(
                "r", 0.0, 0.016, "",
                "Upper limit r < 0.032 (95% CL), BICEP/Keck + Planck (Tristram et al. 2022), "
                "stored as 0 ± half the limit", 2022
            ),
            "f_NL_local": Measurement("f_NL^local", -0.9, 5.1, "", "Planck 2018 IX", 2020),
            "N_eff": Measurement("N_eff", 2.99, 0.17, "", "Planck 2018 VI + BAO", 2020),
            "H_0": Measurement("H_0", 67.36, 0.54, "km/s/Mpc", "Planck 2018 VI", 2020),
            "Omega_m": Measurement("Omega_m", 0.3153, 0.0073, "", "Planck 2018 VI", 2020),
        }

    def _load_gw_data(self):
        self.gw = {
            "graviton_mass": Measurement(
                "m_g", 0.0, 1.27e-23, "eV",
                "GWTC-3 tests of GR; 90% CL upper limit stored as the error", 2021
            ),
            "gw_events_gwtc3": Measurement("N_events", 90, 0, "", "GWTC-3, approximate count", 2021),
        }

    def _load_manuscript_reported(self):
        """The framework's own numbers. They are not measurements of anything independent."""
        self.manuscript_reported = {
            "tau_helicity": Measurement(
                "tau", 0.022, 0.008, "",
                "2025 manuscript §12.4 regression on ACE 1998-2011 data; not reproduced", 2025
            ),
            "beta_critical": Measurement(
                "beta_c", 0.5, 0.0, "", "2025 README and manuscript Table 12.3; no analysis here", 2025
            ),
            "sigma_c_threshold": Measurement(
                "sigma_c", 0.4, 0.0, "", "2025 README and manuscript Table 12.3; no analysis here", 2025
            ),
        }

    def _load_rh_data(self):
        self.rh = {
            "zeros_verified": Measurement(
                "N_zeros", 1.236e13, 0, "",
                "Platt & Trudgian 2021: RH holds to height 3e12 (zero count from N(T))", 2021
            ),
            "de_bruijn_newman_upper": Measurement(
                "Lambda_upper", 0.2, 0, "",
                "Platt & Trudgian 2021, improving Polymath15's 0.22 (arXiv:1904.12438)", 2021
            ),
        }


class ValidationSuite:
    """Runs every comparison and records what each one actually establishes."""

    def __init__(self, data: Optional[EmpiricalData] = None):
        self.data = data or EmpiricalData()
        self.results: List[ValidationResult] = []

    def _record(self, result: ValidationResult) -> ValidationResult:
        self.results.append(result)
        return result

    def validate_three_generations(self) -> ValidationResult:
        measurement = self.data.pdg["N_nu"]
        reanalysis = self.data.pdg["N_nu_reanalysis"]
        return self._record(ValidationResult(
            prediction=Prediction("Three Generations", 3.0, 0.0, "Minimum of the generation penalty"),
            measurement=measurement,
            status=ValidationStatus.CONSISTENT,
            sigma_deviation=measurement.sigma_deviation(3.0),
            notes=(
                f"N_ν = {measurement.value} ± {measurement.error}; re-analysis "
                f"{reanalysis.value} ± {reanalysis.error} ({reanalysis.sigma_deviation(3.0):.1f}σ). "
                "n = 3 is the centre of the penalty exp(α(n-3)²), an input, and is the "
                "minimum only for α > ln(1 + K₁)"
            )
        ))

    def validate_fourth_generation_excluded(self) -> ValidationResult:
        measurement = self.data.pdg["m_tprime_limit"]
        return self._record(ValidationResult(
            prediction=Prediction("No 4th Generation", 0.0, 0.0, "Exclusion by the generation penalty"),
            measurement=measurement,
            status=ValidationStatus.CONSISTENT,
            sigma_deviation=None,
            notes=f"README-stated limit m_t' > {measurement.value} GeV; the excluded n is an input"
        ))

    def validate_helicity_barrier(self) -> ValidationResult:
        reported = self.data.manuscript_reported["tau_helicity"]
        return self._record(ValidationResult(
            prediction=Prediction("Helicity Barrier tau", 0.022, 0.008, "Quadratic coefficient"),
            measurement=reported,
            status=ValidationStatus.NOT_INDEPENDENT,
            sigma_deviation=None,
            notes=(
                "τ is the manuscript's own fitted coefficient; no independent measurement or "
                "reproducible analysis is in this repository"
            )
        ))

    def validate_critical_beta(self) -> ValidationResult:
        reported = self.data.manuscript_reported["beta_critical"]
        return self._record(ValidationResult(
            prediction=Prediction("Critical Beta", 0.5, 0.0, "Barrier activation threshold"),
            measurement=reported,
            status=ValidationStatus.NOT_INDEPENDENT,
            sigma_deviation=None,
            notes="Stated in the README and manuscript, citing McIntyre et al.; no data or analysis here"
        ))

    def validate_primordial_fnl(self) -> ValidationResult:
        measurement = self.data.planck["f_NL_local"]
        return self._record(ValidationResult(
            prediction=Prediction("f_NL < O(1)", 0.0, 1.0, "Suppressed primordial non-Gaussianity"),
            measurement=measurement,
            status=ValidationStatus.CONSISTENT,
            sigma_deviation=measurement.sigma_deviation(0.0),
            notes=(
                f"Planck: f_NL = {measurement.value} ± {measurement.error}, which cannot "
                "distinguish f_NL of order 1 or below from many alternatives"
            )
        ))

    def validate_stelle_ratio(self) -> ValidationResult:
        return self._record(ValidationResult(
            prediction=Prediction("Stelle Ratio", -0.5, 0.0, "gamma_1/gamma_2 = -1/2"),
            measurement=None,
            status=ValidationStatus.UNSUPPORTED,
            sigma_deviation=None,
            notes=(
                "Manuscript Thm 5.1 equates γ₁R² + γ₂R_μν² at ratio -1/4 with Gauss-Bonnet "
                "(dropping its Riemann² term), then asserts -1/2 from unstated loop corrections. "
                "Mod Gauss-Bonnet the Weyl² coefficient is γ₂/2, so any γ₂ ≠ 0 keeps the "
                "spin-2 ghost. No measurement exists"
            )
        ))

    def validate_mass_ratios(self) -> ValidationResult:
        top, charm = self.data.pdg["m_top"], self.data.pdg["m_charm"]
        ratio = top.value / charm.value
        error = ratio * np.hypot(top.error / top.value, charm.error / charm.value)
        measurement = Measurement("m_t/m_c", ratio, error, "", "Ratio of PDG values above", 2022)
        cl6 = Cl6MassHierarchy()
        factors = [c["factor"] for c in cl6.validate_mass_ratios().values()]
        return self._record(ValidationResult(
            prediction=Prediction("Cl(6) Mass Ratios", 1 / cl6.epsilon, 0.0, "Every adjacent ratio is 1/ε"),
            measurement=measurement,
            status=ValidationStatus.FALSIFIED,
            sigma_deviation=measurement.sigma_deviation(1 / cl6.epsilon),
            notes=(
                f"Observed/predicted over six adjacent ratios: {min(factors):.1f} to "
                f"{max(factors):.0f}; {sum(f > 10 for f in factors)} miss by more than 10×. "
                "The manuscript cites RG running without a calculation; schemes are mixed"
            )
        ))

    def validate_all(self) -> Dict[str, ValidationResult]:
        """Run all comparisons."""
        self.results = []
        return {
            "three_generations": self.validate_three_generations(),
            "fourth_generation": self.validate_fourth_generation_excluded(),
            "helicity_barrier": self.validate_helicity_barrier(),
            "critical_beta": self.validate_critical_beta(),
            "primordial_fnl": self.validate_primordial_fnl(),
            "stelle_ratio": self.validate_stelle_ratio(),
            "mass_ratios": self.validate_mass_ratios(),
        }

    def summary(self) -> Dict[str, Any]:
        """Counts by status and one row per comparison."""
        if not self.results:
            self.validate_all()

        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.status.value] = counts.get(result.status.value, 0) + 1

        return {
            "total_claims": len(self.results),
            "status_counts": counts,
            "results": [
                {
                    "name": r.prediction.name,
                    "status": r.status.value,
                    "sigma": r.sigma_deviation,
                    "notes": r.notes
                }
                for r in self.results
            ]
        }

    def generate_report(self) -> str:
        """Markdown table of the comparisons."""
        summary = self.summary()
        lines = [
            "| Claim | Status | σ from claim | Notes |",
            "|-------|--------|--------------|-------|",
        ]
        for r in summary["results"]:
            sigma = "-" if r["sigma"] is None else f"{r['sigma']:.1f}"
            lines.append(f"| {r['name']} | {r['status']} | {sigma} | {r['notes']} |")
        counts = ", ".join(f"{k}: {v}" for k, v in summary["status_counts"].items())
        return "\n".join(["# Empirical Validation Report", "", f"Claims: {summary['total_claims']} ({counts})", ""] + lines)


def main():
    """Print the comparison table."""
    print(ValidationSuite().generate_report())


if __name__ == "__main__":
    main()
