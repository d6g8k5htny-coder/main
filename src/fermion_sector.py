"""
Fermion Sector and Cl(6) Mass Hierarchy
=======================================

Implements the grade-suppression mass model of the retired 2025 manuscript
(body §7 and Appendix B): ε = dim Cl¹(6) / dim Cl(6) = 6/64 and m_f ∝ ε^g(f),
with the third generation at grade 0, the second at grade 1 and the first at
grade 2. Every adjacent-generation mass ratio is therefore predicted to be
1/ε ≈ 10.67; the observed ratios range from about 1.6 to 55 times that value
(see validate_mass_ratios and results/validation_report.md).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import comb


# Masses in GeV, transcribed from memory by the 2025-12-16 session. Central
# values appear to match the 2022 Review of Particle Physics rather than 2024,
# and asymmetric light-quark errors were replaced by their upper error. Schemes
# are mixed: light quarks MS-bar at 2 GeV, c and b MS-bar at their own mass,
# t pole. Not re-verified; consult the PDG for quantitative work.
PDG_MASSES = {
    "up": {"value": 2.16e-3, "error": 0.49e-3},
    "down": {"value": 4.67e-3, "error": 0.48e-3},
    "strange": {"value": 93.4e-3, "error": 8.6e-3},
    "charm": {"value": 1.27, "error": 0.02},
    "bottom": {"value": 4.18, "error": 0.03},
    "top": {"value": 172.69, "error": 0.30},

    "electron": {"value": 0.51099895e-3, "error": 0.00000015e-3},
    "muon": {"value": 0.1056583755, "error": 0.0000000023},
    "tau": {"value": 1.77686, "error": 0.00012},

    "nu_e": {"value": 0.0, "upper_limit": 0.8e-9},
    "nu_mu": {"value": 0.0, "upper_limit": 0.19e-3},
    "nu_tau": {"value": 0.0, "upper_limit": 18.2e-3},
}

# |V_ij|, with the same provenance caveats as PDG_MASSES.
CKM_MATRIX = {
    "V_ud": {"value": 0.97373, "error": 0.00031},
    "V_us": {"value": 0.2243, "error": 0.0008},
    "V_ub": {"value": 0.00382, "error": 0.00020},
    "V_cd": {"value": 0.221, "error": 0.004},
    "V_cs": {"value": 0.975, "error": 0.006},
    "V_cb": {"value": 0.0408, "error": 0.0014},
    "V_td": {"value": 0.0086, "error": 0.0002},
    "V_ts": {"value": 0.0415, "error": 0.0009},
    "V_tb": {"value": 1.014, "error": 0.029},
}

# Adjacent-generation ratios compared in the manuscript (§7.3 and Table B.4)
_RATIO_PAIRS = {
    "top_charm": ("top", "charm"),
    "charm_up": ("charm", "up"),
    "bottom_strange": ("bottom", "strange"),
    "strange_down": ("strange", "down"),
    "tau_muon": ("tau", "muon"),
    "muon_electron": ("muon", "electron"),
}


@dataclass
class CliffordAlgebra:
    """
    Clifford algebra Cl(n): dimension 2^n, grade-k subspace of dimension C(n, k).
    """

    n: int
    dimension: int = field(init=False)
    grade_dimensions: List[int] = field(init=False)

    def __post_init__(self):
        self.dimension = 2 ** self.n
        self.grade_dimensions = [
            int(comb(self.n, k)) for k in range(self.n + 1)
        ]

    def grade_dimension(self, k: int) -> int:
        """Return dimension of grade-k subspace."""
        if 0 <= k <= self.n:
            return int(comb(self.n, k))
        return 0


class Cl6MassHierarchy:
    """
    The manuscript's grade-suppression model of fermion masses.

    Generation g sits at grade 3 - g, so its masses scale as ε^(3 - g).
    """

    def __init__(self, epsilon: Optional[float] = None):
        self.cl6 = CliffordAlgebra(n=6)
        self.epsilon = epsilon if epsilon is not None else 6.0 / 64.0
        self.generation_grades = {3: 0, 2: 1, 1: 2}

    @property
    def suppression_parameter(self) -> float:
        """Return the suppression parameter ε."""
        return self.epsilon

    def mass_suppression(self, generation: int) -> float:
        """Return ε^grade for generation 1, 2 or 3."""
        if generation not in self.generation_grades:
            raise ValueError("Generation must be 1, 2, or 3")
        return self.epsilon ** self.generation_grades[generation]

    def mass_ratios(self) -> Dict[str, float]:
        """Predicted adjacent-generation mass ratios (all 1/ε)."""
        return {name: 1.0 / self.epsilon for name in _RATIO_PAIRS}

    def validate_mass_ratios(self) -> Dict[str, Dict[str, float]]:
        """Compare predicted ratios with PDG_MASSES; factor = observed / predicted."""
        predictions = self.mass_ratios()
        comparisons = {}
        for name, (heavy, light) in _RATIO_PAIRS.items():
            observed = PDG_MASSES[heavy]["value"] / PDG_MASSES[light]["value"]
            comparisons[name] = {
                "predicted": predictions[name],
                "observed": observed,
                "factor": observed / predictions[name],
            }
        return comparisons

    def cabibbo_exponent(self) -> Dict[str, float]:
        """
        Exponent k with ε^k = |V_us|.

        The manuscript (eq. 7.4) fits |V_us| ≈ ε^0.66; the exponent is chosen
        to match, so this is a reparametrisation rather than a prediction.
        """
        v_us = CKM_MATRIX["V_us"]["value"]
        return {
            "V_us": v_us,
            "exponent": float(np.log(v_us) / np.log(self.epsilon)),
            "manuscript_exponent": 0.66,
            "epsilon_to_manuscript_exponent": self.epsilon ** 0.66,
        }

    def fit_epsilon(self) -> Dict[str, Any]:
        """Geometric-mean ε implied by the six observed ratios, versus 6/64."""
        observed = [c["observed"] for c in self.validate_mass_ratios().values()]
        epsilon_fit = float(np.exp(-np.mean(np.log(observed))))
        return {
            "epsilon_estimates": [1.0 / r for r in observed],
            "epsilon_geometric_mean": epsilon_fit,
            "theoretical_value": 6.0 / 64.0,
            "theoretical_over_fit": (6.0 / 64.0) / epsilon_fit,
        }


@dataclass
class FermionGeneration:
    """One fermion generation: two quarks and two leptons (masses in GeV)."""

    generation: int
    quarks: Dict[str, float] = field(default_factory=dict)
    leptons: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_pdg(cls, generation: int) -> "FermionGeneration":
        """Create generation from PDG_MASSES."""
        content = {
            1: (("up", "down"), ("electron", "nu_e")),
            2: (("charm", "strange"), ("muon", "nu_mu")),
            3: (("top", "bottom"), ("tau", "nu_tau")),
        }
        if generation not in content:
            raise ValueError(f"Invalid generation: {generation}")
        quarks, leptons = content[generation]
        return cls(
            generation=generation,
            quarks={q: PDG_MASSES[q]["value"] for q in quarks},
            leptons={lep: PDG_MASSES[lep]["value"] for lep in leptons},
        )

    def total_mass(self) -> float:
        """Return total mass of generation (quarks + leptons)."""
        return sum(self.quarks.values()) + sum(self.leptons.values())


class ThreeGenerationTheorem:
    """
    Generation penalty C(n) = n · K_{1-gen} + exp(α(n-3)²).

    The README states the minimum is n = 3 for all α > 0. It is the unique
    minimum exactly when α > ln(1 + K_{1-gen}); for smaller α it moves to
    n = 2 or n = 1. The penalty is centred on 3 by construction, so even
    above the threshold this restates an input rather than deriving it.
    """

    def __init__(self, alpha: float = 1.0, K_1gen: float = 1.0):
        self.alpha = alpha
        self.K_1gen = K_1gen

    def complexity(self, n: int, alpha: Optional[float] = None) -> float:
        """Compute complexity for n generations."""
        a = self.alpha if alpha is None else alpha
        with np.errstate(over="ignore"):
            return float(n * self.K_1gen + np.exp(a * (n - 3)**2))

    def alpha_threshold(self) -> float:
        """Smallest α above which n = 3 is the unique minimum."""
        return float(np.log1p(self.K_1gen))

    def _argmin(self, alpha: float, n_max: int) -> int:
        values = [self.complexity(n, alpha) for n in range(1, n_max + 1)]
        return 1 + int(np.argmin(values))

    def find_minimum(self, n_max: int = 10) -> Dict[str, Any]:
        """Find the minimising n for the configured α."""
        complexities = {n: self.complexity(n) for n in range(1, n_max + 1)}
        min_n = min(complexities, key=complexities.get)
        return {
            "minimum_at": min_n,
            "minimum_value": complexities[min_n],
            "all_values": complexities,
            "minimum_at_3": min_n == 3,
        }

    def sensitivity_analysis(
        self,
        alpha_range: Tuple[float, float] = (0.01, 100.0),
        n_points: int = 1000,
        n_max: int = 10
    ) -> Dict[str, Any]:
        """Minimising n across log-spaced α, compared with the threshold rule."""
        alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_points)
        min_ns = np.array([self._argmin(a, n_max) for a in alphas])
        threshold = self.alpha_threshold()
        return {
            "alpha": alphas,
            "minimum_n": min_ns,
            "alpha_threshold": threshold,
            "all_equal_3": bool(np.all(min_ns == 3)),
            "matches_threshold_rule": bool(np.all((min_ns == 3) == (alphas > threshold))),
        }

    def experimental_evidence(self) -> Dict[str, Dict[str, Any]]:
        """
        Measurements consistent with three light generations.

        They do not test the penalty itself, whose minimum location is an input.
        """
        return {
            "LEP_neutrinos": {
                "measurement": "N_ν from the Z width (LEP/SLD 2006 combination)",
                "value": 2.984,
                "error": 0.008,
                "sigma_from_3": (3 - 2.984) / 0.008,
            },
            "LEP_neutrinos_reanalysis": {
                "measurement": "N_ν with corrected Bhabha luminosity (Janot & Jadach 2020)",
                "value": 2.9963,
                "error": 0.0074,
                "sigma_from_3": (3 - 2.9963) / 0.0074,
            },
            "LHC_4th_generation": {
                "measurement": "Sequential 4th-generation t' limit, as stated in the 2025 README",
                "limit_GeV": 656,
                "verified_here": False,
            },
            "Planck_Neff": {
                "measurement": "Effective neutrino number (Planck 2018 + BAO)",
                "value": 2.99,
                "error": 0.17,
                "sigma_from_3.046": (3.046 - 2.99) / 0.17,
            },
        }


if __name__ == "__main__":
    cl6 = Cl6MassHierarchy()
    print(f"Cl(6): dimension {cl6.cl6.dimension}, grades {cl6.cl6.grade_dimensions}, "
          f"ε = {cl6.epsilon:.5f}")

    print("\nAdjacent-generation mass ratios (predicted 1/ε for all):")
    for ratio, data in cl6.validate_mass_ratios().items():
        print(f"  {ratio:15s} observed {data['observed']:8.1f}  "
              f"observed/predicted {data['factor']:6.2f}")

    fit = cl6.fit_epsilon()
    print(f"\nGeometric-mean ε implied by data: {fit['epsilon_geometric_mean']:.4f} "
          f"(6/64 is {fit['theoretical_over_fit']:.1f}× larger)")
    cab = cl6.cabibbo_exponent()
    print(f"|V_us| = ε^{cab['exponent']:.3f}; manuscript quotes ε^0.66 = "
          f"{cab['epsilon_to_manuscript_exponent']:.4f}")

    theorem = ThreeGenerationTheorem()
    sens = theorem.sensitivity_analysis()
    print(f"\nGeneration penalty: n=3 for all α ∈ [0.01, 100]: {sens['all_equal_3']}; "
          f"n=3 exactly when α > {sens['alpha_threshold']:.4f}: {sens['matches_threshold_rule']}")
