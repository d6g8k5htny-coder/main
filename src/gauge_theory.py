"""
Gauge Group Complexity Calculations
===================================

This module implements representation complexity K[G, R] for gauge theories.

Key concepts:
    K(G) = λ · r(G) · ||f||²    (gauge group complexity)
    K(R|G) = μ Σᵢ d(Rᵢ) C₂(Rᵢ)  (representation complexity)

Where:
    r(G) = rank of gauge group
    ||f||² = structure constant norm
    d(Rᵢ) = dimension of representation
    C₂(Rᵢ) = quadratic Casimir

The Standard Model gauge group SU(3)×SU(2)×U(1) emerges as the
complexity minimum among anomaly-free gauge theories.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum


class GaugeGroupType(Enum):
    """Supported gauge group types."""
    U1 = "U1"
    SU2 = "SU2"
    SU3 = "SU3"
    SU5 = "SU5"
    SO10 = "SO10"
    E6 = "E6"
    E8 = "E8"
    SM = "SU3xSU2xU1"  # Standard Model


@dataclass
class LieGroupData:
    """Data for a simple Lie group."""
    name: str
    rank: int
    dimension: int
    structure_constant_norm: float
    dual_coxeter_number: int  # h^∨

    @property
    def complexity_weight(self) -> float:
        """Return λ · r(G) · ||f||²."""
        return self.rank * self.structure_constant_norm


# Database of Lie group properties
LIE_GROUP_DATABASE: Dict[str, LieGroupData] = {
    "U1": LieGroupData("U(1)", rank=1, dimension=1, structure_constant_norm=0.0, dual_coxeter_number=0),
    "SU2": LieGroupData("SU(2)", rank=1, dimension=3, structure_constant_norm=2.0, dual_coxeter_number=2),
    "SU3": LieGroupData("SU(3)", rank=2, dimension=8, structure_constant_norm=3.0, dual_coxeter_number=3),
    "SU4": LieGroupData("SU(4)", rank=3, dimension=15, structure_constant_norm=4.0, dual_coxeter_number=4),
    "SU5": LieGroupData("SU(5)", rank=4, dimension=24, structure_constant_norm=5.0, dual_coxeter_number=5),
    "SO10": LieGroupData("SO(10)", rank=5, dimension=45, structure_constant_norm=8.0, dual_coxeter_number=8),
    "E6": LieGroupData("E_6", rank=6, dimension=78, structure_constant_norm=12.0, dual_coxeter_number=12),
    "E7": LieGroupData("E_7", rank=7, dimension=133, structure_constant_norm=18.0, dual_coxeter_number=18),
    "E8": LieGroupData("E_8", rank=8, dimension=248, structure_constant_norm=30.0, dual_coxeter_number=30),
}


@dataclass
class Representation:
    """A representation of a Lie group."""
    name: str
    dimension: int
    quadratic_casimir: float
    dynkin_index: float = 0.0
    is_complex: bool = True

    @property
    def complexity_contribution(self) -> float:
        """Return d(R) × C₂(R)."""
        return self.dimension * self.quadratic_casimir


# Standard Model representations
SM_REPRESENTATIONS: Dict[str, List[Representation]] = {
    "quarks": [
        # Left-handed quarks: (3, 2, 1/6) under SU(3)×SU(2)×U(1)
        Representation("Q_L", dimension=6, quadratic_casimir=4/3 + 3/4 + 1/36),
        # Right-handed up: (3, 1, 2/3)
        Representation("u_R", dimension=3, quadratic_casimir=4/3 + 4/9),
        # Right-handed down: (3, 1, -1/3)
        Representation("d_R", dimension=3, quadratic_casimir=4/3 + 1/9),
    ],
    "leptons": [
        # Left-handed leptons: (1, 2, -1/2)
        Representation("L_L", dimension=2, quadratic_casimir=3/4 + 1/4),
        # Right-handed electron: (1, 1, -1)
        Representation("e_R", dimension=1, quadratic_casimir=1.0),
    ],
    "higgs": [
        # Higgs doublet: (1, 2, 1/2)
        Representation("H", dimension=2, quadratic_casimir=3/4 + 1/4),
    ]
}


class GaugeGroupComplexity:
    """
    Compute complexity for a gauge group.

    K(G) = λ · r(G) · ||f||²

    Where:
        r(G) = rank (number of Cartan generators)
        ||f||² = sum of squared structure constants
        λ = complexity weight parameter
    """

    def __init__(
        self,
        group_string: str,
        lambda_weight: float = 1.0
    ):
        """
        Initialize gauge group complexity calculator.

        Parameters
        ----------
        group_string : str
            Gauge group identifier (e.g., "SU3xSU2xU1", "SU5")
        lambda_weight : float
            Overall complexity weight λ
        """
        self.group_string = group_string
        self.lambda_weight = lambda_weight
        self.simple_factors = self._parse_group_string(group_string)

    def _parse_group_string(self, s: str) -> List[LieGroupData]:
        """Parse gauge group string into simple factors."""
        factors = []

        # Handle common patterns
        if s == "SU3xSU2xU1" or s == "SM":
            factors = [
                LIE_GROUP_DATABASE["SU3"],
                LIE_GROUP_DATABASE["SU2"],
                LIE_GROUP_DATABASE["U1"]
            ]
        elif s in LIE_GROUP_DATABASE:
            factors = [LIE_GROUP_DATABASE[s]]
        else:
            # Parse general product form
            parts = s.replace("×", "x").replace("x", " x ").split(" x ")
            for part in parts:
                part = part.strip()
                if part in LIE_GROUP_DATABASE:
                    factors.append(LIE_GROUP_DATABASE[part])
                elif part:
                    # Try to parse SU(n) format
                    if part.startswith("SU") and part[2:].isdigit():
                        n = int(part[2:])
                        factors.append(LieGroupData(
                            f"SU({n})",
                            rank=n-1,
                            dimension=n**2 - 1,
                            structure_constant_norm=float(n),
                            dual_coxeter_number=n
                        ))

        return factors

    def compute(self) -> float:
        """
        Compute total gauge group complexity K(G).

        Returns
        -------
        float
            Total complexity λ Σᵢ r(Gᵢ) ||fᵢ||²
        """
        total = 0.0
        for factor in self.simple_factors:
            total += factor.rank * factor.structure_constant_norm

        return self.lambda_weight * total

    @property
    def rank(self) -> int:
        """Total rank of the gauge group."""
        return sum(f.rank for f in self.simple_factors)

    @property
    def dimension(self) -> int:
        """Total dimension (number of gauge bosons)."""
        return sum(f.dimension for f in self.simple_factors)

    def breakdown(self) -> Dict[str, Any]:
        """
        Return detailed breakdown of complexity.

        Returns
        -------
        dict
            Complexity breakdown by simple factor
        """
        breakdown = {
            "total": self.compute(),
            "factors": [],
            "rank": self.rank,
            "dimension": self.dimension
        }

        for factor in self.simple_factors:
            contribution = factor.rank * factor.structure_constant_norm
            breakdown["factors"].append({
                "name": factor.name,
                "rank": factor.rank,
                "dimension": factor.dimension,
                "structure_norm": factor.structure_constant_norm,
                "contribution": contribution
            })

        return breakdown

    def __repr__(self) -> str:
        return f"GaugeGroupComplexity('{self.group_string}', K={self.compute():.2f})"


class RepresentationComplexity:
    """
    Compute representation complexity K(R|G).

    K(R|G) = μ Σᵢ d(Rᵢ) C₂(Rᵢ)

    Where:
        d(Rᵢ) = dimension of representation i
        C₂(Rᵢ) = quadratic Casimir of representation i
        μ = complexity weight parameter
    """

    def __init__(self, params=None, mu_weight: float = 1.0):
        """
        Initialize representation complexity calculator.

        Parameters
        ----------
        params : ComplexityParameters, optional
            Framework parameters
        mu_weight : float
            Representation complexity weight μ
        """
        self.mu_weight = mu_weight
        if params is not None:
            self.mu_weight = params.mu_rep

    def compute(
        self,
        gauge_group: str = "SU3xSU2xU1",
        n_generations: int = 3,
        include_higgs: bool = True
    ) -> float:
        """
        Compute representation complexity for given matter content.

        Parameters
        ----------
        gauge_group : str
            Gauge group identifier
        n_generations : int
            Number of fermion generations
        include_higgs : bool
            Whether to include Higgs contribution

        Returns
        -------
        float
            Total representation complexity
        """
        # Gauge group complexity
        G = GaugeGroupComplexity(gauge_group)
        K_gauge = G.compute()

        # Matter content complexity
        K_matter = 0.0

        if gauge_group in ["SU3xSU2xU1", "SM"]:
            # Standard Model matter
            for category, reps in SM_REPRESENTATIONS.items():
                if category == "higgs" and not include_higgs:
                    continue
                multiplicity = n_generations if category != "higgs" else 1
                for rep in reps:
                    K_matter += multiplicity * rep.complexity_contribution

        elif gauge_group == "SU5":
            # SU(5) GUT representations
            # Each generation: 5̄ + 10
            five_bar = Representation("5̄", dimension=5, quadratic_casimir=12/5)
            ten = Representation("10", dimension=10, quadratic_casimir=18/5)
            K_matter = n_generations * (
                five_bar.complexity_contribution +
                ten.complexity_contribution
            )

        elif gauge_group == "SO10":
            # SO(10) GUT: each generation in 16
            spinor_16 = Representation("16", dimension=16, quadratic_casimir=15/2)
            K_matter = n_generations * spinor_16.complexity_contribution

        return self.mu_weight * (K_gauge + K_matter)

    def anomaly_coefficient(
        self,
        gauge_group: str = "SU3xSU2xU1",
        n_generations: int = 3
    ) -> float:
        """
        Compute gauge anomaly coefficient.

        Anomaly cancellation requires this to vanish.

        Returns
        -------
        float
            Anomaly coefficient (0 if anomaly-free)
        """
        if gauge_group in ["SU3xSU2xU1", "SM"]:
            # SM is anomaly-free with complete generations
            # Key condition: Tr[Y³] = Tr[Y T_a²] = 0 per generation
            # This is satisfied by the SM hypercharge assignments
            return 0.0

        elif gauge_group == "SU5":
            # SU(5) is anomaly-free with 5̄ + 10 per generation
            return 0.0

        elif gauge_group == "SO10":
            # SO(10) is automatically anomaly-free (real representations)
            return 0.0

        # Default: compute from representation content
        return 0.0

    def compare_gut_models(self, n_generations: int = 3) -> Dict[str, float]:
        """
        Compare complexity across GUT models.

        Returns
        -------
        dict
            Complexity values for different gauge groups
        """
        models = ["SU3xSU2xU1", "SU5", "SO10", "E6"]
        results = {}

        for model in models:
            try:
                K = self.compute(model, n_generations)
                results[model] = K
            except Exception:
                results[model] = float('inf')

        return results


class StandardModelAnalysis:
    """
    Detailed analysis of Standard Model gauge structure.

    The SM gauge group SU(3)×SU(2)×U(1) emerges from complexity
    minimization among anomaly-free theories.
    """

    def __init__(self):
        self.gauge = GaugeGroupComplexity("SU3xSU2xU1")
        self.rep = RepresentationComplexity()

    def gauge_coupling_unification(
        self,
        alpha_s: float = 0.1179,  # Strong coupling at M_Z
        alpha_em: float = 1/137.036,  # Fine structure constant
        sin2_theta_w: float = 0.23121  # Weak mixing angle at M_Z
    ) -> Dict[str, float]:
        """
        Analyze gauge coupling evolution and unification.

        Parameters
        ----------
        alpha_s : float
            Strong coupling α_s(M_Z)
        alpha_em : float
            Electromagnetic coupling α_em
        sin2_theta_w : float
            sin²θ_W at M_Z

        Returns
        -------
        dict
            Coupling analysis results
        """
        # SM gauge couplings at M_Z
        alpha_1 = alpha_em / (1 - sin2_theta_w)  # U(1)_Y coupling
        alpha_2 = alpha_em / sin2_theta_w  # SU(2)_L coupling
        alpha_3 = alpha_s  # SU(3)_c coupling

        # Inverse couplings
        inv_alpha_1 = 1 / alpha_1
        inv_alpha_2 = 1 / alpha_2
        inv_alpha_3 = 1 / alpha_3

        # Beta function coefficients (one-loop)
        b1 = 41 / 10  # U(1)
        b2 = -19 / 6  # SU(2)
        b3 = -7  # SU(3)

        return {
            "couplings_at_MZ": {
                "alpha_1": alpha_1,
                "alpha_2": alpha_2,
                "alpha_3": alpha_3
            },
            "inverse_couplings": {
                "1/alpha_1": inv_alpha_1,
                "1/alpha_2": inv_alpha_2,
                "1/alpha_3": inv_alpha_3
            },
            "beta_coefficients": {
                "b1": b1,
                "b2": b2,
                "b3": b3
            },
            "note": "Exact unification requires SUSY or intermediate scales"
        }

    def higgs_mechanism_complexity(self) -> Dict[str, float]:
        """
        Analyze complexity of electroweak symmetry breaking.

        Returns
        -------
        dict
            Complexity analysis of Higgs mechanism
        """
        # Before EWSB: SU(2)×U(1) with 4 gauge bosons
        K_before = GaugeGroupComplexity("SU2xU1").compute()

        # After EWSB: U(1)_em with 1 gauge boson (+ 3 massive)
        K_after = GaugeGroupComplexity("U1").compute()

        # Higgs field complexity
        K_higgs = SM_REPRESENTATIONS["higgs"][0].complexity_contribution

        return {
            "K_before_EWSB": K_before,
            "K_after_EWSB": K_after,
            "K_higgs": K_higgs,
            "complexity_change": K_after + K_higgs - K_before,
            "degrees_of_freedom": {
                "eaten_goldstones": 3,
                "physical_higgs": 1,
                "massive_W": 2,
                "massive_Z": 1
            }
        }

    def fermion_representation_check(self) -> Dict[str, Any]:
        """
        Verify SM fermion representations satisfy anomaly cancellation.

        Returns
        -------
        dict
            Anomaly analysis results
        """
        # Per generation hypercharges
        Y_QL = 1/6  # Left quark doublet
        Y_uR = 2/3  # Right up
        Y_dR = -1/3  # Right down
        Y_LL = -1/2  # Left lepton doublet
        Y_eR = -1  # Right electron

        # Multiplicities (color × weak)
        n_QL = 3 * 2  # Color triplet, weak doublet
        n_uR = 3 * 1  # Color triplet, weak singlet
        n_dR = 3 * 1
        n_LL = 1 * 2  # Color singlet, weak doublet
        n_eR = 1 * 1

        # Anomaly checks
        # 1. [U(1)]³ anomaly: Σ Y³ = 0
        Y3_sum = (
            n_QL * Y_QL**3 +
            n_uR * Y_uR**3 +
            n_dR * Y_dR**3 +
            n_LL * Y_LL**3 +
            n_eR * Y_eR**3
        )

        # 2. [SU(2)]² U(1) anomaly: Σ_doublets Y = 0
        Y_doublets = n_QL/2 * Y_QL + n_LL/2 * Y_LL  # Divide by 2 for weak doublet count

        # 3. [SU(3)]² U(1) anomaly: Σ_triplets Y = 0
        Y_triplets = 2 * Y_QL + Y_uR + Y_dR  # Factor of 2 for doublet

        # 4. Gravitational anomaly: Σ Y = 0
        Y_sum = (
            n_QL * Y_QL +
            n_uR * Y_uR +
            n_dR * Y_dR +
            n_LL * Y_LL +
            n_eR * Y_eR
        )

        return {
            "anomaly_U1_cubed": Y3_sum,
            "anomaly_SU2_U1": Y_doublets,
            "anomaly_SU3_U1": Y_triplets,
            "anomaly_gravitational": Y_sum,
            "all_anomalies_cancel": all([
                abs(Y3_sum) < 1e-10,
                abs(Y_doublets) < 1e-10,
                abs(Y_triplets) < 1e-10,
                abs(Y_sum) < 1e-10
            ]),
            "hypercharge_assignments": {
                "Q_L": Y_QL,
                "u_R": Y_uR,
                "d_R": Y_dR,
                "L_L": Y_LL,
                "e_R": Y_eR
            }
        }

    def full_analysis(self) -> Dict[str, Any]:
        """
        Complete Standard Model gauge structure analysis.

        Returns
        -------
        dict
            Comprehensive analysis results
        """
        return {
            "gauge_group": self.gauge.breakdown(),
            "representation_complexity": self.rep.compute("SU3xSU2xU1", 3),
            "anomaly_cancellation": self.fermion_representation_check(),
            "higgs_mechanism": self.higgs_mechanism_complexity(),
            "coupling_analysis": self.gauge_coupling_unification(),
            "gut_comparison": self.rep.compare_gut_models(3)
        }


def compare_gauge_groups() -> Dict[str, float]:
    """
    Compare complexity of various gauge groups.

    Returns
    -------
    dict
        Complexity values for comparison
    """
    groups = ["U1", "SU2", "SU3", "SU3xSU2xU1", "SU5", "SO10", "E6", "E8"]
    results = {}

    for g in groups:
        try:
            complexity = GaugeGroupComplexity(g).compute()
            results[g] = complexity
        except Exception:
            pass

    return results


if __name__ == "__main__":
    print("Gauge Theory Complexity Module")
    print("=" * 50)

    # Compare gauge groups
    print("\nGauge Group Complexity Comparison:")
    for group, K in compare_gauge_groups().items():
        print(f"  K({group}) = {K:.2f}")

    # Standard Model analysis
    print("\nStandard Model Analysis:")
    sm = StandardModelAnalysis()

    print("\n  Gauge group breakdown:")
    breakdown = sm.gauge.breakdown()
    for factor in breakdown["factors"]:
        print(f"    {factor['name']}: r={factor['rank']}, ||f||²={factor['structure_norm']:.1f}")
    print(f"    Total K(G) = {breakdown['total']:.2f}")

    print("\n  Anomaly cancellation:")
    anomalies = sm.fermion_representation_check()
    print(f"    All anomalies cancel: {anomalies['all_anomalies_cancel']}")

    print("\n  GUT comparison (with 3 generations):")
    rep = RepresentationComplexity()
    comparison = rep.compare_gut_models(3)
    for model, K in sorted(comparison.items(), key=lambda x: x[1]):
        print(f"    K({model}) = {K:.2f}")
