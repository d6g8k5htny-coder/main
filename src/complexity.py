"""
Core Complexity Functional Implementation
==========================================

This module implements the central variational principle δC = 0 where:
    C = R + K + B

Components:
    R[H] - Retrodiction complexity (geometric/gravitational)
    K[G,R] - Representation complexity (gauge/algebraic)
    B[H] - Barrier complexity (topological constraints)

The framework unifies spacetime geometry, quantum mechanics, and gauge structure
as emergent from complexity minimization.

Status: exploratory code for the retired 2025 framework. Running it checks
arithmetic, not physics; see results/validation_report.md for what is and is
not established.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize


@dataclass
class ComplexityParameters:
    """Parameters for the complexity functional."""

    # Retrodiction complexity weights
    alpha: float = 1.0  # Einstein-Hilbert weight
    gamma_1: float = 1.0  # R^2 coefficient
    gamma_2: float = -2.0  # R_μν R^μν coefficient; default encodes the manuscript's claimed γ₁/γ₂ = -1/2

    # Representation complexity weights
    lambda_gauge: float = 1.0  # Gauge structure constant weight
    mu_rep: float = 1.0  # Representation dimension weight

    # Barrier complexity weights
    beta_barrier: float = 1.0  # Topological barrier strength
    tau_helicity: float = 0.022  # Helicity barrier coefficient

    # Generation penalty
    alpha_gen: float = 1.0  # Controls sharpness of n=3 minimum

    def stelle_ratio(self) -> float:
        """Return the Stelle ratio γ₁/γ₂."""
        if abs(self.gamma_2) < 1e-10:
            return float('inf')
        return self.gamma_1 / self.gamma_2


class ComplexityFunctional(ABC):
    """Abstract base class for complexity functionals."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Compute the complexity value."""
        pass

    @abstractmethod
    def variation(self, *args, **kwargs) -> np.ndarray:
        """Compute the functional variation δC."""
        pass


@dataclass
class MetricData:
    """Container for spacetime metric data."""

    dimension: int = 4
    signature: Tuple[int, ...] = (-1, 1, 1, 1)  # Lorentzian
    metric_tensor: Optional[np.ndarray] = None
    ricci_scalar: Optional[float] = None
    ricci_tensor: Optional[np.ndarray] = None
    riemann_tensor: Optional[np.ndarray] = None

    def sqrt_minus_g(self) -> float:
        """Compute √(-g) for the metric determinant."""
        if self.metric_tensor is None:
            return 1.0
        g = np.linalg.det(self.metric_tensor)
        return np.sqrt(abs(g))


class RetrodictionComplexity(ComplexityFunctional):
    """
    Retrodiction Complexity R[g]

    The informational cost of reconstructing interior causal structure
    from boundary data. At leading order:

        R[g] = α ∫ R √(-g) d⁴x + γ₁ ∫ R² √(-g) d⁴x + γ₂ ∫ R_μν R^μν √(-g) d⁴x

    This yields Einstein gravity with calculable quadratic corrections.
    The Stelle ratio γ₁/γ₂ = -1/2 emerges from complexity minimization.
    """

    def __init__(self, params: Optional[ComplexityParameters] = None):
        self.params = params or ComplexityParameters()

    def compute(
        self,
        metric: Optional[MetricData] = None,
        ricci_scalar: Optional[float] = None,
        ricci_squared: Optional[float] = None,
        volume: float = 1.0
    ) -> float:
        """
        Compute retrodiction complexity for given metric data.

        Parameters
        ----------
        metric : MetricData, optional
            Full metric data structure
        ricci_scalar : float, optional
            Ricci scalar R (can override metric data)
        ricci_squared : float, optional
            R_μν R^μν contraction
        volume : float
            Spacetime volume element ∫√(-g)d⁴x

        Returns
        -------
        float
            Retrodiction complexity R[g]
        """
        R = ricci_scalar
        R_squared = ricci_squared

        if metric is not None:
            if R is None:
                R = metric.ricci_scalar or 0.0
            if R_squared is None and metric.ricci_tensor is not None:
                # Compute R_μν R^μν
                R_squared = np.sum(metric.ricci_tensor ** 2)

        R = R or 0.0
        R_squared = R_squared or 0.0

        # R[g] = α∫R + γ₁∫R² + γ₂∫R_μν R^μν
        complexity = (
            self.params.alpha * R * volume +
            self.params.gamma_1 * R**2 * volume +
            self.params.gamma_2 * R_squared * volume
        )

        return complexity

    def variation(
        self,
        metric: MetricData,
        delta_metric: np.ndarray
    ) -> np.ndarray:
        """
        Compute the variation δR[g] / δg^μν.

        At leading order, variation yields Einstein tensor:
            δR/δg^μν = α(R_μν - ½g_μν R) + O(R²)
        """
        if metric.ricci_tensor is None or metric.ricci_scalar is None:
            raise ValueError("Metric must have Ricci tensor and scalar for variation")

        g = metric.metric_tensor
        R_uv = metric.ricci_tensor
        R = metric.ricci_scalar

        # Einstein tensor: G_μν = R_μν - ½g_μν R
        einstein = R_uv - 0.5 * g * R

        # Leading order variation
        delta_R = self.params.alpha * np.sum(einstein * delta_metric)

        return einstein

    def stelle_prediction(self) -> Dict[str, Any]:
        """
        Compare the configured quadratic terms with the claimed Stelle ratio.

        Modulo the Gauss-Bonnet density, γ₁R² + γ₂R_μνR^μν equals
        (γ₂/2)C² + (γ₁ + γ₂/3)R², with C the Weyl tensor. Stelle's massive
        spin-2 ghost is absent only when the C² coefficient vanishes, so no
        finite γ₁/γ₂ (including the claimed -1/2) removes it in this basis.
        """
        weyl_squared = self.params.gamma_2 / 2
        return {
            "gamma_1": self.params.gamma_1,
            "gamma_2": self.params.gamma_2,
            "stelle_ratio": self.params.stelle_ratio(),
            "claimed_ratio": -0.5,
            "weyl_squared_coefficient": weyl_squared,
            "ricci_scalar_squared_coefficient": self.params.gamma_1 + self.params.gamma_2 / 3,
            "spin2_ghost_free": weyl_squared == 0,
        }


class BarrierComplexity(ComplexityFunctional):
    """
    Barrier Complexity B[H]

    Topological and consistency constraints that prevent certain
    configurations. Key barriers:

    1. Causality barrier: Prevents closed timelike curves
    2. Helicity barrier: Constrains turbulent cascade (τ ≈ 0.022)
    3. Anomaly barrier: Enforces gauge anomaly cancellation
    4. Generation barrier: Exponential penalty for n ≠ 3 generations

    The helicity barrier constitutive law:
        |Δζ₄| = 0.1843 - 0.2051 C_B + τ C_B²
    where C_B is magnetic compressibility (manuscript §12.4).
    """

    def __init__(self, params: Optional[ComplexityParameters] = None):
        self.params = params or ComplexityParameters()

        # Regression coefficients quoted from the 2025 manuscript; not reproduced here
        self.helicity_a0 = 0.1843
        self.helicity_a1 = -0.2051
        self.helicity_a2 = 0.022  # τ coefficient

    def compute(
        self,
        n_generations: int = 3,
        cross_helicity: float = 0.0,
        plasma_beta: float = 1.0,
        anomaly_coefficient: float = 0.0,
        magnetic_compressibility: float = 0.0
    ) -> float:
        """
        Compute total barrier complexity.

        Parameters
        ----------
        n_generations : int
            Number of fermion generations
        cross_helicity : float
            Normalized cross-helicity σ_c (barrier activation)
        plasma_beta : float
            Plasma β parameter (barrier activation)
        magnetic_compressibility : float
            C_B in the constitutive law
        anomaly_coefficient : float
            Gauge anomaly coefficient (should be 0)

        Returns
        -------
        float
            Total barrier complexity B[H]
        """
        # Generation barrier: exp(α(n-3)²)
        B_gen = np.exp(self.params.alpha_gen * (n_generations - 3)**2)

        # Helicity barrier: active when β < β_c ≈ 0.5
        B_helicity = self._helicity_barrier(cross_helicity, plasma_beta, magnetic_compressibility)

        # Anomaly barrier: infinite if anomaly ≠ 0
        B_anomaly = 0.0 if abs(anomaly_coefficient) < 1e-10 else float('inf')

        return B_gen + B_helicity + B_anomaly

    def _helicity_barrier(self, sigma_c: float, beta: float, C_B: float) -> float:
        """
        Compute helicity barrier contribution.

        Activation thresholds β ≲ 0.5 and σ_c ≳ 0.4 are as stated in the 2025
        README; they are not derived or measured in this repository.
        """
        beta_c = 0.5
        sigma_c_threshold = 0.4

        if beta > beta_c or abs(sigma_c) < sigma_c_threshold:
            return 0.0

        delta_zeta4 = abs(
            self.helicity_a0 +
            self.helicity_a1 * C_B +
            self.helicity_a2 * C_B**2
        )

        return self.params.beta_barrier * delta_zeta4

    def variation(
        self,
        n_generations: int,
        delta_n: int = 0
    ) -> float:
        """
        Compute variation of barrier complexity with generation number.

        δB/δn = 2α(n-3)exp(α(n-3)²)
        """
        n = n_generations + delta_n
        return (
            2 * self.params.alpha_gen * (n - 3) *
            np.exp(self.params.alpha_gen * (n - 3)**2)
        )

    def generation_minimum(self, K_1gen: float = 1.0) -> Dict[str, Any]:
        """
        Locate the minimum of C(n) = n · K_{1-gen} + exp(α(n-3)²) over n = 1..6.

        n = 3 is the unique minimum only when α > ln(1 + K_{1-gen}). The
        penalty is centred on 3 by construction, so this is not a derivation.
        """
        results = {
            n: n * K_1gen + np.exp(self.params.alpha_gen * (n - 3)**2)
            for n in range(1, 7)
        }
        min_n = min(results, key=results.get)

        return {
            "complexity_by_generation": results,
            "minimum_at": min_n,
            "minimum_value": results[min_n],
            "alpha_threshold": float(np.log1p(K_1gen)),
            "minimum_at_3": min_n == 3,
        }

    def helicity_barrier_law(self, C_B: np.ndarray) -> np.ndarray:
        """
        Evaluate helicity barrier constitutive law.

        |Δζ₄| = 0.1843 - 0.2051 C_B + 0.022 C_B²

        Parameters
        ----------
        C_B : array_like
            Magnetic compressibility values (the fitted range is C_B ≲ 1;
            the polynomial is negative for 1.007 < C_B < 8.32)

        Returns
        -------
        array_like
            |Δζ₄| values
        """
        return np.abs(
            self.helicity_a0 +
            self.helicity_a1 * np.asarray(C_B) +
            self.helicity_a2 * np.asarray(C_B)**2
        )


class TotalComplexity(ComplexityFunctional):
    """
    Total Complexity Functional C[H, G, R]

    The complete variational principle:
        C = R[H] + K[G, R] + B[H]

    Where:
        R[H] = Retrodiction complexity (geometric)
        K[G, R] = Representation complexity (gauge/algebraic)
        B[H] = Barrier complexity (topological)

    Physical structures emerge at stationary points δC = 0.
    """

    def __init__(self, params: Optional[ComplexityParameters] = None):
        self.params = params or ComplexityParameters()
        self.retrodiction = RetrodictionComplexity(self.params)
        self.barrier = BarrierComplexity(self.params)

        # Representation complexity requires gauge_theory module
        self._representation = None

    @property
    def representation(self):
        """Lazy load representation complexity to avoid circular imports."""
        if self._representation is None:
            from .gauge_theory import RepresentationComplexity
            self._representation = RepresentationComplexity(self.params)
        return self._representation

    def compute(
        self,
        metric: Optional[MetricData] = None,
        gauge_group: str = "SU3xSU2xU1",
        n_generations: int = 3,
        cross_helicity: float = 0.0,
        plasma_beta: float = 1.0,
        magnetic_compressibility: float = 0.0,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute total complexity and all components.

        Parameters
        ----------
        metric : MetricData, optional
            Spacetime metric data
        gauge_group : str
            Gauge group identifier (e.g., "SU3xSU2xU1", "SU5")
        n_generations : int
            Number of fermion generations
        cross_helicity : float
            Normalized cross-helicity
        plasma_beta : float
            Plasma β parameter

        Returns
        -------
        dict
            Dictionary with R, K, B, and total C values
        """
        # Retrodiction complexity
        R = self.retrodiction.compute(
            metric=metric,
            ricci_scalar=kwargs.get('ricci_scalar'),
            ricci_squared=kwargs.get('ricci_squared'),
            volume=kwargs.get('volume', 1.0)
        )

        # Representation complexity
        K = self.representation.compute(gauge_group, n_generations)

        # Barrier complexity
        B = self.barrier.compute(
            n_generations=n_generations,
            cross_helicity=cross_helicity,
            plasma_beta=plasma_beta,
            magnetic_compressibility=magnetic_compressibility
        )

        return {
            "R": R,
            "K": K,
            "B": B,
            "C": R + K + B,
            "components": {
                "retrodiction": R,
                "representation": K,
                "barrier": B
            }
        }

    def variation(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Compute functional variation δC.

        At a stationary point, δC = 0 yields:
        - Einstein equations (from δR)
        - Gauge field equations (from δK)
        - Constraint equations (from δB)
        """
        return {
            "delta_R": self.retrodiction.variation(*args, **kwargs) if args else None,
            "delta_K": None,  # Requires gauge field variation
            "delta_B": self.barrier.variation(kwargs.get('n_generations', 3))
        }

    def find_stationary_point(
        self,
        initial_params: Dict[str, float],
        constraints: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Find stationary point δC = 0 numerically.

        Parameters
        ----------
        initial_params : dict
            Initial parameter values
        constraints : list, optional
            Scipy-style constraints

        Returns
        -------
        dict
            Optimized parameters and complexity value
        """
        def objective(x):
            # Unpack parameters
            params = dict(zip(initial_params.keys(), x))
            result = self.compute(**params)
            return result["C"]

        x0 = list(initial_params.values())

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints or []
        )

        return {
            "optimal_params": dict(zip(initial_params.keys(), result.x)),
            "minimum_complexity": result.fun,
            "converged": result.success,
            "message": result.message
        }

    def validate_predictions(self) -> Dict[str, Any]:
        """
        Report how the built-in claims behave in this implementation.

        None of these is an empirical test: n = 3 and the Stelle ratio are
        inputs of the functional, and K(G) is an ad hoc definition.
        """
        from .gauge_theory import compare_gauge_groups

        gen_result = self.barrier.generation_minimum()
        stelle = self.retrodiction.stelle_prediction()
        gauge_K = compare_gauge_groups()
        sm_K = gauge_K["SU3xSU2xU1"]

        return {
            "three_generations": {
                "alpha": self.params.alpha_gen,
                "alpha_threshold": gen_result["alpha_threshold"],
                "minimum_at": gen_result["minimum_at"],
                "minimum_at_3": gen_result["minimum_at_3"],
            },
            "stelle_ratio": {
                "claimed": -0.5,
                "configured": stelle["stelle_ratio"],
                "spin2_ghost_free": stelle["spin2_ghost_free"],
            },
            "gauge_group": {
                "K_rep_SM": self.representation.compute("SU3xSU2xU1", 3),
                "K_rep_SU5": self.representation.compute("SU5", 3),
                "K_gauge_by_group": gauge_K,
                "gauge_groups_below_SM": sorted(g for g, k in gauge_K.items() if k < sm_K),
            },
        }


# Convenience functions for quick calculations
def compute_generation_complexity(
    n: int,
    alpha: float = 1.0,
    K_1gen: float = 1.0
) -> float:
    """
    Compute complexity as function of generation number.

    C(n) = n · K_{1-gen} + exp(α(n-3)²)

    Parameters
    ----------
    n : int
        Number of generations
    alpha : float
        Sharpness parameter
    K_1gen : float
        Single generation complexity

    Returns
    -------
    float
        Total complexity for n generations
    """
    return n * K_1gen + np.exp(alpha * (n - 3)**2)


def demonstrate_three_generations(
    alpha_range: Tuple[float, float] = (0.1, 10.0),
    n_points: int = 100,
    K_1gen: float = 1.0
) -> Dict[str, Any]:
    """
    Tabulate the minimising n of C(n) across α.

    The README claims n = 3 for all α > 0. In fact n = 3 is the minimum
    exactly when α > ln(1 + K_{1-gen}) (≈ 0.693 for K_{1-gen} = 1); smaller
    α moves the minimum to n = 2 or n = 1.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    min_n = np.array([
        1 + int(np.argmin([compute_generation_complexity(n, a, K_1gen) for n in range(1, 7)]))
        for a in alphas
    ])
    threshold = float(np.log1p(K_1gen))

    return {
        "alpha": alphas,
        "minimum_n": min_n,
        "alpha_threshold": threshold,
        "all_equal_3": bool(np.all(min_n == 3)),
        "matches_threshold_rule": bool(np.all((min_n == 3) == (alphas > threshold))),
    }


if __name__ == "__main__":
    # Quick demonstration
    print("Complexity Physics Framework - Core Module")
    print("=" * 50)

    # Create total complexity functional
    C = TotalComplexity()

    # Compute Standard Model complexity
    result = C.compute(
        gauge_group="SU3xSU2xU1",
        n_generations=3
    )
    print(f"\nStandard Model Complexity:")
    print(f"  R (retrodiction): {result['R']:.4f}")
    print(f"  K (representation): {result['K']:.4f}")
    print(f"  B (barrier): {result['B']:.4f}")
    print(f"  C (total): {result['C']:.4f}")

    # Demonstrate three-generation theorem
    print("\nGeneration penalty:")
    gen_demo = demonstrate_three_generations()
    print(f"  Minimum at n=3 for all α ∈ [0.1, 10]: {gen_demo['all_equal_3']}")
    print(f"  n=3 exactly when α > ln 2 = {gen_demo['alpha_threshold']:.4f}: "
          f"{gen_demo['matches_threshold_rule']}")

    print("\nBuilt-in claims (not empirical tests):")
    for name, result in C.validate_predictions().items():
        print(f"  {name}: {result}")
