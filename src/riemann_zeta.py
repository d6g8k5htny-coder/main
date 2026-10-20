"""
Riemann Zeta Function and Number Theory Analysis
=================================================

Numerical checks on low-lying zeros of ζ(s) and their spacing statistics.

Established results referred to here: the Riemann Hypothesis holds for all
zeros up to height 3·10¹² (Platt & Trudgian 2021), about 1.24·10¹³ zeros; the
de Bruijn-Newman constant satisfies 0 ≤ Λ ≤ 0.2 (Rodgers & Tao 2020; Platt &
Trudgian 2021, improving Polymath15's 0.22); and Odlyzko's computations show
zero spacings agreeing with
GUE random-matrix statistics, as Montgomery's pair-correlation conjecture
predicts. The 2025 manuscript's claim that complexity minimisation explains
RH is a hypothesis; no argument for it is contained in this repository.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gamma_func


# Imaginary parts of the first 30 non-trivial zeros; they agree with
# mpmath.zetazero to 4e-14. data/riemann_zeros/first_100_zeros.json holds 100.
FIRST_ZEROS = [
    14.134725141734693,
    21.022039638771555,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
    52.970321477714460,
    56.446247697063394,
    59.347044002602353,
    60.831778524609809,
    65.112544048081607,
    67.079810529494173,
    69.546401711173979,
    72.067157674481907,
    75.704690699083933,
    77.144840068874805,
    79.337375020249367,
    82.910380854086030,
    84.735492980517050,
    87.425274613125229,
    88.809111207634465,
    92.491899270558484,
    94.651344040519848,
    95.870634228245309,
    98.831194218193692,
    101.31785100573139,
]

RH_VERIFIED_HEIGHT = 3e12  # Platt & Trudgian (2021)


def zero_count(T: float) -> float:
    """Riemann-von Mangoldt main term N(T) ≈ (T/2π) log(T/2π) - T/2π + 7/8."""
    x = T / (2 * np.pi)
    return x * np.log(x) - x + 7 / 8


@dataclass
class ZetaZero:
    """A non-trivial zero stored by its imaginary part, assumed on Re(s) = 1/2."""
    imaginary_part: float
    index: int = 0

    @property
    def real_part(self) -> float:
        """Real part assumed by the storage format (checked by verify_rh_local)."""
        return 0.5

    @property
    def complex_value(self) -> complex:
        """Return the zero as a complex number."""
        return complex(self.real_part, self.imaginary_part)

    def normalized_spacing(self, next_zero: "ZetaZero") -> float:
        """Spacing to next zero in units of the mean spacing 2π / log(t/2π)."""
        t1 = self.imaginary_part
        return (next_zero.imaginary_part - t1) * np.log(t1 / (2 * np.pi)) / (2 * np.pi)


class RiemannZeta:
    """Evaluation of ζ(s) and checks on a list of stored zeros."""

    def __init__(self, zeros: Optional[List[float]] = None):
        self.zeros = zeros if zeros is not None else FIRST_ZEROS
        self.zero_objects = [ZetaZero(t, i + 1) for i, t in enumerate(self.zeros)]

    def zeta(self, s: complex, terms: int = 1000) -> complex:
        """
        Approximate ζ(s) by truncated series: the Dirichlet series for
        Re(s) > 1, the alternating (eta) series for 0 < Re(s) ≤ 1, and the
        functional equation for Re(s) < 0. Accuracy is limited by `terms`.
        """
        if s.real <= 1:
            if s.real < 0:
                return (
                    2**s * np.pi**(s - 1) *
                    np.sin(np.pi * s / 2) *
                    gamma_func(1 - s) *
                    self.zeta(1 - s, terms)
                )
            eta = sum((-1)**(n - 1) * n**(-s) for n in range(1, terms + 1))
            return eta / (1 - 2**(1 - s))

        return sum(n**(-s) for n in range(1, terms + 1))

    def zero_density(self, T: float) -> float:
        """Approximate number of zeros with 0 < t < T (0 below T = 10)."""
        if T < 10:
            return 0.0
        return zero_count(T)

    def local_density(self, t: float) -> float:
        """Local zero density dN/dt = log(t/2π) / 2π."""
        return np.log(t / (2 * np.pi)) / (2 * np.pi)

    def verify_rh_local(self, tolerance: float = 1e-10) -> Dict[str, Any]:
        """
        Check numerically that the stored values are exactly the first N zeros.

        The values must be strictly increasing, with |ζ(1/2 + iγ)| < tolerance
        at each. mpmath.nzeros(b_k) must equal k at every bracket: b_0 below
        the first value, the midpoints between neighbours, and b_N between the
        last value and zero N + 1 (computed only to place that bracket). The
        Riemann-Siegel Z function must change sign across each bracket. Then
        each interval (b_(k-1), b_k) holds exactly one zero, on the critical
        line, at the k-th value to the residual's accuracy.

        Limits: floating-point mpmath throughout, the library that generated
        the data file. Not interval-certified, and no independent
        implementation or published table is consulted.
        scripts/cross_check_zeros_arb.py and scripts/cross_check_zeros_pari.py
        check the data file separately, with FLINT's ball arithmetic and with
        PARI/GP's independent L-function code.
        """
        import mpmath

        zeros = list(self.zeros)
        n = len(zeros)
        increasing = all(a < b for a, b in zip(zeros, zeros[1:]))
        unique = len(set(zeros)) == n
        with mpmath.workdps(20):
            max_abs = max(float(abs(mpmath.zeta(mpmath.mpc(0.5, g)))) for g in zeros)
            upper = (zeros[-1] + float(mpmath.zetazero(n + 1).imag)) / 2
            brackets = ([min(10.0, zeros[0] / 2)]
                        + [(a + b) / 2 for a, b in zip(zeros, zeros[1:])]
                        + [upper])
            counts = [int(mpmath.nzeros(b)) for b in brackets]
            signs = [mpmath.sign(mpmath.siegelz(b)) for b in brackets]

        mismatches = [k for k, c in enumerate(counts) if c != k]
        sign_changes = all(s != 0 and s == -t for s, t in zip(signs, signs[1:]))
        all_vanish = max_abs < tolerance
        complete = increasing and unique and not mismatches and sign_changes and all_vanish
        return {
            "zeros_checked": n,
            "max_abs_zeta": max_abs,
            "all_vanish": all_vanish,
            "strictly_increasing": increasing,
            "unique": unique,
            "final_bracket": upper,
            "bracket_count_mismatches": mismatches,
            "z_sign_changes": sign_changes,
            "complete_and_ordered": complete,
            "all_on_critical_line": complete,
            "method": "mpmath zeta, nzeros and siegelz (numerical, not interval-certified)",
            "independent_comparison": ("not run here; results/ records checks of the data file "
                                       "against FLINT, PARI/GP, the OEIS tables (20 digits for zeros "
                                       "1-10, integer parts for all) and Odlyzko's zeros6 table "
                                       "(9 decimals for all 100)"),
            "literature_verified_height": RH_VERIFIED_HEIGHT,
        }


class SpectralAnalysis:
    """Spacing statistics of zeta zeros compared with GUE and Poisson."""

    def __init__(self, zeros: Optional[List[float]] = None):
        self.zeros = zeros if zeros is not None else FIRST_ZEROS
        self.rz = RiemannZeta(self.zeros)

    def normalized_spacings(self) -> np.ndarray:
        """Consecutive spacings scaled by the local density at their midpoint."""
        t = np.asarray(self.zeros)
        return np.diff(t) * self.rz.local_density((t[:-1] + t[1:]) / 2)

    def nearest_neighbor_distribution(self, bins: int = 50) -> Dict[str, np.ndarray]:
        """Spacing histogram with the GUE Wigner surmise (32/π²) s² exp(-4s²/π)."""
        spacings = self.normalized_spacings()
        hist, bin_edges = np.histogram(spacings, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        gue_prediction = (32 / np.pi**2) * bin_centers**2 * np.exp(-4 * bin_centers**2 / np.pi)

        return {
            "spacings": spacings,
            "bin_centers": bin_centers,
            "histogram": hist,
            "gue_prediction": gue_prediction,
            "mean_spacing": np.mean(spacings),
            "std_spacing": np.std(spacings)
        }

    def pair_correlation(
        self,
        x_range: Tuple[float, float] = (0.0, 3.0),
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """Montgomery's R₂(x) = 1 - (sin πx / πx)² and sums of up to 9 consecutive spacings."""
        x = np.linspace(x_range[0], x_range[1], n_points)
        gue_prediction = 1 - np.sinc(x)**2

        spacings = self.normalized_spacings()
        sums = [
            spacings[i:j].sum()
            for i in range(len(spacings))
            for j in range(i + 1, min(i + 10, len(spacings)))
        ]
        return {
            "x": x,
            "gue_prediction": gue_prediction,
            "empirical_spacings": np.array(sums),
        }

    def number_variance(
        self,
        L_range: Tuple[float, float] = (0.1, 10.0),
        n_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """Large-L GUE number variance Σ²(L) ≈ (log(2πL) + γ + 1) / π²."""
        L = np.linspace(L_range[0], L_range[1], n_points)
        return {
            "L": L,
            "gue_prediction": (np.log(2 * np.pi * L) + np.euler_gamma + 1) / np.pi**2,
        }

    def level_repulsion(self) -> Dict[str, Any]:
        """Fraction of normalised spacings below 1/2, versus GUE and Poisson."""
        s = self.normalized_spacings()
        gue = quad(lambda u: (32 / np.pi**2) * u**2 * np.exp(-4 * u**2 / np.pi), 0, 0.5)[0]
        return {
            "n_spacings": len(s),
            "fraction_below_half": float(np.mean(s < 0.5)),
            "gue_wigner_surmise": gue,
            "poisson": 1 - math.exp(-0.5),
        }


class RHComplexityConnection:
    """The 2025 manuscript's proposed link between RH and complexity."""

    def __init__(self):
        self.spectral = SpectralAnalysis()

    def complexity_argument(self) -> Dict[str, str]:
        """Outline of the manuscript's argument, with its status."""
        return {
            "premise_1": "Arithmetic structure emerges from complexity minimization",
            "premise_2": "Zeta zeros encode prime distribution information",
            "premise_3": "GUE statistics indicate Hermitian operator spectrum",
            "premise_4": "Hermitian operator ⟹ real eigenvalues ⟹ Re(s) = 1/2",
            "conclusion": "RH follows from complexity minimization in arithmetic space",
            "status": (
                "Hypothesis from the retired 2025 manuscript. Premise 4 is the "
                "Hilbert-Pólya idea, which presupposes an operator nobody has "
                "constructed; premises 1 and 3 are unproved. No proof or evidence "
                "for the conclusion is in this repository."
            ),
        }

    def de_bruijn_newman(self) -> Dict[str, Any]:
        """Known bounds on the de Bruijn-Newman constant Λ (RH ⟺ Λ ≤ 0)."""
        return {
            "lower_bound": 0.0,
            "lower_bound_source": "Rodgers & Tao, Forum Math. Pi 8 (2020) e6",
            "upper_bound": 0.2,
            "upper_bound_source": (
                "Platt & Trudgian, Bull. London Math. Soc. 53 (2021), improving the "
                "Polymath15 bound 0.22 (arXiv:1904.12438)"
            ),
            "rh_equivalent": "Λ ≤ 0, i.e. Λ = 0 given the lower bound",
        }

    def explicit_formula_test(self, x: float, zeros: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Compare Chebyshev ψ(x) with the explicit formula truncated at the given zeros:
        ψ₀(x) = x - Σ_ρ x^ρ/ρ - log 2π - ½ log(1 - x⁻²), pairing each ρ with its conjugate.
        """
        gammas = FIRST_ZEROS if zeros is None else zeros
        zero_sum = sum(2 * (x ** complex(0.5, g) / complex(0.5, g)).real for g in gammas)
        explicit = x - zero_sum - math.log(2 * math.pi) - 0.5 * math.log(1 - x ** -2)

        n = int(x)
        sieve = np.ones(n + 1, dtype=bool)
        sieve[:2] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = False
        exact = 0.0
        for p in np.flatnonzero(sieve):
            exact += math.log(p) * int(math.log(x) / math.log(p) + 1e-12)

        return {
            "x": x,
            "psi_exact": exact,
            "psi_explicit": explicit,
            "relative_error": abs(explicit - exact) / exact,
            "n_zeros_used": len(gammas),
        }


if __name__ == "__main__":
    rz = RiemannZeta()
    check = rz.verify_rh_local()
    print(f"Stored zeros: {check['zeros_checked']}, max |ζ| = {check['max_abs_zeta']:.1e}, "
          f"complete and ordered through t = {check['final_bracket']:.2f}: "
          f"{check['complete_and_ordered']} (numerical; FLINT, PARI, OEIS and Odlyzko-table "
          f"checks of the data file in results/)")

    rep = SpectralAnalysis().level_repulsion()
    print(f"Spacings below 1/2: {rep['fraction_below_half']:.3f} "
          f"(GUE {rep['gue_wigner_surmise']:.3f}, Poisson {rep['poisson']:.3f})")

    ef = RHComplexityConnection().explicit_formula_test(100.5)
    print(f"ψ(100.5) = {ef['psi_exact']:.3f}; explicit formula with 30 zeros "
          f"{ef['psi_explicit']:.3f}")
    print(f"Zeros up to height {RH_VERIFIED_HEIGHT:.0e}: about {zero_count(RH_VERIFIED_HEIGHT):.3e}")
