# Measuring finite H₀ lifetimes on the SIDE24 torus

[Research plan](README.md) · [Exact mathematical sources](SOURCES.json)

**Protocol retained from the original proposal.** A successor [implementation and fixed pilot](../../../experiments/periodic_h0/README.md) now supplies actual observations, controls and a figure. The pilot is inconclusive at the tested resolutions. This protocol itself is not a simulation result or continuum certificate. This protocol measures the unrestricted finite-bar observable. A compact birth/gap or pair-Palm experiment is a different design and must use its own normalizer and coefficient. Begin in d=2; d=3 needs separate resource and source-alignment checks.

## 1. Freeze the observable before generating data

For each independent realization f on a torus of side L=24, record all finite positive H₀ lifetimes of the ordinary superlevel filtration. Implement it as the sublevel filtration of -f. If the library returns finite endpoints u≤v for -f, the superlevel birth is -u, death is -v and lifetime is v-u. Exclude the essential H₀ interval separately, never by dropping the longest *finite* interval. Keep zero-lifetime/tie counts as diagnostics, not silently as positive short bars. GUDHI's default minimum persistence excludes zero-length pairs; collect those separately with `min_persistence=-1` or an explicit prefilter tie counter, while leaving the scientific count restricted to strictly positive finite lifetimes.

For m independent realizations and bins [a_j,b_j), 0<a_j<b_j, save the count N_ij in each realization i. Report

```
mean count per volume in bin j = Σ_i N_ij / (m L^d)
intensity estimate in bin j    = Σ_i N_ij / (m L^d (b_j-a_j)).
```

Do not divide by the total number of bars. That produces a different, normalized distribution and destroys an absolute coefficient comparison. Do not condition away realizations with zero counts. Confidence intervals must use independent **realizations** as replicates; bars within one field are dependent.

If the source density is c ell^(-1/3), the leading prediction for a bin is

```
per-volume bin mass = (3c/2) [b_j^(2/3)-a_j^(2/3)]
expected count across all realizations = m L^d times that mass.
```

Thus compare integrated bin masses, rather than c evaluated at an arbitrary bin center. A source remainder bounded by an unknown C adds at most C(b_j-a_j) to a per-volume bin mass **inside its also unknown valid range**. This is not a numerical error bar until C and the range are evaluated. Predeclare exploratory and confirmation windows; do not select only a window whose fitted slope looks right. Save slope-free coefficient ratios as well as any exploratory slope fit.

## 2. Generate the specified covariance, with truncation visible

The parent Fourier weights are

```
a_n = exp(-2 pi² |n|²/L²) / Σ_j exp(-2 pi² |j|²/L²), n in Z^d.
```

For one representative n from each pair {n,-n}, independent standard normals A_n,B_n and an independent A_0 give the real expansion

```
f(x) = sqrt(a_0) A_0
       + Σ_[one per ±n] sqrt(2a_n)
           [A_n cos(2 pi n·x/L) + B_n sin(2 pi n·x/L)].
```

Its covariance is a_0+2Σ a_n cos(2 pi n·(x-y)/L), exactly the parent's Fourier series before truncation. This specifies the variance factors and avoids accidentally counting conjugate modes twice. A numerical implementation must record its lattice cutoff, denominator approximation, floating-point type and random generator/version/seed. It must retain the same underlying coefficients when comparing spatial resolutions or nested spectral cutoffs.

A finite mode sum is a different field. Report omitted spectral variance and derivative-tail diagnostics; label numerical tail estimates as such unless rigorously enclosed. If using a deterministic truncated-sum normalization, record it and its effect on the covariance. Never rescale each realization by its empirical standard deviation or remove its empirical mean to manufacture agreement with the target model. That alters the ensemble, even when a height shift alone leaves unrestricted lifetimes unchanged. Do not substitute an unperiodized distance kernel or a Gaussian-filtered grid without deriving the resulting covariance.

Before computing persistence, independently check the generator's variance and covariance at zero and several nonzero physical lags against the **declared finite-cutoff analytic covariance**, with uncertainty estimated across independent fields. Compare that analytic finite-cutoff covariance separately with the intended infinite Fourier series; agreement with the truncated target alone does not certify the truncation error. Derive the reference covariance independently of the generator's weight assembly. A deliberately wrong spectral decay and a missing conjugate-mode variance factor must be detected. Barcode and affine-scaling controls alone cannot reveal every wrong ensemble.

Use h=L/n_grid with no duplicated endpoint at L. Choose a grid that resolves the selected Fourier modes; aliasing of a sampled high-frequency field must not be confused with geometric discretization error. Do not promise that a particular grid size is already in the asymptotic regime.

## 3. Use a standard periodic filtration and an independent small oracle

The [GUDHI PeriodicCubicalComplex reference](https://gudhi.inria.fr/python/latest/periodic_cubical_complex_ref.html) documents constructors from vertices or top-dimensional cells and distinguishes their pairing accessors. Pin the tested package version and use **one declared convention**, initially vertices with values -f and periodicity true on every axis. Keep array ordering and periodic gluing in the run metadata. Do not call a coface-pair accessor on a vertex-built complex or interchange the two discretizations while reporting them as the same experiment.

Before any random-field fit, compare finite H₀ intervals against an independently written descending union-find oracle on small periodic grids with known merge trees. On activation, the component with the higher original maximum survives; record the younger maximum's death at the merge level. Handle equal values by a declared deterministic batch/tie convention and compare positive-length intervals. The oracle and the library must agree on the same grid filtration, not merely on total Betti numbers. The successor implementation executes this requirement; see its tests and retained pilot results.

Separate controls should expose:

- **Missing periodic gluing:** a fixture whose connectivity across an edge changes a finite interval when that edge is incorrectly treated as a boundary.
- **Incorrect elder selection:** a small landscape with unequal maxima and two distinct merge levels; reverse the survivor rule and require a changed barcode.
- **Finite/essential confusion:** verify the connected periodic complex has one essential H₀ class and retain every positive finite interval, including its longest one.
- **Units and normalization:** adding a constant leaves lifetimes unchanged. Multiplying all heights by positive A scales every lifetime by A. If nu(ell)~c ell^(-1/3), change of variables gives nu_A(ell)=A^(-1)nu(ell/A), hence c_A=A^(-2/3)c. The reporting code must reproduce this relation, and must fail an intentionally missing volume factor.
- **Bin integration:** for a=t³,b=u³, the shape integral is exactly (3/2)(u²-t²). Use rational t,u to check the integration/reporting algebra independently of fitted data.

These are software/model controls. Their success does not validate the continuum theorem.

## 4. Separate three sources of uncertainty

| Source | Comparison to save | What it can establish |
|---|---|---|
| Monte Carlo variability | Per-realization bin counts and intervals based on independent fields | Sampling uncertainty for the chosen discretized ensemble |
| Spatial discretization | Coupled h,h/2,h/4 on the same Fourier realization and cutoff | Empirical sensitivity to grid refinement; no automatic continuum bound |
| Spectral truncation | Nested mode sets with shared coefficients at sufficiently resolved grids | Empirical sensitivity to the spectral cutoff; no automatic tail certificate |

A stability-based argument requires a stated sup-norm approximation bound between the actual functions/filtrations. If that bound is epsilon, endpoint matching can permit lifetime changes up to 2epsilon; near-diagonal unmatched features need special care. Do not turn a bound on total diagram distance into a direct bound for a sharply cut histogram without handling bin crossings. For a certified comparison, derive the relevant measure/bin inequalities separately.

Use a coarse pilot to estimate memory, runtime and count availability, then select a confirmation sample size without looking at its held-out outcome. Preserve every tested window and resolution. The smallest resolvable lifetime is not the smallest nonzero floating-point difference. Use refinement diagnostics to exclude visibly unstable bins and label the remaining window an empirical choice, not the theorem's proven cutoff.

## 5. Interpret disagreement and publish reproducible evidence

A stable mismatch first triggers checks of sign, volume, amplitude, covariance, essential bars, cutoff and bin integration. A histogram at one grid cannot distinguish failure of the continuum asymptotic from discretization or a pre-asymptotic regime. Conversely, a good fit does not prove pairing or justify removing a parent assumption. If uncertainty cannot be controlled enough to compare the coefficient, report **inconclusive at the tested resolutions** with the data intact.

A discrepancy between a typed contact count and actual elder bars requires a separately defined contact estimator. Candidate contacts, two additional saddle witnesses and finite elder bars must never be pooled into one count. Testing the refined planar failure law would require sampling the original determinant-weighted pinned law and its actual global selector; an unconditional grid histogram is not that experiment.

The reproducibility package should contain source commit, dependency lock, seeds and generator definition, dimensions/side/cutoffs/grids, per-realization counts, essential/zero-bar diagnostics, exact bin edges, all controls, resource usage, and a script regenerating the figure and table. Keep raw data in a versioned artifact with hashes and provenance; keep the small generator, estimator and fixtures in source control. This specification supplies no usable cutoff or simulation acceptance. Actual pilot data are linked above, with their own execution receipt and limitations.
