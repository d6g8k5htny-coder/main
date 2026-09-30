# Following short bars through finer grids

**[Read the refinement result and figure](results/refinement8/RESULTS.md).** Eight original fields are now followed through 1024², with two triangulated-filtration controls. The shortest bins still change; some larger bins stabilize numerically. The [deterministic approximation note](APPROXIMATION.md) supplies a quadratic bound conditional on certified inputs, and [confirmation readiness](CONFIRMATION_READINESS.md) identifies the remaining gates.

The **[first pilot and figure](results/pilot32/RESULTS.md)** remain available. The first experiment is inconclusive at these resolutions: its shortest bins change substantially under grid refinement. All fields, bins and configurations are retained. There is no fitted exponent or selected confirmation window.

The [finite-polynomial certificate](FINITE_CERTIFICATE.md) now evaluates exact
derivative bounds for the rounded Fourier coefficients of the eight refinement
fields. Its [results](results/certificate8/RESULTS.md) provide a certified spatial
interpolation budget. Nodal FFT error and the infinite-field tail remain open, so
this is one input to a diagram-error certificate, not a completed one.

This small experiment samples a truncated version of the periodized Gaussian field on a side-24 square torus and measures finite ordinary superlevel H₀ bars using GUDHI. It accompanies the [manuscript draft](../../docs/research-translation/20260930/MANUSCRIPT.md), [statement crosswalk](../../docs/research-translation/20260930/CROSSWALK.md) and [original protocol](../../docs/research-translation/20260930/EXPERIMENT.md). It is numerical evidence about the declared discretized ensemble, not a continuum certificate or a new mathematical acceptance.

## Reproduce the pilot

From the repository root, with Python 3.12 and a CPU:

```sh
python3.12 -m venv /tmp/periodic-h0-env
/tmp/periodic-h0-env/bin/python -m pip install -r experiments/periodic_h0/requirements.txt
/tmp/periodic-h0-env/bin/python -B -m unittest discover -s experiments/periodic_h0 -p 'test_*.py' -v
/tmp/periodic-h0-env/bin/python -B experiments/periodic_h0/run_pilot.py --output /tmp/periodic-h0-replay
```

Use a new output directory for each run; the runner refuses to overwrite one. The fixed pilot has 32 independent seeds, grids 64²/128²/256² and square mode cutoffs 12/24, giving 192 **coupled** evaluations. Covariance calibration uses a separate 1,024-field sample. Configuration and all bin edges are in [pilot_config.json](pilot_config.json), frozen before inspecting the output. Execution took roughly ten seconds on the recorded host; runtime and platform are in [RUN.json](results/pilot32/RUN.json). No GPU or downloaded research data is required after dependency installation.

The [committed observations](results/pilot32/observations.json) are a small, complete pilot fixture. Compare them to a fresh run:

```sh
cmp experiments/periodic_h0/results/pilot32/observations.json /tmp/periodic-h0-replay/observations.json
/tmp/periodic-h0-env/bin/python -B experiments/periodic_h0/run_pilot.py --verify /tmp/periodic-h0-replay/observations.json
```

Exact replay is tested in the recorded environment. Different FFT implementations or platforms can change floating-point endpoints, ties or bin membership; a byte mismatch must be investigated, not hidden by rebinding the receipt. `--verify` checks retained intervals/counts, derived reports, covariance aggregates and targets, spectral diagnostics and control replay. It does **not** independently regenerate the random fields or establish the authenticity of arbitrary edited observations. The fresh simulation above is the separate execution check. `RUN.json` binds its actual executed sources and observation bytes by SHA-256.

To regenerate the fixed pilot's figure and prose report:

```sh
/tmp/periodic-h0-env/bin/python -m pip install -r experiments/periodic_h0/requirements-plot.txt
/tmp/periodic-h0-env/bin/python -B experiments/periodic_h0/render_report.py /tmp/periodic-h0-replay
```

The renderer accepts only the frozen pilot32 configuration and exact observation hash because its interpretation is specific to those outcomes. A changed result needs a fresh interpretation. Larger follow-up datasets should live in versioned, hashed artifacts; do not accumulate production runs in git.

## Reproduce the resolution comparison

The [frozen design](REFINEMENT_PROTOCOL.md) uses eight existing fields, four
nested grids through 1024², and three filtrations (96 coupled evaluations).
It took about 80 seconds on the recorded CPU host. It is exploratory; all
fields and all bins are retained, with no slope fit or selected test window.

```sh
/tmp/periodic-h0-env/bin/python -B experiments/periodic_h0/run_refinement.py --output /tmp/periodic-h0-refinement
/tmp/periodic-h0-env/bin/python -B experiments/periodic_h0/run_refinement.py --verify /tmp/periodic-h0-refinement/observations.json
/tmp/periodic-h0-env/bin/python -B experiments/periodic_h0/render_refinement.py /tmp/periodic-h0-refinement
```

The renderer requires the exact published observation hash; the verifier
reconstructs derived summaries, matching distances and diagnostics, allowing
only tiny roundoff in derived floats. It does not prove field authenticity or
provide a continuum error enclosure. The small [complete fixture](results/refinement8/observations.json)
is retained for audit; larger future runs belong in hashed artifacts.

New execution receipts use stable role keys, so a custom JSON configuration
named `run_pilot.py` or `requirements.txt` cannot hide another source. Historical
pilot32 receipts remain unchanged and refer to their original source versions.

## What is implemented

| Part | Meaning |
|---|---|
| Gaussian generator | Independent PCG64 normal coefficients, a conjugate Fourier pair for each ±mode, and the declared deterministic denominator through mode64; no empirical recentering or rescaling |
| Coupling | One coefficient bank per seed, reused for all grids and nested cutoffs in that run |
| Filtration | `PeriodicCubicalComplex(vertices=-f, periodic_dimensions=[True, True])`; x/y array axes and row-major oracle indexing, no duplicated torus endpoint |
| Bars | Every strictly positive finite H₀ interval; essential birth and zero-length pairs recorded separately using `min_persistence=-1` |
| Statistical unit | A whole independent field, including zero-count fields; spatial volume is 24², not the number of bars |
| Comparison | Integrated leading bin mass, with all bins half open; standard errors and paired refinement differences are descriptive |
| Oracle and controls | Independent descending union-find; wrong periodicity, elder rule, spectrum, conjugate variance and volume controls; affine heights and exact rational bin-shape check |
| Retained output | Every finite interval, essential/zero count, realization/bin count, calibration sample and derived summary |

[`experiment.py`](experiment.py) defines the model and estimator; [`controls.py`](controls.py) exercises meaningful alternatives; [`run_pilot.py`](run_pilot.py) records the experiment; [`render_report.py`](render_report.py) presents it. The oracle follows axis-edge connectivity, which determines H₀ of this vertex cubical filtration. It is not an independent continuum persistence computation. Deterministic tie order may choose different representatives while leaving positive bar intervals unchanged.

Package conventions were checked against the [GUDHI periodic cubical documentation](https://gudhi.inria.fr/python/latest/periodic_cubical_complex_ref.html) and [NumPy inverse FFT normalization](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html). Dependency versions used here are pinned; this optional environment does not change the repository's standard-library verification lane.

## What remains unresolved

The deterministic spatial interpolation/filtration bound and exact derivative
majorants for the specified rounded finite polynomials are supplied. An evaluated
certified nodal error for the FFT samples, the relation to ideal Gaussian
coefficients, an enclosed infinite-field spectral tail, a justified finite-lifetime
asymptotic window and numerical remainder constants remain open. Marginal
one-standard-error bars do not include these errors. The coefficient is a float64
comparison value from a separate exact enclosure. This experiment does not
estimate typed contacts, weighted-Palm events or candidate-minus-elder defects,
and it supplies no d=3 or higher-homology result. Delegated AI review, personal
owner reading and independent human review retain their separate meanings.
