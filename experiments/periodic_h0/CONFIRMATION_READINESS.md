# When a coefficient comparison becomes interpretable

The [resolution study](results/refinement8/RESULTS.md) is an exploratory follow-up
on eight of the original fields. It is not a held-out test. At 1024², the three
filtrations have identical aggregate counts in the six bins from 0.004 to 0.256;
this agreement does not establish a continuum asymptotic window. Individual
paired differences, finite-sample error, and the last refinement remain relevant.

| Required input | Present evidence | Disposition |
|---|---|---|
| Declared ensemble and observable | Fixed finite Fourier bank, periodic superlevel ordinary H₀, all finite bars | Implemented for the discrete study |
| Spatial approximation theorem | Quadratic interpolation, stability and exact bin sandwich in [APPROXIMATION.md](APPROXIMATION.md) | Deterministic argument supplied |
| Certified finite-polynomial derivative input | [Exact rational certificate](FINITE_CERTIFICATE.md) for the stored rounded coefficients of eight fields | Derivative majorants and spatial interpolation budget evaluated; this defines a finite polynomial, not an ideal Gaussian realization |
| Certified numeric nodal input | [New seed34000 / 128² snapshot](NODAL_CERTIFICATE.md), every node independently enclosed | Nodal error and abstract finite-field diagram bound supplied for this snapshot only; historical samples and computed barcode endpoints remain uncertified |
| Infinite-field truncation | Variance diagnostics through mode64 in pilot32 | No uniform realized or probabilistic tail enclosure |
| Lifetime range | Larger bins exhibit numerical stability; shortest bins still drift | No certified finite-lifetime range for the leading law |
| Remainder | Source-bound qualitative O(1) theorem with its hypotheses | No applicable numerical remainder constant or cutoff supplied here |
| Sampling precision | Eight coupled fields, with per-field counts and paired differences | Descriptive precision only; no simultaneous confidence claim |
| Independent scientific review | Source-bound nonauthor model review of this delivery | Human review and retained organizational-independence requirements remain separate |

The earlier C35 finite-field diagnostic at the finest grid gave `epsilon` between about
0.00882 and 0.00972 from the simple Fourier Hessian majorant. Even if these
floating calculations were rigorously enclosed, the clean bin-count upper bound
would require `a > 2 epsilon`, and its lower bound may be zero when the contracted
bin is empty. Agreement of rounded bin counts cannot replace that requirement.
The new finite-polynomial certificate encloses the derivative inputs and uses
the componentwise alternative to reduce the spatial budget. Its
[eight-field results](results/certificate8/RESULTS.md) retain a null nodal error
and a null total diagram error. They cannot turn numerical agreement into a
continuum certificate or a usable asymptotic window. The separate new
[nodal snapshot](results/nodal1/RESULTS.md) supplies eta+B for its exact finite
polynomial and abstract supplied-sample filtrations; the coarse-grid spatial
term still prevents use in the short-lifetime window.

A later held-out design should be frozen **after** these analytic inputs are
available and **before** its fields are observed. Its specification must record:

1. The exact model, finite-mode/infinite-field relation, grid/filtration, and
   source identities; distinct seed sets from pilot and refinement/calibration.
2. A lifetime interval justified by the approximation and remainder bounds,
   with exact bin endpoints and no later selection by goodness of fit.
3. A field-level sample-size rationale and an appropriate simultaneous error
   procedure; the bar count is not the number of independent observations.
4. The null bin masses including the permitted asymptotic remainder and all
   approximation terms, plus a predeclared discrepancy/indeterminate decision.
5. A policy for empty bins, essential bars, boundary overlaps and failed
   numerical certificates; retain all outcomes and never rebind failed receipts.

No held-out seeds have been consumed by this delivery. Assigning a sample size
or a pass threshold now would conceal missing analytic inputs. The next useful
calculation extends the new nodal certificate to useful finer sample snapshots
and independently checks the barcode computation, followed by a justified model,
tail and remainder budget.
The finite-polynomial derivative input is now available. The source-bound manuscript
and literature work can advance in parallel without waiting for that budget.
