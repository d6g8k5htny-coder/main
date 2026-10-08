# Fifty candidate laws and exact exponent falsifiers

This suite implements the falsification foundation of the
[persistence universality program](../../docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md).
It uses Python's standard library and the existing exact integer periodic-H0
algorithm. The [protocol](PROTOCOL.md) separates exploratory execution from a
genuinely blinded, source-admitted continuum comparison.

`models.json` declares fifty distinct planar finite Fourier law candidates:
ten symmetric positive spectral mechanisms times five circular coefficient
laws. Rotations, amplitude rescaling and different seeds are not counted as new
models. Gaussian and non-Gaussian coefficient laws can share covariance while
having different higher distributions. Every model is a candidate; running it
does not establish the contact/Palm/regularity/global-pairing theorem inputs.

The [finite Gaussian contact proof](finite_gaussian_contact/PROOF.md) now supplies
the named contact covariance, small-separation regression, compact-mark
normalizer and unnormalized all-mark envelope for the ten ideal Gaussian
profiles at cutoffs 2 and 3. It also proves the original two-site pin covariance
bound and gives a local jet coefficient functional. Its finite-r Hessian bridge
and deterministic degeneracy falsifier identify a physical-chart hypothesis
that exact pins alone cannot supply. These are conventional analytic inputs;
the [review record](finite_gaussian_contact/REVIEW.md) preserves their exact scope.
Marked counting, actual bar selection, further residual ranks, critical formal
alignment and numerical coupling remain admission obligations. The sampler and
catalogue admissibility labels retain their original meaning.

`models.py` defines the implemented finite-cutoff normalized sampler. Only
cutoffs 2 and 3 are supported: at cutoff 1, Gaussian and separable spectra
coincide, reducing the fifty declared pairs to forty-five distinct laws.
Accepted coefficient metadata must match the implemented construction, variance,
symmetry, fourth moment and prospective ideal-input contract. Alternate catalogs
and directly supplied definition catalogs undergo the same validation.
Accepted spectrum metadata must also match its entire implemented formula and
mechanism mapping; false mechanisms and extra status labels are refused.
The catalog's root/domain/default/normalization/scope declaration also requires
its complete supported schema, values and strict types.
`run_pilot.py` generates explicitly exploratory fields, quantizes their nodes,
computes exact ordinary H0 on that integer grid and independently checks
connectivity. The field evaluation and sampling law remain floating-point/PRNG
approximations. Exact discrete endpoints are not ideal-field certificates.
The pilot retains all model/sample rows and failures, uses declared units and
does not estimate a slope or compare against reference coefficient digits.
The CLI writes `observations.json` before its final record self-check, then
writes `validation.json` with the observation digest and PASS or FAIL. A saved
observation file alone is not a verification pass. A final verifier failure
preserves the generated rows and original error with a failing exit status.
Invalid execution plans are rejected before reserving the output directory.
The verification CLI reports structural consistency separately from retained
computation failures and returns a failing status if any planned field failed.
CI also checks the committed observation bytes against their sibling validation
receipt and current source, using `verify_published.py`. This refuses stale source
hashes, digest mismatches, false validation records and retained field failures.
Publication also requires the complete frozen default configuration: side24,
grid16, cutoff3, two fields per law, scale65536 and the canonical fixed bin plan.
Other supported plans remain available to the raw retained-record verifier.
Undeclared fields in the observation/environment/configuration/row mappings
are refused, including unsupported scientific-status labels.
It does not authenticate supplied grids as generator outputs.
Failed rows are checked against the atomic stage they reached: completed floats,
quantization and barcodes replay, while a failed connectivity attempt remains
unverified. A later bin-count failure requires the already completed independent
connectivity check. Corrupt raw failures remain saved with final FAIL/exit2.
The audit does not authenticate recorded exceptions or their causal history.
Publication requires the declared sampler mode and refuses even successful
records labelled as injected arithmetic/test inputs. Execution-environment
declarations require nonempty canonical printable strings and a canonical Python
version format. These validate declarations; they do not authenticate the host,
runtime, sampler implementation or supplied grids.
Retained law definitions and derived summaries require recursively exact scalar
and container types, so equal-valued Booleans or floats cannot impersonate the
integer declarations and counts.

The rational mechanism controls are ready independently:

```sh
python3 -B -S experiments/universality/fold_controls.py
python3 -B -S -m unittest discover -s experiments/universality -p 'test_*.py' -v
python3 -B -O -S -m unittest discover -s experiments/universality -p 'test_*.py' -v
```

The controls derive both density and cumulative exponents and exact bin masses.
They show why the same cubic gap can give density powers -2/3, -1/3, 0 or +1
under different radial sampling laws. Their scope is an exact toy pushforward,
not a stationary random-field counterexample or a formal kernel proof.

For the exploratory runner's fixed defaults and output options, use:

```sh
python3 -B -S experiments/universality/run_pilot.py --help
python3 -B -S experiments/universality/verify_published.py experiments/universality/results/exploratory50
```

No held-out seeds have been consumed by this suite. Before a blind experiment,
supply each model's actual law admission, coefficient/remainder/window,
implementation coupling, numerical/count transfer, field-level precision and
independent holdout custody described in the protocol. Higher dimensions,
higher homology and geometry changes are subsequent adapters, with their own
proof and computational contracts.
