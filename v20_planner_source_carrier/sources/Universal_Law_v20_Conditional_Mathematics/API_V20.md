# Exact conditioning lab API and evidence protocol

All inputs are JSON data. No request can supply executable Python/Lean, imports,
commands, checker paths, mathematical approval, or formal success.

## Request

`gaussian_examples.example_requests()` supplies fresh named request objects.
`gaussian_moments.make_certificate(request)` returns an exact certificate or
raises ValueError for invalid/resource-exhausting requests.

The closed request schema is `ul-v20-gaussian-request-1`, with exactly:

- `variables`: ordered unique ASCII identifiers for the full joint model.
- `mean`, `covariance`: full rational vector and symmetric PSD matrix.
- `observed`: `{indices: [...], values: [...]}`. The observed block must be
  positive definite. Target coordinates are unobserved variables in original
  order; they may have singular or deterministic conditional covariance.
- `polynomial`: `{terms: [{coefficient: [n,d], powers: [...]}, ...]}` over target
  coordinates, unique sorted terms and nonzero coefficients.
- `orders`: sorted distinct integers1–4. These are raw moments E[P^k]; only even
  orders automatically equal absolute moments.
- `context`: `{purpose, source_bindings: [{id,sha256}], assumptions: [...]}`.
  Context is provenance data, not source authentication or theorem applicability.

Every rational is `[integer numerator, positive integer denominator]` in lowest
terms. Booleans, floats and fractional aliases are rejected. The exact Schur law
retains nonzero means; no unrecorded independence or pseudoinverse is introduced.

Caps:64KiB request,2MiB certificate,16 joint coordinates,10 observed,6 target,
32 polynomial terms, integrated total degree8,128-bit input rational parts.
Intermediate rationals cap at8192bits, arithmetic at1,000,000 operations and
polynomial support at4096 monomials. Container depth/nodes/serialized bytes are
checked before serialization. These are rejection limits, not a guarantee that
every combination at all outer caps will be accepted.

## Independent check and request binding

`gaussian_checker.verify_certificate(certificate, expected_request=request)`
returns a structured report with `valid`, `state`, `errors`, `raw_moments` and
scope flags. The expected request is mandatory for a valid verdict. A self-hash
cannot identify which question the caller intended. A legitimately recomputed
certificate for another law is checked against that law's expected request,
never silently substituted for the original question.

The generator uses shifted sparse polynomials and centered Wick pairings. The
checker independently rederives conditioning and uses raw Gaussian integration
by parts. No generator import is used by the checker. Neither algorithm proves
Gaussian probability theory inside a formal kernel or establishes actual-field
applicability from finite rational matrices.

## Bounded CLI worker and receipts

`conditioning_lab.run_case(name, output_directory)` accepts one registered example
name, executes the fixed child and returns a fresh receipt path. `verify_run(path,
expected_request=request)` checks source/input/artifact bindings and performs a
new bounded independent verification. The CLI exposes `--case` and `--output`.

The child has CPU10s/address512MiB/file8MiB/fd64 limits, and its parent enforces
wall20s plus retained output caps. Failures preserve request/stdout/stderr and
attempt metadata, including timeout, spawn and malformed-envelope failures.
Oversized retained streams are explicitly truncated with observed byte counts.
A failed job never creates a successful receipt.

Receipt schema `ul-v20-exact-worker-receipt-1` binds job/obligation IDs, explicit
trigger, full request fingerprint, versioned source hashes, environment,
resource policy, stdout/stderr/artifact hashes, exit result and scope. Replay
rejects failed exit metadata, invalid durations, changed sources/inputs/artifacts,
and retained verification/stream contradictions. A valid local receipt remains
execution provenance, not a signed attestation of its past execution.

Successful state is `exact_diagnostic_checked`; formal status is
`not_kernel_formalized`, `actual_field_applicable=false`, and the full theorem
certificate list stays empty. Analytic candidates and their reviewed assumptions
are separate. The additive CLI does not install new canonical graph records or
replace the foundation's durable conductor. The preserved dashboard/queue runs
against its own writable runtime copy.
