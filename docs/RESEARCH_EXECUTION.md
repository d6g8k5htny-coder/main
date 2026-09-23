# Reproducible execution and numerical certificates

This layer implements the useful parts of the assessed Gemini and DeepSeek
proposals: one verification entry point, exact numerical certificate replay,
structured diagnostics and a derived research checkpoint. Dylan Roy authorized
implementation in the current task on 2026-09-19 America/Chicago. Authorization
permits the work; mathematical acceptance still depends on the evidence.

For a source-bound work inventory and per-item execution evidence, use the
[closure pipeline](CLOSURE_PIPELINE.md):

```sh
python tools/closure_pipeline.py run --output-dir /absolute/path/to/new-closure-run
```

It executes all supported checks once and preserves the original criteria for
items that need more mathematics, source reconciliation or independent review.

Run the repository's explicit CI checks and negative controls with the
[verification runner](VERIFICATION_RUNNER.md). It uses the existing checkers
and manifest coverage rather than introducing a competing input manifest.
Each invocation writes a separate report and logs to a new external directory.
The report identifies the actual inputs, environment and outcomes. An unfinished
run, a skipped required check or a malformed input is not a successful run.

```sh
python tools/run_checks.py --output-dir /absolute/path/to/new-run-directory
```

The [RN certificate format](RN_CERTIFICATES.md) supports explicitly scoped
Gaussian determinant-moment bounds. A producer supplies data; replay checks the
coefficient derivation and the complete mark-domain subdivision using exact
rational intervals. A checked polynomial bound, source-byte identity and
applicability to a specified context are distinct results. Their combination
still does not identify the actual RN field law or complete its spatial cover.

```sh
python tools/rn_certificate.py check-candidates
```

The [SIDE24 adapter](RN_SIDE24.md) supplies the missing field construction for
one specified point: normalized image sums with infinite tails, nine-pin
[interval conditioning](RN_CONDITIONING.md), then three portable witnesses for
the corrected determinant powers `4,4,2`. Its candidate is reproduced from the
source kernel and exact pin values on every check. The resulting conditional
determinant factor covers the whole mark interval.

```sh
python tools/rn_side24_check.py
```

The [density/window extension](RN_SIDE24_DENSITY.md) computes the six-pin
gradient density and bounds the full height-window probability after also
conditioning the gradient to zero. It composes these factors with the checked
moment upper and an explicitly imported, hash-bound H3 floor. Its exact
admission margins are diagnostic fields, not sampled counterexamples. The H3
premise is not reproved and the complete spatial cover remains open.

```sh
python tools/rn_side24_density_check.py
```

The [local rectangle extension](RN_SIDE24_CELL.md) replays four spatial-cell
witness sets and the complete partition/area sum. The [parallel candidates](PARALLEL_MATH.md)
reconstruct the fixed-r H3 floor, full LPW tails with a quadratic covariance
modulus, and one normalized H5 jet over a named band. Both commands are required
by CI and the complete closure runner.

```sh
python tools/rn_side24_spatial_check.py
python tools/parallel_math_check.py
```

The [twelve-project continuation](TWELVE_PROJECT_MATH.md) replays the exact
frozen candidate archive in a fresh external directory. The portable wrapper
checks its allowlist, every input identity and the current repository
dependencies before executing the twelve specified mathematical checks.
Fresh reports must match the original candidate bytes. Historical source
scripts remain inert unless explicitly named as a reviewed checker.

```sh
python tools/twelve_project_check.py
```

This execution closes twelve reconstruction tasks at their stated scopes.
Actual weighted-field, event-identification, all-angle and continuum
hypotheses remain separately listed in the mathematical continuation.

[Recurring continuation](RESEARCH_AUTOMATION.md) schedules one bounded private
delivery per run, with source/claim coordination and one complete verification
of the final stable tree. It leaves unchanged or non-actionable state quiet.

The [research frontier](RESEARCH_FRONTIER.md) derives dependency paths and
recorded failures/non-executions from named repository inputs. It preserves
frozen and register-note status layers and records exactly which source bytes
were read. It is a disposable view of those records, not a second writable
claim or refusal register. A hash establishes the selected bytes' identity;
the checkpoint is unsigned and its observation time is not externally attested.

```sh
python tools/research_frontier.py self-check
python tools/research_frontier.py snapshot --output /absolute/path/to/new-frontier.json
python tools/research_frontier.py check /absolute/path/to/new-frontier.json
```

The [LPW amplitude companion](LPW_AMPLITUDE.md) makes the corrected parent
Gaussian moments and conditional radius arithmetic executable. The [recovered
Lean package](../research/formal/candidates/LEAN_RECOVERY_VERIFICATION_20260920_v1.md)
preserves the original failed build and statement-preserving repair, exact locks,
13 compiled theorem declarations and rejection controls. Its recorded build
evidence is attached to OQ-013; Python CI checks custody, while Lean replay is an
explicit separate command.

These mechanisms separate three questions: whether recorded bytes agree,
whether a numerical certificate verifies under its mathematical assumptions,
and whether a governing decision admits its use for a particular claim.
The existing Drive registers remain authoritative for the last question.
Receipts and CI runs carry no automatic promotion or organizational independence.

The implementation builds on existing error-preservation controls, exact
soundness witnesses, law fingerprints and different Gaussian-moment checks.
It does not assert that those component techniques are historically new.
The preserved proposal assessments explain the choices and alternatives:
[Gemini](context/GEMINI_PIPELINE_ASSESSMENT_20260920.md) and
[DeepSeek](context/DEEPSEEK_RESEARCH_ARCHITECTURE_ASSESSMENT_20260920.md).
They describe the assessment-stage repository before these additions.

Unsupported global guarantees remain excluded: numeric confidence weights do
not determine mathematical validity; a structural hash does not decide general
mathematical equivalence; a locally signed or hashed snapshot does not prove
what someone believed at a trusted historical time. No frozen evidence is moved
or rewritten, and no historical claim is downgraded because the new certificate
format does not yet support it.

The [whole-band H3 and RN N6 successor](H3_RN_N6.md) has one portable route.
It replays five mathematical jobs in normal and optimized Python and compares
34 outputs to their frozen exact bytes. The original stronger fixed-radius
H3 floor remains the RN premise.

```sh
python tools/h3_rn_n6_check.py
```
