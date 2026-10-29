# Exact formal scope and alignment review

Scientific effect: NONE. The package offers 13 scalar companion statements from recovered GP-FOR-192. Its original archive has SHA256 a6440511fc259706457b18227734013966251105d538fdeabe633bb73f168631 (10,644 bytes). The two recovered source modules are preserved byte-identically under originals/*.lean.txt. The separately named AlgebraV2 and ProbabilityCompanionsV2 execution successors apply only the three documented compiler repairs in COMPATIBILITY.md; their thirteen theorem statements are unchanged. Neither version is a transcription of the full research theorem. Original Status.lean and metadata are deliberately not imported: labels for EC-012, EC-013 and Theorem-B-Jacobian did not supply those theorems.

| Targets | Exact coverage | Not established |
|---|---|---|
| ec005_fold_gap | Canonical polynomial foldPotential gap for every real s | Random-field normal-form construction or persistence pairing |
| ec008_factor_expansion | Scalar polynomial factor identity in real c,s | Construction/positivity of the actual conditional covariance matrix |
| ec010_generic, ec010_transverse | Polynomial identities for arbitrary real parameters | Validity/coverage of the underlying geometric charts |
| ec011_scalar_cancellation | Commutative real scalar cancellation | An ODE, matrix adjoint transport or Hessian symmetry |
| ec014_contact_power | Scalar inverse-power cancellation for r nonzero | Six-pin determinant, change of variables or Kac–Rice density |
| p02_lm008_r2_cancel | Quotient cancellation when r,cZ are nonzero | A probabilistic conditioning argument |
| p02_lm008_cross_multiplied, p02_lm008_quotient_bound | Implications from supplied numerator and lower-normalizer bounds and displayed sign assumptions | Conditional Cauchy–Schwarz, moment estimates, existence or uniformity of the lower normalizer |
| p02_lm009_bad_event_threshold, p02_lm009_power40_identity | Deterministic threshold and exponent algebra | Markov/moment hypotheses or tail-event measurability |
| p02_lm009_r4_le_r3, p02_lm009_palm_r4_to_r3 | Polynomial comparison on 0 <= r <= 1 and conditional bound with C >= 0 | Existence of a uniform C, all-small-r stochastic theorem or parent closure |

GP214's upper bound Z_r <= 15 r^2 cannot discharge the required lower bound cZ r^2 <= z. None of these targets is the SIDE24 coefficient enclosure, the elder-rule selection theorem, or the joint extremum/merging-saddle law. Some algebraic assumptions are stronger than necessary; preserving them is intentional. Reducing them requires a separately scoped change, not silent strengthening of a parent theorem.

## Review contract
The source manifest is an evidence sidecar, not another scientific register. A trusted successful workflow establishes kernel evidence only for these target declarations and their displayed hypotheses, relative to Lean's kernel/standard foundations/toolchain. Independent review must compare each Lean statement and definition to this scope note and the source actually cited. Review author, provider, family and agent must be explicit. Validate an authenticated record with `python formal/gate.py --alignment review.json`; then the existing controlling gate must decide whether its wider requirements are met. The validator does not authenticate a review merely because a JSON string names a reviewer.

The record contains disposition ACCEPTED, manifest_sha256, scope_sha256, exactly the targets, author/reviewer objects with provider/family/agent, and evidence repository/path/40-character commit/sha256. At publication, no independent alignment review is claimed. A fresh source, scope, toolchain or manifest digest stales the old review. Reject missing or ambiguous lineage, same provider/family/agent, stale digest, partial coverage, or non-ACCEPTED review.
