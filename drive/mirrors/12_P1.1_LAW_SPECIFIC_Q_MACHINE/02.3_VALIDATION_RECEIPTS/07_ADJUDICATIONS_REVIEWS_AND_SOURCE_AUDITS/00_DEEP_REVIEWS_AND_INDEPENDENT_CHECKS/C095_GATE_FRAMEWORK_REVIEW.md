# C095 Review of Gate Framework Master v1.0

**Disposition:** `SUPERSEDED-BY-v1.1-NOT-OVERWRITTEN`

The restriction-drop meta-rule, proof-object discipline, and first 11 semantic gates are retained. The file is not executable as written because three C094-mandated numerical gates are absent and two new gates lack the metadata needed for mechanical checking.

## Findings

| ID | Severity | Topic | Finding | Repair |
|---|---|---|---|---|
| GF-R01 | MATERIAL | Grade/status conflation | Primitive, Proven, Open, and Killed are not points on one total warrant ladder. Primitive is a role; Open/Killed are statuses. | Split assumption role, warrant grade, claim status, and reasoning mode into separate fields. |
| GF-R02 | MATERIAL | Dependency rule ambiguity | The constitutional rule first says only four grades may be dependencies, then allows explicit Open/Conjecture dependencies. | Use a promotion matrix: unresolved dependencies may appear only as explicit hypotheses; a proved implication is Proven-Modulo, otherwise the dependent claim is capped at Plausible. |
| GF-R03 | MATERIAL | Composition witness incompleteness | UNION_BOUND is omitted from the closed witness list even though the text requires a witness for unions. A union bound does not require disjointness or independence. | Add UNION_BOUND, MONOTONE_ASSEMBLY, and INTERVAL_ARITHMETIC; make witness requirements operation-specific. |
| GF-R04 | MATERIAL | Missing C094 numerical gates | The file omits ASSEMBLY, DOMAIN-INFIMUM, and BAND-PROVENANCE, which were made mandatory by frozen C094 failures. | Promote them as gates 14–16 in v1.1. |
| GF-R05 | MATERIAL | Regime field is one-sided | A single regime-of-validity field records establishment but does not explicitly record the deployment regime being checked. | F-1 becomes an establishment/deployment pair with a named bridge for every mismatch. |
| GF-R06 | MATERIAL | COMMON-MODE schema support | Gate 12 cannot be mechanical without instrument IDs, shared components, error-channel declarations, and an independence or external-crosscheck certificate. | Add verification-mode and common-mode-certificate fields. |
| GF-R07 | MATERIAL | HEURISTIC-BRIDGE schema support | Gate 13 cannot be mechanical because the proof object does not carry reasoning mode, load-bearing status, or bridge ID. | Add reasoning-mode, load-bearing, and bridge fields. |
| GF-R08 | MATERIAL | Retrodiction evidence completeness | Appendix A claims two-case retrodiction for COMMON-MODE and HEURISTIC-BRIDGE but gives no frozen failure IDs or hashes. | Retain both gates provisionally; cap framework-level promotion until a hashed retrodiction table is supplied. |
| GF-R09 | MATERIAL | PROVISIONAL semantics | The file says PROVISIONAL caps grade but does not define the cap uniformly. | A provisional gate blocks promotion to Proven/Certified; the claim may remain Derived or Plausible according to the failing gate's explicit cap. |
| GF-R10 | IMPORTANT | Coverage mechanics | Coverage is not checkable without a declared failure universe, chart family, and uncovered-mass certificate. | Add a coverage certificate schema. |
| GF-R11 | IMPORTANT | Band construction | Uncertainty side is present, but method, sample size, confidence, multiplicity, algorithmic tolerance, and seed are not. | Add F-3 uncertainty provenance and Gate 16. |
| GF-R12 | IMPORTANT | Assembly arithmetic | COMPOSITION and POLARITY do not recompute a displayed numerical assembly or enforce conservative rounding. | Add F-2 assembly expression and Gate 14. |
| GF-R13 | IMPORTANT | Uniform-domain claims | DOMAIN and ENDPOINT do not prove an infimum/supremum between sampled rungs. | Add a full-domain extremum certificate and Gate 15. |
| GF-R14 | IMPORTANT | Threshold types | The statement that every threshold must be Derived excludes legitimate normative or user-specified decision thresholds. | Separate theorem/certificate thresholds from policy thresholds; policy thresholds are Primitive-Normative and cannot be cited as mathematical facts. |
| GF-R15 | IMPORTANT | Hash semantics | A content hash is only meaningful if canonical serialization and all load-bearing metadata are included. | Hash canonical proof-object serialization plus dependency hashes. |
| GF-R16 | IMPORTANT | Schema completeness | The file says a missing field is a gate failure but does not define a schema preflight. | Add a mandatory unnumbered SCHEMA preflight before Gate 1. |

## Accepted core

- restriction-drop as organizing lens, explicitly not a theorem
- atomic proof objects and dependency graphs
- supersede-never-overwrite
- freeze-before-execute
- preserve every failure
- retrodiction instead of forward-selected scoring
- survivorship limitation
- separation of exact and deployment checks
- weakest-link discipline
- gates DOMAIN through COVERAGE as core semantic gates

## v1.1 structure

The successor specification contains:

- one mandatory `SCHEMA` preflight;
- 16 numbered gates;
- five schema field groups;
- explicit status/warrant/reasoning-mode separation;
- operation-specific composition witnesses;
- conservative numerical assembly and full-domain certificate gates;
- provisional evidence status for COMMON-MODE and HEURISTIC-BRIDGE until their claimed retrodiction cases are supplied by ID and hash.