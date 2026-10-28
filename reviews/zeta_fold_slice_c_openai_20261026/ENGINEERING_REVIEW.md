# Separate local engineering review

Actual reviewer: OpenAI / Codex subagent `/root/zeta_evidence_review`, a fresh
source-exposed review invocation, same provider/family as this evidence author.
Organizational-independence credit 0. It did not execute the numerical replay,
write this evidence, repeat C8/R1, or perform qualifying scientific review.
This is the reviewer's report, retained by the foreground author. No human
review or independent organizational control is claimed.

**Verdict: PASS for the scoped harness and candid evidence record, pending
final manifest identity verification.** No critical defect or remaining
implementation blocker found.

Reviewed execution source `0a7e5145`, frozen review `8f1d846d`, evidence base
`b7e012c2`. Reviewed core SHA-256 prefixes: `build_evidence.py` `aa064dc6`,
`run_replay.py` `3193e8e1`, `fresh_check.py` `c4e8b558`, README `06296cc6`.
The final manifest binds their complete identities.

## Findings and dispositions

- **Important, ZF-OAI-CUSTODY-01 remains unexplained.** Retained 1511B stdout
  is a prefix of the1816B output recorded by the capture-time hash. The separate
  recovery matches that hash exactly. Strict validation rejects the damaged
  capture; explicit recovery validates both identities, reports one incident,
  rejects changed/unused recovery entries and preserves the damaged file.
  This supports recovered content, not an intact-original-capture claim or an
  explanation. Handling and reporting are adequate.
- **Important validator gap, resolved.** Initial validation accepted wrong
  interpreter/cwd and did not enforce recorded version pins. Current validation
  checks Python/NumPy/mpmath, backend/thread pins, each interpreter/cwd, the
 59-command multiset, job exits and both parent-lane exits. Targeted negative
  controls refuse these changes.
- **Minor historical provenance limit, retained accurately.** The frozen driver
  records stderr size, not capture-time SHA-256. Final package hashes bind retained
  contents but cannot establish a capture-time digest. README/SETUP say so.
- **Minor replay/metadata limits, disclosed.** Executed wrapper records pins
  rather than enforcing them and does not inventory ignored raw files. Packaging
  enforces recorded pins; actual refit lines and generator receipts substantiate
  this run's regeneration. Fresh overall start/wall metadata was not captured
  and remains unknown. No rerun manufactures missing metadata.

## Independent checks actually performed

- Frozen driver/comparator equal #451 byte-for-byte.
- Actual59-command multiset, all job exits and both lane exits validate.
- All66 reproduced hashes/classes recompute:49/15/2 against original,
  49/17/0 against frozen same-provider rerun. Every nonidentical comparison has
  its complete diff.
- All51 source-table entries and9 raw hashes match execution files.
- Fresh CSV contains384 finite points,128 per planned window; maximum errors
  recompute exactly. All six grid hashes/counts verify:66/51/37 at both steps.
- Four current bookkeeping test methods independently pass. Retained normal
  and optimized logs agree.
- Executed wrapper/fresh script identities remain unchanged. Recovery involved
  no numerical re-execution.

The README preserves both residual-trail differences, reports c3435's unmet
step-stop criterion despite rc0, and avoids all-C or acceptance claims.

## Declined scopes

Mathematical correctness; reference certification; uniform error bounds;
zero completeness/RH; convergence or quadrature certification;
persistence/exponent hypotheses; historical trail causes; qualifying scientific
review; prior C8/R1 work; integration readiness; hosted CI. The final manifest
had not been generated when this verdict was returned. Any later identity-only
supplement is recorded separately on the contribution PR without rewriting this
review's chronology.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
