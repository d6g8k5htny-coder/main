# SI01 / SI02 bounded migration convergence proposal

This is an isolated post-audit-cut engineering proposal. It does not change the
historical audit inputs, publish a scientific verdict, or authorize integration.
The two issues are independent: correcting a graph label is not checker evidence,
and a checker refusal is not a proof or a deployed status-transition system.

## Exact base, reference, and authorship

- Migration base: `534f4fce08be919cb969ca87963c5210148497a1`, branch
  `claude/drive-audit-github-migration-rrglpp`, repository
  `d6g8k5htny-coder/main`
- Corrected reference: hardening `a01c72f1937875b94bfd78e183a6dbbdb2b04da4`,
  `claims/graph.json`, `tools/claims_check.py`, and
  `docs/FAIL_CLOSED_CLAIM_AUDIT_REPAIRS_20260925.md`
- Previous source edits identify Anthropic session
  `01Cz7WZybv8znP64SpPj6sWY` in commits `b1335def2615888463e2d134294ef451895b8d0d`
  and `b011cddfa87dee95d3357f5d912ebd38693ed36a`; Git account metadata is not
  evidence of current active ownership
- This proposal: OpenAI/Codex, source-exposed implementation. Nonauthor exact-head
  engineering review is required before integration. Same-provider technical
  review earns zero organizational-independence credit

## SI01: preserve the conditional result and refuse the historical over-grade

The frozen D1-v2.2(1) body calls itself a certified rung and explicitly depends on
RN-UNIF, which is NOT_CLOSED/OPEN. The migration checker did not apply its
unconditional-grade rule to `CERTIFIED_RUNG`. It accepted open, unknown, missing,
and conflicting status columns under that grade.

The selected hardening correction is ported without replacing the whole graph:
operational `CONDITIONAL`, original `source_grade_verbatim: CERTIFIED_RUNG`, and
`HOLD_WITH_DOMAIN`. The statement, source identity, and dependency list remain
unchanged. The reference note preserves Q-RN5-MOMENT-003's annulus-number defect
and RN3's AMEND review with zero credit; neither number is newly certified.

`FW-RUNG-OPEN-PREMISE` is the hardening rule: every transitive premise must have
one of CLOSED, DISCHARGED, PROMOTED, SATISFIED, CERTIFIED in **both** status
columns. Missing, unknown and malformed values refuse. An empty known premise
record is checked rather than skipped by truthiness; the reference profile
rejects it through broader validation, which this bounded port does not import. This rule never writes
those words or determines scientific discharge. Other migration firewalls are
preserved; this is not a complete backport of unrelated hardening behavior.

## SI02: identify the actual consumed source

RN3-FAR and RN5-NEAR-POINT-CERTS now name H3-RUNG-FLOOR. H3-BAND-FLOOR remains a
separate unchanged node; this proposal does not dispute or re-review its claim.
The new node keeps the hardening profile's single-rung scope at r = 0.05,
`AUTHOR_SIDE_CERTIFIED`, original `CERTIFIED_RUNG` label and zero credit. Its
reference note explicitly distinguishes hardening's transition machinery from
this migration proposal.

The actual frozen runner is
`engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py`,
SHA-256 `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930`.
Its `_PINS` binds `../H3_closure/H3_RUNG_FLOOR.md`, it opens that exact file,
and its parser requires lower endpoint `7.7592917375327855e-3`. Thus the source
match is based on executed consumption logic and a byte pin, not a similar title.
No frozen runner or proof body is modified or executed by the new checker.

### Bounded source contract and deliberate update procedure

`FW-RN-FLOOR-SOURCE` validates the single binding against the runner's existing
fixed input contract, then verifies the actual repository file:

- Repository: `d6g8k5htny-coder/main`
- Path: `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md`
- Extraction: whole file, exactly 7,003 bytes
- SHA-256: `6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa`
- No symlink components, missing/non-file input, alternative repository/path,
  unknown extraction, malformed digest/length, or graph-only coherent re-pin

`FW-RN-FLOOR-DEPENDENCY` refuses substituting the band node for either known RN
consumer. Refusals name the floor and reverse consumers, without writing a new
status. Unrelated documentation changes do not invalidate this contract.

This is deliberately narrower and stricter than a generic mutable binding:
migration has no reviewed transition adapter that could justify changing the
consumed scientific input. The constants transcribe an already frozen runner
contract; they create no new mathematical authority. Altering both the file and
the graph cannot silently make a new input validate itself.

For a legitimate successor: retain the original frozen files; identify the
numbered successor runner and source, all affected consumers, and exact byte
identities; provide the review/disposition that supports using that new input;
then propose a scoped update to this contract, graph, and regression fixtures
for nonauthor review. Do not merely update a digest to silence a failed check.
A reviewed deployment of the hardening transition adapter is a separate design,
not something this patch has installed.

## Verification and unresolved boundaries

Run `python3 tools/claims_check.py` and
`python3 -m pytest tests/test_claims.py tests/test_claims_audit_convergence.py -q`.
The new tests use explicit `--graph` CLI calls in normal and optimized Python,
including positive controls, direct/transitive open premises, both status
columns, lost/altered/symlink sources, coherent source-plus-graph re-pinning,
wrong dependencies, exact runner consumption, and unrelated-document controls.

The full repository suite and workflow checks, exact preservation evidence,
and nonauthor review are separate required records at the candidate head.
This document does not imply those checks passed before they are recorded.

No live Drive status is inferred from these historical exports. No premise,
lemma, theorem or prize is promoted; the conditional truth and historical
labels remain. No transition adapter, automatic persistent invalidation,
scientific acceptance, full hardening parity, or live Drive watcher is claimed.
Canonical/default refs and all audit input bytes remain untouched. Integration
requires owner/scope reconciliation, current checks, and audit-coordinator
release; preparing this proposal does not lift that boundary.
