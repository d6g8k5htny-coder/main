# Claim / premise dependency graph

`graph.json` is the program's claim structure as data: 24 claims, 13 named
premises, 9 firewalls. `tools/claims_check.py` turns the firewalls into
assertions; `tests/test_claims.py` proves the checker actually rejects each
violation it is supposed to reject.

This encodes what the Drive sources state in prose. It asserts no mathematics of
its own. When a source status word conflicts with an explicitly open load-bearing
dependency, the graph may carry a weaker fail-closed operational `grade` while
preserving the source word in `source_grade_verbatim`; this is a claim-layer
hold, not a rewrite of the frozen source.

## Why this exists

The Drive's discipline is enforced today by careful prose repeated across many
documents: "packaging is not premise discharge", "do not compose 2D and 3D",
"display is not a certified enclosure", "0 prizes solved". Prose cannot fail a
build. A future contributor — human or model — who relabels
`D1-v2.2(2)` as a theorem, or adds the ratified 3D upper to a 2D dependency
list, gets a red CI run instead of a plausible-looking commit.

## Node schema

**Premise** — `track`, `status_frozen_v2_2`, `status_register_note`, `note`,
optional `depends_on` / `sub_obligations`, `evidence`, `source`.

Two status fields, deliberately. The frozen `D1_ASSEMBLY_v2_2` body and the
register note plus addenda disagree about four of the five premises, and that
disagreement is *correct*: the note's deltas take effect only at the next
issuance. Collapsing them would promote the v2.3 draft by accident. The
firewalls read the **frozen** column, except `FW-NO-RECEIPT-PROMOTION`, which
reads **both columns separately** — a receipt may not move either one.

A premise outside the D1 assembly (`H-B3`, `LM013-JOINT-STACK`) carries the same
transcribed status in both columns and says so in `status_columns_note`. That is
schema uniformity, not a layer disagreement.

**Claim** — `track`, `statement`, `grade`, `depends_on`, optional
`forbidden_extrapolations`, `independence_credit`, `external_review`,
`historical_novelty`, `original_prize_closed`, `evidence`, `note`, `source`, and
optional `source_grade_verbatim` / `audit_disposition` when a frozen source word
is preserved but the live claim layer must fail closed.

Grades in use: `LIVE_ROOT_THEOREM`, `FROZEN_CERTIFICATE`, `CERTIFIED_RUNG`,
`CONDITIONAL`, `PROPOSED`, `AUTHOR_SIDE_CERTIFIED`, `AUTHOR_SIDE_PARTIAL`,
`AUTHOR_SIDE_PROOF_PRESENT`, `ACCEPTED_AT_REVIEW_SCOPE`, `AMEND_REQUIRED`,
`RATIFIED_3D_ONLY`, `REFUTED_AS_WRITTEN`, `OPEN`, and — for the review routes —
the registers' own technical statuses `NEEDS_RECONCILIATION` and
`PASS_TECHNICAL`.

**Review-route claim** (the `RV-LM*` records) — additionally `technical_status`
(the same transcribed word as `grade`; the checker refuses them if they drift),
`exact_object`, `body_bytes`, `body_sha256`, `requires_independent_verdict`,
`independent_review_state`, `independent_review_verbatim`,
`independence_status`, `independence_credit`, `author_provider`,
`reviewer_claim`, `aging_action`, `age_days`, `queue_state`,
`remaining_decisive_work`. Every one is transcribed from
`registers/json/review_queue.json` and `registers/json/easy_closure_queue.json`.

**Evidence record** — `kind`, `ref`, `arithmetic`, `certifying`, optional
`carrier_id`, `note`.

| field | vocabulary |
|---|---|
| `kind` | `proof_body`, `frozen_certificate_body`, `certificate_set`, `register_row`, `review_record`, `exact_rational_module`, and the four that establish nothing: `receipt`, `test`, `carrier_binding`, `reproduction` |
| `arithmetic` | `exact_rational`, `interval_*`, `not_applicable`, `float`, `mpmath_float`, … — or, when it comes from a carrier manifest, that manifest's own sentence |
| `certifying` | what the artifact's own record says about itself. It is used **only to refuse**; it never grants anything |

## The eight firewalls

| ID | Rule | Source |
|---|---|---|
| `FW-UNCONDITIONAL` | a claim graded `LIVE_ROOT_THEOREM`, `FROZEN_CERTIFICATE` or `RATIFIED_3D_ONLY` may not rest, transitively, on a premise whose **frozen** status is OPEN or NOT_CLOSED | `HOLD_OPEN_VALIDITY_PREMISES.md`; OP-GDN-002 §6 |
| `FW-RUNG-OPEN-PREMISE` | a claim cannot remain `CERTIFIED_RUNG` while a transitive named premise is `OPEN` or `NOT_CLOSED`; preserve the source label separately and hold the live claim | fail-closed claim audit 2026-09-25; D1 v2.2 §1 |
| `FW-2D-3D-COMPOSITION` | no claim may depend on both the 2D tracks and the 3D lifetime track | `ERRATA_AND_CLARIFICATIONS_2026-09-13.md` §1 |
| `FW-PRIZE-ISOLATION` | the prize track and the q0/3D tracks may not depend on each other | `LANE_MATH_MAP.md` firewall |
| `FW-NO-PRIZE-CLOSURE` | every prize claim must carry `original_prize_closed: false` | `CLAIM_REGISTRY_VERIFIED_INTAKE.json` |
| `FW-DECIMAL-KILL` | the qualitative rate must forbid a finite decimal `C_Q0`; Theorem B must forbid a numerical `C*` | `Q0_MASTER.md` Part I; C092 §12.3 |
| `FW-LM011-PRECONDITION` | the RV-LM011 synthesis route may not be marked satisfiable while any named prerequisite is unsatisfied, and a technical pass at **zero** organizational independence credit does not discharge an independence-requiring gate | `review_queue.json` RV-LM011-MAIN; `easy_closure_queue.json` P02-LM-011; `docs/OPEN_PROBLEMS.md` §D |
| `FW-NO-RECEIPT-PROMOTION` | a receipt, a green test run, a reproduction or a carrier binding may never raise a grade or move a status, on a claim or on a premise | `engine/README.md`; OP-PROT-012 §4(c); OP-GDN-002 §6 |
| `FW-FLOAT-NOT-CERTIFIED` | high precision is not certification: no evidence may be `certifying` while its arithmetic is float, and no certified/enclosed claim may rest solely on float evidence | `engine/README.md`; `engine/rn_engine/BINDING.json`; `README.md` status discipline |

Plus referential integrity, acyclicity, "a CONDITIONAL claim must name at least
one premise", "`technical_status` and `grade` may not drift apart", and "an
evidence `kind` must be in the vocabulary".

### The LM lemma stack (`FW-LM011-PRECONDITION`)

`docs/OPEN_PROBLEMS.md` §D records, in prose, that `RV-LM011` is the synthesis
route and "needs LM003, LM004-v1.1, LM006, LM009, LM010-v1.1, LM012-v1.1 and
LM013 Carrier-B-v1.1 plus the joint stack **first**." That sentence is now
wired: the eight objects are nodes, `RV-LM011-MAIN.depends_on` names all eight,
and `precondition_routes` names them again so the checker can refuse a
dependency list that quietly loses one.

The joint stack is its own premise, `LM013-JOINT-STACK`, because the register
names it separately from the Carrier B v1.1 conversion
(`PARENT PASS / CARRIER A APPROVE / CARRIER B v1.1 CONVERSION OPEN / JOINT STACK
OPEN / P0.2 OPEN`).

A prerequisite counts as satisfied **only** when its transcribed technical
status is a pass *and*, where the route requires an organizationally distinct
verdict, that verdict is recorded as obtained with non-zero independence credit.
`RV-LM009-MAIN` is the case that makes this worth encoding: its technical status
is `PASS_TECHNICAL`, and it earns **zero** independence credit because the
author provider and the reviewer are both OpenAI. The register says so itself —
"Same-line audits and formal companions earn zero independence credit" — and
keeps `Independent review needed: OPEN`. A same-provider pass is a technical
pass and nothing more.

`synthesis_route_satisfiable` is **transcribed** (`false`, from the register's
`DEPENDENCY VERDICTS OPEN / OWN EXTERNAL VERDICT OPEN / P0.2 NOT PROMOTED`).
The checker can only refuse a `true`. It never sets the field, never computes a
route into satisfiability, and understating is always allowed.

### Receipts and floats

`FW-NO-RECEIPT-PROMOTION` is why `D3-LEMMA-RN-UNIF` carries an `evidence` list
at all. This repository holds no proof body for either piece of that lemma; what
it holds is a byte-exact carrier recovery (`engine/rn_engine/BINDING.json`,
carriers RNENG-01..08) and the receipts of runs against it. Evidence of kind
`carrier_binding` cannot carry a discharging status in either column, so the
premise cannot be closed here by any amount of green CI.

`FW-FLOAT-NOT-CERTIFIED` reads `engine/carriers/MANIFEST.json` when it exists:
for evidence naming a `carrier_id` the manifest lists, that carrier's own
`arithmetic` and `certifying` fields win over the graph's copy, because the
carrier's record is what the run actually used. When the manifest is absent,
unreadable, or does not list the carrier, the graph's own evidence record is
used and the manifest is skipped cleanly. Skipping can only lose a refusal that
the graph's own record would have to state anyway; it cannot manufacture a pass.

`GRADE_STRENGTH` in the checker orders grades for **one** purpose: refusing a
record that claims more than its evidence can carry. It is not a mathematical
hierarchy, it grades nothing, and a grade missing from the table is a failure
rather than a default — a new grade has to be placed explicitly.

## Running it

```bash
python3 tools/claims_check.py                    # check the committed graph
python3 tools/claims_check.py --graph X.json     # check a candidate graph
python3 tools/claims_check.py --manifest M.json  # a different carrier manifest
python3 -m pytest tests/test_claims.py -q        # firewalls + negative controls
```

`tests/test_claims.py` runs the checker on the committed graph and then breaks a
**copy** of it in twenty-two ways — promoting a conditional theorem, composing
2D with 3D, leaking the prize track, dropping the decimal kill, unwiring a
RV-LM011 prerequisite, spending a zero-independence pass on an independence
gate, letting a receipt discharge a premise, calling a float certifying — and
asserts each one is rejected. Two of them are run a second time with a single
field changed back, to prove the refusal came from the mutated field and not
from something incidental. Every mutation is passed to the checker with an
explicit `--graph`: a default-argument bug once made these tests silently
re-check the good graph and pass regardless.

## Updating it

When a source status changes, edit the matching `status_frozen_v2_2`,
`status_register_note` or `technical_status` and cite the exact carrier in
`source`. Do **not** edit a status to make a check pass; that inverts the whole
point. If the graph and the registers disagree, the registers and the frozen
bodies win, and the disagreement is the finding.

## What this does not establish

Nothing mathematical. The graph is a transcription with a checker attached.

* No premise is discharged, closed or promoted here, and none of the five open
  D1 validity premises has moved: all five are still OPEN or NOT_CLOSED in the
  frozen column.
* Adding the LM stack establishes **nothing** about P0.2. Seven component
  verdicts and the joint stack are open, `RV-LM011` has no external verdict, and
  `1 - a_r = O(r^3)` is not asserted here. `P02-LM-011`'s own register row reads
  `P0.2 NOT PROMOTED`.
* A passing `claims_check.py` run is a green run, not a proof. It verifies the
  shape of recorded dependencies and verifies no mathematics whatsoever.
* `independence_credit` is transcribed, never awarded. Every credit in this
  graph is 0.
* The 2D upper/lower tracks and the 3D lifetime track are never composed, and
  no original prize problem is solved.
