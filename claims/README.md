# Claim / premise dependency graph

`graph.json` is the program's claim structure as data: 26 claims, 13 named
premises, 9 firewalls. `tools/claims_check.py` turns the firewalls into
assertions; `tests/test_claims.py` and `tests/test_claims_firewalls.py` prove the
checker actually rejects each violation it is supposed to reject.

Those counts are checked rather than asserted. `tools/claims_check.py` reconciles
three lists that had no reason to agree and did not: the firewalls **declared**
in `graph.json`, the firewalls it **enforces** (`ENFORCED_FIREWALLS`), and the
firewalls named in this file. A run where any of the three disagrees is refused,
and so is a run where this file's claim count or firewall count has drifted from
the graph. All three had drifted: this file documented eight firewalls over nine
enforced, and twenty-four claims over a graph of twenty-six. An undocumented
firewall is not a safeguard a reader can check, and a stale count is how a reader
learns to stop trusting the rest. A **missing** `claims/README.md` is a refusal
too, not a skip: guarding the reconciliation with "if the file exists" would have
made deleting this file the way to switch it off.

This encodes what the Drive sources state in prose. It asserts no mathematics of
its own, and it never changes a status.

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
firewalls read **both columns separately**, and neither column may be moved:
`FW-NO-RECEIPT-PROMOTION` because a receipt may not move either one, and
`FW-UNCONDITIONAL` because a premise open in its register note is open. Reading
only the frozen column, which is what this checker used to do, made a
disagreement in the direction that matters invisible.

Both columns are also read against a **closed** vocabulary. A premise counts as
discharged only when a column says so in a word listed in `PREMISE_DISCHARGED`
(`CLOSED`, `DISCHARGED`, `PROMOTED`, `SATISFIED`, `CERTIFIED`); a word in
`PREMISE_UNDISCHARGED` is a refusal; a word in neither is a refusal too, because
a status this checker cannot read is not evidence that anything was discharged.
The old test asked only whether the frozen column read `OPEN` or `NOT_CLOSED`,
which let `NAMED_HYPOTHESIS` — carried by two premises, and by definition not a
discharge — pass as though it were one. `RESTATED` and `REFINEMENT` are
deliberately in the undischarged set: whether a restatement discharges a premise
is a status decision, and this file does not make status decisions.

A premise outside the D1 assembly (`H-B3`, `LM013-JOINT-STACK`) carries the same
transcribed status in both columns and says so in `status_columns_note`. That is
schema uniformity, not a layer disagreement.

**Claim** — `track`, `statement`, `grade`, `depends_on`, optional
`forbidden_extrapolations`, `independence_credit`, `external_review`,
`historical_novelty`, `original_prize_closed`, `evidence`, `note`, `source`.

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
| `arithmetic` | `exact_rational`, `interval_*`, `not_applicable`, `float`, `mpmath_float`, … — or, when it comes from a carrier index, that index's own sentence. `classify_arithmetic` sorts it into exactly one of `exact`, `float`, `not_applicable`, `unrecognised` and `ambiguous`; the last two are counted in the summary line so an unreadable record is visible rather than inert |
| `certifying` | what the artifact's own record says about itself. It must be a real JSON boolean — a string `"true"` is refused, because the guard is an identity test that such a string walks straight past. It is used **only to refuse**; it never grants anything |

## The nine firewalls

| ID | Rule | Source |
|---|---|---|
| `FW-UNCONDITIONAL` | a claim graded `LIVE_ROOT_THEOREM`, `FROZEN_CERTIFICATE` or `RATIFIED_3D_ONLY` may not rest, transitively, on a premise **either** of whose status columns is anything other than a word in the closed discharged vocabulary | `HOLD_OPEN_VALIDITY_PREMISES.md`; OP-GDN-002 §6 |
| `FW-2D-3D-COMPOSITION` | no claim may depend on both the 2D tracks and the 3D lifetime track | `ERRATA_AND_CLARIFICATIONS_2026-09-13.md` §1 |
| `FW-PRIZE-ISOLATION` | the prize track and the q0/3D tracks may not depend on each other | `LANE_MATH_MAP.md` firewall |
| `FW-NO-PRIZE-CLOSURE` | every prize claim must carry `original_prize_closed: false` | `CLAIM_REGISTRY_VERIFIED_INTAKE.json` |
| `FW-DECIMAL-KILL` | the qualitative rate must forbid a finite decimal `C_Q0`; Theorem B must forbid a numerical `C*` | `Q0_MASTER.md` Part I; C092 §12.3 |
| `FW-LM011-PRECONDITION` | the RV-LM011 synthesis route may not be marked satisfiable while any named prerequisite is unsatisfied, and a technical pass at **zero** organizational independence credit does not discharge an independence-requiring gate | `review_queue.json` RV-LM011-MAIN; `easy_closure_queue.json` P02-LM-011; `docs/OPEN_PROBLEMS.md` §D |
| `FW-NO-RECEIPT-PROMOTION` | a receipt, a green test run, a reproduction or a carrier binding may never raise a grade or move a status, on a claim or on a premise | `engine/README.md`; OP-PROT-012 §4(c); OP-GDN-002 §6 |
| `FW-FLOAT-NOT-CERTIFIED` | high precision is not certification: evidence that declares itself `certifying` must **show** exact arithmetic — float, not-applicable, unrecognised and ambiguous are each refused by name — and no certified/enclosed claim may rest solely on float evidence | `engine/README.md`; `engine/rn_engine/BINDING.json`; `README.md` status discipline |
| `FW-RETRACTED-NOT-UNCONDITIONAL` | a claim carrying a retraction record, or whose transcribed `register_status` says RETRACTED, may not carry an unconditional grade | `ERRATA_AND_CLARIFICATIONS_2026-09-13.md`; `CLAIM_REGISTRY_VERIFIED_INTAKE.json` |

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

`FW-FLOAT-NOT-CERTIFIED` reads **both** carrier indexes —
`engine/carriers/MANIFEST.json` and `engine/rn_engine/BINDING.json` — because the
graph names exactly one `carrier_id`, `RNENG-01`, and it lives in the second one.
Reading only the first meant the override resolved nothing: one lookup, one miss,
every run. For evidence naming a `carrier_id` either index lists, that carrier's
own `arithmetic` and `certifying` fields win over the graph's copy, because the
carrier's record is what the run actually used, and the refusal names which file
it came from. When both indexes are absent, unreadable, or do not list the
carrier, the graph's own evidence record is used and the lookup is skipped
cleanly. Skipping can only lose a refusal that the graph's own record would have
to state anyway; it cannot manufacture a pass.

Activating that override naively would have *weakened* the firewall, which is
worth recording. `BINDING.json` describes RNENG-01 in a sentence that reads "no
`fractions.Fraction`, no `decimal.Decimal` and no interval arithmetic occurs
anywhere" — genuinely float code, whose own **denial** mentions three exact
tokens. Under the old rule that an exact token anywhere wins, that sentence
classified as exact. Hence `ambiguous`: a string carrying tokens from both
vocabularies is prose, not a classification, and the checker says so instead of
picking a side. A record that claims no certification may describe its arithmetic
in prose freely, because nothing rests on it.

A node's `track` is read against a closed set for the same reason. Three
firewalls — `FW-2D-3D-COMPOSITION`, `FW-PRIZE-ISOLATION` and
`FW-NO-PRIZE-CLOSURE` — test `track` by membership against a literal set, so an
unrecognised value drops the node out of all three at once and fails nothing. A
missing `track`, or one outside `TRACKS`, is now a refusal: a typo or a rename
was a silent opt-out from rule 4.

`GRADE_STRENGTH` in the checker orders grades for **one** purpose: refusing a
record that claims more than its evidence can carry. It is not a mathematical
hierarchy, it grades nothing, and a grade missing from the table is a failure
rather than a default — a new grade has to be placed explicitly.

## Running it

```bash
python3 tools/claims_check.py                    # check the committed graph
python3 tools/claims_check.py --graph X.json     # check a candidate graph
python3 tools/claims_check.py --manifest M.json  # a different carrier manifest
python3 tools/claims_check.py --binding B.json   # a different rn_engine carrier index
python3 tools/claims_check.py --readme R.md      # reconcile against different prose
python3 -m pytest tests/test_claims.py tests/test_claims_firewalls.py -q
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

`tests/test_claims_firewalls.py` is the second control file, and it covers the
fail-open cases rather than the firewalls' subject matter: a `certifying` string
that is not a boolean, a certifying record whose arithmetic is not-applicable or
unrecognised or the both-vocabularies sentence, a premise status word in neither
vocabulary, a `NAMED_HYPOTHESIS` premise under an unconditional claim, a premise
open only in its register-note column, an unknown and a missing `track`, a
firewall declared but not enforced, a firewall enforced but not documented, and a
drifted count in this file. Each is checked through the CLI against a mutated
**copy**, and each is paired with the assertion that the committed tree passes.

### The mutants these controls kill

A control that passes against the fixed checker has not been shown to test
anything. Each fix below was reverted in a **copy** of `tools/claims_check.py`
outside the tree, one at a time, in a subprocess with `PYTHONDONTWRITEBYTECODE=1`
— an in-process harness once gave a false result here, because same-size mutants
written in quick succession reused a stale `__pycache__`. Sixteen reverted fixes,
sixteen killed, none surviving:

| reverted fix | control that dies |
|---|---|
| `certifying` need not be a boolean | `..._a_certifying_string_that_is_not_a_boolean` |
| an unlisted arithmetic word reads as exact | `..._an_unrecognised_arithmetic_word` |
| a missing or non-string arithmetic reads as exact | `..._a_missing_arithmetic_field` |
| "an exact token anywhere wins" | `..._a_sentence_from_both_vocabularies` |
| `not_applicable` may certify | `..._not_applicable_arithmetic` |
| one carrier index instead of two | `..._the_graph_s_only_carrier_id_resolves`, `..._overrides_the_graph_and_names_its_source` |
| one status column instead of two | `..._open_only_in_its_register_note_column` |
| the old `OPEN`/`NOT_CLOSED` whitelist | `..._a_named_hypothesis_premise_...`, `..._restated_and_refinement_are_not_discharges` |
| no closed status vocabulary | `..._a_premise_status_word_in_neither_vocabulary` |
| no `track` presence check | `..._a_missing_track` |
| no `track` vocabulary check | `..._an_unrecognised_track` |
| a missing prose document is skipped | `..._an_absent_prose_document` |
| declared-not-enforced unchecked | `..._declared_in_the_graph_and_not_enforced` |
| enforced-not-declared unchecked | `..._enforced_and_not_declared` |
| prose need not name an enforced firewall | `..._does_not_name_an_enforced_firewall` |
| the prose counts are not compared | `..._a_drifted_claim_count...`, `..._a_drifted_firewall_count...` |

Two of the sixteen exist only because the first harness did not reach them: `if
False: ... elif t not in TRACKS:` still evaluates the `elif`, so disabling the
missing-`track` arm left the vocabulary arm live and the matrix looked complete
when it was not. Both arms now have their own mutant.

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
* Closing the fail-open holes moved no status and discharged nothing. Every
  refusal added here is a refusal the committed graph already survives: the
  conversion was measured against the tree before it was adopted, and it adds
  zero refusals to it. A firewall that now fails closed is a firewall that will
  catch a *future* edit — it certifies nothing about the present one.
