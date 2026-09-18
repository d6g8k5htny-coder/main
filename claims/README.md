# Claim / premise dependency graph

`graph.json` is the program's claim structure as data: 16 claims, 12 named
premises, 5 firewalls. `tools/claims_check.py` turns the firewalls into
assertions; `tests/test_claims.py` proves the checker actually rejects each
violation it is supposed to reject.

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
optional `depends_on` / `sub_obligations`, `source`.

Two status fields, deliberately. The frozen `D1_ASSEMBLY_v2_2` body and the
register note plus addenda disagree about four of the five premises, and that
disagreement is *correct*: the note's deltas take effect only at the next
issuance. Collapsing them would promote the v2.3 draft by accident. The
firewalls read the **frozen** column.

**Claim** — `track`, `statement`, `grade`, `depends_on`, optional
`forbidden_extrapolations`, `independence_credit`, `external_review`,
`historical_novelty`, `original_prize_closed`, `note`, `source`.

Grades in use: `LIVE_ROOT_THEOREM`, `FROZEN_CERTIFICATE`, `CERTIFIED_RUNG`,
`CONDITIONAL`, `PROPOSED`, `AUTHOR_SIDE_CERTIFIED`, `AUTHOR_SIDE_PARTIAL`,
`AUTHOR_SIDE_PROOF_PRESENT`, `ACCEPTED_AT_REVIEW_SCOPE`, `AMEND_REQUIRED`,
`RATIFIED_3D_ONLY`, `REFUTED_AS_WRITTEN`, `OPEN`.

## The five firewalls

| ID | Rule | Source |
|---|---|---|
| `FW-UNCONDITIONAL` | a claim graded `LIVE_ROOT_THEOREM`, `FROZEN_CERTIFICATE` or `RATIFIED_3D_ONLY` may not rest, transitively, on a premise whose **frozen** status is OPEN or NOT_CLOSED | `HOLD_OPEN_VALIDITY_PREMISES.md`; OP-GDN-002 §6 |
| `FW-2D-3D-COMPOSITION` | no claim may depend on both the 2D tracks and the 3D lifetime track | `ERRATA_AND_CLARIFICATIONS_2026-09-13.md` §1 |
| `FW-PRIZE-ISOLATION` | the prize track and the q0/3D tracks may not depend on each other | `LANE_MATH_MAP.md` firewall |
| `FW-NO-PRIZE-CLOSURE` | every prize claim must carry `original_prize_closed: false` | `CLAIM_REGISTRY_VERIFIED_INTAKE.json` |
| `FW-DECIMAL-KILL` | the qualitative rate must forbid a finite decimal `C_Q0`; Theorem B must forbid a numerical `C*` | `Q0_MASTER.md` Part I; C092 §12.3 |

Plus referential integrity, acyclicity, and "a CONDITIONAL claim must name at
least one premise".

## Running it

```bash
python3 tools/claims_check.py                 # check the committed graph
python3 tools/claims_check.py --graph X.json  # check a candidate graph
python3 -m pytest tests/test_claims.py -q     # firewalls + negative controls
```

## Updating it

When a source status changes, edit the matching `status_frozen_v2_2` or
`status_register_note` and cite the exact carrier in `source`. Do **not** edit a
status to make a check pass; that inverts the whole point. If the graph and the
registers disagree, the registers and the frozen bodies win, and the
disagreement is the finding.
