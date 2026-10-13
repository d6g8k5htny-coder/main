## Change and purpose

Adopts the useful parts of the closure-superstructure proposal Dylan forwarded on 2026-10-06. The decision record is [main#275](https://github.com/d6g8k5htny-coder/main/issues/275). This PR adds no status word, register or directory. Scope: **engineering/workflow only**. Scientific effect NONE.

- `governance/OP-CLOSURE-EVIDENCE-20261006.md` (new) covers:
  - a derived, never-stored evidence profile, where an axis with no record reads `not recorded`;
  - the blind-reconstruction protocol: a statement packet only, the reconstructor publishes before comparing, and exposure is stated;
  - the adversarial-attack protocol: a checklist plus an `ATTACK STATUS` block;
  - a second, structurally independent implementation for headline constants;
  - novelty refresh through the existing ledger with two adversarial searches;
  - the ≤20-page human-referee packet, where choosing the referee is Dylan's decision;
  - the items explicitly not adopted: a `CLOSED` word, passport files and `closure/` directories.
- `.github/ISSUE_TEMPLATE/review-record.yml` adds two review bases ("Blind reconstruction — statement packet only, proof not read" and "Adversarial attack — counterexample search") plus a pointer to the protocol. The existing three bases and all other fields are unchanged.
- `governance/OP-WORKFLOW-20260930.md` and `docs/WORKSPACE.md` each get a one-paragraph link.

## Current disposition

- Head: `ad6aa1523dd853cee74c80a76552463fe237cac6`, one commit on `claude/nice-maxwell-7i7ok0` over main `cddbb7f6`.
- Author: Anthropic Claude, session `017Mi3hx` (Dylan Roy — delegated AI work). The author will not self-merge.
- Completed review scopes: none.
- Requested: **one nonauthor engineering/governance read**, ideally from a lane whose provider differs from the author's. Then a separate eligible integrator.
- Unresolved findings: none.

## Verification

Run locally at the head above:

- `python -B -S -m unittest discover -s tests -p 'test_*.py'`: 362 tests OK. The same in `-O` mode: 362 OK.
- `python -B -S tools/workspace_landing_check.py`: 0 problems (131 local links, 16 pages).
- `python -B -S tools/navigation_check.py --verify-public`: 0 problems (162 local links, 29 pages, 9 public sources verified).
- The review form parses as YAML, and the dropdown lists the five bases in order.

## Evidence and limits

This is protocol and form text only. It executes no register transition, changes no STATUS/PROOF_INDEX/GRAPH/catalog row, flips no `lemma_closed`, prize or premise, and installs no scheduler. A blind reconstruction or attack record does not by itself earn organizational-independence credit. "Not found" in an attack is not acceptance. Author exposure: this lane wrote the Math-#160/#167/#173/#183 register records named in #275, so it must not be the one to attack or execute them.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
