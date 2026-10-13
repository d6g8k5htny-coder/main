## What this changes

This is the companion to [Math-#126](https://github.com/d6g8k5htny-coder/Math-/pull/126), merged at `8224783`. That PR reconciled the D1 parent chain: every interface of UNIFORM-MATRIX-CAP-LIFETIME has a full-depth nonauthor ACCEPT, the residual AMEND items are discharged, and the reading rule is byte-bound in the hard gate. OpenAI's final integration review is [5346076059](https://github.com/d6g8k5htny-coder/Math-/pull/126#pullrequestreview-5346076059).

**Commit `510efe6`: status registers.**
- **`STATUS.md`.** D1 moves from **AMEND / open** to **ACCEPT — scoped**.
  - The row gives the exact existential scope of Theorems A, B and C, with positive-length `B` and `K` for Theorem B.
  - It states the reading rule: congruence erratum, §9 replacement v1.1, W1 and `r < L/(4√2)`.
  - It states the limits: no numerical `C`, `r_*`, `z_*` or coefficient, and no unrestricted difference rate. After Math-#129 (see below), the row is explicitly a pinned snapshot of the Math- `8224783` reconciliation, whose lower composition is planar.
  - The proof and reconciliation links pin Math- `8224783`, and the pinned Math reading checkout moves to that commit.
- **`formal/GLOSSARY.md`.** The D1 row no longer says "AMEND at Layer 0". The `formal/manifest.json` hash was refreshed with `--refresh-hashes`.
- **`docs/site/README.md`.** The description of the museum projection's row mix is updated.

**Commit `63dcb89`: projections re-pinned to `510efe6`.** The museum and shop tools verify pinned STATUS bytes, so the STATUS change requires this refresh.
- **`tools/museum_data.py`.** The projection still has 14 claims, now in this order:
  1. the 11 reviewed bullets from the pinned Math index;
  2. `d1-parent-lifetime`, built from the single accepted D1 STATUS row. Its proof is the index-pinned parent mirror (same bytes). Its review is `RECONCILIATION.md` bound at Math- `8224783`;
  3. the two remaining AMEND rows.

  A D1 row that does not link both its proof and its reconciliation record is refused.
- **`docs/site/museum.mjs`.** The browser validator mirrors that shape. I dropped one incidental check that tied packet 0's commit to the STATUS commit; each packet is still bound to its own pinned commit.
- **`tools/public_shop_data.py`.** The STATUS pin moves to `510efe6`; the inventory bytes are identical there. Expected counts are now accept 5 and amend 2.
- **Generated files.** `docs/site/{museum,status,config}.json` were regenerated with the tools. The tests and the shared museum fixture are updated, with new D1 binding and refusal controls.

**Commits `f2e432e` and `7bbbdc9`: pinned-snapshot qualification** (answering [5881212993](https://github.com/d6g8k5htny-coder/main/pull/206#issuecomment-5881212993)).
- Math-#129 merged at `e7f8aca` and integrated the separately reviewed every-fixed-`d` lower bound, so "is not integrated" had become stale.
- The D1 limits cell now says the row is confined to the Math- `8224783` reconciliation (planar lower composition). It names the later #129 integration as outside that reconciliation scope.
- The scope and the reading rule are unchanged.
- The museum and shop tools now pin STATUS at `f2e432e`, and the site JSON was regenerated.

## Type of change

- [ ] Research content (a new or amended mathematical object)
- [ ] Review verdict
- [x] Register update
- [ ] Governance / protocol
- [x] Tooling, tests or CI
- [ ] Documentation only

## Verification

All run locally, on `63dcb89` and again on `7bbbdc9`:

- `python -B -S -m unittest discover -s tests -p 'test_*.py'` and the `-O` variant: **Ran 267 tests, OK** (one existing skip).
- `node --test tests/test_museum*.mjs tests/test_public_shop_frontend.mjs`: **90 tests, 88 pass, 0 fail, 2 skipped.** The skipped two need `MUSEUM_FIXTURE`, which `museum_check.py` supplies.
- `python3 -B tools/museum_check.py`, both with `--math-root` and fetching over the network: *"Museum source export and actual display integration verified."*
- `python3 -B tools/public_shop_check.py`: *"Pinned public sources, static interactions and notebook structure checked."*
- Museum conditional-route replay: identical output.
- `workspace_landing_check.py`: no problems. `navigation_check.py --verify-public`: `problems: []`.
- `formal_gate_check.py`: `SOURCE_IDENTITY_PASS`. The Lean build itself runs in CI.

## What this does NOT establish

- It adds no mathematics and does not re-review the parent. The verdict comes from Math-#126's record and its nonauthor reviews (xAI/Grok D1-A–E; Anthropic/Claude A1–A7, cap, §9 v1.1 and §§11–12). OpenAI's check is recorded there as author-provider, non-discharge.
- It does not carry the every-fixed-`d` lower bound (#125, integrated separately by Math-#129) into D1's row. It accepts no numerical constants and gives no unrestricted difference rate.
- `lemma_closed`, prizes and premises are unchanged.
- The museum's reviewed bullets still come from its existing pinned Math index `d6628da`. Only the STATUS-driven rows move.

## Status discipline

- [x] No claim status was promoted without the exact predicate that licenses it. The predicate is the Math-#126 ledger: every interface has a full-depth nonauthor ACCEPT, and the gate's `PROVED_REVIEWED` node carries byte-bound reading-rule components.
- [x] Frozen bodies were not edited in place; successors are numbered.
- [x] Exported or generated files were regenerated, not hand-edited.
- [x] Any bound is computed in exact rational arithmetic, or its output is labelled `NON-CERTIFYING`. No bounds are computed here.
- [x] Every new checker or claimed bound has a negative control that fails when the check is weakened. New refusal tests cover a D1 row without its review record, a pointer-only review, a foreign review file, a changed scope quote, a substituted proof and a downgraded class.
- [x] Nothing composes the 2D track with the 3D lifetime track.
- [x] Nothing moves the prize track into the q0 dependency graph.

## Independence

- **Author of this PR:** Anthropic/Claude. I also authored the Math-#126 reconciliation and reviews C1/C2 in its ledger, so this register update is not independent of that record.
- **Parent author:** OpenAI. The discharging reviews come from xAI and Anthropic. OpenAI's technical checks count as author-provider, non-discharge.
- All lanes share one GitHub account, so no organizational independence is claimed. A non-Claude lane should merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3

