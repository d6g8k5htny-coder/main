# Experience refinement implementation plan

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development. Read the linked spec; preserve its source and scientific boundaries.

**Goal:** Make research tools accessible while reading, make Compare faster to
navigate/load, and make local planner connection status truthful.
**Architecture:** Native disclosure/navigation; two synchronized pager controls;
Compare-only demand loading of the existing source verifier; separate local v24
dashboard state machine. No framework/backend/schema change.
**Spec:** `docs/superpowers/specs/2026-10-26-experience-refinement.md`.

## Global constraints and review focus

No mathematical data, source admission, current.mjs/model, frozen history, worker
control, auth/CORS, workflow/settings or certification changes. Preserve newer
outside focus, no-JS links and script-free Formal. Explicitly distinguish planner
transport, valid data, stale data and worker execution. Check mobile overlay
obstruction, both pager disabled states, source deep links/retry and malformed SSE.

### Task1: Persistent reader navigation (Root)

- [ ] Write/run baseline browser tests for absent Go to, visible access at long
  scroll, native keyboard/links and no-JS/formal behavior.
- [ ] Add reader-jump.css/mjs and consistent native markup to eleven current HTML
  pages; module on ten pages, never Formal. Do initial shared HTML edit before
  delegating Compare's other edits, then leave those hunks alone.
- [ ] Verify menus in desktop/phone schemes, outside/Escape focus, print and native
  destination links. Stage new assets before one final release-key regeneration.

### Task2: Compare ergonomics and acquisition (delegated)

- [ ] Record baseline top-pager absence and two current.json requests.
- [ ] Modify only compare.html/css/mjs, new compare-source-panel.mjs, focused tests
  and driver. Use existing loadPublication and source verifier unchanged.
- [ ] Verify synchronized upper/lower pagination/focus/refusal and exactly one
  initial current payload fetch; second only after source disclosure requested.
- [ ] Verify direct source links and module/network failure, and existing Compare
  regression suite. Return stable files for Root release-key generation.

### Task3: Local v24 dashboard accuracy (separate checkout/delegate)

- [ ] Create isolated source247981b checkout and external runtime; preserve exact
  originals and claim exact dashboard/test/doc scope in existing architecture task.
- [ ] Reproduce premature Live and partial invalid render before changing code.
- [ ] Implement validated atomic snapshot preparation and honest transport/error/
  stale states. Never connect the full legacy Coordinator or formal workers.
- [ ] Run adversarial fixtures and documented real isolated planner GET/SSE flow;
  prove snapshot/journal identity and no worker/certificate promotion.

### Final integration

- [ ] Run required site checks and affected browser regressions after all public
  files stabilize; preserve source identities and test evidence.
- [ ] Separate whole-branch engineering review, current protected integration and
  actual public-byte/browser acceptance. Apply effective existing LOOSEN C without
  claiming a delayed service review passed or bypassing platform gates.
- [ ] Additive verified Drive source/site/local-patch/evidence delivery, one grouped
  register entry, original failure/deletion records and explicit remaining gaps.
