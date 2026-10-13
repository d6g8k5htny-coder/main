import pathlib, sys
W = pathlib.Path(sys.argv[1])
def patch(path, pairs):
    p = W / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, ("MISS", path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s); print("patched", path)

READY = "`reviews/register_execution_readiness_20260930`"
# ---------------- Math-#167 RECONCILIATION.md (v1.4)
patch("reviews/register_alignment_20260930/RECONCILIATION.md", [
("""and labelled as mutable evidence; the eight classifications are unchanged.
**Effect:**""",
"""and labelled as mutable evidence; the eight classifications are unchanged. v1.4 (register execution readiness,
""" + READY + """; re-verified at `dda8991`, Math-#173 merged): the checker accepts the live register in
exactly two states, **baseline** (nothing of the proposal live) or **installed** (every proposed field of the eight nodes,
every component node and every proposed edge live exactly), rejects partial or drifted installs, replays the proposal
from the register without it (`strip`) and, once installed, asserts that `apply` is the identity and that this record's
objects are unheld; the workflow compares the checker's output with `RESULTS.json` in the baseline state and with
`RESULTS_INSTALLED.json` in the installed state; two mutants added (`installed-drift`, `partial-install`). Transitions,
sources and verdicts are unchanged.
**Effect:**"""),
("""- **BASELINE.** Each transition's live node exists with exactly the recorded current classification, fingerprint,
  kind and source; the D1 node is `PROVED_REVIEWED` with its four reading-rule components `PROVED_REVIEWED`.""",
"""- **BASELINE.** The live register is in exactly one of two accepted states. *Baseline:* each transition's live node
  exists with exactly the recorded current classification, fingerprint, kind and source, no component node and no
  proposed edge is live. *Installed:* each transition's live node carries every proposed field exactly (classification,
  `controlling`, fingerprint, source, scope, explicit limits, notes, review disposition, review sources and the derived
  `review_basis`), every component node is live as proposed on every proposed key, every proposed edge is live, and no
  other live fingerprinted node carries a component source. Any node in between (one flipped, one not; a proposed key
  drifted; a component present under another id) is a mismatch and the check fails. In both states the D1 node is
  `PROVED_REVIEWED` with its four reading-rule components `PROVED_REVIEWED`, and every transition is a change (its
  current and proposed classifications differ)."""),
("""  live D1 node and cap component, the interface, the component nodes) are `PROVED_REVIEWED`, every region requires its
  covering node, and every applied node carries its `explicit_limits`, its replacement `notes` and a `review_basis`.""",
"""  live D1 node and cap component, the interface, the component nodes) are `PROVED_REVIEWED`, every region requires its
  covering node, and every applied node carries its `explicit_limits`, its replacement `notes` and a `review_basis`.
  `apply` is idempotent (a node already carrying the proposal is left as it is, a live component or edge is not
  re-created) and `strip` is its pre-image (the eight nodes at their current classification and fingerprint, the
  component nodes and the edges touching them removed): on the baseline register `strip` is the identity, on the
  installed register `apply` is, and `apply(strip(·))` reproduces every object on every proposed key in both. The
  installed state is also simulated (on the baseline register: `apply` of the live graph; once installed: the live graph
  itself) and must be accepted exactly, with re-application a no-op."""),
("""- **GATE.** The proposed graph passes the hard gate's `validate_graph_fail_closed`; `closure_report` gives `gate_ok`
  with no illegal controlling node; `reverse_impact_between(live, proposed)` runs and its impacted set (the eight nodes,
  the eight component nodes and their dependents) is recorded.""",
"""- **GATE.** The proposal replayed from the register without it: the replay passes the hard gate's
  `validate_graph_fail_closed`; `closure_report` gives `gate_ok` with no illegal controlling node on the pre-image, the
  replay and the live graph; `reverse_impact_between(pre-image, replay)` changes and impacts every one of the sixteen
  objects (loss-only rule), and none of them is held once applied (every required premise `PROVED_REVIEWED`). In the
  baseline state the pre-image is the live graph and the impacted set (the eight nodes, the eight component nodes and
  their dependents) is recorded in `RESULTS.json`; in the installed state `reverse_impact_between(live, apply(live))`
  must be empty (nothing left to apply) and the impacted set is not pinned, since after execution it depends on the
  other landed records."""),
("""Twelve mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,
`executed-flag`, `skip-interface-premise`, `holder-reversed`, `baseline-drift`, `duplicate-drift`, `region-unlinked`
(the `covered_by` edges removed before the propagation replay: proof edits no longer reach the regions) and
`review-metadata-only` (the review-record nodes and their edges removed before the replay, leaving the citations as
metadata only: a deleted or edited review record no longer reaches its accepted object).""",
"""Fourteen mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,
`executed-flag`, `skip-interface-premise`, `holder-reversed`, `baseline-drift`, `duplicate-drift`, `region-unlinked`
(the `covered_by` edges removed before the propagation replay: proof edits no longer reach the regions),
`review-metadata-only` (the review-record nodes and their edges removed before the replay, leaving the citations as
metadata only: a deleted or edited review record no longer reaches its accepted object), `installed-drift` (the
simulated installed register with one component fingerprint changed) and `partial-install` (the simulated installed
register with the D4 node back at its current classification). The workflow runs the checker in the state the
repository is in: it compares the output with `RESULTS.json` when the checker reports `register_state: baseline` and
with `RESULTS_INSTALLED.json` when it reports `installed`; the mutants must be rejected in either state (the
composition of Math-#167, #160 and #173 on a scratch copy, with every mutant, is replayed by """ + READY + """)."""),
])
# ---------------- Math-#160 RECONCILIATION.md (v1.8)
patch("reviews/c6_witness_collision_reconciliation_20260929/RECONCILIATION.md", [
("""now live, and records that Math-#151 took the keep-open alternative without defining the regional mechanism.
**Effect:**""",
"""now live, and records that Math-#151 took the keep-open alternative without defining the regional mechanism. v1.8
(register execution readiness, """ + READY + """; re-verified at `dda8991`, Math-#173 merged): the checker
accepts the live register in exactly two states, **baseline** (nothing of §7 live) or **installed** on the main path of §7
(every proposed node live exactly on every proposed key; the residual node either as proposed here, `OPEN_ACTIVE` with its
notes, or `PROVED_REVIEWED` as Math-#173 proposes, with layer, kind, fingerprint and `controlling` unchanged; every
proposed edge not leaving the witness node live; the witness node `OPEN_ACTIVE` while step 4 is pending, or
`PROVED_REVIEWED` with all its proposed evidence edges live and their required targets `PROVED_REVIEWED`), rejects partial
or drifted installs and any other live fingerprinted node on a proposed source, and accepts the selector table either
untouched or carrying every proposed cell with the three region ids covered; the workflow compares the checker's output
with `RESULTS.json` in the baseline state and with `RESULTS_INSTALLED.json` in the installed state; two mutants added
(`installed-drift`, `partial-install`). Nothing in §§1–7 changes. The §1a register-equivalent alternative (no separate
residual node) is not an accepted installed state, since Math-#173 presupposes the separate node.
**Effect:**"""),
("""  reading-rule edges present, and no proposed node duplicates a live source; every proposed node passes the hard
  gate's shape rules; `declarative` is true and `executed` is false.""",
"""  reading-rule edges present, and no proposed node duplicates a live source; every proposed node passes the hard
  gate's shape rules; `declarative` is true and `executed` is false. The live register is in exactly one of two
  accepted states, baseline (nothing of §7 live) or installed (as the v1.8 revision note defines it: every node exactly,
  the residual as proposed or as Math-#173 promotes it, the witness node open or `PROVED_REVIEWED` with its evidence
  edges live); once installed this record's nodes are the live carriers of their sources and nothing else may be. The
  installed state is also simulated (steps 1–3 applied to the live graph, the witness node left as it is) and must be
  accepted exactly."""),
("""- **SELECTOR.** The selector proposal names every selector of the live table for its three regions, assigns each a
  value the hard gate classifies as covered or bypassed, and, applied to the live table, moves exactly those three
  region ids from `open_region_ids` to `covered_region_ids`.""",
"""- **SELECTOR.** The selector proposal names every selector of the live table for its three regions, assigns each a
  value the hard gate classifies as covered or bypassed, and, applied to the live table, moves exactly those three
  region ids from `open_region_ids` to `covered_region_ids`. A live table already carrying every proposed cell is
  accepted as installed when the three region ids are covered and not open; a table carrying part of the proposal is
  rejected; the node-only alternative of §7 step 2 (table untouched) is the baseline case."""),
("""Ten mutants must fail: `allow-symlink`, `no-hash`, `drop-edge`, `stale-fingerprint`, `executed-flag`,
`close-residual`, `drop-lower-bound`, `regional-strict`, `clone-live-node`, `open-cell-left`.""",
"""Twelve mutants must fail: `allow-symlink`, `no-hash`, `drop-edge`, `stale-fingerprint`, `executed-flag`,
`close-residual`, `drop-lower-bound`, `regional-strict`, `clone-live-node`, `open-cell-left`, `installed-drift` (the
simulated installed register with the [PALM] node's fingerprint changed) and `partial-install` (the simulated installed
register without the supersession node's edges). The workflow runs the checker in the state the repository is in: it
compares the output with `RESULTS.json` when the checker reports `register_state: baseline` and with
`RESULTS_INSTALLED.json` when it reports `installed`; the mutants must be rejected in either state (the composition of
Math-#167, #160 and #173 on a scratch copy, with every mutant, is replayed by """ + READY + """)."""),
])
