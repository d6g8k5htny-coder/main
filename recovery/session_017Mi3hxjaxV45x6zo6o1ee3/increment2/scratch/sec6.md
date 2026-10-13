## 6. Checks

`alignment_check.py` (stdlib only, run from the repository root):
- **IDENTITIES.** All fifteen inventoried files (the six sources, the D1 parent, the four in-repo reviews, the two
  consumed sources, and this packet's two review records) exist as regular files, no symlink on their paths, stated
  SHA256 and blob.
- **VERDICTS.** The quoted verdict rows and binding lines of the in-repo reviews (pr22, p15, side24, two-scale), the
  object headers of the sources, the D4 sentence that consumes the interface, the interface note's (N1)/(N4)/(N5)
  displays, the five PROOF_INDEX rows, the interface review's overall verdict and evidence labels, and the seven comment
  ids with their verdict lines and the mutability label in `EXTERNAL_REVIEWS.md` are present as substrings.
- **BASELINE.** Each transition's live node exists with exactly the recorded current classification, fingerprint,
  kind and source; the D1 node is `PROVED_REVIEWED` with its four reading-rule components `PROVED_REVIEWED`.
- **TRANSITIONS.** `declarative` true, `executed` false, `edges_unchanged` false; every proposed classification is
  `PROVED_REVIEWED` with `controlling: false`, a non-empty scope, explicit limits, replacement notes and at least one
  review source; every candidate node's fingerprint equals the SHA256 of its inventoried source; every component node
  is a `reading_rule_component`, `PROVED_REVIEWED`, non-controlling, fingerprinted to its inventoried source, with a
  role, a provider, a review basis and a consumer, one node per byte identity and none for bytes the live graph already
  carries; every added edge is well formed, new, and leaves one of this record's objects; applied to a copy of the live
  graph, no live node other than the eight changes, no live edge is removed, every flipped node's required premises (the
  live D1 node and cap component, the interface, the component nodes) are `PROVED_REVIEWED`, every region requires its
  covering node, and every applied node carries its `explicit_limits`, its replacement `notes` and a `review_basis`.
- **GATE.** The proposed graph passes the hard gate's `validate_graph_fail_closed`; `closure_report` gives `gate_ok`
  with no illegal controlling node; `reverse_impact_between(live, proposed)` runs and its impacted set (the eight nodes,
  the eight component nodes and their dependents) is recorded.
- **PROPAGATION.** Nine two-commit cases on the proposed graph with source snapshots in the shape of
  `git_transition_audit.read_snapshot`: a D4 proof edit reaches the fixed-remote region and the mesoscopic consumer; an
  annulus proof edit reaches its region; deleting the interface review record reaches the interface, D4, the annulus and
  both regions; editing the external-review record reaches D2, D3, D4, D6 and the fixed-remote region; editing the
  two-scale addendum reaches the annulus and its region; editing the realized-covers source reaches D6; a D4 fingerprint
  edit reaches the region; an unrelated edit reaches none of the eight; a no-op impacts nothing.
- **INTERFACE.** The finite content of §3, in two labelled parts: exact rational checks (Hölder for `p = 2, 3` on
  finite spaces, the (N2) exponent identity, the (N3) identity, the sharp-`p` example at `r = 2^(−pm)`, the integer
  monotonicity in (N4)) and floating-point numerical corroborations (the (N4) grid maximum, the (N5) Simpson quadrature
  at three parameter points, the (N4) example's uniform tail on a grid). The analytic arguments are in the review; the
  numerical part corroborates, it does not prove.
- **NEGATIVES.** Real filesystem faults on a temporary copy of the inventory (delete, change one byte, symlink, symlink
  parent of the D4 proof) are rejected.
- **D4_DUPLICATE.** The live graph carries `math.d5-component.remote-window-proof` as `PROVED_REVIEWED`,
  non-controlling, with the same SHA256 fingerprint and source as the D4 transition and as the inventory, a `review_basis`
  citing main#76 with verdict ACCEPT, `component_of` `math.d5-pin-neighborhood-first-moment`; and the older live node
  `math.rn-fixed-remote-window` still carries that fingerprint at `AUTHOR_SIDE_CANDIDATE`, the lag this record aligns.

Twelve mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,
`executed-flag`, `skip-interface-premise`, `holder-reversed`, `baseline-drift`, `duplicate-drift`, `region-unlinked`
(the `covered_by` edges removed before the propagation replay: proof edits no longer reach the regions) and
`review-metadata-only` (the review-record nodes and their edges removed before the replay, leaving the citations as
metadata only: a deleted or edited review record no longer reaches its accepted object).

