## 4. Proposed transitions

Eight node transitions and, since v1.3, the evidence that carries them, represented in the graph itself: eight
fingerprinted review-record / consumed-source component nodes and sixteen required edges (`PROPOSED_TRANSITIONS.json`:
`transitions`, `proposed_graph_nodes`, `proposed_graph_edges`; `edges_unchanged: false`). The change of packaging answers
OpenAI 5360320845 item 1, OpenAI 5360327058 items 1–2 and Codex threads 4139807558/4139807562/4139807567: the production
loss-only audit (`git_transition_audit.read_snapshot`) reads `node.source` only, so acceptance evidence and consumed
sources that live only in `review_sources` metadata would never revalidate a promoted object when they change. The eight
classifications and every scope exclusion are unchanged from v1.0.

| Node | Current | Proposed | Scope (STATUS row) | Review sources | Required premises after |
|---|---|---|---|---|---|
| `math.rn-count-interface` | `AUTHOR_SIDE_CANDIDATE` | `PROVED_REVIEWED` | (N1)–(N5) and the §3 counterexamples as self-contained probability statements; (N6)–(N7) and §5 as formula-level implications conditional on the parent's regression law / §9 convention and on the cap support source | this packet (`RN_COUNT_INTERFACE_REVIEW.md`, a component node); main#76 items 2–3 | `math.uniform-matrix-cap-lifetime` and `math.d1-component.marked-cylinder-cap` (both live `PROVED_REVIEWED`; the note's consumed premises); `math.align-component.rn-interface-review` |
| `math.lifetime-remainder` | `AUTHOR_SIDE_CANDIDATE` | `PROVED_REVIEWED` | Theorem R, existential `O(1)` remainder | main#67 5841270276, 5841782206 | D1 (live `PROVED_REVIEWED`); `math.align-component.external-issue-reviews` |
| `math.side24-coefficient` | `AUTHOR_SIDE_CANDIDATE` | `PROVED_REVIEWED` | coefficient arithmetic, `d = 2, 3` | main#65 5841269490; in-repo Anthropic record | D1; `…external-issue-reviews`; `…side24-review-claude` |
| `math.rn-fixed-remote-window` | `AUTHOR_SIDE_CANDIDATE` | `PROVED_REVIEWED` | fixed `ρ`, `η`, between-pin window | main#76 5841783172, 5841861362 | `math.rn-count-interface`; `…external-issue-reviews` |
| `math.rn-region.fixed-remote` | `COVERED_BY_CANDIDATE` | `PROVED_REVIEWED` | the D4 region | as the covering node | `math.rn-fixed-remote-window` (new required `covered_by` edge) |
| `math.rn-fixed-annulus-window` | `AUTHOR_SIDE_CANDIDATE` (blob fingerprint) | `PROVED_REVIEWED` (SHA256 fingerprint) | `d = 2` fixed scaled annulus, window | `reviews/pr22_…/REVIEW.md` | `math.rn-count-interface`; `…pr22-annulus-review`; `…two-scale-addendum` (the S6–S21 stitching input, which itself requires `…two-scale-review`) |
| `math.rn-region.fixed-annulus-window` | `COVERED_BY_CANDIDATE` | `PROVED_REVIEWED` | the covering node's scope | as the covering node | `math.rn-fixed-annulus-window` (new required `covered_by` edge) |
| `math.p15-full-price` | `AUTHOR_SIDE_CANDIDATE` (label fingerprint) | `PROVED_REVIEWED` (SHA256 fingerprint) | realized family, `d_i ≥ 2` | `reviews/p15_…/REVIEW.md`; main#74 5842112010 | `…external-issue-reviews`; `…p15-review`; `…p15-realized-covers-setwise` ((P1), (P2), Theorem P2 setwise only, which itself requires `…p15-review`) |

Component nodes (`kind: reading_rule_component`, `PROVED_REVIEWED`, `controlling: false`, `fingerprint` = SHA256 of the
source; one node per byte identity; none duplicates a source the live graph already carries with a fingerprint):

| Node | Source | Role | Provider | Consumed by |
|---|---|---|---|---|
| `math.align-component.rn-interface-review` | `reviews/register_alignment_20260930/RN_COUNT_INTERFACE_REVIEW.md` (this packet) | review record | Anthropic (nonauthor) | `math.rn-count-interface` |
| `math.align-component.external-issue-reviews` | `reviews/register_alignment_20260930/EXTERNAL_REVIEWS.md` (this packet: seven issue comments transcribed with id, account, timestamps and body; mutable external evidence, labelled) | external review record | xAI/Grok via Cursor (posted by `cursor[bot]`) and owner; recorded by Anthropic | D2, D3, D4, D6 |
| `math.align-component.side24-review-claude` | `reviews/side24_v1_coefficient_claude_20260929/REVIEW.md` | review record | Anthropic (other session) | D3 |
| `math.align-component.pr22-annulus-review` | `reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md` | review record | xAI | annulus |
| `math.align-component.two-scale-addendum` | `frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md` (`079f9399…`, blob `89cae3a9`) | consumed proof: the S6–S21 stitching input ((S4) on `A ≤ |u| ≤ B`) | OpenAI | annulus |
| `math.align-component.two-scale-review` | `reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md` (`04cf4c98…`, blob `89882516`) | review record of the addendum (S6–S21 and (S3)–(S5) ACCEPT) | xAI | `…two-scale-addendum` |
| `math.align-component.p15-review` | `reviews/p15_full_price_nonauthor_20260926/REVIEW.md` | review record | xAI | D6, `…p15-realized-covers-setwise` |
| `math.align-component.p15-realized-covers-setwise` | `frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md` (`c0dbb821…`, blob `17388191`) | consumed interface: (P1), (P2), Theorem P2 setwise at `K ≥ K_H(d)` only, as the D6 review checked them; the note's `c_v ≤ p_v` price theorem is not included | OpenAI | D6 |

These are exactly the support dependencies the canonical claim manifest records for the annulus (`two-scale-s6-s21`)
and for the full-price theorem (`p15-realized-covers:setwise-interfaces`). The interface's two consumed premises are
the live D1 node (regression law `Q`, §9 Borel-mark convention, read with its four reading-rule components) and the live
cap component (`0bf922b9…`, the SHA256 the note quotes for its §5 support statement); the (N6) cross-reference to the
C6 Palm route's Lemma 5.1 in the review is for the reader and is not consumed.

Every proposed node keeps `controlling: false`; each candidate node's `fingerprint` becomes the SHA256 of its source in
§2, and `apply()` carries `scope`, `explicit_limits`, a replacement `notes`, `review_disposition`, `review_sources` and a
structured `review_basis` onto the node, so no promoted node keeps a "review open" or "covered by candidate only" note
(Codex 4139807551; OpenAI 5360327058 item 3). Region nodes keep `coverage_source` and gain a required `covered_by` edge
to their covering node, so that a change to the covering proof reaches the region in `reverse_impact_between` (Codex
4139807558; OpenAI 5360327058 item 1). No `hist.*`, `eng.*` or `regional.*` node moves; no live node other than the
eight changes; no live edge is removed; every added edge leaves one of this record's objects. The order in
`execution_order` creates the component nodes and edges first, then the premise, then each region after its covering
node; the D4 theorem node and its region are executed only after Math-#151 is on main (satisfied at `ec6db8c`): the D4
bytes are now carried by two live nodes on the same review, and the flip aligns the older one with the newer one; the
executing lane may then collapse `math.d5-component.remote-window-proof` onto the live node or keep both (xAI
5901912476, 5360240245). STATUS, PROOF_INDEX, SELECTOR_REGION and the catalog need no edit for this record: they
already say what the graph is being aligned to.

`alignment_check.py` applies the proposal to a copy of the live graph and replays it through the hard gate's own
validators (`validate_graph_fail_closed`, `closure_report`, `reverse_impact_between`, imported from
`frontiers/downstream_gate_20260925/hard_gate.py`): the proposed graph is well formed, no node is controlling,
`gate_ok` holds, and the loss-only reverse-impact proposal names exactly the changed nodes and their dependents.
`PROPAGATION` then runs nine two-commit mutations through `reverse_impact_between` with source snapshots in the shape of
`git_transition_audit.read_snapshot` (proof edits reach their regions and consumers; a deleted review record and an
edited external-review record reach the accepted objects and their dependents; consumed-source edits reach their
consumers; an unrelated edit and a no-op reach none of the eight), the guarantee OpenAI 5360320845 asked to be tested
rather than inferred from old-versus-new GRAPH metadata.

