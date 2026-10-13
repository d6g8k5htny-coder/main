import pathlib, sys, hashlib, json
W = pathlib.Path(sys.argv[1])
def patch(path, pairs):
    p = W / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, ("MISS", path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s); print("patched", path)
# ---- Math-#167 record
patch("reviews/register_alignment_20260930/RECONCILIATION.md", [
("""`RESULTS_INSTALLED.json` in the installed state; two mutants added (`installed-drift`, `partial-install`). Transitions,
sources and verdicts are unchanged.""",
 """`RESULTS_INSTALLED.json` in the installed state; two mutants added (`installed-drift`, `partial-install`); after Codex
review 5365065925 on Math-#183 (thread 4143661511) the baseline state also pins the proposal-written fields of the eight
nodes to their digests at `dda8991`, so the proposal's metadata installed under the current classification is a partial
install (mutant `text-preinstalled`; three added in all). Transitions, sources and verdicts are unchanged."""),
("""  exists with exactly the recorded current classification, fingerprint, kind and source, no component node and no
  proposed edge is live. *Installed:*""",
 """  exists with exactly the recorded current classification, fingerprint, kind and source, its proposal-written fields
  (scope, explicit limits, notes, review disposition, review sources, review basis, coverage source, source) carry the
  digest recorded at `dda8991` (`BASELINE_TEXT`), no component node and no proposed edge is live. *Installed:*"""),
("""Fourteen mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,""",
 """Fifteen mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,"""),
("""simulated installed register with one component fingerprint changed) and `partial-install` (the simulated installed
register with the D4 node back at its current classification). The workflow runs the checker in the state the""",
 """simulated installed register with one component fingerprint changed), `partial-install` (the simulated installed
register with the D4 node back at its current classification) and `text-preinstalled` (the baseline text-field rule
disabled: the pre-image of the applied graph, which carries the proposed metadata under the current classifications,
would pass as baseline; every run asserts that it is partial). The workflow runs the checker in the state the"""),
])
# ---- Math-#160 record
patch("reviews/c6_witness_collision_reconciliation_20260929/RECONCILIATION.md", [
("""proposed edge not leaving the witness node live; the witness node `OPEN_ACTIVE` while step 4 is pending, or
`PROVED_REVIEWED` with all its proposed evidence edges live and their required targets `PROVED_REVIEWED`), rejects partial""",
 """proposed edge not leaving the witness node live; the witness node `OPEN_ACTIVE` with none of its proposed edges live
while step 4 is pending, or `PROVED_REVIEWED` with all of them live and their required targets `PROVED_REVIEWED`), rejects partial"""),
("""with `RESULTS.json` in the baseline state and with `RESULTS_INSTALLED.json` in the installed state; two mutants added
(`installed-drift`, `partial-install`). Nothing in §§1–7 changes.""",
 """with `RESULTS.json` in the baseline state and with `RESULTS_INSTALLED.json` in the installed state; two mutants added
(`installed-drift`, `partial-install`); after Codex review 5365065925 on Math-#183 (threads 4143661492, 4143661519) the
installed selector table is accepted only when the graph carries the proposal, and the open witness node may carry none
of its proposed edges (mutants `selector-ahead`, `witness-edges-partial`; four added in all). Nothing in §§1–7 changes."""),
("""  the residual as proposed or as Math-#173 promotes it, the witness node open or `PROVED_REVIEWED` with its evidence
  edges live); once installed""",
 """  the residual as proposed or as Math-#173 promotes it, the witness node open with none of its proposed edges live, or
  `PROVED_REVIEWED` with all of them live; one witness edge toggled is asserted partial in every run); once installed"""),
("""  accepted as installed when the three region ids are covered and not open; a table carrying part of the proposal is
  rejected; the node-only alternative of §7 step 2 (table untouched) is the baseline case.""",
 """  accepted as installed when the three region ids are covered and not open and the graph carries the proposal (a table
  ahead of the graph is rejected, and the rule itself is asserted in every run); a table carrying part of the proposal
  is rejected; the node-only alternative of §7 step 2 (table untouched) is the baseline case."""),
("""Twelve mutants must fail: `allow-symlink`, `no-hash`, `drop-edge`, `stale-fingerprint`, `executed-flag`,""",
 """Fourteen mutants must fail: `allow-symlink`, `no-hash`, `drop-edge`, `stale-fingerprint`, `executed-flag`,"""),
("""simulated installed register with the [PALM] node's fingerprint changed) and `partial-install` (the simulated installed
register without the supersession node's edges). The workflow runs the checker in the state the repository is in: it""",
 """simulated installed register with the [PALM] node's fingerprint changed), `partial-install` (the simulated installed
register without the supersession node's edges), `selector-ahead` (the selector-to-graph tie disabled) and
`witness-edges-partial` (the witness-edge count ignored: one edge toggled would pass). The workflow runs the checker in
the state the repository is in: it"""),
])
# ---- Math-#173 record
patch("reviews/c6_residual_closure_20260930/RECONCILIATION.md", [
("""differs between the two by construction); no other content changes; v1.5 at `02772ec` applies the direct-statement amendment""",
 """differs between the two by construction); after Codex review 5365065925 on Math-#183 (thread 4143661501) `LIVE` accepts
exactly two register states, baseline (residual absent or `OPEN_ACTIVE`, none of this record's edges, no component but
the cross-record palm node) and installed, so partial installs fail and the interim shape is a replay, not an accepted
live state (mutant `partial-install`); no other content changes; v1.5 at `02772ec` applies the direct-statement amendment"""),
("""  execution cannot turn this check red (Codex 4140125558); on the installed register that simulation is the live graph
  itself (v1.6).""",
 """  execution cannot turn this check red (Codex 4140125558); on the installed register that simulation is the live graph
  itself (v1.6). The live register must be baseline (residual absent or `OPEN_ACTIVE`, none of this record's edges, no
  component except the cross-record palm node) or installed; anything else, the interim shape included, fails; the
  installed graph without one component is asserted partial in every run (v1.6)."""),
("""Fourteen mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-required-edge`, `executed-flag`,""",
 """Fifteen mutants must fail: `partial-install` (partial installs labelled baseline), `allow-symlink`, `no-hash`,
`stale-fingerprint`, `drop-required-edge`, `executed-flag`,"""),
])
# ---- readiness README
patch("reviews/register_execution_readiness_20260930/README.md", [
("""  covered. Math-#167's `apply()` is idempotent and its pre-image `strip()` lets the hard-gate replay run from the register
  without the proposal in either state; once installed, `apply` is the identity and the record's objects are unheld.""",
 """  covered. Math-#167's `apply()` is idempotent and its pre-image `strip()` lets the hard-gate replay run from the register
  without the proposal in either state; once installed, `apply` is the identity and the record's objects are unheld.
  After Codex review 5365065925 on Math-#183: Math-#167's baseline also pins the eight nodes' proposal-written fields
  to their digests at `dda8991` (metadata installed under the old classification is partial); Math-#160 accepts the
  installed selector table only when the graph carries the proposal, and the open witness node with none of its
  proposed edges; Math-#173 accepts exactly baseline or installed (its interim shape is replayed, not accepted live)."""),
("""- **Mutants:** `installed-drift` and `partial-install` added to Math-#167 and Math-#160; Math-#173's
  simulation of the installed state (and its `installed-drift` mutant) now runs on the installed register too. Every
  mutant of every checker is rejected on the baseline register and on the executed copy (`--mutants`).""",
 """- **Mutants:** `installed-drift`, `partial-install` and `text-preinstalled` added to Math-#167; `installed-drift`,
  `partial-install`, `selector-ahead` and `witness-edges-partial` to Math-#160; `partial-install` to Math-#173, whose
  simulation of the installed state (and its `installed-drift` mutant) now runs on the installed register too. Each new
  rule is asserted by a permanent self-test in every run, and its mutant disables the rule, so it is rejected in either
  register state. Every mutant of every checker (44) is rejected on the baseline register and on the executed copy
  (`--mutants`)."""),
("""From the repository root (standard library only; a temporary copy is used unless `--out DIR`, outside the repository):""",
 """From the repository root (standard library only; a temporary copy is used unless `--out DIR`: a path outside the
repository, neither inside it nor one of its parents, that does not exist or is an empty directory; the script never
deletes anything). `--write-installed-results` (maintenance of this packet) writes the pinned installed outputs only
after every checker, both interpreter modes and, if requested, every mutant have passed."""),
("""- **Order.** Math-#167 at any time. Math-#160 steps 1–3 **before** Math-#173 (the palm-proof node's fields are
  Math-#160's; Math-#173 reuses it). Math-#160 step 4 and Math-#173's deferred edges together, or step 4 later: both
  shapes are accepted (§3).""",
 """- **Order.** Math-#167 at any time, in one commit. Math-#160 steps 1–3 **before** Math-#173 (the palm-proof node's fields
  are Math-#160's; Math-#173 reuses it); the witness node's own evidence edges belong to step 4 (none of them while it is
  open, all of them once it is `PROVED_REVIEWED`). Math-#173 in one commit (steps 1–2 together: the interim shape with the
  residual still `OPEN_ACTIVE` is not an accepted live state). Math-#160 step 4 and Math-#173's deferred edges together,
  or step 4 later: both shapes are accepted (§3)."""),
("""## 5. Exposure and independence""",
 """## 5. Review of this packet

Codex review 5365065925 on Math-#183 (six findings, all taken): selector table accepted ahead of the graph (4143661492);
baseline nodes not checked on the proposal-written fields (4143661511); Math-#173 labelling partial component installs
baseline (4143661501); the open witness node accepting a subset of its edges (4143661519); `--out` deleting an existing
path (4143661496); installed results written before validation (4143661525). Each checker rule now has a self-test in
every run and a mutant that disables it; the helper never deletes and writes only after a fully valid run.

## 6. Exposure and independence"""),
])
# ---- workflows
for name, old, new in (
    ("register-alignment.yml", "'installed-drift', 'partial-install']", "'installed-drift', 'partial-install', 'text-preinstalled']"),
    ("c6-witness-collision-reconciliation.yml", "'installed-drift', 'partial-install']", "'installed-drift', 'partial-install', 'selector-ahead', 'witness-edges-partial']"),
    ("c6-residual-closure.yml", "mutants = ['allow-symlink',", "mutants = ['partial-install', 'allow-symlink',"),
):
    patch(".github/workflows/" + name, [(old, new)])
# ---- re-pin Math-#173's own bytes
D = W / "reviews/c6_residual_closure_20260930"
spec = json.loads((D / "PROPOSED_TRANSITIONS.json").read_text())
rec = next(c for c in spec["proposed_graph_nodes"] if c["id"] == "math.c6r-component.residual-closure-record")
old_sha = rec["fingerprint"]
src = (D / "residual_check.py").read_text()
import re
m = re.search(r'"reviews/c6_residual_closure_20260930/RECONCILIATION.md": \[\n\s+"([0-9a-f]{64})",\n\s+"([0-9a-f]{40})"', src)
assert m and m.group(1) == old_sha, (m and m.group(1), old_sha)
old_blob = m.group(2)
data = (D / "RECONCILIATION.md").read_bytes()
sha = hashlib.sha256(data).hexdigest(); blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
for name in ("residual_check.py", "PROPOSED_TRANSITIONS.json"):
    p = D / name; s = p.read_text(); n1, n2 = s.count(old_sha), s.count(old_blob)
    p.write_text(s.replace(old_sha, sha).replace(old_blob, blob)); print(name, "repinned sha x%d blob x%d" % (n1, n2))
print("173 RECONCILIATION sha", sha, "blob", blob)
