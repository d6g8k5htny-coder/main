# PR #712 exact-head adversarial findings

Subject: 15e8f689d00ca03e28e9bc5d97b6aba3f19ec900.
Actual executor: OpenAI Codex, chat 01a103c8-7a8e-7461-8e9f-36e3b02757c5,
owner-delegated engineering. Scientific effect NONE. Shared principal; no
organizational-independence claim.

The original 43-case suite passes in normal and optimized parent interpreters
(67.917s / 67.824s, exit0). It includes the designated 35-mutant controls.
That preserves the value of the original evidence. Its mutation children always
use normal -B; an optimized parent is not proof of optimized mutant children.

The new six-case suite fails by six assertions and zero errors in BOTH modes.
Production source and existing tests are byte/stat unchanged before and after.
No Lean, mathematical proof, live private federation or canonical gate was
executed or changed by these fixtures.

## Reproduce

From tests/: python3 -B -m unittest test_cbz_drift_adversarial_01a103c8 -v
and python3 -B -O -m unittest test_cbz_drift_adversarial_01a103c8 -v.

These are EXPECTED FAILURES on the recorded subject. They must be repaired at
source and rerun; this packet is not a green admission artifact.

## R1: incomplete transitive graph and skipped scans (blocking, P1)

crawl.py seeds once and snapshots list(self.crawled()) before its Lean pass.
Lake handling considers a dependency only when it is already in the repository
or local-clone maps; it then opens that dependency AFTER declaration comparison.

Three real-Git cases establish distinct failures:

- A declared file-remote Lake dependency absent from search directories is
  ignored even with clone-on-demand enabled.
- A locally held Lake dependency is opened and reported resolved, but its changed
  shared definition is not scanned.
- Pins/.gitmodules/workspace records reached inside a child are not traversed.

The actual CLI case returns **exit0 / 0 BLOCK / five repositories** while
scanning only two Lean files. The third Lean repository contains Fed.f = 7;
the first two contain Fed.f = 3. All are literal definitions inside the
claimed lexical comparison domain. This is not an elaborator limitation.

Expected repair: reach a bounded graph fixed point; read every discovered
repository's supported manifests/registries; validate and scan each selected
repository/ref exactly once; terminate cycles without dropping new nodes.
Unreachable declared edges must be reported, not omitted.

## R2: unresolved Lake revisions pass pin sync (blocking, P1)

A Lake dependency on an available repository records an absent forty-hex revision.
The result contains no PIN-MISSING/BLOCK because the Lake path compares revision
strings but never calls object resolution. A different existing historical
revision may be informational; a nonexistent revision is a broken pin.

## R3: absent declared canonical input drops artifact coverage (blocking, P1)

A workspace explicitly names fed/reg:absent-registry.json as its canonical
repository registry. The path is absent. The crawl emits only an informational
observation and returns blocking = 0; artifact traversal disappears.

Expected repair: distinguish optional historical observations from required
current inputs. A missing declared current registry is not a successful
zero-artifact verification.

## R4: current-tip qualification lacks refresh (G7 applicability gap, P2)

An existing full clone stays at cached default ref A after its local file remote
advances to B. Repo(fetch=True).default_tip() still returns A. This is an actual
reproduction. The README explicitly describes missing-object-only fetch, so the
finding does not claim that narrow implementation violates its own documented
fetch behavior. It means that command alone cannot qualify CURRENT tips.
Refresh default refs or bind an independently fresh ref snapshot for the final
gate; do not relabel cached refs as current.

## Additional admission boundaries

The delivery intentionally compares literal Lean source, not elaborated ASTs.
Ambient section/universe context, generated declarations and general Python/JSON
type contracts remain outside that scope. Literal structure fields ARE compared.
The full owner addendum therefore requires complementary typed coverage and
explicit unsupported results; this component alone cannot establish it.

Workflow/configuration enforcement is separate: PR path filters omit several
topology/registry-only changes. The independently reported ruleset observation
is at PR712/6510934948; legacy protection remained UNKNOWN after403. No hosted
bypass was demonstrated by this packet, and a green job alone is not proof that
every relevant federation merge is blocked.

## Evidence

RESULTS.json records actual source hashes, times, commands and exits. Raw logs
remain privately retained. Published stderr replaces only the checkout prefix
with $CHECKOUT; these normalized files do not claim raw-byte identity. Raw
SHA-256s are in RESULTS.json. The CLI report is summarized without transient
filesystem paths.

B-003 shared canonical files and the implementation branch remain untouched.
Route these tests to the actual crawler writer, repair, then rerun this same
case set on the frozen successor. All original failures remain historical.
