# Reproduce the federation checkpoint

This is a reproduction guide for the Partial Baseline. It is not a final G9
PASS, a complete mathematical auditor, or a CBZ-0 declaration. STATUS.md is the
entry point; it states the closed and quarantined domains and immutable rules.
No final baseline vector/tag or single complete G0–G11 run exists at this cut.

## Inputs and layout

Use Git, CPython 3.11, Node (the actual version is recorded by the Node runner),
and the repository-pinned Lean 4.34.1 toolchain. Use the exact dependency lock
files; do not substitute a newer Mathlib. Current checked Lean release commit:
5045d0056413266e57c625dcd7c365b10e377c52. Mathlib:
d13f23b723b8a846827a245b89c10fc7d3f11612. Each formal package owns its own build
outputs. Sharing a verified dependency package cache does not authorize sharing
or trusting another package's owned compiled objects.

The final source vector is external to the source trees and has exactly this
shape (all twelve values are full lowercase 40-hex commits):

    {"schema":"cbz.federation-vector.v1","repositories":{
      "main":"COMMIT","Math-":"COMMIT","governance-":"COMMIT",
      "Universal-Law-Workspace":"COMMIT","meta-framework":"COMMIT",
      "query-":"COMMIT","google-drive":"COMMIT","trial":"COMMIT",
      "sandbox":"COMMIT","d6g8k5htny-coder":"COMMIT",
      "openclaw-custodian-validation":"COMMIT",
      "research-hosted-qualification":"COMMIT"}}

The final vector must be the published exact vector named by BASELINE.md, not a
fresh selection of moving default branches. It is deliberately not embedded in
a commit whose own hash it would have to contain. Until that vector exists,
component commands establish only their explicitly recorded source cuts.

Clone complete history into twelve sibling directories with the names above,
all from https://github.com/d6g8k5htny-coder/NAME.git. Existing configured
credentials must authorize the two private repositories; credentials are never
written to a report. Public repositories alone are not the complete federation.
For a new empty parent directory, the commands are:

~~~sh
export CBZ_ROOT=/absolute/new/federation
export CBZ_EVIDENCE=/absolute/new/evidence
mkdir -p "$CBZ_ROOT" "$CBZ_EVIDENCE"
for repo in main Math- governance- Universal-Law-Workspace meta-framework query- google-drive trial sandbox d6g8k5htny-coder openclaw-custodian-validation research-hosted-qualification
do
  git clone --no-checkout "https://github.com/d6g8k5htny-coder/$repo.git" "$CBZ_ROOT/$repo"
done
~~~

Checkout each repository at its exact vector commit. Use detached checkouts;
do not substitute the current default branch or silently accept a missing
object. A shallow source needs complete history or explicit immutable fetches.
Run the finished federation-vector verifier's documented --root/--vector
commands after checkout; the network mode must observe all required source and
dependency commits. Its admission is separate from this guide.


Install the pinned elan release only after checking its archive digest; preserve
the existing TLS/proxy trust configuration:

~~~sh
curl -sSfL https://github.com/leanprover/elan/releases/download/v4.2.4/elan-x86_64-unknown-linux-gnu.tar.gz -o "$CBZ_EVIDENCE/elan-v4.2.4.tar.gz"
printf '%s  %s\n' 42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63 "$CBZ_EVIDENCE/elan-v4.2.4.tar.gz" | sha256sum -c -
mkdir "$CBZ_EVIDENCE/elan-installer"
tar -xzf "$CBZ_EVIDENCE/elan-v4.2.4.tar.gz" -C "$CBZ_EVIDENCE/elan-installer"
"$CBZ_EVIDENCE/elan-installer/elan-init" -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
elan toolchain install "$(cat "$CBZ_ROOT/main/formal/lean-toolchain")"
~~~

For each Mathlib formal package, run lake exe cache get in that package before
its documented fresh-owned-object gate. These commands use the checked-in
lake-manifest.json/lakefile pins. Never treat a fetched cache as evidence that
owned declarations were freshly built or kernel-replayed.

Historical display sources are intentionally older than the baseline. The
prepared Node runner reads exact Git objects and fails if any is absent.
Full public history supplies these objects; explicit fetches for a shallow
input are:

~~~sh
git -C "$CBZ_ROOT/main" fetch origin 71400b94f6cb354a8cf7aba73ffede2138a64efa
git -C "$CBZ_ROOT/main" fetch origin a12c178c0f857a130cf434e9efd44233a038195b
git -C "$CBZ_ROOT/main" fetch origin f2e432ea5c86742e480c66624775bc9103343314
git -C "$CBZ_ROOT/Math-" fetch origin d6628da09384728992dcbe6e921cc28ba85aebb0
git -C "$CBZ_ROOT/Math-" fetch origin 82247833b5dd58e04291d55353b52938d73ec614
git -C "$CBZ_ROOT/Math-" fetch origin 9d7b6802424fb4715b31999066aafca8ee2f3cca
git -C "$CBZ_ROOT/query-" fetch origin c88768bb11efd1f7d6bda188f13064bedec54a06
~~~

Historical proof/quarantine bindings also require their referenced Git objects,
including main tags cbz-quarantine-q-8336, cbz-quarantine-q-8340 and
cbz-quarantine-q-8429. Preserve them as historical inputs; do not check them out
over the selected baseline or import their mathematical assertions as proofs.

## Prepared Node regression

From the clean selected main root:

~~~sh
python3.11 -B -S -m unittest discover -s tests -p test_cbz_node_regression.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_node_regression.py -v
python3.11 -B -S tools/cbz_node_regression.py --math-root "$CBZ_ROOT/Math-" --query-root "$CBZ_ROOT/query-" --output "$CBZ_EVIDENCE/node"
~~~

The output directory must be new and outside all three source roots. The helper
verifies every historical descriptor before writing that fixture set, creates
separate shop/museum trees, and passes SHOP_MATH_FIXTURE, SHOP_QUERY_FIXTURE and
MUSEUM_FIXTURE to every tracked tests/*.mjs file. It clears NODE_OPTIONS and
NODE_PATH. It runs Node directly without a shell and retains node.tap plus
result.json. Only a real zero exit and a complete TAP summary with tests=pass>0
and fail=cancelled=skipped=todo=0 produce PASS. Missing objects, changed bytes,
ambiguous summaries, source/index changes and missing executables fail.

A RUNNING report is an incomplete attempt. File existence, a prior PASS, or a
partial log is not success. Reusing an output directory is rejected so a new
run cannot overwrite a retained failure. The helper checks all tracked main
bytes and modes against HEAD before/after, including changes hidden by
assume-unchanged, and binds unchanged HEAD. This is an endpoint check, not a
claim against malicious simultaneous writers, untracked ignored dependencies,
a hostile Git/Node/Python runtime, or arbitrary supplied transcripts.

## Gate ownership and commands

The following are exact commands for the existing component tools. Run from the
indicated selected repository and record actual source, output, exit and time.
A component PASS does not qualify a global gate. All final gates still require
the final vector, full declared domain and one observed final run.

| Gate | Executable component / exact command | Remaining final obligation |
|---|---|---|
| G0 | main: python3.11 -B -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_auditor_hardening tests.test_cbz_layer_a_scan_mutants tests.test_cbz_a3_float_census; repeat with -O | All admitted certifier self-tests/mutations, including package compiled auditors |
| G1 | main: python3.11 -B -S tools/cbz/layer_a_scan.py --git REPOSITORY --rev EXACT_COMMIT --fail-on all --json EXTERNAL.json; Python signal inventory: python3.11 -B -S tools/cbz/a3_float_census.py --git REPOSITORY --rev EXACT_COMMIT --json EXTERNAL.json | Complete pin-aware owned/child domain and semantic mathematical review; an AST zero does not certify exact arithmetic |
| G2/G3/G4 | main: python3.11 -B -S tools/formal_gate_check.py --run-lean; Math-: LEAN_NUM_THREADS=2 python3.11 -B -S formal/gate.py --execute | All declared closed packages, exact compiled inventories/types/axioms, no receipt-source substitution |
| G5 | main: python3.11 -B -S -m unittest discover -s tests -v; repeat with -O; prepared Node command above; Math-: python3.11 -B -S -m unittest discover -s tests -v; repeat with -O | Full active federation suite and explicit dormant quarantine contract, optional/container coverage and all declared package suites |
| G6 | G0 source mutations; main replay helper: cd cbz/evidence/kernel/additional/replay-tool then python3.11 -B -S run_mutations.py --output NEW_EXTERNAL; Math downstream and meta-framework: documented run_validation.py campaigns | Every declared mutation target/threshold, intended assertion failures only |
| G7 | ULW finished tools/verify_federation_vector.py --root ROOT --vector EXTERNAL_VECTOR, then network mode --network | Final exact twelve-repository vector, eight child pins, nine Lean locks and required remote availability |
| G8 | Source census below plus direct reconciliation of STATUS/PROOF_INDEX/GOVERNANCE/TOOLING/BASELINE to the final observed reports | No current single tool proves semantic truth of every status sentence; final reconciliation must be recorded |
| G9 | New clones at published baseline tag/vector; read STATUS.md only, answer its eight entry-point questions, then execute the commands and retain all results | Final clone/tag test has not been performed by this guide |
| G10 | Each clone: git status --porcelain=v1 --untracked-files=all; git stash list; git worktree list --porcelain; git branch -a; git tag; GitHub current open PR/ref enumeration | Fresh complete remote enumeration, recoverable retirement and actual absence readback |
| G11 | New Node helper self-tests above; census/replay/canonical tests below; every modified tool's registered verification | Registry completeness and no abandoned tool stubs across the exact final source vector |

ULW gitlinks deliberately fail the standalone lexical scanner. Do not suppress
or skip them. A passing pin check and independent child scans must be composed
by an admitted pin-aware domain tool before claiming global G1. This composition
was a recorded tool gap at this checkpoint.

Source census, from main (output outside all twelve clones):

~~~sh
python3.11 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -S tools/cbz_inventory.py generate --root "$CBZ_ROOT" --classifications cbz/classifications.json --output "$CBZ_EVIDENCE/inventory.json"
python3.11 -B -S tools/cbz_inventory.py verify --root "$CBZ_ROOT" --manifest "$CBZ_EVIDENCE/inventory.json"
~~~

Census generation and verify establish their stated byte/binding properties,
not kernel execution authenticity, complete obligation semantics or CBZ green.
Replay helper self-tests use the following commands from
cbz/evidence/kernel/additional/replay-tool:

~~~sh
python3.11 -B -S -m unittest -v test_replay_tool.py
python3.11 -B -O -S -m unittest -v test_replay_tool.py
python3.11 -B -S run_mutations.py --output "$CBZ_EVIDENCE/replay-python"
~~~

Actual Lean guard controls and six generated auditor mutations use the exact
installed-Lean/collector arguments in that directory's README.md. Exact package
build/replay commands are in additional/HOW_TO_REPLAY.md and
determinant/HOW_TO_REPLAY.md. Those package boundaries are retained; no default
Mathlib rebuild or reused owned olean is a substitute.

## Regression quarantine contract

The archived trial suite is Q-9001. It reproduced 87 failures in each isolated
mode; this is retained red evidence, not an active green suite. Preserve every
test and source. It remains dormant until deliberately reopened and contributes
no closed mathematical dependency. Q-9002 through Q-9006 record five unexercised
real-record paths: evidence URL shape, owner/released URL, artifacts, depends_on
and live-head freshness. Their real data/schema prerequisites are absent.
Never fabricate records, relax the production schema, delete these checks or
label them exercised to obtain zero skips.

The original main discovery had 24 skips per mode; 19 methods were actually
exercised separately, including all five real restricted-container controls.
The remaining five are precisely the quarantined coverage gaps above. A final
regression contract must reconcile these recorded outcomes explicitly; this
guide does not relabel a skipped test as a passed test or declare G5 green.
