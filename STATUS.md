# CBZ-0 execution status

## Later Main and pin observation — 10 November 2026, 01:48 UTC

Main #787 merged as `6d928a3216233fb75dc717ee0b2d2e4bb08db266`, adding a
bounded SDGA source and declared-protocol library. Its classifications remain
BLOCKED; it does not authenticate Lean execution, establish a kernel result or
change scientific status. Workspace #66 had pinned the preceding Main
`ae8de850475f6924aef0abe0fa3670f5e4360751`; the later Main merge makes that
pin stale and requires another guarded refresh. Main `ae8de850` Pages run
`42736437394` subsequently observed public bytes and appended receipt
`81b3998534827e2c2e45188a1a52672aa6666001`; this is a publication observation,
not a complete cold-start or scientific acceptance.

## Historical tag custody and Pages retry — observed 10 November 2026, 01:25 UTC

A candidate custody record preserves selected Git objects and source files from
five Math and two Main non-ancestor tags; the tags themselves remain in place.
Its nine-manuscript index records exact blobs, tag/commit bindings and literal
heading locations. It does not admit current proof or quarantine status,
classify all obligations or establish scientific review. The earlier 01:09
Pages deployment failure remains historical: retry job `129563572226`
succeeded and observed the exact public candidate, appending receipt
`7e88f668216e5dcdf9aa8d3e957d2e3e87428a2f` after prepared receipt
`443fe72d76d05671de5dd9079ee8a17c0582eeab`. This publication observation is
for Main `da92c8290d99b7ab1db686ce9294e20c915f01a7`, not a final CBZ gate.

The scoped tag readback commands, from Main, are:

```sh
(cd cbz/evidence/tag-custody-20261110 && sha256sum -c SHA256SUMS)
python3 -B -S cbz/evidence/tag-custody-20261110/verify_math_tag_native_custody.py cbz/evidence/tag-custody-20261110/math-tag-native-custody-20261110.tar.gz
python3 -B -O -S cbz/evidence/tag-custody-20261110/verify_math_tag_native_custody.py cbz/evidence/tag-custody-20261110/math-tag-native-custody-20261110.tar.gz
python3 -B -S cbz/evidence/tag-custody-20261110/verify_main_tag_native_custody.py cbz/evidence/tag-custody-20261110/main-tag-native-custody-20261110.tar.gz
python3 -B -O -S cbz/evidence/tag-custody-20261110/verify_main_tag_native_custody.py cbz/evidence/tag-custody-20261110/main-tag-native-custody-20261110.tar.gz
```

To rerun the two specified adverse campaigns without writing inside the
checkout, run from Main and retain each actual output and exit:

```sh
CBZ_TAG_TEST_DIR=$(mktemp -d) || exit 1
cp cbz/evidence/tag-custody-20261110/* "$CBZ_TAG_TEST_DIR/" || exit 1
python3 -B -S "$CBZ_TAG_TEST_DIR/test_tag_custody_adverse.py" after-repair || exit 1
python3 -B -S "$CBZ_TAG_TEST_DIR/test_tag_custody_code_mutations.py" || exit 1
```

The initial one-off verifiers falsely accepted corrupted source under `-O`
and some duplicate or relabeled bindings; the original failures are retained.
The repaired bounded campaign accepted four clean controls, rejected 28 altered
inputs and caught eight executions of two designated helper-code mutants. These
checks and their limits are detailed in [TOOLING](TOOLING.md#historical-tag-custody-verifiers--10-november-2026)
and the [source record](cbz/evidence/tag-custody-20261110/README.md). They do
not establish complete changed-path enumeration, logical isolation or G0.

## Current handoff and gate limits — observed 10 November 2026, 00:57 UTC

**Active intermediate baseline. CBZ-0 and terminal Partial are not established.**
The observations in this section are time-stamped, not promises that refs stay
fixed. The dated sections below retain earlier observations, including headings
that used “current” or “candidate” at the time; they do not override newer
observations. Before integrating anything, re-read actual refs, open PRs and
required checks. A Main commit containing this handoff advances Main, so the
Workspace pin must be compared with the resulting default tip again.

The non-atomic source observation at 00:56–00:57 UTC was Main
`da92c8290d99b7ab1db686ce9294e20c915f01a7`, Math-
`9bdfc54b76173b57912503bfcf901a5f9d43b09c`, and Workspace
`a89cf416ce1741af4153c8ce389ae864a0a7245f`, whose Main pin was `da92c829`.
The other nine observed defaults were:

| Repository | Observed default commit |
|---|---|
| `meta-framework` | `0f5894f81a5462c10ee5ec8f8109f466e712e47e` |
| `query-` | `fb4c457b5861dc49696fd0a8e11c7c405056c05f` |
| `google-drive` | `ad08584b3ea9d533af10113e85169f471fa8ec93` |
| `trial` | `3738c859d519d2dbf1e388fe48fe0ae60c9487a3` |
| `governance-` | `1b62ca2f64b3012c2fdbbf7f55ced236c3042891` |
| `sandbox` | `dce04a4174835107126ab4f36c199a2dc320f964` |
| `d6g8k5htny-coder` | `d59de18e5fb4d087ed7a2a1c1b68536a4b8b5f15` |
| `openclaw-custodian-validation` | `83e0ba034aa3999d12554c1583732573ec09e8da` |
| `research-hosted-qualification` | `a6b1e6a6039f9cac596ef130719f6104b4c5ecf1` |

The twelve reads were not an atomic federation snapshot. Main #787 and #788
were separate open drafts at this observation. A clone or successful component
check is not a complete repository, ref, PR, CI, administration or proof inventory.

Further observation at 01:09 UTC: Main `da92c829` current-site build succeeded,
but its automatic Pages deployment job `129553245949` timed out while syncing
files and cancelled the deployment. The public-byte readback was skipped, so
that run does not establish current public bytes. Its prepared receipt commit
`443fe72d76d05671de5dd9079ee8a17c0582eeab` and state artifact `13725767427`
were retained; the failure remains part of the record.

The 66 determinant companions remain closed only for their exact conditional
Lean statements and original kernel/custody evidence. Their parent continuum,
Palm, lifetime and scientific applications are not thereby closed. At Main
`da92c829`, the formal component reported 59 named targets with empty axiom
lists and 18 rejection controls; this is not whole-federation qualification or
scientific alignment. The preserved, closed-unmerged #695 catalog is historical,
not an admitted set of current quarantines. Its 184 rows include six coverage
records without an explicit quarantine status; the remaining rows have not
been fully reconciled with current mathematical dispositions and logical-isolation
evidence. The three cusp archive
tags and their original source bytes are preserved in the custody record below.

Layer A remains binding: do not present placeholders as closed; admit no
unchecked axioms; certify bounds by exact rational or justified real/interval
arguments, not floating point; do not weaken bounds or silently disable
failures; mark vacuous regimes; report actual sources, executions and limits;
and test certification tools against adverse inputs without silent skips.
Kernel evidence establishes its exact statements and axioms, not scientific
alignment or independent review.

### What can be run now

The observed source cut includes twelve repositories: `main`, `Math-`,
`Universal-Law-Workspace`, `meta-framework`, `query-`, `google-drive`, `trial`,
`governance-`, `sandbox`, `d6g8k5htny-coder`, `openclaw-custodian-validation`
and `research-hosted-qualification`. This list is not proof that no other
associated infrastructure exists. For a fresh source observation, choose new
absolute paths outside existing work and use complete, authorized clones:

```sh
export CBZ_ROOT=/absolute/new/federation
export CBZ_EVIDENCE=/absolute/new/evidence
export GIT_TERMINAL_PROMPT=0
mkdir "$CBZ_ROOT" "$CBZ_EVIDENCE" || exit 1
for repo in main Math- Universal-Law-Workspace meta-framework query- google-drive trial governance- sandbox d6g8k5htny-coder openclaw-custodian-validation research-hosted-qualification; do
  git clone "https://github.com/d6g8k5htny-coder/$repo.git" "$CBZ_ROOT/$repo" || exit 1
done
```

Some repositories may require existing credentials. A denied or missing source
is a failure to obtain evidence, never a skip. Record each actual commit and
recheck remote defaults; independently sampled refs are not atomic.

The commands below are **component checks**, not a complete or final G0–G11
runner. They require the installed project dependencies and pinned Lean toolchain.
Set `CBZ_ROOT` to the absolute parent of the twelve checkouts and `CBZ_EVIDENCE`
to a new absolute directory, then create that directory. At this observed cut,
both formal packages pin `leanprover/lean4:v4.34.1`; Math's formal package also
pins Mathlib. With an installed `elan`, prepare the pinned toolchain and Math
cache using:

```sh
elan toolchain install leanprover/lean4:v4.34.1
(cd "$CBZ_ROOT/Math-/formal" && lake exe cache get)
```

A fetched dependency cache is not proof that owned declarations were freshly
built or kernel-rechecked. From the Main root, run each line independently and
retain its actual exit status and output; the exit status of the last line is
not an aggregate result:

```sh
python3 -B -S tools/cbz_inventory.py generate --root "$CBZ_ROOT" --repositories main Math- Universal-Law-Workspace meta-framework query- google-drive trial governance- sandbox d6g8k5htny-coder openclaw-custodian-validation research-hosted-qualification --output "$CBZ_EVIDENCE/inventory.json"
python3 -B -S tools/cbz_inventory.py verify --root "$CBZ_ROOT" --manifest "$CBZ_EVIDENCE/inventory.json"
python3 -B -S tools/cbz/layer_a_scan.py --git "$CBZ_ROOT/main" --rev HEAD --fail-on all --json "$CBZ_EVIDENCE/lean-main.json"
python3 -B -S tools/cbz/layer_a_scan.py --git "$CBZ_ROOT/Math-" --rev HEAD --fail-on all --json "$CBZ_EVIDENCE/lean-math.json"
python3 -B -S tools/cbz/a3_float_census.py --git "$CBZ_ROOT/main" --rev HEAD --json "$CBZ_EVIDENCE/a3-main.json"
python3 -B -S tools/formal_gate_check.py --run-lean
python3 -B -S -m unittest discover -s tests -v
python3 -B -O -S -m unittest discover -s tests -v
python3 -B -S -m unittest tests.test_reader_links -v
python3 -B -O -S -m unittest tests.test_reader_links -v
python3 -B -S tools/cbz/g11_mutation_selftest.py --json > "$CBZ_EVIDENCE/g11-mutations-normal.json"
python3 -B -O -S tools/cbz/g11_mutation_selftest.py --json > "$CBZ_EVIDENCE/g11-mutations-optimized.json"
python3 -B -S tools/cbz/g11_census.py --rev HEAD --json "$CBZ_EVIDENCE/g11-census.json"
```

For the scoped auditor-origin and census mutation controls, still from Main,
create four separate empty directories and run both groups in both modes:

```sh
mkdir "$CBZ_EVIDENCE/adapter-normal" "$CBZ_EVIDENCE/adapter-optimized" "$CBZ_EVIDENCE/census-normal" "$CBZ_EVIDENCE/census-optimized"
CBZ_AUDITOR_ORIGIN_MUTATION_EVIDENCE_DIR="$CBZ_EVIDENCE/adapter-normal" python3 -B -S -m unittest tests.test_cbz_auditor_origin tests.test_cbz_inventory_origin tests.test_cbz_auditor_origin_mutations tests.test_cbz_auditor_origin_workflow -v
CBZ_AUDITOR_ORIGIN_MUTATION_EVIDENCE_DIR="$CBZ_EVIDENCE/adapter-optimized" python3 -B -O -S -m unittest tests.test_cbz_auditor_origin tests.test_cbz_inventory_origin tests.test_cbz_auditor_origin_mutations tests.test_cbz_auditor_origin_workflow -v
CBZ_AUDITOR_EVIDENCE_DIR="$CBZ_EVIDENCE/census-normal" python3 -B -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_auditor_hardening tests.test_cbz_layer_a_scan_mutants tests.test_cbz_a3_float_census -v
CBZ_AUDITOR_EVIDENCE_DIR="$CBZ_EVIDENCE/census-optimized" python3 -B -O -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_auditor_hardening tests.test_cbz_layer_a_scan_mutants tests.test_cbz_a3_float_census -v
```

The inventory reports only supplied obligation classifications; an empty list
does not mean no obligations. The Lean scanner is lexical and reports retained
negative fixtures under `--fail-on all`; that output needs precise fixture
isolation and cannot by itself clear Layer A. The Python census is also scoped.
The Main formal command requires the pinned toolchain and performs its own build,
kernel, axiom and rejection controls; it does not qualify other packages. Run
Math-'s formal component from its repository root with
`python3 -B -S formal/gate.py --execute` after installing its pinned dependencies.

From the Workspace root, run:

```sh
python3 -B -S tools/verify_pins.py --require-current
python3 -B -S tools/verify_canonical_inputs.py
python3 -B -S - "$CBZ_ROOT" "$CBZ_EVIDENCE/vector.json" <<'PYVECTOR' || exit 1
import json, pathlib, re, subprocess, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
names = ('main', 'Math-', 'Universal-Law-Workspace', 'meta-framework',
         'query-', 'google-drive', 'trial', 'governance-', 'sandbox',
         'd6g8k5htny-coder', 'openclaw-custodian-validation',
         'research-hosted-qualification')
heads = {}
for name in names:
    head = subprocess.check_output(['git', '-C', str(root / name),
                                    'rev-parse', '--verify', 'HEAD^{commit}'],
                                   text=True).strip()
    if re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('invalid commit for ' + name)
    heads[name] = head
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps({'schema': 'cbz.federation-vector.v1',
                             'repositories': heads}, indent=2) + '\n')
PYVECTOR
python3 -B -S tools/verify_federation_vector.py --root "$CBZ_ROOT" --vector "$CBZ_EVIDENCE/vector.json" --network --output "$CBZ_EVIDENCE/vector-receipt.json"
```

The generated vector records the twelve local HEADs; it is not a published
baseline or an atomic remote observation. Network verification must actually
complete and report its own scope. The first two commands check eight public
pins and four canonical input identities;
they do not establish all twelve source tips, complete Git hygiene or G7. For
each of the twelve clones, `git status --porcelain=v1 --untracked-files=all`,
`git stash list`, `git worktree list --porcelain` and `git for-each-ref` are
starting observations, not semantic dispositions of all remote refs, tags or
other hosts. Preserve the outputs and any failure.

| Gate | Observed boundary and remaining work |
|---|---|
| G0 | Auditor-origin controls and mutation workflows exist; the complete federation auditor and its adverse qualification are not established. |
| G1 | The lexical and Python censuses above are scoped; a verified full Layer A audit is not established. |
| G2, G3, G4 | The Main and Math formal components have bounded commands above; a fresh current-composition axiom, Lean and kernel pass across every admitted proof remains open. |
| G5 | Main navigation ran 2,620 tests with 24 explicit skips at #794's tested merge. Separate component passes do not make that run no-skip; a complete federation regression remains open. |
| G6 | Recorded designated mutation campaigns are source-specific. The declared federation-wide threshold has not been established in a final pass. |
| G7 | Workspace pin and canonical-input checks are available above; a current twelve-repository vector and dependency qualification must be observed together. |
| G8 | Reader-link tests are available above; complete consistency and current disposition checks for the five canonical surfaces remain open. |
| G9 | No STATUS-only fresh-clone exercise has yet established the boundaries and rerun every gate. There is no admitted command that runs the complete matrix. |
| G10 | Local Git commands above are starting observations only; complete ref, worktree, stash, tag, PR, CI and administration dispositions remain open. |
| G11 | The census and its designated mutations above are bounded tooling checks, not a final federation-wide tooling qualification. |

The final run must execute all applicable gates together on the same observed
source vector, retain actual logs and failures, and satisfy all twelve CBZ-0
criteria at once. Older receipts, conditional checks, skipped tests and this
handoff do not substitute for that run.


## Historical foundation custody — 10 November 2026

Workspace #64 merged as `a3ac03329c6c54f0e3df4f159d894b2a8e70de44`, pinning
Main `eed4e646565bf670d87b5ec41b9ae18f8a20d35a` and Math `9bdfc54b`.
Its hosted candidate contract ran 254 tests per mode without skips, and its
pin job checked eight current pins and four canonical inputs. These were
sequential observations; postmerge checks require their own observation.
This later Main candidate would require another guarded Workspace refresh.

The closed-unmerged Main #695 catalog is preserved as exact historical bytes,
not admitted as current proof status. Its 184 records contain the eleven
named fields; six are regression or real-record coverage records without an
explicit status field. A non-admitting Git-tree comparison at Main `c32e6f27`
and Math `9bdfc54b` found 354 identical and five changed entries among 359
directly parsed historical locations. This checks object identity, not the
meaning, isolation or completeness of a proof. The three changed directory
trees and two changed files are detailed in [the custody record](cbz/evidence/foundation-custody-20261110/README.md).

The three historical cusp archive tags are now indexed below in
[PROOF_INDEX](PROOF_INDEX.md#historical-cusp-archive-custody--10-november-2026),
and 87 exact source files from their named trees are retained as a compressed
archive. Their source review records are preserved as history; no new review,
scientific acceptance or current CLOSED disposition follows. Two unspecified
README-row paths and five real-record globs/live states remain unresolved.
To verify the retained file bytes from this checkout, run
`sha256sum -c cbz/evidence/foundation-custody-20261110/SHA256SUMS`;
this is custody verification, not a final G0–G11 gate.

## Lean option-member scanner correction — 10 November 2026

At published Main `c32e6f27d318e25708939105285366a7a0daf971`, the scoped
Lean scanner returned no findings and exit 0 on the two retained
`BadPublic.lean` and `BadPrivate.lean` negative fixtures, although each uses
`debug.skipKernelTC.set`. Its previous Main census therefore missed those two
fixture-local A2 signals; that observation and the failed regression controls
remain in [the evidence record](cbz/evidence/lean-option-member-20261110/README.md).
The fixtures are negative test data, not closed mathematical modules, and no
transitive isolation or federation-wide axiom clearance is inferred.

This candidate reports `skipKernelTC` in any dotted identifier component,
including escaped components, so a method suffix cannot hide the option name.
It conservatively reports uses such as `.get` as well as `.set`; it does not
resolve aliases or determine whether a use enables the option. Main source at
`c32e6f27` then yields three violations in the retained negative fixtures and
one `unsafe` review token across 18 Lean blobs; the 1,974-blob Math source
`9bdfc54b` yields no listed tokens. These are lexical observations, not Lean
compilation, kernel/axiom proofs, scientific acceptance, or full Layer A audit.
The candidate requires exact-head hosted testing and readback before admission.

Workspace #63 merged as `ca321af7df4b63a832b0a13c5929eab99c84a25a`, pinning
Main `c32e6f27` and Math `9bdfc54b`; its hosted contract ran 254 tests per
mode without skips and its pin job checked eight current pins and four
canonical inputs. Those observations were sequential, not a final G7 cut. A
later Main merge requires another guarded pin refresh. No CBZ-0 or terminal
Partial is declared.

## Auditor-origin repair and published pin cut — 9 November 2026

Main #791 merged as `05f2289478a0ee9492adc1eb501466e0f66866df`; its hosted
navigation run executed 2,609 tests with 24 skips, and the selected historical
#525 binding test passed in each mode. Workspace #62 merged as
`bda15fcd29d6d484f9b2e1233663dbd8fdd8b796`, pinning that Main commit and Math
`9bdfc54b76173b57912503bfcf901a5f9d43b09c`. The Workspace tested and
postmerge contract runs each executed 254 tests per mode without top-level
skips; pin checks covered eight current pins and four canonical inputs. These
sequential observations do not establish final G5 or G7.

At Main `05f22894`, actual normal and optimized subject-origin probes showed
that both Lean and Python-census mutation graders could count a subject's
`AssertionError` as a test-oracle kill when the subject was under `tests/`.
The original failures and source identities are preserved in
[the evidence record](cbz/evidence/auditor-origin-20261109/README.md). This
candidate uses live pre-cleanup code provenance and requires a clean baseline
for each source mutant; current-head qualification remains required. Earlier
mutation counts are historical observations, not sound final G0 credit.

The scoped Lean lexical scans found a deliberately unapproved Main test axiom,
an `unsafe` review token and no listed tokens in 1,974 Math Lean blobs; the
Workspace scan refused a gitlink. The Math Python census failed on one
intentionally preserved historical writer parse error. These are scoped
observations, not whole Layer A clearance or proof that float signals certify
bounds. Exact commands and limits are in [TOOLING.md](TOOLING.md#auditor-origin-controls).
No CBZ-0, terminal Partial or scientific acceptance follows.


**State: active intermediate baseline; CBZ-0 is not established. No terminal Partial Baseline is declared.**

This entry adopts the owner's expanded [Master Directive](governance/MASTER_DIRECTIVE_20261029_SWARM.txt)
and [team protocol](GOVERNANCE.md#team-structure-and-coordination).
The inherited scientific material below is explicitly the 29 September
snapshot, not a fresh CLOSED decision or an empirical-validation receipt.
Its mathematical statements, source cuts and review limits are preserved.

## Graph-binding recovery and pin observation — 9 November 2026

Main #790 merged as `27da96bd9546e1dcc08345ae2a0fc3842c0f5753` and
Workspace #61 merged as `35b4a63281697e1656c63a663d8e58eb8215d2c5`,
pinning that Main commit and Math- #559's
`9bdfc54b76173b57912503bfcf901a5f9d43b09c`. Workspace's tested merge
ran 254 tests per Python mode without skips and its pin job checked all eight
current pins and four canonical inputs. The source-bound observations are
sequential; a later Main merge requires another guarded pin refresh.

The prior Main navigation run executed 2,601 tests with 25 explicit skips,
including the historical #525 graph-binding composition. This candidate
fetches the exact Math object `dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde`
into a new bare Git repository, requires Main's historical data commit
`29b7d5969e2cd707debb7449c69a71bbcd39cf63`, and refuses a missing, skipped
or failed selected test in either Python mode. A local real-block replay
fetched the Math object, ran 37 normal binding tests and the selected
optimized test without skips. An earlier replay failed because its harness
started outside the Main repository; that failure remains retained. The
33 focused workflow and history controls passed per mode, including eight
actual guard-removal comparisons against synthetic subprocess outcomes.
Hosted execution of this candidate remains required. The earlier 25 skips
are not erased, and no full G5, final G7, mathematical closure or scientific
acceptance is inferred from this engineering check.

The [tooling entry](TOOLING.md#pinned-525-graph-binding-composition) gives
the exact standalone commands, source identities and supported scope.

## Later integrated origin and source-cut observations — 9 November 2026

Main #789 merged as `b0a6c12abe87ae2a43163b3fe6ba4fa14adf389d` and
Workspace #60 merged as `defb18ed4745432d3a2f1bc7a17cc28a1292a334`;
the latter pins that Main commit. Math default was separately observed at
`f532c3e2d91a0c7679a37c733741660917a9b21b` before #559 merged as
`9bdfc54b76173b57912503bfcf901a5f9d43b09c`; Workspace's Math pin therefore
requires refresh. These observations predate this documentation composition:
a later Main merge also requires a fresh guarded Workspace pin comparison.
The first #789 merge request was rejected while an
actual required `public-intake` check was in progress; after that check
succeeded, the normal expected-head retry merged without bypass.

At Main #789's tested merge `de9db13d21b53ab33dda09818e01d53979d90f8a`,
the hosted crawler ran 50 frozen controls and 23 grader controls per mode,
and its six shards and merges reported 109/109 designated source mutations
in each mode. The hosted prepared Node run had 33 helper controls and 767
passing Node tests per mode without skips, cancellations or todo cases.
Original artifact hashes, source identities, crawler provenance and Node TAP
and 39 fixture bytes per mode were independently checked. These are bounded
component results, not a no-skip whole-federation regression or final G0/G6.

Workspace #60 repairs a reproduced vector mutation false kill caused by a
subject under a directory named `unittest`. Its grader uses pre-cleanup live
code identity for the selected body, dispatch and standard unittest helpers;
arbitrary helpers and hostile runtime manipulation remain unsupported. At
tested merge `1f5ae4e0c507ee2133da8dc03f16f4e0a8da63b7`, hosted contract
run `42712020137` ran 254 tests per mode with no skips and recorded 13 clean
baselines and 13 designated source mutants per mode. Pin run `42712020429`
checked eight strict-current pins and four canonical inputs. The original
artifact and all 52 nonce/source/stderr-bound events were checked. These are
sequential observations, not a final G7 or G5.

A new offline vector run on a separately observed twelve-repository cut at
those Main, Math and Workspace commits passed its selected structural checks:
12 commits, 30 selected inputs, eight pointers, four canonical bindings and
nine lock packages. Receipt SHA-256
`d5a4672c7a07835a72d4edd68ddbd670e9b0bd24f8124b601359c3ab3619ddd5`.
Separate before/after API reads matched all twelve heads; they were not atomic.
The earlier cut first failed on a missing historical Main Git blob; after
recovering and verifying that exact object it passed offline. Its network run
fetched 20 of 22 exact commits and failed because two private Git transports
returned exit 128. No current private Git transport success, full tracked
working parity or Git hygiene is inferred.

The bounded remote inventory observed 2,078 branches and 39 tag refs across
the twelve visible repositories. Ancestry comparisons found 1,441 branch
tips ancestral to the captured defaults and 637 non-ancestral; neither result
decides semantic integration or deletion eligibility. The
[machine-readable observation](cbz/inventory/federation-refs-20261109.json)
records public refs and private aggregate counts; private branch names and
commit IDs are withheld from the public projection. Twenty-nine tag targets were ancestors of
the captured defaults; ten were not. Eight non-ancestors are explicitly
named preservation/quarantine snapshots and remain retained; non-ancestry
does not establish that any tag is unreferenced or safe to remove. The local
cut's clean statuses and empty stash lists cover only those checkouts and their
registered worktrees, not every clone or orphan worktree.

Math #559 merged as `9bdfc54b76173b57912503bfcf901a5f9d43b09c` after
tested merge `79ad170de796404f6a8cd5273ce0fc4229da361e` completed its
kernel and byte-integrity workflows. Its receipt records 66
literal target axiom lists, seven real rejection controls and no scientific
effect; the original conditional statements and prior outcomes are not
rebound. Full-regression run `42708585331` completed 1,144 normal and 1,144
optimized Math tests with no skips. This is not a full federation G5.
Neither the parent continuum claims nor independent scientific acceptance
follows.

At the separately observed Main #787 head `d561405d936e81d2f77e9308b4e176cb994c77aa`,
the hosted scope contract ran 23 tests per mode and failed two
submicrosecond-fraction controls in each. Its normal discovery ran 2,599
tests with two failures and 25 skips. At Main #788 head
`f5f301b652e222395dc45cf300ae6697c4fcf8ae`, the hosted offline contract
ran 23 tests per mode successfully, while the 18 CLI tests had 33 failures
in each mode; normal discovery ran 2,617 tests with 33 failures and 25 skips.
These are observations at those exact draft heads, not results for later
changes. Both deliveries remain separate open scopes; the original failure
artifacts are retained and neither is credited as a completed gate.

From a clean Workspace checkout, the bounded vector commands are:

```sh
python3 -B -S -m unittest discover -s tests -p 'test_verify_federation_vector*.py' -v
python3 -B -O -S -m unittest discover -s tests -p 'test_verify_federation_vector*.py' -v
python3 -B -S tools/verify_federation_vector.py --root /absolute/siblings --vector /outside/vector.json --output /outside/new-offline.json
python3 -B -S tools/verify_federation_vector.py --root /absolute/siblings --vector /outside/vector.json --network --output /outside/new-network.json
python3 -B -S tools/verify_pins.py --require-current
python3 -B -S tools/verify_canonical_inputs.py
```

For the vector commands, place twelve clean Git checkouts directly below the
absolute sibling root. The JSON file must have exactly `schema` set to
`cbz.federation-vector.v1` and `repositories` mapping these exact names to
their selected full lowercase 40-hex commit IDs: `main`, `Math-`,
`Universal-Law-Workspace`, `d6g8k5htny-coder`, `google-drive`, `governance-`,
`meta-framework`, `openclaw-custodian-validation`, `query-`,
`research-hosted-qualification`, `sandbox` and `trial`. The output must be a
new nonsymlink file outside all checkouts. Historical canonical objects must
also be locally present; an unavailable object or private remote is a real
failure, not a skip.

The vector is an explicitly selected immutable cut, not a default-tip or
scientific certificate. Full proof/quarantine reconciliation, complete Layer
A audit, no-skip regression, authenticated private transport, all final
G0–G11 gates and the STATUS-only cold start remain open. No CBZ-0 or terminal
Partial Baseline is declared.

## Later Node, inventory and pin observations — 9 November 2026

Before this source change, Main #785 merged as
`c81b7572f3c690338db9d66fdea582fc68f4a73f` and Main #784, preserving
#781's original Claude lineage, merged as
`1902aac113761974c4569f78acb6a8baadabb7f1`. Workspace #58 merged as
`8a4d2248e9646ba3181a61defc90d13a01fe08ac` and pins that Main. Its hosted
premerge federation job ran 187 tests in each Python mode without skips; its
pin job checked eight strict-current pins, four canonical inputs and 187 tests.
The postmerge federation job repeated 187 tests per mode without skips; its
pin job also passed eight strict-current pins, four canonical inputs and 187
tests. These were sequential observations. A merge of this Main candidate will
require another guarded Workspace comparison; they do not award final G7.

At #784's tested tree, the hosted prepared Node job ran 33 admitted helper
controls and 767 Node tests in each preparation mode, with no Node failures,
skips, cancellations or todo cases. It checked 39 fixture files per mode.
The original earlier 766/767 failure and the historical REJECTED alignment
remain unchanged. The unadmitted Node mutation/control launcher was not used;
these are not full G0/G6, a continuum proof or scientific acceptance.

At tested merge `8b4c8656bf35bdccfb9decddef29194ff68e6289`, hosted
run `42705939654` also executed 33 helper controls and 767 Node tests per
mode, with no skips, failures, cancellations or todo cases. Its 39 fixture
files per mode and TAP digests were independently checked. This observation
is scoped to that tested merge; later changes require their own qualification.

The Main hosted normal full regression ran 2,576 tests with 25 skips; this is
not a no-skip G5. Separately on native `2b93a6d2279e5f3edff0eb991e4fdb25d64a6b8f`
whose tree matches #785, ten pytest cases and the historical #525 binding
case passed in each Python mode, and two browser cases passed per mode in a
hosted postmerge job. Local browser attempts instead failed before execution
with socket permission errors. A separate live node-status check exited 1:
`eng.hard-gate` was stale while three other records were fresh. No historical
source pin was rewritten. Also on native `2b93a6d2`, two separate Lean 4.34.1
package runs reported 59 target axiom lists empty and 18 rejected negative
controls per mode; the receipt's scientific alignment remains pending. None
of these separate observations becomes a final same-pass federation gate.

A later A8 probe on Main `1902aac1` found that both the inventory and
node-status mutation-origin graders accepted a distinct code object equal to
an authorized unittest oracle as a kill. A separate freshness probe found
that setup or teardown could call the selected body and have its failure
credited as an oracle. A further selected test could relabel a runtime error
as its failure type and obtain a false kill. The previous execution results
and these red findings are retained. The candidate requires the actual
exception to be exactly AssertionError, uses object identity and execution-phase tracking,
and inventory rejects a skipped target; its status-isolation control again requires the
named failure and one failure. Arbitrary test-class helper methods remain unsupported, including helper
methods that perform assertions; a nested subject or callback is not an oracle.
The reference-crawler mutation grader on the same parent separately accepted
an assertion with a spoofed unittest filename, failed to propagate optimized
Python to its child and treated skipped/expected-failure tests as passes. Its
old failmaps and outcomes remain historical. The local candidate repairs
those paths and uses a new failmap schema, but fresh crawler controls and the
full designated mutation campaign must pass before it is treated as sound.

The local inventory suite subsequently ran 83 tests in each Python mode on
the candidate's inventory source, without top-level skips; its full-suite
mutation children explicitly retain four declared recursion-controller skips.
The expanded inventory recorder self-test killed 28 cases per parent mode,
covering both child modes and binding the actual helper, controller, controls
and freshness-checker hashes. A later driver-adverse attempt failed because
its isolated copy omitted two newly required files; after copying those files,
the driver refused both the no-op and subject-assertion probes in both parent
modes. The first failure remains retained. These are component observations,
not a no-skip whole-Main G5 or a final G0/G6.

On earlier native candidate `5b174715bfcf035a0ed9fe8ac78e5bdab1f7bc06`,
under CPython 3.12.14, the freshness grader controls ran 23 tests per parent
mode without skips, detecting ten grader mutations in both child modes; its
eleven original source mutations produced 22 qualified child kills per parent
mode. After the exception-type correction, the local CPython 3.12.14 suite
ran 24 tests per parent mode without skips, detected eleven grader mutants in
both child modes, and recorded 22 original freshness-mutation kills per parent
mode. The crawler grader controls ran 23 tests per
parent mode without skips, including 16 grader-source mutations per parent
mode. Hosted run `42705939370` at tested merge
`8b4c8656bf35bdccfb9decddef29194ff68e6289` ran 50 frozen tests and 23
grader controls per mode, with 16 grader mutations per mode; both merged
failmaps contain 109/109 designated source mutations and 50 selected tests.
The nine artifact ZIPs, source hashes and outcome records were checked. This
component result is bound to that tested merge, not to a later tree. Earlier normal-parent self-mutation timeout and
the first overlapping-guard mutation failure remain preserved.

At Math `f532c3e2d91a0c7679a37c733741660917a9b21b`, three mutation
graders also accepted equal-but-distinct unittest code as an oracle. Draft
Math #559 contains a separate identity-based correction and real adverse
controls; its complete fresh shared-source matrix and hosted kernel run remain
separate from the historical 66 conditional Lean statements and receipts.
Math run `42706816998` at tested merge `e8dfae1543f312b7569a0dac31932025b6f7f501`
captured all 17 expected hidden evidence files, 42 paired mutation records and
168 raw logs; it is byte integrity evidence, not kernel or scientific
qualification.

This component has a bounded synchronous-unittest domain, is not a hostile
sandbox and needs its own fresh tests and mutations. See
[TOOLING.md](TOOLING.md#later-provenance-and-prepared-node-tools--9-november-2026).

For the prepared Node component, use clean Main, Math- and query- checkouts
under `CBZ_ROOT`, with Math- at `f532c3e2d91a0c7679a37c733741660917a9b21b`
and query- at `fb4c457b5861dc49696fd0a8e11c7c405056c05f`. Set
`CBZ_EVIDENCE` to a new directory outside all three source roots, and ensure
Python 3.11 and Node 24 are available. Fetch these immutable historical
commits without changing the selected checkouts:

```sh
for sha in 71400b94f6cb354a8cf7aba73ffede2138a64efa a12c178c0f857a130cf434e9efd44233a038195b f2e432ea5c86742e480c66624775bc9103343314; do git -C "$CBZ_ROOT/main" fetch origin "$sha"; done
for sha in d6628da09384728992dcbe6e921cc28ba85aebb0 82247833b5dd58e04291d55353b52938d73ec614 9d7b6802424fb4715b31999066aafca8ee2f3cca; do git -C "$CBZ_ROOT/Math-" fetch origin "$sha"; done
git -C "$CBZ_ROOT/query-" fetch origin c88768bb11efd1f7d6bda188f13064bedec54a06
cd "$CBZ_ROOT/main"
python3.11 -B -S -m unittest -v tests.test_cbz_node_regression.NodeRunnerTests
python3.11 -B -O -S -m unittest -v tests.test_cbz_node_regression.NodeRunnerTests
python3.11 -B -S tools/cbz_node_regression.py --math-root "$CBZ_ROOT/Math-" --query-root "$CBZ_ROOT/query-" --output "$CBZ_EVIDENCE/node-normal"
python3.11 -B -O -S tools/cbz_node_regression.py --math-root "$CBZ_ROOT/Math-" --query-root "$CBZ_ROOT/query-" --output "$CBZ_EVIDENCE/node-optimized"
```

The runner checks selected source and fixture bytes at its endpoints, executes
all tracked Node tests and requires a complete zero-failure TAP summary. It
is not a hostile-writer sandbox or a final cold-start/federation gate. Its
runnable source, purpose, observed verification and limits are registered in
[TOOLING.md](TOOLING.md#later-provenance-and-prepared-node-tools--9-november-2026).
The whole-federation inventory, proof dispositions, no-skip regression,
complete tool audit and final G0–G11 pass remain open.

## Inventory provenance and pin refresh — 9 November 2026

At the sequential observation before this source change, Main #783 was signed
merge `7738e4c2e0881dd952772421877d9854364d272f`, tree
`486b7f3223301ffba716e6524a069b5058b188e3`; Workspace #56 was signed merge
`5dcb544c1847ea2a0b7724201f747627c2746235`, tree
`db4b46b06b43fed5c832696c68d04ac775044fd6`, pinning that Main. Workspace
pre- and post-merge federation jobs each ran 187 tests in both modes without
skips; the separate pin jobs checked eight strict-current pins and four
canonical inputs. These observations were not atomic, and a later Main merge
requires another guarded pin refresh. They do not award final G7.

Main #783's six observed post-merge workflows succeeded. Its ordinary site
publication recorded 33 snapshots, with prepared receipt
`7e515b3dbc74eb8ffaf45039b2472f6e0a7124b9` and observed receipt
`fd191f57d33cabc73aec8091df837d7695cb316a`; the restored receipt history is
retained in the evidence below. Publication is not scientific acceptance or
the STATUS-only cold-start gate.

The inventory mutation controllers now use a bounded recorder that captures
traceback code identities before unittest cleanup. In local CPython 3.12.14,
the revised inventory suite ran 83 tests without top-level skips in each mode,
with twelve authorized source-mutation kills and eleven passing baselines per
mode. The full-suite path explicitly accounts for four inner recursion skips.
A real subject-assertion contamination caused 83 full-suite failures in each
mode and was refused. Four recorder mutations were killed in each child mode
by the separately qualified node-origin checker; its driver also refused a
no-op and a subject assertion in both parent modes. Initial missed-mutation and
driver-parser failures are preserved. These are bounded synchronous unittest
observations, not a hostile-code sandbox or full G0/G6 qualification.

From a Main checkout, run the bounded recorder and inventory controls with:

```sh
python3 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3 -B -S -m unittest discover -s tests -p test_cbz_inventory_origin.py -v
python3 -B -O -S -m unittest discover -s tests -p test_cbz_inventory_origin.py -v
python3 -B -S cbz/evidence/inventory-origin-20261109/run_self_mutations.py
python3 -B -S cbz/evidence/inventory-origin-20261109/run_driver_adverse.py
```

The inventory CLI's source-state commands and limitations are in
[the inventory guide](docs/cbz/INVENTORY_RECOVERY.md). The exact source,
logs, native delta and receipt bundle were independently read back from
`libfile_fde3_f-a1d4cb841f208191aafdc61057043076_59a0f9f234e48191bc19fddcc0f80648`,
SHA-256 `bea63fdd7dc5e53174e2af4e08f3db91973546a45392def6763446f097f026d3`.
Earlier dated tip descriptions and unqualified-classifier observations below
remain historical. Whole-federation audits, no-skip regression, proof
dispositions, final G0--G11 and the STATUS-only cold start remain unawarded.

## Latest observed inventory integration — 9 November 2026

Main #780 merged at 19:26 UTC as `b1d36c80403acc96d257b9e71140a6cb83a595af`,
tree `10bc30f82661ebdd763384496afffbe7f575459a`, with parents `f1ef3a2b…`
and `24ad8a143862fb9255fc6e2e7bf08c12f989f71d`. Workspace #55 subsequently merged at 19:43 UTC as
`8a7aac8626438967830ea5a516a944ce48759018`, tree
`d8db9ab8dd764d6f564b1d46fee03990be7b770a`, with parents `1c552538…` and
`9f0fbbbb615c7d39207f7bfaef3fd53a7e561fc3`; its Main gitlink now pins
`b1d36c80403acc96d257b9e71140a6cb83a595af`. The tested merge
`238b379f671b0d0f0d4c471befdfa7dc1ebf4c35` had that exact tree. Hosted
job `129389209449` passed eight strict-current pins, four canonical inputs
and 187 tests; job `129389215646` ran 187 tests in each Python mode without
skips. Local strict pins and canonical inputs also passed; local normal
regression ran 187 tests without skips, while its optimized run was still in
progress at this observation. These are sequential observations, not final
G7. Post-merge jobs require their own observed outcomes. A later Main merge,
including this documentation change, requires another guarded pin comparison.

The recovered inventory CLI is present in the merged source, with commands and
scope in [TOOLING.md](TOOLING.md#recovered-tracked-source-inventory--9-november-2026).
Its ten historical designated mutation counts remain subject to the adverse
assertion-origin findings described below. The merge does not qualify those
classifiers as sound. No historical test or mutant outcome is reclassified.
The #780 author separately reported a Node run with 766 passes and one failure
among 767 cases; this report is not a passing full regression and must be
preserved and independently resolved or isolated. See [Main #780](https://github.com/d6g8k5htny-coder/main/pull/780).

The 41 T10 branch histories have since been retained in a self-contained Git
bundle with 17,990 reachable Git objects, restored without network or
alternates and checked with `git fsck`; this does not adopt any proposal,
include external LFS payloads or separate submodule repositories, or authorize
remote branch removal. Bundle SHA-256
`d44d3d2d81be98ca27fbc2636a3a34d8d486ce1e64a8a12830a214880c6ae284`,
Library ID `libfile_fde3_f-19a1ea0bf3ec81918654bfd1baefd2e4_59a0f9f234e48191bc19fddcc0f80648`;
restore evidence SHA-256 `1ffc5edf89390caa6e65d587bddcf3d16dfc61247013dc55880e25503b42e7a6`.
A later nine-repository public ref snapshot found 2,067 heads and 39 tags in
separate observation intervals; all 41 recorded Math heads matched and eight
Workspace pins matched the then-observed defaults. It does not establish
current pins after #780, clean Git, full PR/admin inventory, safe deletions,
proof truth or final G7/G10. The raw snapshot and independent parse readback
are `libfile_fde3_f-38ba247049888191b936dde2c664f54a_59a0f9f234e48191bc19fddcc0f80648`,
SHA-256 `8c7b6fc75f1a4b9218361adb44895166ec6b6dfc3b5e61893367b730e50e5efd`.

## Observed integration boundary — 9 November 2026

This section records sequential observations through 19:02 UTC, before this
documentation change. Main #782 merged as
`f1ef3a2babecca0f0d4eb35b018dc81d30e9d62c`, tree
`5b771047cb384a9bb167ed2df505489cee1cbf7d`, with parents `6ab126be…` and
`9bf048e9aeee3b846da92fba6e4d0708dee310e7`. Workspace #54 merged as
`1c552538e6a8952548e2bd5297a0031dda1f94e7`, tree
`e069b39ba25ffc2e3c6e2838a9b53c1e4e8484a6`, pinning that Main commit.
Math- was `f532c3e2d91a0c7679a37c733741660917a9b21b`. These observations
were not simultaneous. A later Main merge, including this documentation
change, requires another guarded Workspace pin refresh before current-tip
agreement may be claimed. The older dated sections below remain historical.

The repaired freshness mutation grader rejects the real subject-origin false
kill; the original failure and all eleven original mutations remain. Current
Main #782 hosted controls job `129353108258` passed 20 focused controls,
eight designated guard detections and 22/22 authorized original-mutation
kills in each parent mode, with real subject-origin refusal in both child
modes. This is selected synchronous unittest and named-pattern evidence, not
a hostile-child sandbox, full Layer A audit or final G0/G6. Its bounded
commands and limits are in [TOOLING.md](TOOLING.md). The old false-kill
result is not retroactively rebound. The `eng.hard-gate` engineering record
remains explicitly stale under the cited-content check; its historical source
pin is preserved. A strict freshness check is expected to return nonzero
while that observed drift remains.

At earlier native tree `1c765f1aae84cd91c2d26706b967690b73823e2c`, full
local discovery completed 2,489 tests in each Python mode with 25 explicit
skips each. At the current tested Main #782 merge, hosted navigation ran
2,489 tests with 25 skips and a separate 20 history tests without skips. The
25 local skips comprised five restricted-container checks, two browser
checks, two Lean-dependent checks, one historical composition check, four
real-record checks with no applicable fields, one real-record live-source
comparison not performed, and ten pytest-dependent checks. Other separately executed controls do not turn these into one
no-skip run. Criterion 5 remains unawarded. Exact logs and limits are in
checkpoint v152 and the independently read-back supplement
`libfile_fde3_f-9cfc31236cd88191b63533d7eb3b0729_59a0f9f234e48191bc19fddcc0f80648`.

Workspace #54's exact-head hosted federation job ran 187 tests in both modes
without skips; its pins job passed eight network and strict-current pins,
four canonical inputs and 187 tests. Local exact-tree tests also ran 187 per
mode without skips. Its post-merge pins job `129366156775` passed the same
eight current pins, four canonical inputs and 187 tests; the separate
post-merge federation job `129366157128` subsequently ran 187 tests in
each mode without skips. Main's post-merge required aggregate job
`129366584981` reported success at `f1ef3a2b…`. Its current-site deployment
job `129367351564` completed: the published-source receipt was prepared at
`29a8b2be282ebfe02c2bc71743ac477a63f74842` and observed at
`4a92fb8a90fa2f0a54ffdd1b009e0771201c0fe9` for 32 snapshots. The hosted
rendered Chromium report recorded 39/39 passing cases and no page errors;
this is bounded emulation, not physical-device, complete screen-reader or
scientific certification, nor the STATUS-only cold-start test. Previously
failed GitHub startup, abandoned-job and retry outcomes remain recorded
below and are not replaced by later successes.

Independent mechanical T10 readback reproduced 65 **proposed** obligation
records and 559 other items byte for byte from retained source. It matched
all 46 source statements in five selected records. A restored selective
Git-object pack resolved all 1,872 recorded source bindings; it is not full
ref/history retention and authorizes no branch deletion. Those proposals
have not thereby been adopted or shown mathematically isolated, closed or
scientifically accepted. The exact source, logs and selective pack were
independently read back from
`libfile_fde3_f-5a2970c2901881918ed36aad44f1610a_59a0f9f234e48191bc19fddcc0f80648`,
SHA-256 `2bebe703fba2b7538e127182fb91dbd4c32d2a53ecf0d768982cb94c62a0a10a`.
A later structural-only readback found nonempty, basic-type values for all
eleven required fields in the 65 proposals, with 26 adverse malformed-input
controls refused in each mode. This does not establish the truth of any
field, an exact mathematical boundary, a blocking reason or isolation.
Proposals 019, 020 and 049 mention absent or unresolved review/novelty in
their blocking-reason text alongside source or proof limits. Those historical
notes remain, but missing procedural review alone is not a mathematical
blocker. The scope, commands and retained probe are in
[TOOLING.md](TOOLING.md#t10-structural-readback-probe--9-november-2026).

At the earlier read-only #780 A8 probe, the then-unmerged inventory candidate at
head `24ad8a143862fb9255fc6e2e7bf08c12f989f71d` accepted subject-origin
`AssertionError` failures under its mutation classifiers. Twenty selected
original-source baselines passed; sixteen combined nominal-mutation and
subject-assertion selected children met the RF312/replay/status classifier
conditions in both modes. The two older inventory conditions also accepted
all four full-suite adverse children, each reporting 81 tests, 95 failures and
four controller skips. These are adverse probes, not the historical mutant
results; those outcomes remain unchanged. The test-name and message checks do
not establish selected-oracle origin. The landed inventory classifier must
qualify any later provenance repair and its own tests before future mutation
counts are treated as sound. Source, baselines and raw adverse logs are
retained in separately read-back evidence files identified in DECISIONS.
No peer branch or mathematical disposition was changed by the probe.

For available bounded checks, from a Main checkout run:

```sh
python3 -B -S -m unittest tests.test_node_status_mutation_grader -v
python3 -B -O -S -m unittest tests.test_node_status_mutation_grader -v
python3 -B -S cbz/evidence/node-status-freshness/run_mutations.py
python3 -B -O -S cbz/evidence/node-status-freshness/run_mutations.py
python3 -B -S tools/cbz/g11_census.py --rev HEAD --json /tmp/cbz-g11.json
python3 -B -O -S tools/cbz/g11_census.py --rev HEAD --json /tmp/cbz-g11-opt.json
python3 -B -S tools/node_status_refresh.py records/node_status --check
python3 -B -S -m unittest discover -s tests -p 'test_*.py' -v
```

From the Workspace checkout, run `python3 tools/verify_pins.py
--require-current` and `python3 tools/verify_canonical_inputs.py`. These
commands need the documented dependencies and network where applicable.
They do not implement the complete G0–G11 matrix; no single documented
command currently runs that final pass. Retain actual skips, errors, sources
and exits. The proof index, quarantine inventory, whole Layer A audit,
full regression, final kernel/axiom pass, mutation threshold, final pin and
Git/admin inventory, tooling admission, consistent surfaces and STATUS-only
cold start have not been observed together. No CBZ-0 or terminal Partial is
claimed.

At 19:06 UTC the independently read-back contribution checkpoint was version
153, SHA-256 `ed780a4dd034b648f96d8e959587ef17f0270d5a85e65b52fdbfb82f1df5c7e4`;
later versions may exist.
The original large evidence ZIP remains at its v139 identity and SHA; newer
supplements are separate, not a matched replacement.

## Freshness provenance repair candidate — 9 November 2026

Main #779 merged as signed `6ab126beda65e19865b8468b52f8757ac2324e1f`,
tree `df5509f1f25dc406bad95ab4b88c288300901a98`. This isolated repair
began from its tree-equivalent hosted parent `2d4403838f6d697e671834f12a9a87b379a93ad6`;
native source commit `9486c850d0c22fd1438e18fb392dbe02d7bd214f` has tree
`d9a794488b34168cad49580a40460b44a5b33dfd`. Draft Main #782 hosted head
`c65dc2d3ee52038a0804f6c7455ff8b5844d17ab` tested synthetic merge
`0f76c40fac38af52aed6b32693bb3e1c6ea902d3`, tree
`cfc7bca8d65eec3592e22be0530ddbdcc406f134`; at that earlier observation it was not yet integrated. The later integration is recorded above.

Workspace #53 merged as signed `2441baf051312878ca28e462d455cfbcd8e91b20`,
tree `13dd6b73201627177c568cb4233a33fb60acbbab`, pinning Main `6ab126be…`.
Its exact-head federation job ran 187 tests per mode; its pins job ran 187
tests, eight network and strict-current pin checks, and four canonical-input
checks. Post-merge pins job `129351557433` passed eight network and
strict-current pins, four canonical inputs and 187 tests. Post-merge
federation run `42674110867` failed before jobs with GitHub internal-server
correlation `63dc5ffa-474e-4405-925b-574e391c43c6`; its failed-job retry
was refused with HTTP 403. Four earlier hosted Workspace runs
`42671456239`, `42671456244`, `42672115232`, and `42672114965`
stopped before any job with GitHub internal-server annotations; targeted
retries of the first two were refused with HTTP 403. Those failures remain
evidence; they are not passing checks. Sequential tips are not atomic.
This section supersedes older dated descriptions of current tips below.

The old node-status freshness mutation grader accepted a real subject-origin
`AssertionError` as a kill in both child modes. Its historical mutation
results are not retroactively provenance-qualified. The candidate captures
the actual traceback and code identities before unittest cleanup, accepts
only the selected test body or explicitly captured standard assertion code,
and rejects subject, helper, nested callback and lifecycle origins. The real
adverse subject replacement was refused in both modes while its baseline
passed. A malformed non-object report also exposed a classifier exception;
the candidate now refuses it. Local CPython 3.12.14 runs passed 20/20 focused
controls in each parent mode, including designated grader-guard mutations.
The extracted workflow block exited zero and observed 22/22 authorized kills
in each parent mode: eleven unchanged subject mutations in both child modes.
Each parent run also observed eight authorized detections of four grader-guard
mutations across both child modes. Separate existing subject/refresh/projection
controls passed 72/72 per mode. These are local, bounded observations, not
full regression. The source bundle, raw logs, original
failures and provenance are independently read back from
`libfile_fde3_f-0332e0543bcc819194e66f37fbd29b81_59a0f9f234e48191bc19fddcc0f80648`,
SHA-256 `9e980d03e16d94716d62df10c00a39c88f0c0717b9de414d60a13ebc5b3d7411`.
The hosted current-site controls job `129350315907` also passed on the tested
merge. Its downloaded artifact `13700485135`, SHA-256
`478e130f46398aa00756f4b78692a8a34e6897eda4d2ceb63bba43b74e71c303`,
contains the eight expected files: 20 controls, eight guard detections and
22/22 original-mutation kills in each parent mode, with real subject-origin
refusals in both child modes. This observed component is not final G0 or G6.
The exact native composed-tree G11 censuses passed in both modes with 69 tools,
zero failing tools and zero skips; that is not the final federation G11 gate.
At that earlier observation the full local regressions were still running. Their later outcomes are recorded above.

The hosted navigation workflow `42673203393` failed before jobs with a
GitHub internal-server annotation; its failed-job retry was refused with
HTTP 403. Public-shop `42673202992` and federation-drift `42673203388`
also reported workflow failure with internal-server annotations even though
their individual jobs succeeded. Do not erase or turn these failures into
passing workflow checks. The hosted artifact, full job logs, actual strict
refresh, Workspace merge and failed-run records were independently read back
from `libfile_fde3_f-e95f98760b048191833a8514e364aab2_59a0f9f234e48191bc19fddcc0f80648`,
SHA-256 `528f5686c2ef37c214142d5d2ab00a2a4b8a5642aae320a5304002abbde464ce`.

The read-only command `python3 -B -S tools/node_status_refresh.py
records/node_status --check` observed Math- default
`f532c3e2d91a0c7679a37c733741660917a9b21b` on 9 November and exited 1:
the cited directory of `eng.hard-gate` changed from tree `715449bb…`
to `13272515…`. The other three records were fresh under the cited-content
contract. This candidate marks only that record and its published projection
stale while retaining the historical source pin. The strict freshness check
remains nonzero until the stale condition is resolved; it is not a proof or
scientific-status decision. The actual observation is retained in
`libfile_fde3_f-ca14368b86b88191ae1d95fae1baeeba_59a0f9f234e48191bc19fddcc0f80648`;
the original false-kill evidence is retained in
`libfile_fde3_f-606b8b159414819198b7a213ca93d32c_59a0f9f234e48191bc19fddcc0f80648`.
The site atlas tests passed 10/10 and both asset-release commands produced
digest `efff4cc5ef57e4ece003e919c7e9449cae97b7e539ffb515f819de4a2621ed3e`.
After the record correction, a second actual strict observation ran
18:25:16–18:26:26 UTC with no input drift and again exited 1: the corrected
record remained stale, and the other three records remained fresh under the
cited-content contract. The report is in the hosted evidence supplement.

For this bounded component run:

```sh
python3 -B -S -m unittest tests.test_node_status_mutation_grader -v
python3 -B -O -S -m unittest tests.test_node_status_mutation_grader -v
python3 -B -S cbz/evidence/node-status-freshness/run_mutations.py
python3 -B -O -S cbz/evidence/node-status-freshness/run_mutations.py
python3 -B -S tools/node_status_refresh.py records/node_status --check
python3 -B tools/site_asset_release.py --check
```

The grader covers selected synchronous unittest cases and these explicit
mutation/oracle patterns; it is not a sandbox against a malicious child
process. It does not establish a full Layer A audit, complete regression,
Lean/kernel pass, final G7/G11, STATUS-only cold start, same-pass G0–G11,
scientific acceptance or CBZ-0. No terminal Partial is declared.

## Navigation history candidate — 9 November 2026

The candidate starts from signed Main
`5ebe5771d44160f56d4f2fd1e11726858d17dc90`. The last sequential Workspace
default `a3e6552104d89917105a668a296121680a7f03bc` pins that Main commit;
Math- remains `f532c3e2d91a0c7679a37c733741660917a9b21b`. At 17:33 UTC the
stable contribution status was v141. Its stable evidence archive was v139 after
two failed v140 transfer attempts; the v140 integration supplement is
`libfile_fde3_f-577788f2c9a88191a2493c330a8c0d2b_59a0f9f234e48191bc19fddcc0f80648`.
The status explicitly records this split; the archives are not a matched
single-version pair.

Landed navigation artifact `13660495414`, SHA-256
`b2b93e4a1792fef2c994559baee9852927ed26f3a00484cd301eee99de0e6f5a`,
recorded 2,459 tests and 37 skips, including twelve missing historical Main
Git-object skip events. The candidate requests a full checkout and adds a
separate fail-closed execution of the two affected reader classes. On an
actual full clone of the base, the new shell block passed 20/20 tests with no
skips; on a depth-one clone it exited 1 and retained the 20-test log with
twelve skip events. The existing full discovery remains unchanged.

Focused workflow controls passed 25/25 in each Python mode. A local bounded
probe killed nine designated workflow mutants per mode, requiring each named
assertion to originate in the selected test body; body, helper, nested and
subject-origin controls checked that boundary. The probe is an evidence
artifact, not a general grading service. Its source, reports and both actual
history logs are in
`libfile_fde3_f-f232855b2dc081919315ab3c31a32f56_59a0f9f234e48191bc19fddcc0f80648`,
SHA-256 `6fa1b43e564baa27270e311f44b56cd2e580b2564c8253805789ab17f4c39fcf`.
At this record's creation, hosted candidate execution and integration had not
been observed; inspect the current pull request and its actual checks. The other
observed skip categories remain. No full-regression 100%, final G5/G11,
whole Layer A clearance, final G7, STATUS-only cold start, same-pass G0--G11,
CBZ-0, terminal Partial or scientific acceptance is claimed. Any later Main
merge requires a guarded Workspace pin refresh.

## Current integrated v139 binding — 9 November 2026

The current serial integration boundary is signed Main merge
`ed5d83b7eff354cd7025117f4d2371422acd816b`, tree
`87fdabafadfdb1ee2b766222b6d0c313de8d9a89`; Math- remains
`f532c3e2d91a0c7679a37c733741660917a9b21b`; and signed Workspace merge
`7019083126d3e9fb732447dee5c2895d8c5b7475`, tree
`5d591958e516a582367deb4433cbedf300207d7d`, pins that exact Main revision.
The latest organization-visible open-pull-request search returned zero.
Durable v139 was independently byte-read back at stable status identity
`libfile_9551d0182bcc819188edfe9f998165b3`, 553,155 bytes, SHA-256
`68c00bab7590630ddaae8f67e0d9e1dc081d5168bc65809b18cce5cf0518affc`,
and evidence identity `libfile_1f9ec8ce62708191972f9e2bf6295816`,
358,589,343 bytes, SHA-256
`02e6efe1360f2e2b93ea9c64687db14b6b610961ca93434d1527665461c0fbb0`.
The ZIP has 13,124 unique members and preserves all 13,122 v138 members by
name, CRC and size.

At the integrated Main tree, author-observed normal and optimized G11 censuses
are byte-identical at SHA-256
`a11ec958365f7eb2fa4fd35672398d866e051d08a0d62b88692f7d10b5dbec56`
and pass with 69 tools, zero unresolved tools and zero production findings.
The auditor passes 263/263 controls and kills 328/328 designated mutants per
mode. Complete discovery passes 2,459 tests per mode with 25 explicit skips;
therefore full-regression 100% is not awarded. This is an observed component
result, not invented independent review or the final federation G11 gate.

All eight exact-head Main jobs passed before normal protected integration.
Five landed Main workflows passed on their first attempt. The landed
workspace workflow `42581324023` preserved an HTTP-504 Elan-download failure
on attempt 1 after source controls passed; a targeted retry on the identical
signed commit succeeded on attempt 2, including formal/kernel, architecture,
landing and verification jobs. Workspace exact-head and post-merge pin and
federation-contract workflows passed. The failed attempt remains evidence and
is not overwritten by the retry.

No Lean/axiom/kernel replay for the whole federation, final G7, full Layer A
clearance, STATUS-only cold start or same-snapshot G0--G11 matrix is recorded.
No theorem, proof disposition, quarantine, bound, axiom, authorship, failure,
kernel outcome or scientific status changes. All twelve criteria remain
unawarded; CBZ-0 and terminal Partial are not established. A later Main merge,
including this documentation refresh, requires another guarded Workspace pin
refresh before current-tip agreement may be claimed.

## Current G11 tool-surface closure candidate — 9 November 2026

This candidate starts from signed Main default
`848cd66bbf4a73e77a6f3e1fff64ce4796150f6d`, tree
`2f48833cbd051545be1a7d383137c99e5fa57a12`. Workspace default
`24c7c72c342d9b66275bfc5b30d3ae5df4ed55cf` pins that exact Main revision;
Math- remains `f532c3e2d91a0c7679a37c733741660917a9b21b`. The latest
organization-visible open-pull-request search returned zero. Durable v138 was
independently byte-read back at stable status identity
`libfile_9551d0182bcc819188edfe9f998165b3`, 548,909 bytes, SHA-256
`1f61a595a7bbbc0e4a3c661c3f860e0355c3fbe7db8c9d810ca42e25cff6f059`,
and evidence identity `libfile_1f9ec8ce62708191972f9e2bf6295816`,
358,585,987 bytes, SHA-256
`3af309a2ca2f951e5cca74f5647776822bb0de7459b0575ea0e9b486faae169c`.
Its ZIP has 13,122 unique members and retains all 13,118 v137 members.

The final thirteen current-tree G11 failures were documented historical
research/browser drivers with no current executable reference or exercise.
They are removed from the operational surface while all plans, reports,
preserved RED/GREEN artifacts, failures, authorship and product source remain
unchanged. Their exact native recovery identities are:

| Removed path | Git blob | Bytes | SHA-256 |
|---|---|---:|---|
| `tools/compare_experience_browser_check.py` | `75c9128806335b10e5862ecfae8eb6bab7495557` | 15,350 | `962713a9786e624b13a1e3588dae19570ab21679f4e3a25224a515836f38d80c` |
| `tools/figure_png_browser_check.py` | `dc89bd0284727ca15a62894516ffc1b87a7732cf` | 21,400 | `67105c43d0d095d5e870c228019bc09ca351c884fde0716125735b955d0e2b8a` |
| `tools/library_discovery_browser_check.py` | `09b0ee84ce9282cf04e0d8c346e947754f26f662` | 23,489 | `a424629d30a7f0a14635ca38f6f89493cd627023b30947348c5e4f4f81dd222c` |
| `tools/library_source_search_browser_check.py` | `b28742869ca3088d2accb19e4177ec202a660c3a` | 9,116 | `49885136b7128ba3542bf32bc173b9a007309ea2e4d703371d92cc336d49275e` |
| `tools/persistent_status_browser_check.py` | `b4bea1d4d4b73afd78a06c28b72939cad64029b5` | 7,466 | `c7916ef456fd5ff37d9e367f4e6b5cccc9987e05be8c1c66e463a103bf4815c7` |
| `tools/reader_jump_browser_check.py` | `71f69d25c11acdcbd57af18d91efac0a778457d1` | 7,206 | `7c430e27c07c02e7e067192a23ea00be28b82dd6bcd443c7c631966b32064ebd` |
| `tools/reading_list_citation_browser_check.py` | `dbfa049d2dd824235ea1d07e919fc3ca878bb05c` | 11,027 | `a34c2dd4c9823bae717f29e5fcea636f8610316dd3bb3e0cc137fe2425cb36ea` |
| `tools/research_desk_browser_check.py` | `d03f3428ed914442c3029aaf19ab786ee40db447` | 15,882 | `c2c22f0357b3e3dab65bac784b4289dad9d94c23a800f6412cb16310b7b0451e` |
| `tools/research_desk_continuity_browser_check.py` | `632beb9c7cad7b1cdc3523dab3caacaed88b6d51` | 13,961 | `f76679193dd4035476aa4b35afeebd05d9c12d4d8cc28ce7f3332dda981e099b` |
| `tools/research_desk_undo_browser_check.py` | `49f2815a91806b34668d14ff133a4cb2737a3f95` | 14,222 | `09a298f133bfccd0a746b0ea9bb34bbf7bd8986f7baa07b946d401b0578e9a5f` |
| `tools/research_reading_browser_check.py` | `516d60889a527b212c7b68b7b8292b21c576ab92` | 8,903 | `1fdb1a9804460dfb1153be7abc777b573976f8c43a484479f379aaa6860914f1` |
| `tools/snapshot_compare_browser_check.py` | `7ffd5c6bf0769523016f0c33227edcaab23f7706` | 18,427 | `b8b4e1c49368f8ae68862d4515b7805e7219d846193b7bfc6d4c6183e7e9cdc4` |
| `tools/source_list_continuity_browser_check.py` | `00e6a259b3579eff34fe61beaf1b515dc258798d` | 17,514 | `54cc653827ffdb48b1779b6f846046c1944c7c89678b6c06a3fcdb823cb4b0ea` |

Fresh pre-change normal and optimized censuses were byte-identical at SHA-256
`5ce8979ff1b0ba1a5c77aaefcf276d135f4cc3d4314a4a00d9756cd3d2260342`
and both exited 1 with 82 tools, 13 unresolved tools and zero production
findings. At exact deletion commit
`26bd26594c6bf7d1678ae0ea40483b61e07fac61`, tree
`473cd9a018993327536c6510ac78122eb830020f`, both censuses pass, are
byte-identical at SHA-256
`793a7f09cdafb40b22283d284722a495628c22b707631ae0e887b10fb123adc6`,
and report 69 tools, zero unresolved tools and zero production findings. The
auditor passes 263/263 controls and kills 328/328 designated mutants in each
mode; the byte-identical mutation receipt has SHA-256
`e73ecb9f1a31208d4a92dfc5359e11ea8c6de833ab0df1495cc3c5675f05b97e`.
Complete discovery runs 2,459 tests in each mode with exit 0 and 25 explicit
skips; normal and optimized log SHA-256 values are
`1cf4c2d78b85a7317e8e60e15a6fc382cc538950c6dc9ad65b57859c8a27a411`
and `0d0122b4080e969d1d0795dde8a912d95c6aa2c52f3726999bb66fe261fee6d3`.
The older two-error/23-skip executions below remain their original historical
outcomes and are not rebound.

This closes the bounded current G11 component at the exact candidate tree; it
does not establish full-regression 100% because 25 tests were skipped, rerun
Lean/kernel/axiom checks, complete a STATUS-only cold start, or execute the
final G0--G11 federation matrix in one pass. No theorem, proof disposition,
bound, axiom, authorship, historical failure, kernel outcome or scientific
status changes. CBZ-0 and terminal Partial are not established.

## Current G11 Atlas-driver consolidation candidate — 9 November 2026

This candidate starts from signed Main default
`31150f4a0736cb79bf33e366b6b725b061126000`, tree
`b3f103e72a12cc766519c28bae92265162f2955e`. Freshly integrated Workspace
default `83623bff94918021a00d8f42c9618dc549015d6c` pins that exact Main revision;
Math- remains `f532c3e2d91a0c7679a37c733741660917a9b21b`. The latest
organization-visible open-pull-request search returned zero. Durable v137 was
independently byte-read back at stable status identity
`libfile_9551d0182bcc819188edfe9f998165b3`, 545,109 bytes, SHA-256
`907b726558a856276cbeaf88bb3f521df9d7c27c975af847a6a8882ff186f770`,
and evidence identity `libfile_1f9ec8ce62708191972f9e2bf6295816`,
358,579,834 bytes, SHA-256
`00288242715743d878efec482a78920fc348b11e28223c49aa9bbb07a20c0307`.
Its ZIP has 13,118 unique members and retains all 13,114 v136 members.

Eight further documented historical Atlas browser drivers had no current
executable reference or exercise. They are removed from the operational
surface while every plan, report, failure and authorship record remains
unchanged. Their exact native recovery identities are:

| Removed path | Git blob | Bytes | SHA-256 |
|---|---|---:|---|
| `tools/atlas_approachable_browser_check.py` | `0eaa7e488a1ca9cd3c0075981a9223b1fad291fe` | 11,659 | `99f567499dcee942b11a627afeb7a02589a8c8a41d787453b46d19b725313103` |
| `tools/atlas_claim_summary_browser_check.py` | `67a2df8dd2c1c4f3bf4c1838e4e03247d2ec0c9b` | 8,339 | `55b15ed1ccd16b5276afe8c1d23d1538da837f611621e70dfbcd1952d51ccbc1` |
| `tools/atlas_live_controls_browser_check.py` | `e6daa8f1cb52539036cfa2f86f8ce01e26a96351` | 8,876 | `b7c318a19c79f3da06734723a0f65d831d0ee140bd07dc451e11c25c4193aeca` |
| `tools/atlas_refusal_browser_check.py` | `1f9c48448c80b42e74f916429a7bc47f2c10a2f7` | 8,030 | `639b9f4295273ff2c6fb5c0221dc1bd998804b84d38edb6f1efffa12c6fb6c28` |
| `tools/atlas_relationships_browser_check.py` | `fd56cfdcc7170ce2653fa5f65c8e8e254bd76885` | 12,995 | `8a6a1c57fd5382a61b0cb11af1ed632bf6a1c71cc70098afe340c5607e4f5b87` |
| `tools/atlas_spatial_browser_check.py` | `306edcf855366826ce76fb57466e46e6b8bcff53` | 18,113 | `915b6f6bca3e81fb39ebf250363eee3daf4f2a8a76f41c2e90f2754b1dcea881` |
| `tools/atlas_spatial_labels_browser_check.py` | `bf3f5f68631aa13579c3efb7b0fd404c95b3f3e6` | 7,113 | `97d13995e52e584b06ffa2cd842661a4326fd32612610e3a7a3004a3e76d612e` |
| `tools/atlas_webgl_fallback_browser_check.py` | `c5bc1bee12bd47c49f949e18462d751ebc2cb0d0` | 15,536 | `7abd99767a6acb456c29be3e92210d13a2ce2d507c344639a304fba9a9055959` |

At exact deletion commit `98dfecca599aa05b2007ec41d72fd605dd13f99e`,
normal and optimized censuses are byte-identical at SHA-256
`5a59528f3db330596c0b7df13d0bb03a95e2971183a1a008547cf42e09a2da80`.
They have zero production findings, reduce the registry from 90 to 82 tools
and the unresolved residue from 21 to 13 tools, and still exit 1. G11 remains
FAIL. The auditor passes 263/263 controls and kills 328/328 designated mutants
in both modes; the byte-identical mutation receipt has SHA-256
`e73ecb9f1a31208d4a92dfc5359e11ea8c6de833ab0df1495cc3c5675f05b97e`.
Complete discovery ran 2,459 tests per mode and preserved exactly two managed
Chromium singleton-socket errors and 23 skips. Its normal and optimized logs
have SHA-256 `2249bc0015f087432ca9a09572fc983b0a46d03ef80cad351d7455876888c955`
and `b9a2d3fa3f12609284a9828e5c2b9aa95889220fb61e4c7291daa2d250e991e9`.
The preceding zero-test commands exited 5 and receive no credit.

This candidate does not rerun Lean/kernel verification, establish
full-regression 100%, complete a STATUS-only cold start or run G0--G11 in one
pass. No theorem, proof disposition, bound, axiom, authorship, historical
failure, kernel outcome or scientific status changes. All twelve criteria
remain unawarded; CBZ-0 and terminal Partial are not established.

## Current G11 consolidation candidate — 9 November 2026

This candidate starts from signed Main default
`7712277ced9170c8d61d8f324319f40177c145b1`, tree
`1bafc1d181ddc161c51d58bfe3084a118d2d9bbf`. The fresh sequential Math-
and Workspace defaults are `f532c3e2d91a0c7679a37c733741660917a9b21b`
and `6064d7b96eb60f5f4e50f410c89c1f0dee813a52`; Workspace pins that exact
Main revision. The organization-visible open-pull-request search returned zero.

`tools/atlas_accessibility_audit.py` and
`tools/atlas_resilience_browser_check.py` were both current-tree
`referenced=false`, `documented=false` historical browser drivers. The first
is retained by exact name, byte count and SHA-256 only in the frozen F973
evidence-source record; the second has no current reference. They are removed
from the operational surface without altering that historical record. Their
exact sources remain recoverable at native blobs
`df4738e0df42b7f6ae320482be86194ce2d4bd66` (21,485 bytes, SHA-256
`e133b8c5a7df0b261d676cd3d686c92607b90a9305f92d0d0bbca4675e5e3bac`)
and `6b36d770624c846f2d6411b7a55d20885954019e` (15,397 bytes, SHA-256
`1c760214db84a1ec011be66f78d62f569ec191f15e0ccbb93b14f94651a8e972`).

At exact deletion commit `0a36115eefd1d4d26dd575e720e6e9ded172999a`,
normal and optimized censuses are byte-identical at SHA-256
`67a6c46c4752b1a9276502dd6ebfe8c9e127a38adefc5c3f7b8da64a5d1312c5`.
They contain zero production findings, retain 21 documentation or
no-current-exercise tools and exit 1, so G11 remains FAIL. The exact workflow
suite passes 265/265 and the designated self-test kills 328/328 mutants in
both modes. Complete discovery ran 2,459 tests in each mode and preserved the
same two managed-executor Chromium singleton-socket errors and 23 skips; it is
not a 100% regression pass. The preceding zero-test discovery invocation is
preserved and receives no credit.

Durable checkpoint v136 is byte-read back at stable status identity
`libfile_9551d0182bcc819188edfe9f998165b3`, 540,053 bytes, SHA-256
`423a1b558c99a73b43cedfb10765d810ad8a6a2e9dd5eb74899bedb0b8aa4fae`,
and evidence identity `libfile_1f9ec8ce62708191972f9e2bf6295816`,
358,573,202 bytes, SHA-256
`2b53f6cfae4343f468e2f3c80c6ac7211fff7c2e8f47ff8c12b3c908fc41b83d`.
The ZIP has 13,114 unique members and retains all 13,110 v135 members. This
candidate does not rerun Lean/kernel verification, establish full-regression
100%, complete a STATUS-only cold start or run G0--G11 in one pass. No theorem,
proof disposition, bound, axiom, authorship, historical failure, kernel outcome
or scientific status changes. All twelve criteria remain unawarded; CBZ-0 and
terminal Partial are not established.

## Current G11 retirement candidate — 9 November 2026

This candidate starts from signed Main default
`e18798544f16479436283d71689ae635a1817b20`, tree
`b8a678df6ea8b03c28e1ccef0e34baf89bc7274d`. The freshly observed
sequential defaults are Math-
`f532c3e2d91a0c7679a37c733741660917a9b21b` and Workspace
`400e03ca5f154e8d236e7126ea1525e78b3b595c`; the latter's published
`repos/main` gitlink is exactly `e18798544f16479436283d71689ae635a1817b20`.
The fresh organization-visible open-pull-request search returned zero. A merge
of this later Main candidate will again require a guarded Workspace pin refresh
before final G7 or criterion 8 can be awarded.

`tools/atlas_formal_evidence_browser_check.py` was an unqualified historical
acceptance driver whose only current executable references imported its byte
identity helpers; no current workflow exercised the driver. It is removed from
the operational tool surface and remains exactly recoverable at Git blob
`9576fd34806528484a3898665c19cabe9eb3797c` (20,118 bytes, SHA-256
`4ce981b7b9c0d422f099fcd8556a144d01de8a6c18de9431a6a045fecce55af7`)
and in Main ancestry. The two current qualified browser tools now carry the
same small hashing functions locally. Their 11 runner/adverse tests pass in
normal and optimized modes. The bounded G11 controls pass 265/265 and the
designated self-test kills 328/328 mutants in each mode. At exact retirement
commit `724cb93bd6f3fdd0e407d48c5c3f267eb4c5bef0`, normal and optimized
censuses are byte-identical at SHA-256
`7ad72c9771a5859b465f7a4c2dc668a59c917a7a7d3770ddb0b6614ea8583455`,
contain zero production findings, retain 23 documentation or
no-current-exercise tools, and exit 1. G11 therefore remains FAIL.

Durable checkpoint v135 preserves the preceding Main #773 and Workspace #47
integrations. Stable status identity `libfile_9551d0182bcc819188edfe9f998165b3`
is version 135, 536,616 bytes, SHA-256
`c97603a63431df8cbbacf25e83cdcd908c3b246c760ead12ffd6d6ceffa09fe3`.
Stable evidence identity `libfile_1f9ec8ce62708191972f9e2bf6295816` is
version 135, 358,567,704 bytes, SHA-256
`230bc21590ce8171890269201ac9cd1675d93f849a8e1f91b8d2380f2f805df6`.
Byte readback matched; the ZIP has 13,110 unique members and zero duplicate
names. This candidate does not rerun federation Lean/kernel checks, establish
full-regression 100%, complete a STATUS-only cold start, or run G0--G11 in one
pass. No theorem, proof disposition, bound, axiom, authorship, historical
failure, kernel outcome or scientific status changes. CBZ-0 and terminal
Partial are not established.

## Current serial integration correction — 9 November 2026

This candidate starts from signed Main default
`6deca3198c4b11b2d1208df904f5b942cd3e4bfb`, tree
`3f615867f7d04f2b3e772760c25af479d26c7948`. The latest sequentially
observed defaults are Math-
`f532c3e2d91a0c7679a37c733741660917a9b21b` and Workspace
`a3d4c50b8b94ae1e02a032f68c5411007d695fa0`, tree
`a9f007b18fe496c63c74680a2005acb5766535cd`. At that observation the
Workspace `repos/main` gitlink equalled the Main default exactly, all eight
strict-current public pins and all four canonical inputs passed, and the
organization-visible open-pull-request search returned zero. These facts are
sequential, not atomic or perpetual; merging this candidate would advance Main
again and require another guarded Workspace pin refresh before final G7 or
criterion 8 could be awarded.

Durable checkpoint v134 is recorded at
[Main #684 comment 6730912447](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6730912447).
The stable status identity
`libfile_9551d0182bcc819188edfe9f998165b3` is version 134, 532,600 bytes,
SHA-256 `a33611e2434f508e249c4effed5cbd33380c6b2bcc894ab6222cbe25101d8d5c`.
The stable evidence identity
`libfile_1f9ec8ce62708191972f9e2bf6295816` is version 134, 358,562,188 bytes,
SHA-256 `f99bcae69fd7ae7d6117e66ba6454cae6423647d32c926b8a60e518f9c854e30`.
Byte readback matched, the ZIP contains 13,106 unique members, and all 13,102
retained v133 non-status members are byte-identical.

The current exact G11 census has zero production findings and 24 retained
documentation or no-current-exercise tools, so it exits 1 and G11 remains
FAIL. Full local Main discovery is also not green: normal execution failed at
Chromium startup and optimized execution observed 2,459 tests, two errors and
23 preserved skips; both errors are the managed executor denying Chromium's
singleton socket with `Operation not permitted`. Hosted Chrome component
success does not convert that local failure into full-regression 100%.

No fresh federation-wide Lean build, axiom audit, kernel replay, STATUS-only
cold start or same-pass G0--G11 matrix is recorded for this candidate. No
proof disposition, theorem statement, bound, authorship, historical failure,
kernel outcome or scientific status is changed. The older dated maintenance
snapshot below remains historical evidence and is superseded only for current
repository, pin, pull-request, G11-count and durable-checkpoint facts.

## Current maintenance snapshot — 7 November 2026

The Main candidate containing this file has native source parent
`8038863c04cc96acfcfc0d2953ab8575ccce213e`; its isolated work began from the
then-current Main default `299c7fad72f1079b87b2d044b2d30f2098678f29`.
The sequentially observed Math- and Workspace defaults are
`f532c3e2d91a0c7679a37c733741660917a9b21b` and
`05f3096d077a6fa4ff6cf763c074451cb1ac9972`. Workspace #41 refreshed its
Main pin to `299c7fad72f1079b87b2d044b2d30f2098678f29`, but this later isolated
Main candidate is newer, so strict-current pin agreement is again unawarded
until a guarded post-integration refresh lands. Open-PR counts are dynamic and must be read from native refs rather
than inferred from this file. The 12:04 UTC local checkout observation found
14/14 clean checkouts; the later v79 observation found 19 checkouts, zero
dirty trees, zero stashes, valid HEADs and no missing registered worktree
paths. This current isolated worktree is outside those instants. The exact
remote census still contains 2,049 non-default branches requiring explicit
integration, archival or dormant disposition.

The five canonical reading surfaces are this file,
[PROOF_INDEX.md](PROOF_INDEX.md), [GOVERNANCE.md](GOVERNANCE.md),
[TOOLING.md](TOOLING.md) and [BASELINE.md](BASELINE.md). The proof index is
explicitly incomplete; the tooling registry has no admitted final G0–G11
orchestrator; and the baseline ledger marks every unawarded criterion. The
latest Git-native durable receipt is v79 at
[Main #684 comment 6685706413](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6685706413).
Neither that checkpoint nor this documentation repair establishes CBZ-0.
Main #762 repaired the G11 `defaults.run` workflow-scope false positive with
39/39 tests and 76/76 designated mutants passing in both modes; #763 then
closed seven documentation-only gaps. The review-repaired G11 attribution
native source `8038863c04cc96acfcfc0d2953ab8575ccce213e` (parent
`d962c4e1593cd3b35d655aee9c7a582da6b63ca3`, tree
`de8e2c664116803643cfda2f324756e7716c57af`) was qualified locally at exact-tree
commit `1616b31889a8b242ff6950dbd7af4d437e54648c`. Fresh execution on that exact
tree passed 256/256 tests and killed 321/321 designated mutants in both modes
with zero skips; the sealed
mutation-inventory SHA-256 is
`d7021d5cc7e464c423335c7b56b67b5ef1844c09865683d728a998e1661a5652`.
The normal and optimized mutation receipts are byte-identical, 199,677 bytes,
SHA-256 `367a13fed18c0880fcc73590ffb1f615bb48ba1baf76ecc621f427637f31f687`.
The exact-tree byte-identical normal/optimized whole-tree reports (SHA-256
`bcd9605e448a4e3d121f8e9a03733190f376367e5c2ab690b17a90675605ef8f`,
56,059 bytes each)
contain zero production findings and retain exactly 32 no-current-exercise
or documentation tool candidates; both census runs exit 1. The stricter
attribution newly removes false exercise credit from
`tools/architecture_evidence_adapter.py`,
`tools/architecture_graph_artifact_readback.py` and
`tools/artifact_index_export.py`; the prior 29-tool result remains historical.
Five later exact-head defects are also repaired: assignment targets evaluate
their executable subexpressions before rebinding; unresolved chained-comparison
suffixes and eager-comprehension bodies/guards invalidate possible effects
without receiving execution credit; uninvoked JavaScript class/object method
bodies remain deferred; and literal-true JavaScript conditions exclude their
dead `else` arms. The duplicate comparison review report is retained as review
history but counted once. Three subsequent exact-head defects are also
repaired: helper-mediated writes on unresolved Python branches invalidate
downstream provenance; quoted JavaScript child-process imports are not trusted;
and unreachable JavaScript URL initializers cannot create provenance. An
adjacent adverse control then exposed parameter-alias mutation through a
helper called on an unresolved branch. The repair maps only aggregate mutation,
not local parameter rebinding, back to the caller argument. A paired positive
control preserves the latter distinction. Three further review defects now
require every enclosing JavaScript branch to select a call, reject `while` and
`for` bodies without a proven first iteration, and refuse process calls after
direct `return` or `throw` in an invoked helper. The duplicate parameter-alias
review record is preserved but counted once. The adverse total is therefore
147 distinct review/adverse defects. Two older unresolved review findings then
required augmented-assignment targets to be captured before RHS helper effects
and JavaScript parameter default initializers to remain deferred until their
declaring helper is actually invoked. Both were reproduced red in normal and
optimized modes before repair, bringing the then-current total to 149 distinct defects.
Three still-older unresolved review findings were then reproduced red in both
modes: a possible Python branch rebind retained trusted import identity; a
proven JavaScript loop credited calls after direct `break`/`continue`; and an
unselected `switch` case received module-root activation. Possible-effect
deletions now invalidate local imports, direct loop termination cuts off the
remaining loop body, and unsupported switch selection fails closed. Four fresh
review findings were then reproduced red in both modes. Python now
captures a helper's callable identity before evaluating side-effecting call
arguments; JavaScript parameter scopes use delimiter-balanced discovery;
default initializers execute only for a missing or explicit `undefined`
argument; and unproven `catch` handlers receive no execution credit. Three
adjacent exact-head findings now exclude skipped nullish-coalescing operands,
mask JavaScript regular-expression literals before call discovery, and
propagate direct termination from completed standalone blocks. Three further
exact-head findings retain definition-time effects of functions declared on
unresolved branches, map aggregate mutation through a helper-local alias back
to the bound caller argument, and refuse optional-call argument credit without
proof that the call executes. Five latest controls interleave
chained-assignment target stores with later
target evaluation, retain lambda-default effects in unresolved branches,
reject optional property/computed-chain subexpressions without proven
evaluation, defer unconstructed JavaScript instance-field initializers, and
mask regular-expression literals following control headers. Four newest
controls include unresolved-branch class construction, helper definition-time
parameter mutation, continuous optional property chains and computed instance
fields. Three latest controls continue suppression through computed accesses
and intermediate calls in one optional chain, defer computed-name class and
object methods, and recognize semicolonless instance-field boundaries. The
newest three controls propagate helper class-construction mutations through
bound caller arguments, refuse downstream child-process activation after an
unsupported JavaScript `await`, and balance nested computed method names before
deferring their bodies. Five newest controls interleave every leaf store inside
a destructuring target, defer quoted class/object methods, balance nested
computed instance-field names, snapshot destructured leaf values before target
effects while preserving nested aliases, and keep executable assignments in
class static blocks at class-definition reachability. Three still-newer
exact-head controls unpack
nested Python targets only when each nested target is reached, defer quoted and
numeric JavaScript instance-field initializers until construction, and apply
unconditional termination inside executable JavaScript class static blocks.
Two newest exact-head review controls recognize exponent, leading-decimal and
numeric-separator JavaScript instance-field names as deferred until construction,
and evaluate the operand of a direct static-block `throw` before suppressing its
unreachable tail. Four further exact-head controls propagate mutation through an
omitted mutable-default helper parameter, discover helper-parameter aliases
inside nested statements, defer numeric-name class/object methods, and
distinguish an instance field literally named `static` from the static modifier.
Two newest exact-head controls preserve the definition-time identity of an
omitted mutable default even when the name used by its default expression is
later rebound, and select helper-alias origins branch-sensitively: literal
branches contribute only their taken origin while unresolved branches retain
the union of possible origins. Mutation-time origin snapshots now preserve an
already-executed alias mutation even after a later local rebinding, and
recursive invalidation of a captured mutable default reaches nested aggregates
and their surviving external aliases. Two newest exact-head controls union
mutation-time origins with a helper alias's final formal origins so a later
helper-induced mutation after rebinding cannot lose either caller aggregate,
and propagate formal aliases through value-bearing annotated assignments while
leaving annotation-only assignments nonmutating. Two fresh review controls now
record nested-helper mutations against the alias origins present at the call
site and locate JavaScript class bodies only after delimiter-balanced heritage
expressions. The first native repair `3f3dc9166a79b7d0bb7ec97b6422621a7da90f85`
failed both whole-tree censuses with `RecursionError`; that failure is preserved
and receives no credit. The replacement precomputes signature-checked per-call
mutation sets and has a recursive-helper termination control. The older
defect-196 union mutant became behaviorally equivalent under this stronger
call-site capture and is explicitly retired; its historical RED control remains.
Six further exact-head repairs skip nested class/function expressions while
locating an outer JavaScript class body, preserve helper mutation provenance by
visiting assignment right-hand sides before ordinary or annotated rebinding,
balance multiline computed names, fail closed on malformed class structure,
admit valid private names `#class`/`#function`, and exclude reserved words in
import/export specifiers from expression-keyword reachability. The strengthened
assignment mutants preserve right-hand-side traversal and change only ordering;
earlier 319/320-case attempts receive no credit. The current total is 206 distinct
review/adverse defects. The former 20 active
static-attribution gaps are closed without crediting mere mentions, fake APIs,
interpreter command payloads, same-line JavaScript decoys or unreachable code.
Generator methods are not credited from object creation/closure alone;
comprehension targets are isolated from same-named outer static bindings; and
forwarding-wrapper discovery obeys literal Boolean short-circuit reachability.
Absolute-parent traversal now stops at the filesystem root and reachable
function-local process imports are recognized. Destructured assignments bind
only corresponding values and do not credit the unused aggregate. Rebound
wrapper parameters, unresolved local shadows, and JavaScript names occurring
only in strings or properties cannot credit a tool. No final G11 or combined
G0–G11 pass is claimed. Forwarded interpreter command payloads receive the
same command-mode filtering as direct calls; augmented assignments replace or
clear prior static values; and property calls cannot impersonate a named
child-process import. Four further exact-head review findings are repaired:
unresolved command elements cannot shift the executable, Boolean `or` follows
left-to-right short-circuit order, a local callee cannot inherit a same-named
module wrapper, and quoted JavaScript call text is not a call. Position-preserved
dynamic suffixes after an already proven tool path, resolved starred command
elements, `unittest` lifecycle methods and trusted loader-wrapper paths are now
recognized with dedicated negative controls. Three final-head review findings
are also repaired: class methods cannot register as same-named module wrappers;
known conditional commands select only their taken branch while unknown
alternatives earn credit only when every branch selects the same tool; and an
intervening JavaScript assignment invalidates a prior URL binding. The stricter
source newly exposes `tools/atlas_formal_source_bridge.py`, so the earlier
28-tool report remains historical rather than being rebound to this 29-tool
result. Three subsequent exact-head findings are also repaired: every
`pathlib.Path` constructor component participates in path resolution; assigning
through a command element or slice invalidates its aggregate static binding;
and a JavaScript function parameter cannot impersonate a same-named imported
child-process callee. Four latest exact-head findings are also repaired:
interpreter mode switches are interpreted only before the selected script;
element or slice writes invalidate wrapper forwarding as well as direct static
bindings; a helper-local invalidation cannot fall back to a same-named module
binding; and identifiers in JavaScript default initializers are references,
not parameter bindings. Three subsequent review findings are repaired as well:
recognized interpreter options consume their operands before script selection;
an element write invalidates every resolved identity alias of the command list;
and a function parameter shadowing a JavaScript URL binding blocks that stale
module binding inside the function. No intermediate result is rebound as a
passing gate. Three additional exact-head findings are now repaired: supported
command-list mutator calls invalidate the list and all resolved aliases before
direct or forwarded attribution; a function-local JavaScript declaration
shadows a same-named child-process import across its function scope; and a
function-local `new URL` binding cannot escape to an outer call. The supported
Python mutator set is finite and documented; this remains bounded static
analysis rather than general Python or JavaScript execution.
Three later exact-head findings are also repaired: a write through an alias of
a wrapper parameter invalidates the shared forwarding proof; reassignment of a
registered wrapper binding prevents later calls from inheriting the original
function's evidence; and a JavaScript declaration in any enclosing function
shadows a same-named child-process import for nested calls. Each counterexample
was reproduced red and has a dedicated control and designated mutant. Two
further exact-head defects are now repaired: wrapper discovery stops after an
unconditional `return` or `raise`, and module-wrapper availability inside a
deferred caller is evaluated at that caller's actual activation point rather
than the nested call's earlier source line. Both counterexamples were
reproduced red and have dedicated controls and designated mutants.
Three later exact-head findings are also repaired: a non-forwarding duplicate
function definition clears an earlier wrapper registration; list augmented
assignment preserves Python's shared list identity so later alias mutation is
visible; and bounded JavaScript reachability refuses literal-false branches
and uncalled named helpers. Each supplied counterexample was reproduced red in
both modes and now has a dedicated control and designated mutant.
Two newer review counterexamples are also repaired. Deferred Python function
bodies now consume module bindings in activation order instead of the final
module environment, and JavaScript helper reachability requires an invocation
chain rooted in module execution. The stricter JavaScript rule briefly exposed
`tools/museum_conditionals.mjs`; an explicit adverse-tested rule admits only
callbacks passed to the imported `node:test` identity, restoring that actual
test execution without treating arbitrary callback APIs as roots. The
intermediate 30-tool census remains historical; the then-current review-defect
total was 76 and exact-head review remained required. Three latest
counterexamples are also repaired: mutually exclusive `is None` branches no
longer manufacture a post-join binding; a `node:test` registration must itself
be reachable from module execution; and a shadowed `test` name cannot inherit
the imported registrar's identity. Each was reproduced red in both modes and
now has a dedicated control and designated mutant.
Two subsequent exact-head defects are also repaired: mutually exclusive
cache-guard alternatives use isolated fail-closed binding environments before
their results are joined, and a nested Python helper's binding effects occur at
its call position before analysis resumes in the caller. Both supplied
counterexamples were reproduced red in both modes and now have dedicated
controls and designated mutants.
Three still-later exact-head defects are repaired as well: a Python helper's
declared `global` assignment updates the module binding used by later calls;
deleting a command element or slice invalidates the aggregate and its resolved
identity aliases; and a JavaScript helper invocation cannot provide execution
evidence after that helper binding has been replaced. Each supplied
counterexample was reproduced red in both modes and now has a dedicated
control and designated mutant.
Three final exact-head counterexamples are also repaired: deletion inside a
forwarding wrapper invalidates that wrapper's command provenance; a declared
Python `global` is excluded from the helper's local static-shadow set so a
same-scope read observes the module write; and JavaScript helper calls resolve
the nearest lexical declaration instead of a shadowed module helper. Each was
reproduced red in both modes and now has a dedicated control and designated
mutant.
Three newer exact-head counterexamples are repaired as well: helper side
effects in Python call arguments are applied before attribution of the enclosing
process call; a reassigned Python helper cannot execute its former body; and a
deferred JavaScript helper validates module URL bindings at its reachable
activation points rather than the child call's textual position. Each was
reproduced red in both modes and has a dedicated control and designated mutant.
Two latest exact-head counterexamples are repaired: each Python call argument's
static value is captured at its evaluation point while later in-place list
mutation still invalidates the shared object, and assignment RHS expressions
execute before their targets invalidate active helper bindings. Both were
reproduced red in both modes and have dedicated controls and designated mutants.
Two still-newer exact-head counterexamples are repaired: list and tuple element
values are captured in left-to-right evaluation order before later element
effects can rebind their source names; and a declaration in a module-level
JavaScript lexical block shadows a same-named imported child-process callee.
Both were reproduced red in normal and optimized modes and have dedicated
controls and designated mutants.
Three subsequent exact-head counterexamples are repaired as well: captured
sequence-element values survive supported list concatenation; ordinary helper
expansion stops after unconditional `return`/`raise`, including termination in
a selected literal branch; and destructured declarations in a JavaScript
lexical block shadow same-named imported child-process callees. All three were
reproduced red in both modes and have dedicated controls and designated mutants.
Three newest exact-head counterexamples are repaired too: a named operand in a
supported concatenation is captured before later operand effects can rebind its
source name; traversal of a selected Python branch stops after definite
termination inside that branch; and a semicolonless JavaScript lexical
declaration shadows the imported child-process callee under ASI. All three were
reproduced red in normal and optimized modes and have dedicated controls and
designated mutants.
Three latest exact-head counterexamples are also repaired: sequential calls to
the same helper are each expanded while direct recursion remains bounded by an
active-call guard; a helper's declared `global` reassignment invalidates the
module helper identity used by later calls; and literal Boolean short-circuiting
refuses an unreachable loader operand. All three were reproduced red in normal
and optimized modes and have dedicated controls and designated mutants.
Two subsequent exact-head counterexamples are repaired as well: a literal
conditional expression traverses only its selected arm, with unknown conditions
left uncredited; and calling an async helper without awaiting it does not execute
its body. The bounded scanner does not infer coroutine execution. Both supplied
fixtures were reproduced red in normal and optimized modes and have dedicated
controls and designated mutants.
Five later exact-head counterexamples are also repaired: async forwarding
wrappers and generator object creation do not execute their bodies;
comprehension bodies require established iteration and generator expressions
require consumption; a chained comparison stops after a proven false prefix;
and a JavaScript `const`/`let` URL binding cannot escape its lexical block.
Every supplied fixture was reproduced red in normal and optimized modes before
the repair and has a dedicated designated mutant.
Four final source defects are repaired after three exact-head review findings
and one adjacent adverse case were reproduced red: forwarding-wrapper
discovery now follows literal conditional-expression reachability; wrapper
reassignments declared `global` or `nonlocal` invalidate the destination
binding at actual execution order rather than lexical helper order; lambda
creation evaluates defaults but not its deferred body; and declared `nonlocal`
helper reassignment invalidates the nearest enclosing active helper identity.
The exact results for that source stage remain historical. Exact-final-head
review and hosted checks remain required before integration.
Three subsequent exact-head findings are also repaired: forwarding-wrapper
discovery evaluates callee, positional and keyword argument effects before it
registers forwarding; chained-comparison discovery stops after a proven false
prefix; and unproven exception handlers/`else` bodies do not count as execution
while the `try` body and `finally` remain reachable. Each supplied fixture was
reproduced red in both modes and has a dedicated control and designated mutant.
Six newest exact-head findings are repaired after all supplied fixtures were
reproduced red in both modes: possible writes in unproven exception handlers
and `else` bodies invalidate downstream bindings without crediting those
branches; wrapper discovery applies the same `try`/`finally` boundary; empty
comprehensions and unconsumed generators cannot prove forwarding; a definitely
terminating `finally` stops the enclosing sequence; a destructured JavaScript
lexical declaration shadows an outer URL binding; and Python `nonlocal` static
assignment updates the nearest enclosing binding. The exact 140-control,
192-mutant results remain bound to that historical source stage.
Seven later review findings were reproduced red and repaired: unknown-branch
writes retained stale provenance; a handler-free terminating `try` did not
terminate its enclosing sequence; destructured helper binding did not shadow a
module helper; composite JavaScript child-process arguments credited nested URL
names; decorated helpers retained their original bodies; an untrusted local
`fileURLToPath` was accepted; and a non-string executable reached interpreter
membership. Those repairs raised the bounded suite to 148 controls and 199
mutants and truthfully exposed three additional retained tools.
Eight further unresolved review findings were then reproduced red before the
current repair: exception-boundary possible effects did not invalidate helper
and forwarding-wrapper identities; exception target names left stale bindings;
truthy non-iterable literals credited comprehension bodies; unsupported or
exit-only interpreter options could select a later script; assignment
expressions did not update static bindings; nested aggregate writes retained
reachable alias provenance; decorated forwarding wrappers were registered; and
JavaScript URL names in child-process options positions were credited. The
current source fails closed at all eight boundaries. Ten additional adverse
findings were then reproduced red and repaired: multiline execution-order
resorting, loop-write retention, subprocess executable override, class-scope
leakage and class runtime shadowing; plus shell-command attribution, shadowed
JavaScript `URL` construction, stale URL-property provenance, unreachable
short-circuit calls and reused Python helper-local frames. The bounded
subprocess rule now accepts exactly one ordinary argv positional argument,
rejects dynamic/true `shell`, `**kwargs` and executable overrides; URL credit
requires the intrinsic unshadowed constructor and an unmutated binding;
JavaScript call reachability handles only proven single literal short-circuit
operators; and each helper invocation receives fresh local bindings, imports
and nested definitions. Seven further exact-head review findings were then
reproduced red and repaired. Unknown Boolean operands and unknown wrapper
branches now apply possible effects and invalidate stale direct and forwarding
bindings; URL attribution requires a complete initializer rather than a
property suffix; JavaScript conditional expressions traverse only the selected
literal arm; aggregate `str(...)` conversion preserves only values already
known to be strings; direct helper calls must statically satisfy the required
signature before their bodies are traversed; and nested closures resolve
lexical bindings from their enclosing owned scopes without falling back to an
unrelated module name. Six further review findings then closed unknown
conditional-expression effects, wrapper calls that cannot bind, stale import
identity after assignment expressions, tuple/list augmented-assignment
conflation, unbraced literal-false JavaScript branches, and deferred concise
arrow bodies. The final five findings preserve annotation-only imports, bind
positional, keyword, default, `*args` and `**kwargs` aggregates by identity,
apply `unittest` lifecycle effects in execution order for each test, exclude
quoted URL declarations, and invalidate mutable URL-object aliases on escape.
The current source passes the 183-control, 249-mutant suites above. The current
review/adverse-defect total is 135;
exact-final-head review and hosted checks remain required before integration.

## Current verification boundary

The full federation inventory, proof dispositions, required tooling admission,
cross-repository synchronization and final common G0-G11 pass remain unfinished.
Source-only checks and individual successful jobs do not complete the twelve
baseline criteria. The closed-unmerged
[foundation candidate #695](https://github.com/d6g8k5htny-coder/main/pull/695),
preserved at head `82d9e480a51b946b3e74525ccf32139129d37d28`, is a recoverable candidate
handoff, not the current default-branch baseline.
The pre-integration Main source was `11e28815163e15223e8827fe6ff47a1faaf36ebd`.
[PR #713](https://github.com/d6g8k5htny-coder/main/pull/713) merged on 2026-10-31 at 10:18:44 UTC
as `9abd2c100ae105103945073a7fa6c47b15679269`; its qualification evidence
remains bound to the sources actually checked, not automatically to later
compositions. The proof catalogue and other foundation material in #695 are
preserved outside the default branch, not an admitted current baseline.
This intermediate entry is not a completed cold-start acceptance test.

## How to verify

The existing formal component can be run with
`python -B tools/formal_gate_check.py --run-lean` using the toolchain pinned
in `formal/lean-toolchain`; consult [the formal guide](docs/FORMAL_VERIFICATION.md)
for setup and exact scope. Retain its actual build, kernel, axiom and negative
control outputs. This command alone is not the full G0-G11 federation run.
For the restored bounded replay controls in this PR, use Python 3.11.16 and
run `python3.11 -B -S run_mutations.py --output /new/output` from
`cbz/evidence/kernel/additional/replay-tool/`; see its [README](cbz/evidence/kernel/additional/replay-tool/README.md)
and [governance scope](GOVERNANCE.md#restored-bounded-replay-controls).
This checks the documented Python validator and its three mutants per mode;
it neither authenticates a historical execution nor performs a Lean kernel replay.
The retained Python 3.9 mutation-grader failure is not a passing qualification.

The empirical fuzzing harness and contextual federation crawler are required
but not admitted by this entry. Applicable source-bound empirical validation is
necessary for a new CLOSED decision, alongside the formal gates; unsupported,
empty or missing coverage cannot silently pass. #712's original green controls
do not resolve its later graph and Lean-context findings. A bounded repair is
not full AST/type/theorem synchronization. New gating tools require documented
commands and self-verification before admission; unavailable or unsupported
coverage remains explicit. No documented command here runs all G0-G11 yet.

The empirical statement check (G12) is a registered bounded component on `main`
([PR #726](https://github.com/d6g8k5htny-coder/main/pull/726) `d0d384b1`, extended by
[#742](https://github.com/d6g8k5htny-coder/main/pull/742), [#743](https://github.com/d6g8k5htny-coder/main/pull/743)
and [#752](https://github.com/d6g8k5htny-coder/main/pull/752), merged `a7295bf20741c111da06d72faa846536ff76fa1c`).
From a checkout, run:

```
python3 -B -S -m unittest tests.test_cbz_statement_fuzz -v
python3 -B -O -S -m unittest tests.test_cbz_statement_fuzz -v
python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json report.json
```

It reads every registered Lean statement of a formal package (and, in census mode `--lean-dir`, every
declaration of a directory) independently of Lean in exact arithmetic with Lean's conventions, samples binders
before the colon at exact pathological values and on finite discrete models, grades itself by a mutation
contract, and reports PASS / CORROBORATED / FAIL / UNSUPPORTED fail-closed; the
[registration](GOVERNANCE.md#empirical-statement-check-for-core-lean-arithmetic-skeletons-bounded-registration)
records commands, hosted self-verification, limits and the row field
`G12: PASS | CORROBORATED | not applicable (<construct>)`. At `a7295bf2`: main `formal/` 59/59 PASS; Math-
`formal/` 3 PASS · 56 CORROBORATED · 4 UNSUPPORTED (named); every Lean declaration of Math- at `dfa9b30a`
(3,782): 7 · 128 · 0 · 3,647, each UNSUPPORTED classified by construct. CORROBORATED is admitted evidence,
never a proof; a PASS is statement-level evidence only: not kernel evidence, not scientific acceptance, not
the common G0-G11 run.

The finite MomentTail component is a registered bounded candidate in
[Math PR #545](https://github.com/d6g8k5htny-coder/Math-/pull/545), exact source
`e1cf617b6bc979b75d420f28df17ddad59426522`. In an isolated checkout of that Math
source, run all six commands:

```
python3 -B cbz/finite_moment_tail_20261030/test_adapter.py -v
python3 -B -O cbz/finite_moment_tail_20261030/test_adapter.py -v
python3 -B -m unittest discover -s formal/tests -p test_moment_tail.py -v
python3 -B -O -m unittest discover -s formal/tests -p test_moment_tail.py -v
python3 -B cbz/finite_moment_tail_20261030/adapter.py --source-root . campaign --seed 20261030 --count 10019
python3 -B -O cbz/finite_moment_tail_20261030/adapter.py --source-root . campaign --seed 20261030 --count 10019
```

The [contract](https://github.com/d6g8k5htny-coder/Math-/blob/e1cf617b6bc979b75d420f28df17ddad59426522/cbz/finite_moment_tail_20261030/CONTRACT.json)
and [bounded governance registration](GOVERNANCE.md#finite-momenttail-empirical-component-bounded-registration)
record the six-command hosted job, exact source/test evidence and limits.
Its CPython 3.11.16 run actually passed; it covers one finite specialization,
not a federation-wide empirical gate or a new CLOSED decision. Existing pins
and the global unsupported inventory are unchanged by this registration.

## Team and reconstitution

Use the discovery, proof-closure, tooling, verification, integration and
documentation roles in [GOVERNANCE.md](GOVERNANCE.md). Roles are fluid;
ownership is not a lock. Start from the current pins, dependency manifests
and registry graph, then assign the highest-value closable work to available
agents. Record actual actors, source cuts, conflicts, decisions and handoffs
in [DECISIONS.md](DECISIONS.md) and [CONFLICTS.md](CONFLICTS.md).
The following is a bounded execution observation at 2026-10-30 00:13:19 UTC,
not an exhaustive list of all agents or a claim of independent scientific review:

| Actual actor | Current role and delivered scope | Source or execution boundary |
|---|---|---|
| OpenAI/Codex chat `01a11d37-16a7-7d32-a8b1-8131336243bf` | Documentation composition; R3 delivery coordination | This additive documentation contribution; [task record](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6511442656) |
| `/root/cbz712_r3_registry_repair_20261029`, delegated in that chat | Tool repair | Two-file staged R3 tree `d6f8eb15af4216eb93352b96382264e5d39525cf` on `15e8f689`; nine focused and original 43 controls pass in both modes; original six retain five separate failures |
| `/root/cbz712_r3_source_review_20261030`, delegated in that chat | Fresh nonauthor engineering verification | Exact R3 tree only; independent five real-Git/CLI controls per mode; same provider, no organizational independence or scientific acceptance |
| OpenAI/Codex chat `01a0bbb5-2fcb-77f0-b78b-4d220ddd7ab2` | Integration and replay-source recovery | [PR #713](https://github.com/d6g8k5htny-coder/main/pull/713); source-qualified handoff and final composed hosted checks remain distinct |
| OpenAI/Codex chat `01a11c48-e9e4-7113-a03f-24b2155da25e` | Tool repair and compatible composition | [#712 source repair pickup](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511525318); separate [PR #715](https://github.com/d6g8k5htny-coder/main/pull/715) trigger contribution |
| OpenAI/Codex chat `01a103c8-7a8e-7461-8e9f-36e3b02757c5` | Completed bounded adversarial qualification | [#712 original six findings](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511043716); [bounded qualification completed and only that pickup released](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6511605192); this does not assert that the actor is globally idle |

Re-read native records and exact refs when reconstituting these roles; names
and a shared account are not proof of current capacity or source qualification.
At phase boundaries reconcile evidence and re-rank work. A final baseline
must record its observed team composition and runnable final gate commands;
this role description does not claim that acceptance has occurred.

---

## Historical scientific status snapshot

**Snapshot date:** 2026-09-29  
**Purpose:** one human-readable status page for the public front door.  
**Rule:** this page summarizes source-bound review outcomes; it is not an independent promotion register.

The D5, SARD-G successor and C6 rows below use the [29 September source-bound reconciliation](docs/STATUS_RECONCILIATION_20260929.md), read at Math- `4d062e2a976abc8800a1ae63a49a943526b6bc0d`. They do not execute Math- GRAPH transitions or refresh the museum/shop's explicitly pinned historical snapshots.

## ACCEPT — scoped

| Object | Accepted scope | Source and review | Explicit limits |
|---|---|---|---|
| **D1 — parent lifetime theorem (Theorems A, B, C)** | Theorem A `0 <= 1-p_r <= C r^3`, uniform over compact births, compact positive gaps and all frames; Theorem B compact-mark candidate/elder densities `~ c_{B,K} ell^(-1/3)` with difference `O(ell^(2/3))`, for `B` and `K` of positive length; Theorem C unrestricted leading term `c_{d,L} ell^(-1/3)`. Fixed `d>=2` and `L`, existential constants, read with the congruence erratum, the Section 9 replacement v1.1, wording W1 and `r < L/(4 sqrt 2)` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md) · [reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md) · [Math-#126](https://github.com/d6g8k5htny-coder/Math-/pull/126) | No numerical `C`, `r_*`, `z_*` or coefficient; no unrestricted difference rate; this row is confined to the Math- `8224783` reconciliation, whose lower composition is planar (`d=2`); the separately reviewed every-fixed-`d` lower bound `1-p_r >= c r^3` was integrated afterwards by [Math-#129](https://github.com/d6g8k5htny-coder/Math-/pull/129) at `e7f8aca41c308f8ad8614b587f75916943c86f27` and is not part of this row's reconciliation scope; no RN/24-jet certificate |
| **D2 — unrestricted lifetime remainder** | Theorem R at its stated existential `O(1)` remainder scope: candidate and elder densities have leading `c ell^(-1/3)` term with bounded remainder for sufficiently small lifetime | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206) | No numerical remainder constant or numerical lifetime cutoff; no RN/JETMOD closure |
| **D3 — SIDE24 coefficient calculation** | The coefficient expression for SIDE24 in dimensions 2 and 3, including cone moments, all-direction periodization comparison, covariance/Schur transfer, and outward special-function arithmetic | [Proof/package](https://github.com/d6g8k5htny-coder/Math-/tree/main/coefficients/side24_v1) · [review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841779222) | Coefficient arithmetic acceptance is separate from any imported parent theorem and from a finite-radius error band |
| **D4 — fixed-remote RN count theorem** | Fixed positive spatial separation `rho`, fixed witness separation `eta`, between-pin height window; conditioned covariance, three determinant factors, endpoint normalizer, contact kernel, and fixed-separation factorial moments | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md) · [review](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841861362) | Does not cover shrinking `rho` or `eta`, pin neighborhoods, intermediate scales, all remote heights, or global RN/24-jet closure |
| **D6 — P15 full-price theorem** | Realized disjoint capacity/clutter family with `d_i >= 2`, all independent probabilities, `c_v <= phi(p_v)`, same palette `K >= K_H(d)`, sharp `rho*=1/(3-log(3e-2))` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/full_price_20260924/PROOF.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/74#issuecomment-5842112010) | Demand one remains an obstruction; no arbitrary-downset or unrestricted prize-problem extension |
| **D5 — planar pin neighborhoods and global window first moment** | Original weighted six-observation pin law, fixed torus and compact positive-gap marks, all planar frames and indices, counting only critical points other than the conditioned pins M and S: all-height punctured pin disks and fixed scaled-ball/collar first moments are `O(r^3)`; nested `kappa*r^2` microdisks are `O(r^5)`. Intermediate shells and the torus-wide first moment are `O(r^3)` **for the stated between-pin height window**. Existential constants only | [Source-bound reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/79f9a73596b1ce925034bd426c005a3251693c1b/reviews/d5_reconciliation_20260929/RECONCILIATION.md) · [planar P2 continuum review](https://github.com/d6g8k5htny-coder/Math-/pull/141#issuecomment-5894512272) · [source ledger](docs/STATUS_RECONCILIATION_20260929.json) | Planar `d=2`, fixed torus, gap bounded away from zero; no all-height global count, higher-dimensional transfer, numerical RN/24-jet certificate, elder-pairing conclusion or sharp C6 factorial order |
| **SARD-G — qualitative robust-chart successor** | The unconditioned planar periodized field and each fixed admissible original two-pin law with its original determinant tilt are almost surely Morse with distinct critical values and no saddle-saddle gradient connection, hence planar Morse-Smale in the completed-measure sense. Uses first-closed-contact charts, fixed atlas axes, corrected v2 A2, R3/R4 and the pinned-law transfer with its genericity input | [Integrated reading rule](https://github.com/d6g8k5htny-coder/Math-/blob/e947ade51defcec1fb75444efbe91ac7fcf43f60/reviews/sard_closure_integration_20260929/READING_RULE.md) · [transfer review](https://github.com/d6g8k5htny-coder/Math-/blob/e947ade51defcec1fb75444efbe91ac7fcf43f60/reviews/sard_closure_integration_20260929/TRANSFER_REVIEW.md) | Per fixed admissible planar law; historical A1/A6 remains AMEND as written. No common probability-one set over uncountably many targets, C103 regional estimate, R0 Boolean transition, near-connection rate, RN/JETMOD closure or quantitative pin-coalescence uniformity |
| **C6 — Fourier count tail and planar factorial upper bound** | At fixed `d>=2`, torus and compact positive-gap marks, Theorem F gives count tail `C exp(-c x^(2/d))` under the original pin law and determinant tilt. Theorem W composes with the dimension's global window mean. In `d=2`, the reviewed planar mean yields `E[N(N-1)] <= C r^3 log(1/r)` and fixed higher factorial orders `C_p r^3 log(1/r)^(p-1)` for the stated window count N, excluding M and S; the tail cap includes all critical points | [Exact reviewed proof](https://github.com/d6g8k5htny-coder/Math-/blob/4d59319b7fbe0790ebd5a1482bc91c4d9cc2d4f3/frontiers/c6_fourier_cutoff_20260929/PROOF.md) · [nonauthor review 5356335148](https://github.com/d6g8k5htny-coder/Math-/pull/148#pullrequestreview-5356335148) · [reading note](https://github.com/d6g8k5htny-coder/Math-/blob/4d062e2a976abc8800a1ae63a49a943526b6bc0d/reviews/c6_fourier_completion_20260929/READING_NOTE.md) | Higher-dimensional factorial bounds remain conditional on `G_d`; sharp Gaussian `O(r^3)`/`Theta(r^3)` and the marked-Palm route are not accepted here. No numerical cutoff or parameter-uniform constants; the abstract barrier is not a Gaussian counterexample |

## AMEND / open

| Object | Current reason | Source |
|---|---|---|
| **D5 — remaining quantitative obligations** | The scoped planar first moments above do not complete the historical numerical RN annulus partition/area-weighted budget, all-radius RN or RN/24-jet certificates. Separately, Math-#141's P_d/C_d/I_d/G_d continuum counts were accepted and merged for fixed `d>=2` and torus, compact birth and positive-gap marks with `k>=k_->0`, sufficiently small `r`, the original `W_r/Z_r`, and counts excluding M and S. Pin/collar estimates are all-height; intermediate shells and the global first moment retain the stated between-pin height window. Constants remain existential. These later results are separate from the planar reconciliation above. Proposed GRAPH transitions have not been executed by this summary. | [D5 reconciliation and exclusions](https://github.com/d6g8k5htny-coder/Math-/blob/79f9a73596b1ce925034bd426c005a3251693c1b/reviews/d5_reconciliation_20260929/RECONCILIATION.md) · [merged #141 proof](https://github.com/d6g8k5htny-coder/Math-/blob/e1ca400b3414e5a7bec13b91bc4f1e5293df32be/frontiers/d5_dimension_lift_20260929/PROOF.md) · [A/B review5357858391](https://github.com/d6g8k5htny-coder/Math-/pull/141#pullrequestreview-5357858391) · [C review5357882570](https://github.com/d6g8k5htny-coder/Math-/pull/141#pullrequestreview-5357882570) (OpenAI, source-exposed) |
| **SARD-G A1/A6 — original formulation** | Remains AMEND as written. The separately reviewed robust-chart/fixed-frame successor establishes its own qualitative fixed-law theorem; it does not validate the defective original A1 premise or supply a regional C103 estimate. | [Original-formulation review](reviews/sard_g_successor_a1_a6_20260926/REVIEW.md) · [historical PR](https://github.com/d6g8k5htny-coder/main/pull/122) |

## Engineering only

These are useful infrastructure, but they carry **no theorem-acceptance meaning**.

| Surface | What it provides |
|---|---|
| **[query-](https://github.com/d6g8k5htny-coder/query-)** | Public read-only package and exact-source lookup; default commit `76e1ca09` includes the package and the canonical no-dependency unittest command in CI |
| **[Universal-Law-Workspace](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace)** | Supporting federation/navigation map with `scientific_status_authority: false` |
| **main CI/navigation** | Link, structure, and engineering checks; green runs are not mathematical review |

## Where to read — public source custody

The two human maps are this file and the [Math- proof index at the pinned public Math commit](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/PROOF_INDEX.md). Neither file replaces the underlying proof or review.

For exact source identity, use [`docs/public-math/sources.json`](docs/public-math/sources.json). It indexes **2,138 intended public text artifacts** and points to 14 JSON shards. Each artifact row carries:

`repository + path + 40-char commit + Git blob + bytes + sha256`.

The frozen inventory includes the public `Math-` and `main` mathematics selected for publication and explicitly excludes sandbox, quarantine, and personal paths. It is source visibility/custody only.

The machine catalog at [`meta-framework/registry.json`](https://github.com/d6g8k5htny-coder/meta-framework/blob/main/registry.json) currently contains only public GitHub sources (`Math-` plus one declared public `google-drive` replica). The query portable stubs likewise point only to public `Math-` commits. Candidate query stubs are not silently promoted into the catalog.

### Default-branch boundary

- Pinned Math reading checkout: `82247833b5dd58e04291d55353b52938d73ec614`.
- Pinned query checkout: `89456c5033173c69a304c5fffa781888185675cc`.
- [Math- PR #70](https://github.com/d6g8k5htny-coder/Math-/pull/70) landed nine byte-custody copies at `9d7b6802424fb4715b31999066aafca8ee2f3cca`; [PR #75](https://github.com/d6g8k5htny-coder/Math-/pull/75) added their proof-index pointer. These are public default-branch imports, **not newly accepted mathematics**. Source self-labels were not adopted.
- The `main#59` issue-body file in PR #70 is explicitly a transcription, not original source bytes, so it is not counted as byte custody.
- Open draft AMEND/review PRs remain visible but are not represented here as landed mathematics.

## Reading rule

When a reviewed child result imports a parent that remains AMEND or open, keep those facts separate. “ACCEPT — scoped” never means “everything upstream and downstream is accepted.”
