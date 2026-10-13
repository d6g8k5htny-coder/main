# C254 nonauthor integration source review

Disposition: **HOLD FOR COMBINED INTEGRATION — DOWNSTREAM SOURCE GUARDS REQUIRE
REVIEWED REBINDING OR A SEPARATELY ADMITTED ISOLATED ROUTE**. The sparse C254
overlay itself has no blocking accounting finding.
This is an inert review of a sparse candidate overlay and its baseline manifest,
not an executed source gate, mathematical proof acceptance, or an authenticated
independent alignment record. C254 remains **NOT_COMPILED**; tests, compilation,
kernel checking, transitive axiom audit and executable controls remain **NOT_RUN**.
Scientific effect: **NONE**. Organizational-independence credit: **0**.

Reviewer: OpenAI/Codex `/root/c254_integration_review`, fresh nonauthor of the
candidate and integration, delegated AI work for Dylan Roy. This reviewer had no
role in the two modules, API contributions, target registration, or candidate
edits. The same provider as the authors supplies no organizational independence.
The assignment is integration accounting only; a separate mathematical source
review is underway. No primary API research or mathematical acceptance is claimed.

Parent-supplied claim: R17 Work Events 1559, claim
`8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`, expiry 2026-10-11T17:57:44Z.
The reviewer did not independently read Drive or renew/release that claim.
Parent-supplied source base: Math- main
`2bbce4b65a4bccece8f234eeac7efba0f4bcd72b`, tree
`8df166540bee51334dc06d65494dc63d64d4d223`. Local SOURCE_STATE.json repeats this
identity and records observed cut 2026-10-11T16:03:00Z and formal-manifest blob
`9d6f71bd547049b06b0ac00574d60dd176fe8386`. This reviewer did not perform a live
main-ref resolution or independently Git-blob-hash those remote records.
The bounded downstream reads below were independently fetched through native
GitHub at the exact stated base commit.

## Exact reviewed identities

All paths below are relative to `work/continuation254/` unless otherwise stated.
SHA-256 was computed from the actual local bytes, without project execution.

| File | SHA-256 |
|---|---|
| baseline/formal/manifest.json | 40cb6ffc6776717a26d2a0a66815aeef508b0c8411588fff06ff2cec43a752a1 |
| candidate/formal/manifest.json | d2c8f9a29a83c1bee4bb5ffebc085b4e84845713b688407d8b04780ae4526fcd |
| candidate/formal/ResearchFormalCoreR1/TorusTopology.lean | efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4 |
| candidate/formal/ResearchFormalCoreR1/ClosedSuperlevel.lean | 1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e |
| candidate/formal/ResearchFormalCoreR1.lean | aa1ce27f685d17eea45e35dc3d3ac5201ffa128e89e594d1f5181f44e6ba3a50 |
| candidate/formal/README.md | a6b095e320e7ffadf23549b4e0f3941f11863b4a9d9763e90f65f6d8bf60bee9 |
| candidate/formal/SCOPE.md | a67fc4fbd8e5fecb865e1a301926adc69094b245c98e85143200a804ac8e1603 |
| candidate/formal/TORUS_FILTRATION.md | 971e70d69a96661e5be077eb7a2991dfc69288be73b1ebcaa62415bb943ccc8b |
| candidate/formal/blueprint/src/content.tex | 7822720f75a3d81a1bb5aeb5e6f670142a96e89bba6653f04dbc9f7a1540049c |
| candidate/formal/tests/test_d2_schur.py | 202fd5f124bcf266295194794433f8d33e5ec0658516e403da53cfcc8e0115eb |
| baseline/formal/gate.py | 278989bbba7e1f90870ea0d197a996bc0a65b2a4b217e2eb284accb1282356c1 |
| baseline/formal/tests/test_gate.py | 94aaf74693ddb817a3d728344209f9b87f47367a31d2efdf35afd5aca6f2aae3 |
| baseline/formal/tests/test_d2_schur.py | f87bcd034192c1abbc2d4c0cbc0704c744c2865636a5486a19630823301b607a |

All eight candidate files covered by `manifest.files` match their recorded
hashes. The manifest itself is separately bound by its hash above; it is not
listed inside its own `files` map. The five modified predecessor file records
are exactly README, root imports, SCOPE, blueprint and the D2 test. The three
new file records are exactly the two Lean modules and TORUS_FILTRATION.md.
There are 42 bound file records, 10 registered source modules and 104 targets.

## Prefix, pins and predecessor preservation

The baseline target list has exactly 63 entries. An inert structured JSON
comparison confirms that candidate targets `[0:63]` equal the entire baseline
list in the same order. Its suffix `[63:104]` equals the column-one theorem/lemma
inventory from TorusTopology followed by ClosedSuperlevel, in source order:
21 torus targets plus 20 superlevel targets, no omissions or duplicates.
The first eight `source_modules` retain their baseline order, followed by the
two new module paths. Root source is exactly the ten import lines generated
from that ordered module list, with no root declarations or extra imports.

Every baseline `files` record other than the five intentional changed records
is retained unchanged. In particular, the following predecessor module records
are identical in the baseline and candidate manifests:

| Predecessor module under ResearchFormalCoreR1/ | Preserved SHA-256 |
|---|---|
| AlgebraV2.lean | 4480708263c40a2f7f03f12ef7e15f0eb4f8193c2dccbb253921a7ec251863c3 |
| ProbabilityCompanionsV2.lean | 338452a0c2ba24d016059354d543cc995b8cb29c9d4a41d98bb93499f5f8cb8f |
| MeasureBridge.lean | 28ac130db490c3c3e2fbfabf3ac5be43c854ab1468bf573bf5a8c0dd2c50955f |
| WeightedLaw.lean | f31dfd2cd048cc0c0446c54c46d89c9cdd5d7bb9d76e9d8d16b7c034f351d736 |
| MomentTail.lean | 87fe3ee90afc52d8c2652881f1ae195d1b0b15a2ffc18c84d6166d1a4d0de2aa |
| MomentGenerality.lean | fdec55f1602938cd8fdb5b71535962e76a9671bf230dea53868b14147fb3b509 |
| WeightPerturbation.lean | 50ce15709a133da12a0b4333cc653d26c5fd8a412976c975a7c2f3b25ed5c00f |
| D2Schur.lean | d3fb3ed7b4043227f1d5a06e208de9d94eb413c067e9d0f8802388407c9e15b9 |

Gate and gate-test local baseline bytes match their retained manifest hashes
above. Compiler/lock hash records are unchanged: lean-toolchain
`d5edba4e4b8faad9c1baeadb265716d20d03be4d1a2647dc5e35b0c0325bea7b`,
lake-manifest.json
`d63753befccc21923783a5085148e3ff028d97134ea52f5a08bd87ecf1b0b043`, and
lakefile.toml
`d629891716c169404d1f0d76492441c8dd66f1e790b1b8b0819e51c8d5059255`.
The local baseline lakefile bytes were also rehashed and match. Toolchain and
lock bytes, and all eight predecessor Lean module bytes, were not available in
this sparse overlay/baseline for independent byte comparison. Their hash-record
preservation is confirmed, not a fresh check of a fully assembled checkout.

All nine dependency revision mappings are identical:

| Dependency | Revision |
|---|---|
| plausible | 118aa17ee84656b8bd727fef7c458ee8c833385c |
| LeanSearchClient | ddf04cf3949fa556442341e87d47f9f6e6074707 |
| importGraph | e928b72544873815af278d38681b31c0293588e3 |
| proofwidgets | 106ff4fafc74ef4ac99d81dbf3ab399118f497a5 |
| aesop | 355695d523e41d0554926416cba2a2b3544fbbc9 |
| Qq | 6a489d9af5d0c47e5b259e2e8bcdfc1811b5a259 |
| batteries | f2effa3d803fda822b1f97b806c47cf2adfbcbc2 |
| Cli | e92c9f15fdfacc8536f31cfb3b7ad26c3c8cd204 |
| mathlib | d13f23b723b8a846827a245b89c10fc7d3f11612 |

## Definition consumption and Lean inventory

Both new modules have one `ResearchFormalCoreR1` namespace, all 41 targets start
at column one, and no additional theorem/lemma declaration escaped the unchanged
gate inventory pattern. Inert inventory of candidate `.lean` files finds only
the root and the two registered new modules. Thus no unregistered Lean file is
present in the exposed overlay. The complete base-plus-overlay inventory was
not assembled or checked by this reviewer.

The exported torus definitions/abbreviations are linked into registered theorem
statements or proofs: `torus_E`, `torus_latticeVec`, `torus_periodBasis`,
`torus_periodLattice`, `torus_quotient`, `torus_proj`, `torus_isPeriodic`, and
`torus_descend`. The basis-sum and membership targets consume the scaled basis
and lattice; projection/covering/compactness/path targets consume the quotient;
periodicity and descent targets consume the predicate and quotient eliminator.
The three lattice instance declarations supply typeclass construction text.
Their elaborated selection and dependencies remain unobserved.

The exported superlevel definitions are likewise consumed: `superlevel_set`
by controls/monotonicity/closedness and the later constructions;
`superlevel_inclusion` by identity/composition targets;
`superlevel_componentMap` and `superlevel_componentInclusion` by their registered
computation/identity/composition targets. `superlevel_filtration` is used by
`superlevel_h0Filtration`, which is explicitly present in both registered H0
map-law theorem statements and proofs. Therefore the actual H0 definition is
not merely an unused declared object outside target accounting. This is a
textual consumer graph, not an observed transitive axiom report.

No new source contains `sorry`, `admit`, `native_decide`, `axiom`, or `unsafe`.
That text observation is not a substitute for elaboration or transitive audit.

## D2, blueprint, documentation and gate

The complete baseline and candidate D2 test files were read and diffed. Exactly
three assertion lines change: `[47:]` becomes `[47:63]`; exact total 63 becomes
at least 63; last-module equality becomes module membership. The exact 16-target
D2 window and its ordered names remain asserted. All mathematical test bodies
and their 35/135 case-count assertions, denominator/positivity assumptions,
Gaussian/singular controls, and wrong-sign/missing-factor controls are byte
unchanged. No test was run. This generalization admits an appended module suffix;
the package's unchanged exact source-declaration inventory gate still supplies
the full target/module registration constraint.

All 11 blueprint `\\lean{...}` links name registered targets, including the four
new links. No `\\leanok` occurs. Root imports, blueprint and both new modules
match their candidate manifest hashes. Link validity is static; neither compiled
declaration resolution nor semantic equivalence is claimed.

README/SCOPE accurately state 104 package targets, retained 63-target predecessor
layer, 41 new targets, and 21/20 new-module counts. They distinguish ordinary
formal logical-control declarations from the unchanged five executable rejection
controls. They retain the open smooth/Morse, critical endpoint, finite-event,
H0/component naturality, elder-label, tameness and finite-bar obligations.
TORUS_FILTRATION explicitly limits Field correspondence to the literal formula
and its cited source hash, without an imported/compiled theorem naming Field.
This reviewer did not re-read or hash Field itself.

There is no source evidence-label conflict requiring an integration change:
`formalization_status: proved` is required by the unchanged source gate and
defined by the README as proof text supplied without a successful trusted run.
The candidate documentation simultaneously and explicitly says NOT_COMPILED
and execution NOT_RUN; a run receipt would separately claim kernel-checked.
Consumers must retain that stated interpretation. This review supplies no new
run receipt or status promotion.

The entire 237-line gate and 302-line gate test were read. No candidate gate or
gate-test overlay edit is present, and retained hash records match baseline
bytes. The gate retains exact root imports and ordered theorem inventory,
unregistered-Lean exclusion, source hashes/path/symlink checks, fixed compiler
release/commit, dependency-lock equality, fresh build, leanchecker, transitive
allowlisted axiom audit, all five negative controls, post-execution source/version/
dependency rechecks and source-specific alignment/lineage validation. The D2
generalization does not weaken those package constraints. No gate invocation,
test invocation or process result was observed.

## Actual exposure and limits

### Important finding: downstream exact-source consumers reject the new identity

After the initial overlay comparison, root reported an existing downstream
guard. This reviewer independently fetched and inspected the following exact
base-commit sources through the native GitHub connector. The check.py reads
cover lines 1–75; workflows and replay scripts were requested at 1–100 and are
shorter than that bound, so their full returned text was read.

| Source at base commit | Returned Git blob SHA |
|---|---|
| companions/d2_square_schur_20261006/check.py | 4e02c2d939d1f5b19a72c562df6174c2cb4c4dbc |
| companions/d2_moment_bridge_20261006/check.py | 882130d0a5428799c654b60c3ba01903df970d7d |
| companions/d2_atom_integral_20261006/check.py | d7e7df7dfc49de50596059db3d11dd3f552cb0c0 |
| .github/workflows/d2-square-schur.yml | 8ea3f10edd8b73bb7c98af622304c7994e5373ba |
| .github/workflows/d2-moment-bridge.yml | 069d028c5080f9f81728d6aca0c557d0fa2abeaf |
| .github/workflows/d2-atom-integral.yml | bcba3e01b82a8d7194a2a5c10e6dad230b2dc4d6 |
| companions/d2_square_schur_20261006/replay.sh | f5bc850272b3823fde320894142078abafdc6d0d |
| companions/d2_moment_bridge_20261006/replay.sh | 85c0de87f883386a8a36f5db79c51391313f09c9 |
| companions/d2_atom_integral_20261006/replay.sh | a4247be9ff3eff0229fef62c9b4a6b792159a6c7 |

All three checkers require the exact whole `formal` Git tree
`38b6d1a7746075b560349d184a461933745c11ce` and manifest SHA-256
`40cb6ffc6776717a26d2a0a66815aeef508b0c8411588fff06ff2cec43a752a1`.
Square pins are at lines 21–23 and enforced at 59–61; moment pins are at
16–18 and enforced at 59–61; atom pins are at 21–23 and enforced at 64–66.
Their core gate pin is unchanged and compatible with C254, but the full tree
and manifest deliberately change in this candidate. Candidate manifest hash
`d2c8f9a29a83c1bee4bb5ffebc085b4e84845713b688407d8b04780ae4526fcd` is unequal
to the pinned predecessor. This is a definite source-identity mismatch, not a
speculative elaboration concern.

Each workflow has `formal/**` in pull_request and main-push path triggers
(square/moment line 7 and 13; atom line 9 and 17). Each replay invokes its
`check.py source` before suites and `formal/gate.py --execute`, under
`set -euo pipefail`. Therefore an ordinary C254 PR/main integration is statically
expected to stop those three replay jobs at the intentional source guard,
before their Lean execution. The prior installation/cache workflow step can
still occur; no claim is made that no process at all would run. No workflow was
triggered and no observed failure or mathematical proof failure is claimed.

Required resolution: coordinate these existing owners and obtain a reviewed
consumer-source rebind, including the associated source metadata/contracts and
validation, or use a separately admitted isolated execution route that does not
pretend these old consumers cover the new core. Do not remove the guards, weaken
the gate, blindly replace hashes, or trigger a knowingly incompatible master/PR
route as evidence. This reviewer made no edits to those owner sources. Root is
publishing an actionable integration task with the native source/review handoff.

This finding changes combined readiness to HOLD while leaving the overlay's
local accounting conclusion intact. The nine returned blob identifiers are
connector metadata; this reviewer did not independently hash full remote blobs.

### Full actual exposure

Substantively read in full: workspace AGENTS.md; PLAN.md and LEDGER.md; baseline
formal manifest, gate.py, tests/test_gate.py, tests/test_d2_schur.py and
lakefile.toml; candidate formal manifest, README.md, SCOPE.md,
TORUS_FILTRATION.md, root imports, full blueprint, full tests/test_d2_schur.py,
full 244-line TorusTopology.lean and full 195-line ClosedSuperlevel.lean; and
task-1-report.md. Baseline README/SCOPE/root/blueprint were inspected through
complete diffs of the candidate changes. SOURCE_STATE.json was parsed as inert
JSON; only its top-level identity/runtime summaries and baseline-source-file
metadata were substantively inspected. The full 2294-line metadata record was
not substantively read. The integration author's Perl script and API briefs
were not read or run.

A quick memory registry search for this workspace/C254 returned no matching
entries and supplied no factual basis for the review. The using-superpowers
entrypoint was read; its explicit dispatched-subagent stop instruction applied.
No other skill workflow was invoked.

Operations used: cat/sed/rg/wc for inert text and inventory reads, diff for inert
comparisons, shasum and in-memory Perl Digest::SHA for hashes, and Perl JSON::PP
for structured text comparisons. No Python, Node, Lean, Lake, project scripts,
project tests, compiler, Docker, runtime, external write or child-agent spawn
was used. The sole written artifact is this review. No candidate source was edited.

No blocking defect was found in the exposed overlay accounting. Combined
integration is HOLD for the independently confirmed downstream guard finding.
Before this evidence can be consumed as executed formal evidence, the authoritative
base-plus-overlay bytes, full inventory and the unchanged gate must undergo the
admitted exact-source hosted route. Mathematical source review and any required
independent alignment remain separate. This scoped disposition does not release
ownership, admit runtime, authorize a merge, close C253/C254 mathematical gates,
or supply scientific acceptance.
