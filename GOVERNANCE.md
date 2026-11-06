# CBZ-0 governance

Effective 2026-10-29, by the owner's **MASTER DIRECTIVE — CRYSTALLINE
BASELINE PROTOCOL**. This is the current operational authority throughout
the federation. Earlier owner mandates, freezes, family-review gates,
approval forms and custody ceremonies are historical where they conflict.
Platform permissions and denials remain binding. Historical mathematical
statements, test failures, authorship and verification outcomes are facts;
this governance change does not rewrite them.

## Current owner source

The latest complete master is preserved [verbatim in this repository](governance/MASTER_DIRECTIVE_20261029_SWARM.txt)
and [in main#686](https://github.com/d6g8k5htny-coder/main/issues/686#issuecomment-6511339492):
36,617 UTF-8 bytes, SHA256
`83ac6552fc6406e95213a1cfd6e8636095215e11aac6e3db473cd42c22275a8b`.
Its team-model sections 1.5, 7.4 and 8 supersede older exclusive-owner and
monitor-specific integration restrictions. Historical decisions retain their
original timestamps and outcomes; the latest directive controls future action.

## Mission and authority

The sole mission is Clean Baseline Zero (CBZ-0), not additional research.
Agents may build, repair, replace, integrate, archive or remove operational
material needed for that mission. No owner question, selection, approval,
external review quorum or waiting period is required. Resolve ambiguity
conservatively for mathematical rigor and aggressively for consolidation.
Coordinate independent tasks without blocking on a peer's approval.

Priority: Layer A; CBZ-0; reduced maintenance; cold-start legibility;
auditor soundness; speed; aesthetics. Select work by closure, unblocking
and maintenance benefit, less effort and ten times the rigor risk. On
conflicts preserve the mathematically stronger valid result; prefer the
simpler operational structure; combine compatible work. Archive a genuinely
incompatible alternative with its exact boundary and evidence.

## Layer A: immutable

1. No logical placeholders or stubbed theorems presented as closed.
2. No unverified axiom injection.
3. No floating-point certification of mathematical bounds: exact rational
   or justified interval/real reasoning only.
4. Never loosen a mathematical bound to pass a check.
5. Preserve failing tests; repair or isolate them with a written boundary.
   Consolidation requires evidence of redundancy under a stricter proof.
6. Mark conclusions relying on unsatisfiable hypotheses in the proof index.
7. Every status statement must describe observed source/build/audit state.
8. Certifying tools require their own tests, including adverse mutations;
   they must not skip cases silently or manufacture a passing result.

Kernel verification establishes only its exact formal statement and
declared axiom dependencies. Prose review, finite algebra checks, hashes,
merges and engineering CI cannot establish kernel closure or a stronger
mathematical interpretation.

## Layer B: operational statutes

- **B-001 (2026-10-29):** Canonical federation authority is this file;
  `STATUS.md` is the complete cold-start entry. `PROOF_INDEX.md` records
  exact CLOSED/kernel evidence or QUARANTINED boundaries. `TOOLING.md`
  registers mission tools; `BASELINE.md` binds snapshots, commits and gates.
  Rationale: replace dispersed procedural approval chains with reproducible
  evidence. Scope: all in-scope repositories.
- **B-002 (2026-10-29):** Decisions are append-only in `DECISIONS.md`.
  Date, phase, layer, action, rationale, reversibility and evidence are
  required. Discovery evidence is timestamped and source-bound; a changing
  federation is not represented as an atomic observation.
- **B-003 (updated 2026-10-30):** Agents share accountability for CBZ-0.
  Ownership identifies the actual writer and current source scope; it is a
  coordination hint, not an exclusive lock or approval gate. Preserve compatible
  active work, communicate exact file/ref scopes and use guarded updates.
  Reassign or absorb blocked work explicitly in the decision ledger.
  Canonical documents remain single coherent surfaces; no agent identity,
  including a shared `/root` label, confers exclusive integration authority.
  Rationale: implement the owner's dynamic team model without silent overwrite.
- **B-004 (2026-10-29):** Snapshot before recoverable/destructive operations;
  one extra verification cycle before irreversible published history work.
  No silent deletion. Preserve removed material in a documented tag,
  recoverable archive or bundle. No history rewrite is needed merely for
  aesthetic cleanup.
- **B-005 (2026-10-29):** A gate is PASS only after its command ran and its
  output and exit were observed at the recorded source. Source-only checks
  and historical hosted receipts must not substitute for fresh execution.
  Final CBZ gates must all be observed in one final run.
- **B-006 (2026-10-29):** An auditor certifies only the properties and
  domain its implementation and adverse tests establish. Explicitly
  report unsupported claims or cases; never call them proven by a lexical
  scan. Quarantine classifications do not erase detected hazards.

## Quarantine

Every proof quarantine records an ID, title, exact source location, Layer A
status, exact proven/unproven boundary, mathematical blocking reason,
attempts, resumption cost, isolation, owner `none` and maintenance `zero`.
Closed formal code must not import a quarantined logical dependency.
Quarantine is dormant until deliberately reopened. No new research is
authorized merely because a quarantined question exists.

## Execution and termination

Discover all repositories/refs/PRs/proofs/CI/admin material and tool gaps;
write the inventory. Finish or quarantine obligations, integrate compatible
work and archive obsolete ceremony. Run G0 through G11; repair discrepancies
at source. Publish the canonical handoff and perform a fresh-clone test
using `STATUS.md` alone. Each iteration must make a measurable change;
after two no-op iterations diagnose and change strategy.

Do not declare CBZ-0 until all twelve owner criteria are observed together:
clean Git, no dangling deliveries, all proofs closed/quarantined, no Layer A
violations, all regressions green, Lean/kernel/axioms green, mutation target
met, pins consistent, tools documented/verified, canonical surfaces truthful,
cold-start reproducibility and complete decisions/no administrative residue.
A resource-limited terminal state must explicitly be a Partial Baseline,
with precise isolated residue and unmet criteria; it must never be called
CBZ-0.

## Tool registry

No new certifying tool has yet been admitted by this mission. Existing tools
are being inventoried against their actual tests and execution coverage.
Mission tools must be finished, documented with exact usage and verification,
and registered here and in `TOOLING.md` before final admission. Tools are a
means to closure, never a reason to open a general infrastructure project.

### G11 bounded static-attribution repair

**Registered 2026-11-06; review-repaired 2026-11-06; G11 admission remains
pending.** The repair at source commit
`06b81c26498b092e3ff600c02a2ac985fb5fbb50` (tree
`54da40e6e17a46e7abbedc2252c2b86448cf4ce1`) extends the existing bounded
G11 census only far enough to recognize explicit tool paths delivered to
trusted Python execution/loading APIs or named Node child-process imports.
Run `python3 -B -S [-O] -m unittest tests.test_g11_census`,
`python3 -B -S [-O] tools/cbz/g11_mutation_selftest.py`, and then
`python3 -B -S [-O] tools/cbz/g11_census.py --rev <commit> --json <path>`.
The exact source commit passed 94/94 controls and killed 145/145 designated
mutants in each mode; the sealed mutation-inventory SHA-256 is
`ee49fb05808c750d45cf8dd954be84e6b0f6f46cb4c7fa0951406e4ade5cac07`.
Its normal and optimized whole-tree reports were
byte-identical (SHA-256
`b38bf67f776d6e2467a2b8931200bbc1124f3b6f5cc279ca11b973089fd9dcf5`,
56,078 bytes each):
zero production findings and 29 retained tool candidates, so both
census runs correctly exited 1 and G11 remains FAIL. The review repair rejects
interpreter command-mode payload arguments and unreachable wrapper bodies,
binds JavaScript references only inside the selected child-process call,
stops absolute-parent traversal at the filesystem root, and recognizes
reachable function-local process imports. Static attribution is not runtime
execution, completeness of arbitrary Python/JavaScript semantics, Layer A
clearance or scientific evidence. Tuple/list assignment targets are bound only
to their corresponding resolved values; unsupported destructuring is left
unresolved rather than credited as one aggregate. Wrapper parameters lose
forwarding status after reachable rebinding, unresolved local names cannot
inherit module-level path values, and quoted or property-only JavaScript names
do not count as binding expressions. Wrapper arguments use the same
interpreter command-mode filter as direct process calls; augmented assignment
replaces or clears the prior static value; and a property call cannot match a
named child-process import. Unresolved sequence elements preserve their command
position: an unknown value before a candidate tool refuses attribution, while a
dynamic suffix after a statically selected tool does not erase that proven
prefix. Boolean `or` uses left-to-right short-circuit semantics; local callees
cannot inherit module wrappers; quoted JavaScript call text is masked; resolved
starred command elements, `unittest` lifecycle methods and trusted
`spec_from_file_location`/`run_path` wrappers are recognized. These are bounded
static facts, not runtime or scientific certification. Class methods remain
separate from bare module wrappers; statically known conditional expressions
select one branch, while unknown alternatives earn only intersection evidence;
and a JavaScript URL binding loses provenance after an intervening assignment.
All `pathlib.Path` constructor components must resolve before a path is
credited; a subscript or slice assignment clears the root static aggregate;
and a named child-process import is not credited inside a function whose
parameter shadows that import. Interpreter mode switches are rejected only
before script selection, so later `-c`/`-m` tokens remain program arguments.
Element and slice writes invalidate wrapper forwarding as well as direct
aggregates; helper-local invalidation cannot fall back to a module binding;
and JavaScript default initializer references are not parameter bindings.
Recognized interpreter options consume their operands before script selection;
element writes invalidate every identity alias of a resolved command list; and
a function parameter shadowing a JavaScript URL binding blocks the outer
binding inside that function.
Supported command-list mutator calls invalidate the root binding and its
resolved identity aliases before direct or wrapper attribution. A same-function
JavaScript declaration shadows a named child-process import throughout that
function, and a function-local `new URL` declaration is visible only inside its
own function. The recognized mutator methods and function parser are explicit
bounded rules, not a claim of complete Python or JavaScript semantics.
Writes through aliases of wrapper parameters invalidate their shared
forwarding provenance. Reassignment of a wrapper name invalidates the recorded
wrapper before subsequent attribution. JavaScript callee resolution checks
declarations in every containing function, not only the innermost scope.
Wrapper discovery stops scanning a statement sequence after an unconditional
`return` or `raise`; unreachable process calls therefore cannot prove
forwarding. Calls made from deferred functions use the caller's activation
point when deciding whether a module wrapper has been defined or reassigned,
instead of treating the nested call's lexical line as module execution order.
When a later function definition is not a forwarding wrapper, it clears an
earlier same-named wrapper registration. List augmented assignment preserves
the resolved list object's identity so subsequent mutations through aliases
remain visible. JavaScript process calls inside literal-false branches or
uncalled named helpers are not execution evidence. These three review
counterexamples have dedicated adverse controls and designated mutants.
Deferred Python bodies are evaluated in activation order, so later module
assignments cannot rewrite an earlier invocation's static value. JavaScript
helper calls require a reachable chain rooted at module execution; callbacks
are roots only when passed to the imported `node:test` identity. That explicit
positive rule restores the actual `tools/museum_conditionals.mjs` test path
without admitting arbitrary callback APIs. The intermediate 30-tool census is
preserved as a diagnostic rather than rebound.
At Python branch joins, bindings written under the supported `is None` guard
are invalidated rather than combining mutually exclusive states. A
`node:test` callback root is admitted only when the registration call is
itself reachable from module execution, and only when the registrar still
resolves to the unshadowed imported identity. These bounded rules have
dedicated adverse controls and designated mutants.
Each supported cache-guard alternative starts from an isolated fail-closed
binding environment before the join invalidates alternative writes. Reachable
Python helper bodies are expanded at their call position, so their binding
effects precede the caller's continuation. These are bounded source rules,
not general symbolic execution.
The stricter source newly exposes `tools/atlas_formal_source_bridge.py`; the
earlier 28-tool result remains historical and is not rebound.

### Empirical statement check for core-Lean arithmetic skeletons: bounded registration

**Registered 2026-11-01 14:40 UTC; amended 2026-11-06 after main PRs #742, #743 and #752; global
admission of the empirical layer remains pending.**
[main PR #726](https://github.com/d6g8k5htny-coder/main/pull/726) (merged
`d0d384b18a409ed51061458287dde22816782210`) supplied `tools/cbz/statement_fuzz.py`;
[#742](https://github.com/d6g8k5htny-coder/main/pull/742) (merged `d479620b`),
[#743](https://github.com/d6g8k5htny-coder/main/pull/743) (merged `fe327e02`) and
[#752](https://github.com/d6g8k5htny-coder/main/pull/752) (merged
`a7295bf20741c111da06d72faa846536ff76fa1c`) extended it. Current source at that merge:
`tools/cbz/statement_fuzz.py` (blob `6e7d6c1ecd489bcddf5f3b819ed6a9327fcda4a6`, SHA-256
`78ec967a2a2957a203c57b412769d7685f9d953817b5173d4102a813c7120f1d`),
`tests/test_cbz_statement_fuzz.py` (blob `daab0995bdbd0e4779ba2c4b336a2eaf53a6c22b`, 86 tests) and the run
records under `cbz/evidence/g12-statement-fuzz-01n/`. History: the 2026-11-01 registration bound tool blob
`020cb194` (35 tests; hosted run 41242533228 / job 124659189973) with coverage main `formal/` 31/31 PASS at
`4712fc13`, #740's candidate `formal/` 59/59 PASS at `43b10c6f`, Math- `formal/` 63/63 UNSUPPORTED at
`19a55750`; those receipts stay under the evidence directory and in Git history.

Purpose: the addendum's empirical layer for *statements*. Kernel verification is not strengthened by
sampling, so the tool reads each Lean statement and the package's own `def`s — by Lean-qualified name,
resolved from the statement's namespace outward and through its `open`s, with definition bodies evaluated
in their own declaration scope — in an exact interpreter: Python integers, `Fraction`, and the
multi-quadratic field ℚ(√d₁, …, √dₖ) for `Real.sqrt`; a float anywhere is an assertion failure. Lean-core
and Mathlib conventions are implemented where they differ from naive arithmetic: `Nat` truncated
subtraction, totalized division (`x / 0 = 0`, `x % 0 = x`), Euclidean `Int` `/` and `%`, `0⁻¹ = 0`,
`Real.sqrt x = 0` for `x ≤ 0`, exponents typed on their own (`rightact%`), ascriptions elaborated at the
ascribed type, numeral typing by the `binop%` maximal-type rule. Verdicts, fail-closed: **PASS** — a closed
statement (finite quantifier domains enumerated) evaluates true exactly; **CORROBORATED** — a statement
with binders before the colon (ℕ ℤ ℚ ℝ under decidable hypotheses; measure-theoretic statements on finite
discrete models Ω = `Fin n` with rational weights, tables and subsets) was refuted by no sampled instance
drawn from an exact pathological pool, the hypotheses' own boundary values and seeded rationals — admitted
evidence, never a proof and never PASS; **FAIL** — an exact witness, a misstatement or transcription defect
in the Lean text, never kernel evidence; **UNSUPPORTED**, naming the construct the interpreter does not model
(structures, transcendental functions, finite vectors `Fin n → ℝ`, unbounded quantifiers inside a
statement, `∑`, …). UNSUPPORTED is never upgraded and makes the exit code non-zero; `--admit-corroborated`
admits CORROBORATED into exit 0 and is recorded in the report. Census mode (`--lean-dir`) evaluates every
`theorem`/`lemma` of a directory without a manifest, accounts for every declaration (one the splitter cannot
separate is an UNSUPPORTED target, never an omission) and classifies the UNSUPPORTED reasons.

From a checkout of this repository, run:

```
python3 -B -S -m unittest tests.test_cbz_statement_fuzz -v
python3 -B -O -S -m unittest tests.test_cbz_statement_fuzz -v
python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json <report.json>
python3 -B -S tools/cbz/statement_fuzz.py --lean-dir <directory> --exclude /raw --json <report.json>
```

Self-verification: the 86 tests (tokenizer; Lean semantics with the six Euclidean division goldens of
main#684/6511137449; exact real algebra; sampled binders and hypothesis boundaries; finite models;
qualified-name resolution and declaration scope; declaration splitting; census mode; fail-closed behaviour;
the mutation contract; the real packages) ran on the merge commit of #752 in the hosted `navigation` job
([run 42165008899](https://github.com/d6g8k5htny-coder/main/actions/runs/42165008899), job `127719321978`,
CPython 3.11.18, artifact `research-navigation-42165008899` whose `commit.txt` is `a7295bf2…` and whose
`tests-normal.log` shows the 86 `test_cbz_statement_fuzz` cases `ok` among 2,051) and locally in normal,
`-O` and `-W error` modes. `--selftest` grades every module by the three-class mutation contract:
guaranteed-false mutants must FAIL (each comparison replaced by its negation one conjunct at a time, the
right-hand literal of an equality shifted by one — never `<`→`≤`), truth-preserving weakenings (`<`→`≤`,
`>`→`≥`) must hold, and the manifest's declared negative controls that change a statement must FAIL; an
UNSUPPORTED or errored mutant is a survivor, never a kill, and a stubbed evaluator fails the unmutated
control. Mutants carry the binders, so a class-1 mutant of a sampled statement must be refuted by sampling.
Reviews of record: #726 — same-family 01DKKe
([726/6547986788](https://github.com/d6g8k5htny-coder/main/pull/726#issuecomment-6547986788); one P1
fail-open domain inference, fixed by its follow-on) and different-family Grok Bot agent 12
([726/6548050253](https://github.com/d6g8k5htny-coder/main/pull/726#issuecomment-6548050253)); Codex 01a103c8
supplied the division-semantics controls. #742 — Grok Bot agent 8 (engineering, PASS_WITH_NITS at
`48f3a461`), agent 12 (integration), agent 10 (readback). #752 — Codex automated review in two rounds, four
findings (returned-function parameter types; definitions pooled by short name; declarations the splitter
missed; definition bodies evaluated in the caller's scope), all repaired at `95e5de31` and `55571918` and
answered in their threads; Root (OpenAI/Codex, B-003) exact-head qualification and integration
([752/6676699342](https://github.com/d6g8k5htny-coder/main/pull/752#issuecomment-6676699342)).

Coverage actually observed, per package and cut (receipts under `cbz/evidence/g12-statement-fuzz-01n/`):

| Package | Cut | Targets | PASS | CORROBORATED | FAIL | UNSUPPORTED | Mutation contract |
|---|---|---:|---:|---:|---:|---:|---|
| main `formal/` (SIDE24 + P15 skeletons) | `ec6962ff` (`formal/` unchanged since) | 59 | 59 | 0 | 0 | 0 | 184/184 killed, 24 weakenings kept, 15 declared controls FAIL |
| Math- `formal/` (`ResearchFormalCoreR1`, Mathlib real analysis) | `19a55750` = `dfa9b30a` | 63 | 3 | 56 | 0 | 4 (named measure-theoretic constructs) | 72/72 killed, 6 weakenings kept; receipts at seeds 1 and 2, seed 3 recorded in the evidence README |
| Math- every Lean declaration, 16 groups (census, `--lean-dir`) | `dfa9b30a` | 3,782 | 7 | 128 | 0 | 3,647 (classified by construct) | — |

Gate mapping: G0/G5/G6 use the documented tests and the mutation contract; G8/G9/G11 require this exact
scope, runnable commands and self-verification record. For a CLOSED row on a Lean statement, the field this
registry recognises is `G12: PASS | CORROBORATED | not applicable (<construct>)`, recorded with the report
file's SHA-256 and the `manifest_sha256` (package) or source-file SHA-256 (census) it binds, at the same
source cut as the kernel receipt: PASS and CORROBORATED are statement-level evidence, *not applicable*
keeps the row's kernel status and names the construct, and a FAIL blocks the row until the Lean text is
repaired. A per-record annotation of the #695 literal kernel index on this rule is
`cbz/evidence/g12-statement-fuzz-01n/census_math_dfa9b30a_20261104/g12_annotations_literal_index_82d9e480.json`
(367 records: 34 PASS · 69 CORROBORATED · 264 not applicable · 0 FAIL). None of this adds to or takes from
G2–G4. The recorded gap (an exact-rational interval backend for `Real.exp`/`log`) is narrowed, not
withdrawn: no statement of the registered `formal/` packages uses those functions, but the census classifies
49 declarations as *transcendental function* (33 `Real.exp`, 9 `Real.log`, 4 `Real.cos`, 3 `Real.sin`; 39 of
them in the P15 proofs of `specifications/lean_statement_proofs_20261011`), which therefore carry
`G12: not applicable (transcendental …)`; the other UNSUPPORTED classes are named in the census. This
component installs no global CLOSED guard and does not complete the common final G0–G11 run. Maintenance owner: none; stdlib only, self-contained. Actual executor: Anthropic Claude,
Claude Code session `session_01NMeKEismAyeqgdB4sy2NJU`; the declaration-scope repair of #752 by a successor
lane under the owner's account; scientific effect NONE.

### Finite MomentTail empirical component: bounded registration

**Registered 2026-10-30 00:36:59 UTC; global admission remains pending.**
[Math PR #545](https://github.com/d6g8k5htny-coder/Math-/pull/545) supplies
`cbz/finite_moment_tail_20261030/adapter.py`, its self-tests and complete
[CONTRACT.json](https://github.com/d6g8k5htny-coder/Math-/blob/e1cf617b6bc979b75d420f28df17ddad59426522/cbz/finite_moment_tail_20261030/CONTRACT.json)
at head `e1cf617b6bc979b75d420f28df17ddad59426522`, tree
`87cd19c612906a61bca7659fa24e50a3958e0eb3`. This registry entry does not
represent that candidate as merged or update the federation's existing pins.

Purpose: empirical falsification of the finite rational specialization of
`ResearchFormalCoreR1.p02_lm009_moment40_tail`. It uses exact Fraction/integer
decisions for normalized finite laws, signed jet values, any positive r and
epsilon, an actually sufficient fortieth-moment bound, and the strict event.
Actual Lean/test byte hashes are checked before evaluation. Invalid inputs,
false premises, unsupported targets, altered source and empty coverage refuse
success. The two source-file checks do not attest dependencies or run Lean.

From an isolated checkout of that exact Math source, run:

```
python3 -B cbz/finite_moment_tail_20261030/test_adapter.py -v
python3 -B -O cbz/finite_moment_tail_20261030/test_adapter.py -v
python3 -B -m unittest discover -s formal/tests -p test_moment_tail.py -v
python3 -B -O -m unittest discover -s formal/tests -p test_moment_tail.py -v
python3 -B cbz/finite_moment_tail_20261030/adapter.py --source-root . campaign --seed 20261030 --count 10019
python3 -B -O cbz/finite_moment_tail_20261030/adapter.py --source-root . campaign --seed 20261030 --count 10019
```

Self-verification: [run 40980378198, attempt 1](https://github.com/d6g8k5htny-coder/Math-/actions/runs/40980378198),
job `123818653248`, actually ran CPython 3.11.16 on merge
`f38124cbef8f8950dcc59fe0e0155d45cd2b767b` with the same tree. Component
20/20 and original MomentTail 12/12 tests passed in each mode; all six documented
commands exited zero. Each campaign observed the same 10,019 distinct valid
cases with zero violations/refusals. Five implementation mutants produced 20
intended child assertion failures with zero errors across parent/child modes:
three test wrong bounds on valid premises, one tests strict-event output, and
one tests false-premise admission. The latter two are separate semantic/admission
controls, not mathematical bound counterexamples. Artifact `12944860659` was
whole-byte verified against SHA256
`dcf13fa0124c51091731786aec81c03a32b5fe53a8e49f689b1e5901c17f415d`;
its actual source hashes, outputs, counters and child identities were read back.

Gate mapping: G0/G5/G6 use the documented self-tests, unchanged regressions and
adverse controls; the native `finite-moment-tail-self-test` job runs on every PR,
main push and manual dispatch and retains failed outcomes. G8/G9/G11 require
this exact scope, runnable commands and self-verification record. G2-G4 retain
the full axiom/build/kernel requirements, and G7 still requires federation
pins and contextual AST/type/theorem synchronization. This component neither
installs a global CLOSED guard nor supplies the general-real, Gaussian,
weighted/family or remaining unsupported coverage. The full empirical harness,
drift tool admission and common final G0-G11 run remain unfinished.
Maintenance owner: none; self-contained. Actual executor: OpenAI/Codex chat
`01a11d37` with delegated implementation and nonauthor engineering review;
same provider, scientific effect NONE.

## Team structure and coordination

**B-007 (owner expansion dated 2026-10-29):** Operate as a fluid team, not a
sequential coordinator queue. These are mandates, not permanent seats:

| Role | Mandate | Required handoff |
|---|---|---|
| Discovery | Enumerate current topology, obligations, refs, deliveries and gaps | Exact source cut, coverage and unknowns |
| Proof closure | Finish or precisely quarantine existing obligations without new research | Exact statement, hypotheses, kernel/axiom evidence and boundary |
| Tool building | Build the narrow tool needed for closure; include adverse self-tests | Source, commands, exits, supported domain and residuals |
| Verification | Execute the actual gate population and retain failures | Tested composition, raw outcomes and limits |
| Integration | Compose compatible work with guarded, recoverable mutations | Actual integrated source, checks and remaining discrepancies |
| Documentation | Keep canonical surfaces consistent with observed evidence | Source-bound status, runnable commands and team reconstruction |

An agent may change roles or absorb blocked work under the owner's ranking
function. Roles and a shared `/root` alias confer neither exclusive file
ownership nor scientific or organizational independence. Automated checks
and self-review do not become evidence of a broader mathematical claim.

### Conflict protocol

1. Register the incompatibility in [CONFLICTS.md](CONFLICTS.md), with both
   positions, actual actors, source cuts, constraints and affected paths.
2. Apply the hierarchy: mathematically stricter valid content wins for
   Layer A; simpler compliant structure wins for Layer B; combine compatible
   contributions; retain the stronger verified alternative and precisely
   quarantine a genuinely irreconcilable one.
3. Log the resolution, rationale and evidence in [DECISIONS.md](DECISIONS.md)
   before calling it resolved. Preserve the losing work's recovery path and
   all observed failures; no vote, silence or shared account supplies proof.
4. When a conflict class recurs, add a dated Layer B rule here that prevents it.

Agent questions may coordinate work for one work cycle; they are not
approval gates. If a peer is unresponsive or its output is unusable, proceed
with the best source-bound judgment, record the assumption and take useful
work. Do not wait for a human or coordinator. A real platform denial remains
binding; an expected-head guard is race protection, not a permission lock.

### Shared state, handoffs and convergence

Publish current task intent and source changes in the existing ledger and
task-local record. Prepare concurrent reversible changes in isolated branches;
re-read the current tree before mutation and compose rather than overwrite.
A handoff records from-agent, to-agent or available role, task, exact state,
what was done, what remains, discoveries, tools and evidence. The receiver's
actual action is separate from a delivery notification.

At each of the four phase boundaries, reconcile the inventory, conflict
registry and gate evidence, then re-rank work. Phase convergence is an
evidence synchronization point, not a provider quorum or waiting period.
Private or local working files must be represented by an accessible,
appropriately scoped source/evidence handoff rather than invisible state.

### Reconstituting the team

A cold-start agent begins with [STATUS.md](STATUS.md), the exact owner source,
this role table and the decision/conflict records. It discovers the current
federation from pins, manifests and registry edges; assigns the highest-value
closable tasks to available roles; uses isolated reversible work; records
handoffs; and converges on the same final G0-G11 run. The current bounded team observation is in [STATUS.md](STATUS.md#team-and-reconstitution).
The final baseline must record the actual team composition and evidence, not
only this role template.

## Required addendum tools: admission boundaries

The owner's compatible intent-locked addendum requires both an empirical
randomized/pathological falsification harness and a dynamic federation drift
crawler before final baseline lock. Discovery follows active root pins,
dependency manifests and workspace registry graphs, including topology changes.
Neither tool is admitted merely by this requirement or by a green badge.

- G0/G5/G6 require meaningful positive/adverse self-tests and source-bound
  mutation grading. Invalid executions cannot count as kills, and a
  truth-preserving mutation cannot be assigned an unjustified false verdict.
- New CLOSED claims require applicable empirical validation at the exact source
  in addition to kernel, axiom and statement evidence. Empirical success never
  proves a mathematical bound; decisive numerical violations require exact
  arithmetic or justified intervals. Missing or unsupported coverage is explicit.
- G7 includes contextual AST signatures, type definitions and theorem statements
  across declared federation relationships. Lexical equality alone does not
  establish semantic agreement; missing topology cannot silently pass.
- G8/G9/G11 require truthful coverage and limitations, documented runnable
  commands, self-verification and tool registration. The final G0-G11 run remains
  required; separate proposed gate numbers do not replace it.

Implementation and repair continue through the existing mission main#684 and
tool/component task records. This governance change records requirements, not
completed implementations, theorem closure or CBZ-0.

## Restored bounded replay controls

The four Python sources under
`cbz/evidence/kernel/additional/replay-tool/` are restored byte-for-byte from
public commit `260271793f0d53e4348390c5e8708f3280ac4964`, matching their
published SHA256SUMS. The existing README defines use, coverage and limitations;
the Python validator checks transcript consistency and authenticates no execution.

Run `python3.11 -B -S run_mutations.py --output /new/output` from that directory
using Python 3.11.16. The frozen suite contains 31 baseline tests. Preserve its
original reports and logs: the historical runner's `reason_bound_kill` fields
alone do not establish intended-cause qualification. A recorded negative-control
probe showed that it accepts unrelated assertion causes or extra runtime errors
when the expected method header appears. The original six hosted observations
were separately inspected and have their intended causes; they are not withdrawn.

**Recorded mutation qualifier:** `tools/cbz_replay_mutation_qualify.py` is a
separate, bounded post-check; all four historical Python files and SHA256SUMS
remain unchanged. After the historical command, run from the repository root:

```sh
python -B -S tools/cbz_replay_mutation_qualify.py --input /new/output --output /new/qualification.json
```

The qualification output must be fresh and outside the input/source directories.
The optional `--source-dir` supplies the same frozen source set explicitly. The
qualifier pins source, test and runner bytes, recomputes the three mutant-source
hashes, and requires the complete two-baseline/six-mutant population. Every log
must match its full 31-test census, exact recorded exit, no skips/errors/foreign
failures, and the intended final cause. The owned-names mutation checks diagnostic
loss while the module-assignment guard still rejects; it is not a validator-bypass
claim. Duplicate/unknown/missing/nonfinite fields, bool/int exit aliases,
missing/extra/symlink files, altered digests and nonempty stdout are rejected.

Self-verification at the initial source: 14 tests passed normally and optimized
on Python 3.9.6; six source mutants produced named assertion failures in both
modes, with no import/error kills. A separate source-exposed OpenAI/Codex reviewer
ran the 14 tests in both modes and checked the authentic and adverse inputs.
All six wrong-cause/mixed-error probes reject on cause after explicitly labeled
synthetic reported-runtime adjustment; their untouched Python 3.9.6 reports
instead reject at the declared-runtime boundary. These parser checks do not
execute the historical replay program. Exact source and raw records accompany
the task-local review; a hosted result remains bound to its actual source/run.

The `bounded-replay-python-controls` job in
`.github/workflows/cbz-p15-historical-readback.yml` installs Python 3.11.16,
requires both 14-test qualifier suites without skips, runs the historical
controls, then requires the separate qualification output. It retains complete
results on success or failure. The P15 job is unchanged. This binding serves
criteria 6, 9 and 11 only within the recorded scope; it is not actual Lean replay,
whole-federation G0, general grader soundness, or mathematical proof. Actual Lean
controls use the separate documented harness and pinned Lean 4.34.1.

Unsigned internally consistent records do not authenticate execution. This tool
checks recorded consistency; independent native artifact/source custody remains
necessary. The unsupported local Python 3.9 historical-grader failure, synthetic
false-credit cases, source-mutant controls and corrected failed attempts remain
in the evidence. No original test, bound or expected verdict is weakened.
Maintenance owner: none; frozen profile, self-contained. Requalify changed
source/profile inputs; do not convert this registration into global admission.
