# Lean verification specification (LVS) — design and conventions for all workers

Read this whole file before writing anything. It is the contract between the workers of this
programme. Scientific effect of everything here: **NONE**. Organizational-independence credit: 0
(Anthropic / Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`). No claim is promoted, closed,
discharged or reclassified. A Lean statement is not a review; a kernel receipt is not acceptance.

## 0. Non-negotiable vocabulary (existing lane values only — never invent a new value)

* formal-progress labels (docs/FORMAL_VERIFICATION.md:35): `none`, `specified`, `proved`,
  `kernel-checked`. A committed manifest of a *proof* package says `proved`; `kernel-checked`
  appears only in a run receipt. A *statement* package says `specified` and nothing stronger.
* alignment dimension: the only value a manifest/register may carry is `PENDING_INDEPENDENT_REVIEW`.
  Per-target review verdicts (ALIGNED / ALIGNED AT NARROWER SCOPE / MISALIGNED) are written only by
  an independent reviewer, never by us.
* `scientific_effect: "NONE"`, `scientific_status_authority: false` on every sidecar.
* Allowed transitive axioms for proofs: exactly `propext`, `Classical.choice`, `Quot.sound`.
  Rejected: `sorryAx`, `Lean.ofReduceBool` (native_decide), any custom `axiom`.
* #95 ladder (evidence levels, recorded separately from scientific status): L0 prose/LLM argument;
  L1 executable numerical/reproduction checks; L2 exact arithmetic/symbolic checks; L3
  property/mutation/falsification suite; L4 SMT/decision-procedure certificate; L5 proof-assistant
  checked theorem. A statement-only Lean `def : Prop` that elaborates is NOT L5; it is "specified".
* Dispositions in LANDING_CLAIMS.json (`HOLD_WITH_DOMAIN`, `FAIL_CLOSED`, `EXACT_COUNTEREXAMPLE`,
  `REVIEWED_SCOPED`, `ENGINEERING_HOLD`) are transcribed verbatim, never changed or re-derived.
* Never write "verified", "proved", "closed", "certified" about any informal theorem. Write what
  the Lean object is: a statement, or a kernel-checked arithmetic identity, etc.

## 1. Pinned environment (all workers use exactly this)

* Lean: `leanprover/lean4:v4.34.1`, release commit `5045d0056413266e57c625dcd7c365b10e377c52`.
  `lean --version` must read `Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)`.
  Installed locally: `export PATH="$HOME/.elan/bin:$PATH"` (elan 4.1.2; toolchain linked from
  `<SCRATCH>/lean-toolchain/lean-4.34.1-linux`).
* Mathlib: rev `d13f23b723b8a846827a245b89c10fc7d3f11612`; full dependency lock = Math- `formal/lake-manifest.json`
  (plausible 118aa17e…, LeanSearchClient ddf04cf3…, importGraph e928b725…, proofwidgets 106ff4fa…,
  aesop 355695d5…, Qq 6a489d9a…, batteries f2effa3d…, Cli e92c9f15…, mathlib d13f23b7…).
* Pinned Mathlib SOURCE tree (readable now, oleans are still being built from source):
  `<SCRATCH>/mathf/.lake/packages/mathlib/Mathlib/`.
  **Every Mathlib identifier you use must be confirmed by grep in that tree** (record file:line).
  Do NOT rely on memory of Mathlib names. Prefer the most stable API (see §4).
* The Mathlib build (`scratchpad/mathf`, `lake build ResearchFormalCoreR1`) is running in the
  background; do not start another lake process inside `scratchpad/mathf`. Core-Lean (dependency-free)
  packages CAN be built now: main's `formal/` builds in ~12 s.
* Mathlib cache hosts are blocked in this sandbox (cache.mathlib.org, mathlib4.lean-lang.org,
  lakecache.blob.core.windows.net, release.lean-lang.org, reservoir.lean-lang.org). GitHub
  `releases/download` and git clones work. CI (GitHub-hosted) uses `leanprover/lean-action@f061402b…`
  with `use-mathlib-cache: 'true'`, as every Math- Lean workflow does.

## 2. Repositories, branches, worktrees (do not touch anything else)

* main worktree: `<SCRATCH>/lvs_main`
  on branch `claude/lean-verification-specification-20261010` from origin/main `f6deeba7`.
  Existing core-Lean package: `formal/` (gate `tools/formal_gate_check.py`, tests `tests/test_formal_gate.py`).
* Math- clone: `<SCRATCH>/repos/math`
  on branch `claude/lean-specifications-20261010` from origin/main `6c020d6a`.
  New packet directory: `specifications/lean_statements_20261010/` (nothing under `formal/` is touched).
* Layer 0 byte-pinned sources already copied (Math- commit `760340e921ac4ceda296b8118da936f1133e956e`,
  the LANDING_CLAIMS audited default commit):
  `lvs_main/formal/sources/p15_v1/{three_fronts_20260924_P15_PRICE_BOUNDARY.md (2266 B, sha256 498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b, blob 1d6946658e00…), three_fronts_20260924_P15_REALIZED_COVERS.md (11467 B, c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9, blob 173881916ccd…), full_price_20260924_PROOF.md (11352 B, 87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9, blob 582180e41dca…), price_budget_20260924_PROOF.md (7935 B, 3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535, blob 9b17d179d240…)}`.
  Other Layer 0 texts are read with `git -C <math clone> show 760340e921ac4ceda296b8118da936f1133e956e:<path>`
  (parent lifetime proof: commit `c39164254ad1529ed49543743598d5dcb24bab98`, path
  `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`, blob dfed3b8d318a…).
* Open PRs that touch the formal lane and MUST NOT be collided with: main #315 (gate version
  preflight: tools/formal_gate_check.py, tests/test_formal_gate.py, formal/manifest.json hash lines),
  main #347 (formal-verification.yml evidence retention), Math- #312 (formal/gate.py declaration
  admission). We therefore: never edit `tools/formal_gate_check.py`, never edit Math- `formal/**`,
  keep the single unavoidable `tests/test_formal_gate.py` edit to the one `31` assertion.

## 3. Deliverables

### 3a. main `formal/` — new kernel-proved core-Lean targets (P15 exact arithmetic)
Modules (namespace `UniversalLaw.P15`, one namespace per module, theorems at column 0):
* `formal/UniversalLaw/P15/PriceBoundary.lean` — exact finite arithmetic of the demand-one
  counterexample: mu_p(D)=3/4 at p=1/2 on {x,y} (as integers over denominator 4: good sets ∅,{x},{y}
  have mass 1+1+1 of 4), generator prices 1, 2/3, 2/3, 4/9 (over denominator 9: 9,6,6,4), the
  minimum 4/9 and `1/3 < 4/9` (3·4 > 9… state cross-multiplied), and the two facts about log the
  source PROVES being analysis are NOT targets (does_not_claim them).
* `formal/UniversalLaw/P15/RealizedCovers.lean` — (P10) `6*binom(409,2)=500616`, `15*409^4=419743994415`;
  §6 numbers `409 = 1*408+1`, `6*409 = 2454`, `K_H = 816 = 2*408`, `815*3 < 2448 = 2454-6`,
  `2454/3 = 818`; (P11) `500616/40900^2 + 15/100^4 = 2449227/8180000000 < 3/10000` cross-multiplied;
  (P9) elementary case for the §6 instance: `max(408, ceil(6*408/3)) = 816`.
* `formal/UniversalLaw/P15/FullPrice.lean` — (F3) endpoints `0.84547981724898672067 / …68` as Nat
  in 10^-20 units: adjacent, and `…68 * 7 < 6 * 10^20` (i.e. upper endpoint < 6/7); `16/27 < 6/7`
  (16·7 < 6·27); (T3) `4^(a+1)/3^n` at a=1,n=3 equals 16/27 and `(4/3)(4/9)^1 = 16/27`;
  `p_star` is analysis (not a target). Source for (T3): price_budget PROOF.md.
Each target needs: docstring quoting the source, a manifest row {name, module, title, source,
informal_anchor (VERBATIM substring of the pinned bytes — copy-paste, keep odd spacing like
"at most15"), does_not_claim}, and at least one manifest `negative_controls` entry per module
(a textual mutation that must make Lean fail). Sources go in manifest `sources` with
`repository: "d6g8k5htny-coder/Math-"`, `commit: "760340e921ac4ceda296b8118da936f1133e956e"`,
original `path`, `bytes`, `sha256`, `local_copy`. Then: add modules to `source_modules` and the root
`formal/UniversalLaw.lean` import list (exact order), `python3 tools/formal_gate_check.py --refresh-hashes`,
`--write-alignment`, plain gate, `--run-lean` (real kernel check; quote the receipt's
`checked_commit` = HEAD of the worktree; commit first if needed — the gate wants a 40-hex HEAD, any
HEAD works for a local run). Update `formal/SCOPE.md` (new section with the "Not established"
column), `formal/README.md` (module list), and `tests/test_formal_gate.py` line asserting
`len(m["targets"]) == 31` → the new count. Run `python3 -m unittest tests.test_formal_gate -v`.

### 3b. Math- `specifications/lean_statements_20261010/` — statement-only Mathlib package
Layout:
```
specifications/lean_statements_20261010/
  README.md  SCOPE.md  REGISTER.json  check.py  test_check.py  replay.sh
  lakefile.toml  lean-toolchain  lake-manifest.json  Specifications.lean
  Specifications/{Field,Lifetime,Side24,RN,P15}.lean
  sources/<id>/<file>   (byte-pinned Layer 0 copies)
```
Rules for every `.lean` module in the packet:
* ONLY `def`, `noncomputable def`, `structure`, `abbrev`, `namespace`, `open`, `section`,
  `variable`, docstrings. **No `theorem`, `lemma`, `example`, `instance`, `axiom`, `sorry`,
  `native_decide`, `unsafe`, `opaque`, `partial`, `macro`, `notation` tricks, `set_option` that
  weakens checking.** The checker refuses them. (A proof belongs in Math- `formal/` via the lane.)
* Every specification is `def <Name> ... : Prop := ...` whose docstring has exactly these labelled
  lines: `Source:` (repo, commit, path, sha256), `Anchor:` (short verbatim quote), `Interface:`
  (which inputs are *given* objects whose defining property is a hypothesis field, not derived),
  `Does not claim:`.
* Anything the informal text leaves informal (Kac–Rice representation, "versions of the densities",
  elder pairing, the exact weighted law Q_r^W, the field itself) becomes a FIELD of an interface
  `structure` with a docstring naming the source clause. Never an axiom. The Prop is then stated for
  all/any object satisfying the interface. Say so in `Does not claim:`.
* Namespace: `UniversalLaw.Spec.<Module>`. One namespace per module.
* Use the pinned Mathlib names only after grepping them (see §1, §4). Record for each module a list of
  the Mathlib declarations used with file:line in your final structured output.

Register `REGISTER.json` (schema v1): top-level `{schema_version:1, object:"LEAN-STATEMENT-REGISTER-20261010-v1",
scientific_effect:"NONE", scientific_status_authority:false, alignment_status:"PENDING_INDEPENDENT_REVIEW",
meaning:"...inventory of formal statements per registered claim; formal-progress labels only…",
toolchain:{lean, lean_commit, mathlib}, vocabulary:{formal_progress:[none,specified,proved,kernel-checked]},
sources:[{id, repository, commit, path, bytes, sha256, local_copy}], claims:[ROW…]}` where ROW =
`{claim_id, kind (landing_claim|graph_node|proof_index_result|open_obligation|existing_lean_package|main_experiment_proof|exact_certificate),
title, layer0:{source_id, heading}, disposition_transcribed (verbatim from LANDING_CLAIMS or null),
formal_progress (none|specified|proved), statements:[{package, module, declaration, kind
(def_prop|structure|theorem), formal_progress, anchor, does_not_claim}], kernel_evidence:[{package,
receipt_or_workflow, note}], not_established:"…", next_step:"…"}`.
`formal_progress` of a claim = `specified` only if a Prop stating THE CLAIM ITSELF exists;
`proved` only if a theorem proving the claim itself (not an arithmetic sub-identity) exists — in
this programme that is **none of the research claims**; arithmetic skeletons are listed under
`statements` with their own `proved` label and the claim stays `none` or `specified`.

check.py (stdlib only, modelled on Math- companions/*/check.py and main's gate): `source` mode:
packet file identity vs git HEAD; lexical refusal list above on every .lean; root imports exactly
the modules; every top-level `def …: Prop`/`structure` in the modules is registered in REGISTER and
vice versa; vocabulary check; `formal_progress` never `proved`/`kernel-checked` for statements in
this package; anchors verbatim in `sources/` bytes (and sha256 match); lake-manifest ==
formal/lake-manifest.json of the same commit (same lock). `--execute`: `lake build`, then
`#check` of every registered declaration (and `#print axioms` — defs may use none), elaborated
types log, receipt with `formalization_status:"specified"`, lean version, checked_commit, log
digests. Negative controls (real, executed in `--execute`): a module with `theorem injected : True := trivial`
refused by the source scan; a Prop with a type error fails `lake env lean`; a register row naming
a missing declaration fails; an unregistered declaration fails; a register saying `proved` fails.
test_check.py: unittest, positive + every negative above, no Lean needed except where guarded.
replay.sh: like companions' replay (check source → tests both modes → check --execute).
Workflow `.github/workflows/lean-specifications.yml`: like `cap-first-exit-lean.yml` with
`lake-package-directory: specifications/lean_statements_20261010`, `use-mathlib-cache: 'true'`,
paths filter on the packet and the workflow, artifact upload of `.lake/evidence`.

### 3c. main `docs/LEAN_VERIFICATION_SPECIFICATION.md`
The user-facing specification: environment (versions, hashes, install commands, hosts to allow,
resource needs, time to build), the two gates and their receipts, the ladder, the complete claim
table rendered from REGISTER.json (claim → Layer 0 bytes → Lean statement(s) → formal_progress →
not established → next step), the existing Lean coverage (main 31, Math- 63, companions 26,
CapFirstExit 152 declarations, W3 scratch 101 unregistered), what remains at `none`, and the exact
commands to reproduce every check. Scientific effect NONE paragraph at the top.

## 4. Mathlib API guidance (confirm each by grep before use)
* Measures: `MeasureTheory.Measure Ω`, `[MeasureTheory.IsProbabilityMeasure P]`, `∫ ω, f ω ∂P`,
  `MeasureTheory.Integrable`, `MeasureTheory.Measure.map`, `MeasureTheory.Measure.real`? (prefer
  `(P A).toReal`), `MeasurableSet`.
* Gaussian: `ProbabilityTheory.gaussianReal (μ : ℝ) (v : NNReal) : Measure ℝ`
  (Mathlib/Probability/Distributions/Gaussian/Real.lean). State Gaussianity of a field via all
  finite real linear combinations having law `gaussianReal 0 v` under `Measure.map`.
* Reals: `Real.Gamma`, `Real.pi`, `Real.sqrt`, `Real.log`, `Real.exp`, rpow `x ^ (y : ℝ)`,
  `|x|` is `abs`.  Rationals as `(3 : ℝ) / 4`. 20-digit decimals as `(7340691930603427103 : ℝ) / 10 ^ 20`.
* Smoothness: `∀ n : ℕ, ContDiff ℝ n f` (avoid `⊤`/`∞` notation pitfalls); derivatives
  `iteratedFDeriv ℝ k f x`, `fderiv ℝ f x`, `gradient`? (prefer `fderiv`).
* Euclidean space: `EuclideanSpace ℝ (Fin d)`; inner product `⟪x, y⟫_ℝ` needs `open InnerProductSpace`? —
  grep `RealInnerProductSpace` notation; norms `‖x‖`.
* Asymptotics: `Asymptotics.IsEquivalent l u v` (`u ~[l] v`), `Asymptotics.IsBigO`, filter
  `nhdsWithin 0 (Set.Ioi 0)` (`𝓝[>] 0` needs `open Topology`).
* Matrices: `Matrix (Fin n) (Fin n) ℝ`, `Matrix.PosSemidef`, `Matrix.det`.
* Finite combinatorics: `Finset`, `Finset.powerset`, `Finset.sum`, `Finset.prod`, `Fintype`,
  `Finset.card`, `Nat.choose`, `⌈x⌉₊` (`Nat.ceil`).
* Periodic lattice torus: either `Fin d → AddCircle (L : ℝ)` or functions on `EuclideanSpace ℝ (Fin d)`
  with an explicit periodicity field; the latter is simpler and robust.

## 5. Honesty checklist applied to every statement (verifiers use this)
1. Does the Prop say what the anchor says — same quantifiers, same constants, same side conditions?
2. Are "versions", "Borel", "existential constants", "compact marks", "fixed L", "all frames" present
   as explicit hypotheses or quantifiers, not dropped?
3. Is anything the source proves taken as a hypothesis (that is fine, say so) — and is anything the
   source ASSUMES turned into a conclusion (not fine)?
4. Is the statement vacuous (contradictory hypotheses) or trivially true (e.g. `∃ C, … ≤ C` with no
   uniformity)? Existential constants must still be quantified OUTSIDE the ∀ over r, b, k, frames.
5. `Does not claim:` names the parent theorem, the identification of interfaces, review, acceptance.
6. No new vocabulary values; no "verified/closed/certified".

## 6. Additions after reconnaissance (2026-10-10 18:05Z)

* main (origin/main) also holds mathematical work outside Math-: `experiments/universality/<name>/PROOF.md` + `REVIEW.md`
  (ten merged prose proofs from codex PRs #319/#321/#323/#326/#329: finite Gaussian contact, candidate lifetime,
  fold chart, weighted fold window, contact genericity, elder transfer, finite H0 lifetime, fold structural
  universality, periodized Gaussian H0, iid short-bar process) and `experiments/periodic_h0/` (exact integer H0 and
  nine stdlib Fraction certificates with results/*/CERTIFICATE.json). These get register rows of kind
  `main_experiment_proof` (formal_progress none, next_step) and `exact_certificate` (L2 evidence, not Lean; note
  that a Python certificate is not a kernel proof). Unmerged codex PRs #330–#349 are listed by PR number only.
* An alignment record already exists on main: `reviews/lean_probability_alignment_20261005/` (REVIEW_272_20.md +
  alignment_272_20.json): xAI/Grok 4.7 statement-alignment ACCEPTED for the 20 Math-#272 formal targets at Math-
  `270167900b71539d0e2135131a3a96f1f04289f6`, Lean not re-executed, organizational-independence credit 0, and
  STALE for the current 63-target manifest (digest changed). Mention it in the document's independence section and
  as kernel_evidence/alignment notes on the Math- existing_lean_package row; do not treat it as current.
* The claims reader's inventory (journal label read:claims, data.objects) carries verbatim statements for every
  reviewed scoped result; packets integrated 27 Sep–3 Oct are listed by id only — give them rows with
  statement "not extracted" and formal_progress none rather than omitting them.
