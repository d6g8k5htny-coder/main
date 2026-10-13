# Companion brief: `companions/torus_filtration_20261011/` (isolated Lean companion for the C254 modules)

Scientific effect: NONE. Organizational-independence credit: 0. Nothing here promotes, closes or reclassifies any claim.
The companion gives the two C254 modules (torus quotient / closed superlevel filtration, proposed by the OpenAI/Codex
lane as NOT_COMPILED source, then compile-fixed by this lane) a hosted execution route that does NOT modify `formal/`
and therefore does not disturb the three D2 companions, which pin the formal tree `38b6d1a7…` and manifest `40cb6ffc…`.

## Where (S=<SCRATCH>)

- Worktree: $S/repos/math_c254 (Math- at main 2bbce4b6, branch claude/c254-torus-filtration-20261011). It currently
  carries the C254 overlay inside `formal/` — DO NOT TOUCH anything under `formal/` or `.github/workflows/d2-*` or the
  existing companions; the overlay will be reverted from `formal/` by the integrator. Create only
  `companions/torus_filtration_20261011/**` and `.github/workflows/torus-filtration.yml`.
- Pattern to mirror, read it completely first: `companions/d2_moment_bridge_20261006/{check.py,replay.sh,test_bridge.py,
  SOURCE_FILES.json,README.md}` and `.github/workflows/d2-moment-bridge.yml` (same worktree). Also read
  `formal/gate.py` (audit_axioms regex, lineage/identity style) and `companions/d2_atom_integral_20261006/SCOPE.md`.
- The two Lean modules (statement/theorem inventory fixed): originals from the C254 handoff at
  $S/lvs/c254/overlay/ResearchFormalCoreR1/{TorusTopology,ClosedSuperlevel}.lean (NOT_COMPILED, hashes efa80d28… /
  1fed3eda…); compile-fixed versions are being produced in parallel and will be placed as
  `companions/torus_filtration_20261011/TorusTopology.lean` and `ClosedSuperlevel.lean` by the integrator, with the
  namespace changed from `ResearchFormalCoreR1` to `TorusFiltration` and nothing else renamed. Theorem names (21 +
  20 = 41, in file order) and `def`/`abbrev`/`instance` names are exactly those of the originals: derive the inventory
  lists from the originals with `^theorem`, `^(?:noncomputable )?def`, `^abbrev`, `^instance` scans.
- Toolchain: `export PATH=$HOME/.elan/bin:$PATH`; from `formal/`: `lake env bash` gives an environment where
  `lean --root=<SIDE> -o <SIDE>/X.olean <SIDE>/X.lean` compiles a companion module against the pinned Mathlib (the
  D2 replay does exactly this). Locally `formal/.lake/packages` is a symlink to the prebuilt dependency tree.
  Run at most one Lean process at a time; never run `lake build`, `lake update`, `lake exe cache`.
- Provenance to record (README/SCOPE/SOURCE_FILES): C254 claim 8a5c7a24-784b-4aaa-b951-cd9f37b0dc64, handoff comments on
  Math- PR #419: 6111056103 (modules), 6111059166 (overlay), 6111059498 (nonauthor source review, COMPILATION HOLD),
  6111059848 (implementer report), 6111255007 (C255 route finding: D2 pins; companion route recommended); local
  elaboration report 6111093545; pickup 6111139060 (this lane). Author of the source: OpenAI/Codex lane for Dylan Roy;
  compile fixes and companion scaffold: Anthropic/Claude lane; credit 0 for both; no human review.

## Deliverables (write them into the worktree; nothing else)

1. `companions/torus_filtration_20261011/TorusFiltration.lean`: root module, exactly
   `import TorusTopology\nimport ClosedSuperlevel\n` (module names are file stems under the companion root).
2. `check.py` (stdlib only), commands `source | emit <out> | negative <log> --control C --status N | finish <out>`,
   mirroring the D2 checker but WITHOUT any pin on `formal/` tree, manifest or gate (the companion consumes nothing
   from the core). Instead: (a) `source`: HEAD is 40-hex; tracked membership of the companion dir + workflow equals
   SOURCE_FILES `files` (+ SOURCE_FILES.json itself); every file's bytes/sha256/git_blob match; `git diff --exit-code
   HEAD -- <companion> <workflow> formal/lakefile.toml formal/lake-manifest.json formal/lean-toolchain`; under
   GITHUB_ACTIONS the GITHUB_SHA/REPOSITORY/RUN_ID/RUN_ATTEMPT checks as in D2; records `environment`: text of
   formal/lean-toolchain and the package revisions of formal/lake-manifest.json. (b) `emit`: Audit.lean (`#print
   axioms TorusFiltration.<name>` for every theorem AND every def/abbrev/instance of both modules), Types.lean
   (`#check` with `set_option pp.explicit true` for every theorem), Positive.lean (3-6 `example`s that apply the
   companion's theorems to concrete inputs, e.g. the lattice membership of `latticeVec`, path-component map on `mk`,
   closed-superlevel monotonicity), and the negative-control files. (c) `negative`: exact-diagnostic parsing as in D2
   (`fullmatch`), one pattern per control; controls to design and VERIFY LOCALLY (compile them!): at least one
   false mathematical statement about the companion's own definitions rejected by Lean with an exact diagnostic you
   pin (e.g. `unsolved goals ⊢ False`), one `sorry` injection caught by the axiom audit (`sorryAx` must be refused by
   your audit parser), one custom `axiom` injection caught the same way, one `native_decide` injection caught the same
   way. (d) `finish`: every build status 0 (`build-TorusTopology`, `build-ClosedSuperlevel`, `build-TorusFiltration`,
   `positive`, `axioms`, `types`, `leanchecker`), toolchain.log matches `Lean (version 4.34.1, …, commit
   5045d0056413266e57c625dcd7c365b10e377c52, Release)`, packages.log (one `name rev` line per installed package from
   `git -C formal/.lake/packages/<name> rev-parse HEAD`) equals formal/lake-manifest.json; axioms of every declaration
   within {propext, Classical.choice, Quot.sound} using your own copy of the gate's AXIOM regex and rules (unknown
   or duplicate names refused; unrecognized output refused); types inventory equals the theorem list; negative
   controls as designed; receipt.json with schema 1, scientific_effect NONE, formalization_status kernel-checked,
   alignment_status PENDING_INDEPENDENT_REVIEW, scientific_acceptance false, checked_commit, repository, run id and
   attempt, lean_version, dependency_revisions, sources, declarations, axioms, negative_controls, log_sha256,
   provenance (the C254 ids above and the original NOT_COMPILED hashes).
3. `replay.sh` mirroring the D2 one: `python3 -B -S check.py source`, both test modes, `emit`, then inside `lake env
   bash` from `formal/`: toolchain.log, packages.log, `lean -DwarningAsError=true --root=$SIDE -o … TorusTopology.lean`,
   same for ClosedSuperlevel and TorusFiltration, positive, axioms, types, `leanchecker --fresh TorusFiltration`,
   the negative controls (each with `.status`), then `finish`. Output dir `.lake/torus-filtration-evidence` at the
   repo root (must not pre-exist). It must NOT run `formal/gate.py` or the formal tests.
4. `test_torus_filtration.py` (unittest, stdlib): inventory equality against the Lean files (theorems, defs), the
   SOURCE_FILES lists, no-admission token scan on comment-stripped code, workflow path triggers, and NEGATIVE CONTROLS
   for check.py itself (each test must fail if the check is weakened): negative() rejects a wrong diagnostic, a status
   0, an unknown control; the axiom parser rejects sorryAx, an unknown name, a missing name, unrecognized text;
   types() rejects a missing or extra name; `source` membership mismatch detected (use a temporary git repository
   fixture as test_bridge.py does if it does; otherwise unit-test the pure functions).
5. `SOURCE_FILES.json`: schema 1, scientific_effect NONE, alignment PENDING_INDEPENDENT_REVIEW, `definitions`,
   `theorems` (per module, in order), `files` {path: {bytes, sha256, git_blob}} for every tracked companion file and
   the workflow (write a generator script OUTSIDE the repo at $S/lvs/c254/companion/gen_source_files.py that
   recomputes it from the worktree; the integrator runs it last), `provenance` (ids and original hashes above,
   `namespace_mapping`: "ResearchFormalCoreR1.<n> (handoff) -> TorusFiltration.<n> (companion)").
6. `README.md` and `SCOPE.md`: purpose, bound sources, what the 41 statements establish (topology and scaffolding
   only: quotient, descent, path lifting with existential endpoint, closed superlevel inclusions and component/H0
   maps), what is NOT established (copy the handoff's own list: critical-level attainment, finite critical events,
   label tracking, naturality to singular H0 generators beyond what is proved, interval decomposition, bar
   bijection, elder events, once-counting, any identification of `PairLawInterface.p`), the compile-fix record
   (placeholder section `## Compile fixes at the pin` to be filled by the integrator), the namespace mapping, the
   route rationale (D2 pins), the vocabulary rules (`proved` is a source label; `kernel-checked` only in a receipt;
   never "verified"/"closed"/"accepted"), and how to run. Author lines: "Dylan Roy — delegated AI work. Source
   author: OpenAI/Codex lane (C254). Compile fixes and companion: Anthropic/Claude lane." Credit 0.
7. `.github/workflows/torus-filtration.yml` mirroring d2-moment-bridge.yml: pull_request and push(main) on
   `companions/torus_filtration_20261011/**`, the workflow file, `formal/lakefile.toml`, `formal/lake-manifest.json`,
   `formal/lean-toolchain` (NOT `formal/**`); lean-action pinned exactly as the D2 workflow (same SHA), lake-package-
   directory formal, use-mathlib-cache true; run replay.sh; upload `.lake/torus-filtration-evidence/` as
   `torus-filtration-${{ github.run_id }}-${{ github.run_attempt }}`, retention 30, if-no-files-found error.

House style: stdlib only; checkers print one JSON/one-line summary and exit nonzero on failure; every check has a test
that fails when the check is weakened. Do not write `sorry` anywhere except inside the negative-control fixtures that
the checker must refuse. Report exactly what you created and what remains for the integrator (module placement,
SOURCE_FILES regeneration, README compile-fix section).
