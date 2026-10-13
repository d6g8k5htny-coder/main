# Lean statement register: statement-only Mathlib specifications of the registered claims

**Scientific effect: NONE.** No `lemma_closed`, prize, premise, LANDING_CLAIMS disposition, PROOF_INDEX, GRAPH or STATUS change; nothing under `formal/**` is touched. Nothing here is a proof, a review or an acceptance. Author: Anthropic / Claude (Claude Code cloud session), same account as every lane: organizational-independence credit **0**. Draft; this session does not merge.

## What this adds

`specifications/lean_statements_20261010/` — a statement-only Lean 4.34.1 + Mathlib `d13f23b7…` package (same lock as `formal/`):

- `Specifications/{Field,Lifetime,Side24,RN,P15}.lean`: 150 registered declarations (141 `def … : Prop` specifications and 9 interface structures for the registered claims (D1 Theorems A/B/C, D2 Theorem R, D3 SIDE24 coefficient with the exact `c_{d,ref}` definition, D4/D5 RN regional theorems and the D5 reconciliation, C6 count theorems, D6/P15 Theorems P2/T1/F and ρ*). Every docstring carries `Source:`, `Anchor:` (verbatim in the byte-pinned copies under `sources/`), `Interface:` and `Does not claim:`. **Zero theorems**; no `sorry`, no axioms. Objects the sources leave informal (Kac–Rice representations, the weighted law, elder pairing, the field) are fields of interface structures, never axioms.
- `REGISTER.json`: the statement register — 171 claims (every LANDING_CLAIMS row with its disposition transcribed verbatim, every GRAPH `math.*` node, every PROOF_INDEX reviewed result and open obligation, every existing Lean package) → byte-pinned Layer 0 source → Lean statement(s) → formal-progress label (`specified` at most; a claim is `specified` only when a Prop states the claim itself) → not established → next step.
- `check.py` + `test_check.py` + `replay.sh`: stdlib gate (source identity, lexical refusal of any proof construct, inventory = register, vocabulary, verbatim anchors, lock identity; `--execute`: `lake build`, `#check` of every declaration, real negative controls, receipt that can only say `specified`).
- `.github/workflows/lean-specifications.yml`: lean-action (pinned) with the Mathlib cache, runs the gate, uploads evidence.

## What it does not establish

Anything. A statement is a precise question, not evidence of truth. The register is an inventory of formal statements, not a status register (`scientific_status_authority: false`; dispositions transcribed, never derived). Independent alignment review (different provider/family/agent) is pending for every declaration.

## Evidence

- `python3 -B -S check.py source` and the tests in both Python modes at `<<COMMIT>>`: `check.py` SOURCE_CHECK_PASS (171 claims, 228 statement entries, 150 registered local declarations, 23 pinned sources verified against git history); `test_check.py` 94 tests OK in normal and -O modes; the repository test suite shows the same 166 failures / 3 errors with and without this packet (all in the git-isolation tests `test_d5_c6_git_isolation` family, environment-dependent and pre-existing on `main` 6c020d6a).
- Lean elaboration of the five modules against the pinned Mathlib: all five modules elaborate with warnings as errors (`lake build` of the packet against the pinned Mathlib built from source here: 38 s); a local replica of `--execute` without `leanchecker` passed every other step (fresh build; `#check` and `#print axioms` of all 150 declarations, axiom sets within {propext, Classical.choice, Quot.sound}; environment audit 141 statements / 9 structures / 123 auxiliary / 0 refused; every negative control refused). `leanchecker` on the full Mathlib environment was killed at this sandbox's 13.5 GB memory limit three times, so no local receipt exists (this sandbox cannot reach `cache.mathlib.org`; the workflow run on this PR is the authoritative elaboration check).

## Coordination

Claim: main #307 comment 6100419142. Companion PR on main: https://github.com/d6g8k5htny-coder/main/pull/358.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
