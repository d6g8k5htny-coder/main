## Change and purpose

Mathematics + engineering + documentation; **scientific effect NONE** (no `lemma_closed`, prize, premise, LANDING_CLAIMS disposition, PROOF_INDEX, GRAPH or STATUS change; nothing here is a review or an acceptance). Owner instruction of 10 October: make sure all the mathematical work can be verified in Lean and provide the specifications and versions. After this change a reader can:

1. Read `docs/LEAN_VERIFICATION_SPECIFICATION.md`: exact pinned versions (Lean `v4.34.1` release commit `5045d005…`, Lake 5.0.0, elan, Mathlib `d13f23b7…` with its lock, action SHAs), two install routes (standard; restricted-network via GitHub release assets with the hashes observed here), the hosts a build machine must reach for the Mathlib cache, every package/gate/receipt of the lane, the complete claim table rendered from the new statement register (Math- `specifications/lean_statements_20261010/REGISTER.json`, companion PR <<MATH PR>>), a coverage summary and a tractability-ordered roadmap from `specified` to `proved`.
2. Run the main core-Lean gate on <<N>> new kernel-proved targets in `formal/UniversalLaw/P15/{PriceBoundary,RealizedCovers,FullPrice}.lean`: the exact integer/rational arithmetic displayed in the P15 notes (demand-one counterexample masses and prices; the 816/818 benchmark with (P10)/(P11); the (F3) 20-digit endpoints of ρ* and `< 6/7`; 16/27 and (T3)). Sources are byte-pinned under `formal/sources/p15_v1/` (Math- `760340e921ac4ceda296b8118da936f1133e956e`); every target has a verbatim `informal_anchor` and a `does_not_claim`; <<K>> new executable negative controls; `SCOPE.md`, `README.md`, `ALIGNMENT.md` (regenerated) and the single target-count assertion in `tests/test_formal_gate.py` are updated. `tools/formal_gate_check.py` is **not** edited (main #315 owns it).
3. Follow one descriptive pointer line each in `docs/FORMAL_VERIFICATION.md` and `docs/RESEARCH_INDEX.md`.

Discussion: main #307 claim [6100419142](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6100419142); lane contract `docs/FORMAL_VERIFICATION.md`.

## Current disposition

- Head and integration owner: <<COMMIT>>, Anthropic / Claude (Claude Code cloud session `session_01Cz7WZybv8znP64SpPj6sWY`); draft, this session does not merge.
- Claim / handoff: main #307 6100419142 (lease to 23:55Z 10 Oct); release posted there on completion.
- Completed review scopes and source identities: author-side only. Same-account authorship: organizational-independence credit 0. Independent alignment review (different provider/family/agent) of the <<N>> new targets: none yet (`PENDING_INDEPENDENT_REVIEW`).
- Unresolved finding IDs and next action: none open; next action is cross-provider alignment review of the new targets and of the Math- statement register.

## Verification

Run locally at <<COMMIT>> in the sandbox (Ubuntu 24.04, Lean 4.34.1 commit 5045d005…, no Mathlib needed for this package):

```
python3 tools/formal_gate_check.py              # SOURCE_IDENTITY_PASS … <<MANIFEST>>
python3 tools/formal_gate_check.py --run-lean   # build, leanchecker, axiom audit (all []), <<CONTROLS>>, receipt
python3 -B -S -m unittest tests.test_formal_gate -v   # <<TESTS>>
python3 -B -O -S -m unittest tests.test_formal_gate -v
```

Hosted: the required `verify` aggregate runs the same gate on the merge commit (`formal-verification.yml`, unchanged).

## Evidence and limits

- The receipt establishes the <<N>> integer statements only, relative to the Lean kernel and the pinned toolchain. It does not establish Theorems P2, T1 or F, the counterexample's logarithm inequalities, any probability or measure semantics, or anything about the P15 prize; each target's `does_not_claim` says what it leaves out, and `formal/SCOPE.md` carries the "Not established" column.
- The specification document is an inventory of formal statements and an environment specification, not a status register (`scientific_status_authority: false`; dispositions transcribed, never derived). A `specified` statement is a precise question, not evidence of truth.
- Known integration note: both this PR and main #315 re-hash the `tests/test_formal_gate.py` line in `formal/manifest.json`; the second to land re-runs `--refresh-hashes`.
- Authorship/exposure: Anthropic / Claude read every pinned source; no human review; independence credit 0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
