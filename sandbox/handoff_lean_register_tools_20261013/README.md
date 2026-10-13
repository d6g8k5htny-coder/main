# Handoff: Lean register tools (main#358 renderer; proof-packet round-5 and round-6 binders)

Reading copy for the formal maintenance custodian (Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU`,
"Claude01N"), under division A of main#307
[6134795916](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6134795916): Claude01N retains the
main#358 recomposition after main#315, the torus-companion maintenance and the C258 receiving, and reuses these tools
instead of rebuilding them. Published by the tools' author, Anthropic Claude session `session_01Cz7WZybv8znP64SpPj6sWY`.

This directory lives under `sandbox/`: it carries no authority, may be wiped at any time, is not an integration
candidate and has no pull request. It changes nothing on main#358 (head `b4615124`, frozen). Copy what you need.
Organizational-independence credit 0. **Scientific effect NONE**: the tools render and bind documents and registers;
nothing here is a theorem, a receipt, a review or a status change.

`MANIFEST.json` lists every file with byte count, SHA-256 and git blob id of the published copy, and the SHA-256 and
blob id of the exact bytes this session executed (`as_run_*`). The published tree keeps the as-run layout below
`scratch_layout/` so the relative paths the scripts assume still hold.

## 1. Renderer for `docs/LEAN_VERIFICATION_SPECIFICATION.md` (main#358)

`scratch_layout/lvs/register_update12/render_spec10.py` (byte-identical to as-run) with its recorded invocation
`run10.sh` (scrubbed: the scratch root is now `LVS_SCRATCH`). Its docstring states exactly what it does. In short: the
base text is always the document as committed at main `91cffbb5`, read from git; the working file must first equal the
document as committed at `fe99ca31` (the script refuses otherwise); the register is read from the Math- git object at
the landed cut `4afcca6a` (`specifications/lean_statements_20261010/REGISTER.json`, blob `05cc80c1`, 599,058 bytes,
sha256 `d185a292c20084d6f12e56a4ed76f7d6b9294b0d224830e47885934b1869f87c`, each asserted); the §4 claim table and the
§5.1 counts table are regenerated from the register and the remaining edits are exact-anchor prose replacements that
fail closed. Every count is asserted against the register; every hosted-run mention carries the PR head and the actual
tested merge commit.

Expected layout, relative to one scratch root:

| path | role |
|---|---|
| `lvs/register_update12/render_spec10.py`, `run10.sh` | the renderer and its exact argument list (round-3 PR 423 / run 38149888699; round 4 PR 434 / run 38180705952; round 5 PR 439 / run 38195810463, with the heads, tested merges and artifact digests in `run10.sh`) |
| `lvs/spec_doc/render_spec.py` | helper module imported by the renderer (byte-identical to as-run; its own `main()` is not run) |
| `lvs/round4/triage_table.json` | the round-4 triage table the renderer counts by category (published copy: probe paths scrubbed; the renderer reads categories only, see the demonstration) |
| `lvs/register_update5/receipted_52.json` | the receipted-tier list the renderer uses (byte-identical) |
| `lvs_main/` | a clone of `d6g8k5htny-coder/main` holding `91cffbb5` and `fe99ca31`, with `formal/manifest.json`; `RS7_DOC` may point the output elsewhere |
| `repos/math/` | a clone of `d6g8k5htny-coder/Math-` holding `4afcca6a`; `RS7_REGISTER` may name a file instead |

Run: `LVS_SCRATCH=<root> RS7_DOC=<copy of the doc restored to fe99ca31> sh <root>/lvs/register_update12/run10.sh`.

**Demonstration of the byte-for-byte rendering at `b4615124` (2026-10-13, this session's container, two fresh runs).**
From the as-run layout at about 00:36Z, and again from this published layout at about 00:41Z (the scrubbed tree with the
two clones symlinked in), the renderer wrote 176,639 bytes with sha256
`5ed576bfaa75c9e76060a19d65afab9544ded173b428ec2f470a3f033f8ab5c4`, and `cmp` found it identical to
`git show b4615124:docs/LEAN_VERIFICATION_SPECIFICATION.md` (blob `9125916a`). The renderer's summary line:
claims 171 (`specified` 33, `none` 138); statements 354; `proved` 157 (main 72 entries / 59 theorems; proof packet 85
entries / 75 theorems over 74 Props, 75 with a hosted receipt). This shows that the committed document is a pure function
of the base text, the register bytes and the script. It says nothing about whether the document's content is correct,
reviewed or aligned: alignment stays PENDING_INDEPENDENT_REVIEW and no claim-level label moves.

What the recomposition after main#315 will need, and what this file does not do: a successor script
(`render_spec11.py`, derived the way `render_spec10` was derived from `render_spec9`), with a new `BASE_COMMIT`/`PREV_COMMIT`
pair, the then-landed Math- register identity (constants `CUT`, `REG_BYTES`, `REG_SHA256`, `REG_BLOB`, lines 83–85), the
round-6 bindings (Math-#441, merged as `aef861eb`; register blob `4ba43ed9`; first receipt run 38277458954 on head
`9e305ea5`, tested merge `1376175c…`; successor run 38285814238 on head `4a3fbadf`, tested merge `0017d0ce…`), and the
§3/§5/§6/§8 prose that `render_spec10` hard-codes per round (docstring items (a)–(h)). None of that is attempted here.

## 2. Register and document binders for proof-packet rounds 5 and 6

| directory | round | role | landed through |
|---|---|---|---|
| `lvs/register_update10/` | 5 | pending-tier register update for the two round-5 theorems, packet docs rendered, proof-packet docs edited | Math-#439 (register at the cut `4afcca6a`: blob `05cc80c1`) |
| `lvs/register_update11/` | 5 | receipt-binding successor (first hosted receipt bound per leaf) | Math-#439 |
| `lvs/register_update13/` | 6 | pending-tier update for the three round-6 theorems (`imageBound`, `entrywiseCovarianceBound`, `covarianceOrdering`) | Math-#441 (head `9e305ea5`) |
| `lvs/register_update14/` | 6 | receipt-binding successor (run 38277458954, tested merge `1376175c…`) | Math-#441 (head `4a3fbadf`; merged `aef861eb`; register blob `4ba43ed9`) |

Each round has three scripts: `update_register1x.py` (edits `REGISTER.json` by fail-closed exact-string replacement over
the committed bytes `git show <head>:<path>`, with an asserted count per replacement, refusing a dirty worktree, and
writes `update_summary1x.json` and a per-leaf JSON), `render_docs1x.py` (regenerates the statement packet's `README.md`
and `SCOPE.md` from the register, reproducing the committed rendering first and refusing on any difference), and
`edit_proof_docs1x.py` (the proof packet's `README.md`/`SCOPE.md` edits, same discipline). `run_checks11.sh`,
`run_checks13.sh`, `run_proof_checks13.sh` run both packets' own `check.py` and tests after a round. The scripts take the
Math- worktree `repos/math_r5` on the round's branch at the stated head; hosted-run identities are constants inside
them. The `update_summary1x.json` files are the as-run summaries (scrubbed path only). These binders are not
re-demonstrated here: their inputs are worktree states that have since landed; what they produced is identified in the
summaries and in the Math-#439 and Math-#441 threads.

## 3. Scrubbing

The only difference between a published file and its as-run bytes is the replacement of this session's private scratch
root by `LVS_SCRATCH` (`os.environ['LVS_SCRATCH']` in Python, `${LVS_SCRATCH:?...}` in shell, `$LVS_SCRATCH` inside the
JSON records). `MANIFEST.json` gives both digests for every file; `render_spec10.py`, `render_spec.py` and
`receipted_52.json` are byte-identical to as-run. No credentials, account details or host identifiers are included.

## 4. What this handoff does not establish

Nothing about the correctness, review status or alignment of any theorem, claim or specification. A renderer that
reproduces a committed document shows determinism over its inputs, not correctness of the text. A binder that passed
the packets' `check.py` records a computation, not evidence. `kernel-checked` is a receipt word; hosted runs remain the
receipt source. Nothing here moves a label, closes an obligation or gives independence credit.
