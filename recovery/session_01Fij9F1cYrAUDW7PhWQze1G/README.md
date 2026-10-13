# Recovery checkpoint: Claude session 01Fij9F1cYrAUDW7PhWQze1G

Answers the owner's recovery request ([main#307/6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988)). Anthropic Claude, session `session_01Fij9F1cYrAUDW7PhWQze1G` ("Claude 01Fi"). Written 2026-10-13. Scientific effect NONE. Nothing here is a proof, a review verdict or a status change. The verdicts live on the linked PRs.

## What is here

| directory | what it is | cited by |
|---|---|---|
| `ulw_guard_20261013/` | `receipt.py`, the harness that runs the original and candidate `unittest discover` steps of Universal-Law-Workspace `ci.yml` and `pins.yml` under the shells GitHub Actions uses. `receipt.json` is its CPython 3.11.15 output, 3,164 bytes, SHA-256 `a7b77192…`. `mutations.py` holds the two mutation controls. | [ULW#16](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace/pull/16) body and review 5509491588 |
| `intake404_20261013/` | `default_branch_404_probe.py`: the probe of a 404 on the default branch's own compare, run against main#391 head `66776948`. | [main#391/6154187346](https://github.com/d6g8k5htny-coder/main/pull/391#issuecomment-6154187346) |
| `c160_path_01_20261013/` | `reproduce_and_mutate.sh`: the symlinked-`TMPDIR` reproduction and the production mutant for C160-PATH-01. | [main#392 review 5509417996](https://github.com/d6g8k5htny-coder/main/pull/392#pullrequestreview-5509417996) |
| `intake_panel_20260926/` | Probe scripts and harnesses of the 26 September adversarial panel on `tools/public_intake_check.py`, with one directory per finder lens and per refuter. 35 files that carry synthetic credential-shaped test strings are not uploaded; `EXCLUDED.json` lists each one's path, size and SHA-256. | [main#86/5851057735](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-5851057735) |

The three 13 October scripts were reconstructed from the exact commands run that day. They are not byte copies of earlier files, because those commands ran inline.

## Already recoverable elsewhere (not duplicated)

- The intake checker repairs that came out of the panel are merged: main#153, #156 and #240. They carry their own tests and weakening proofs.
- C160-PATH-01's repair is GROK-H01's commit `322c8291` on `grok/c160-path-01-canonical-0cc2`, published as main#392.
- The formal-guide routing change is GROK-H02's commit `d7bebec5`, published as main#393.
- The ULW guard is GROK-H04's commit `8e0f5fde`, published as ULW#16.

## Not uploaded

- **The per-vote panel transcripts.** They are session logs that contain host paths and tool output, so only their conclusions are public, in #86.
- **The rest of the scratchpad, about 1.8 GB.** It is mostly checkouts, worktrees and intermediate copies for PRs that have since landed.

## State at this checkpoint

- **Claims held:** none.
- **Open lines this session moved, all waiting on other lanes:**
  - **main#392:** an OpenAI integrator after a base carry.
  - **main#393:** GROK-H02's amendment of its route to the capped #229.
  - **ULW#16:** an integrator.
- **Next reproducible step for any of them:** the commands in the matching directory. Each needs CPython 3.11 and git.
