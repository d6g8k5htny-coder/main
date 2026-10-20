# Correction to this session's integration guidance (20 October 2026)

Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017), on Dylan Roy's instruction in session (19 October, about 23:55Z). This is retention only, not for merge. Credit 0. Scientific effect NONE. This file is the one explanation of the correction. The historical records it cites are unchanged.

## What was wrong
On 19 October this session told Dylan in chat that it does no integration "unless dot allocates it". Its task list also said the remaining stages were "integrations needing dot's allocation".

No rule says either. Both overgeneralized two notes from 14 October that concerned only C8's eligibility.

## Where it came from
| record (unchanged) | what it said | basis |
|---|---|---|
| `increment3_20261014/STANDING_RULES.md`, entry of 20:33Z; posted as main#404 comment 6177843592 | Do not post merge-tree or composition data touching PRs with live integration offers unless dot asks | C8's eligibility line (main#401 comment 6176304839): a merge or composition result counted as content exposure |
| local notes `rules_017.md`, entry of 21:25Z, from C8's main#307 comment 6178816289 | On a PR that might later need an integrator, stay metadata-only unless dot allocates a review | It protected C8's chance to integrate that PR |
| dot, main#307 comment 6177962483 | The restriction is C8-specific, not provider-wide | — |

## Related lines, kept as history
- **`increment2/HANDOFF_STATE.md` (13 October).** It says "the coordinator (OpenAI "dot" on main#307) routes work; offer → ACK → activation → execution → readback → own release". That described how work was coordinated on 12–13 October. It is not a standing condition for integration eligibility.
- **`increment2/scratch/audit_*.md` P6 and P9 (6 October).** These treat integration as closed to this lane "until the owner changes the session's permission rule". The cause was this session's permission layer refusing actions. That is a platform-permission matter and stays relevant under (3) below. It never required dot's allocation.

## What applies now
Integration eligibility comes from three things:
1. **The actual assignment for the PR.** That is an owner instruction (such as Dylan's in-session assignment of main#451 R3), a coordinator activation, or a self-pickup where the rules allow one.
2. **Current project rules:**
   - `governance/OP-WORKFLOW-20260930.md`. A self-started pickup does not authorize a protected integration, an occupied role, a private-access expansion or a security-sensitive change. Author, reviewer and integrator stay separate. Merges use expected-head guards. The Ready-triggered automatic review must finish, or its next step must be settled on the PR.
   - `CONTRIBUTING.md`, including the binding check on `public-intake`.
   - Owner decisions recorded on the board.
3. **Platform permissions:** this session's permission mode and any denial. A denied step is never retried through another tool, never transferred to another agent, and never worked around.

dot's allocation is needed only where (1) or (2) requires it for the PR in question.

## Where the correction is applied
- **Local notes `rules_017.md`:** the correction is appended and the earlier entries are unchanged. Copies are kept here: `guidance/rules_017.md`, and the pre-correction copy `guidance/rules_017.pre_correction_20261020.md` (SHA-256 `de1f211a…`).
- **Task list entry #282:** reworded.
- **This file.**
- **GitHub:** I searched this session's saved post texts and found no comment repeating the overgeneralized wording, so there is nothing to correct there. Earlier retention increments are not modified.
