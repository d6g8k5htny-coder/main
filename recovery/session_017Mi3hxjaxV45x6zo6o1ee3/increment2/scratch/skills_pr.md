**Owner request.** Dylan Roy asked this session directly at about 02:56Z to add these skills. Claim: main#229 [5975964415](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5975964415). Actual author: Anthropic Claude, session `017Mi3hx…`. **Scientific effect NONE.** The author will not merge.

## What this adds

Two new files and nothing else:
- **`.claude/skills/steward/SKILL.md`.** Repo conventions that Claude Code agents read before acting on CI, review or merge-state events on a Math- PR they opened or drive. It summarizes, and links as normative, the rules that already bind this repository: [`AGENTS.md`](../blob/main/AGENTS.md), `main`'s [`AGENTS.md`](https://github.com/d6g8k5htny-coder/main/blob/main/AGENTS.md), [OP-WORKFLOW-20260930](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-WORKFLOW-20260930.md), [OP-PRIVACY-20260927](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-PRIVACY-20260927.md) and [`docs/FORMAL_REQUIRED_CHECKS.md`](../blob/main/docs/FORMAL_REQUIRED_CHECKS.md). Sections:
  1. identity, roles and independence credit, including the owner ruling that Grok Bot agent reads count;
  2. the PR body as the one current disposition;
  3. reads, verdicts and AMENDs: frozen heads while a read is open, AMENDs only after the verdict, byte-diff readbacks, optional points never starting a push;
  4. checks: `math-downstream-gates`, packet checkers in `-B -S` and `-B -O -S`, strict mutant rejection, `pins_on_main`, the Lean `gate.py` source and `--execute` modes;
  5. branches: merge commits only, no change to another owner's branch without a handoff, stacked-base reconciliation;
  6. integration (for the integrator, never the author);
  7. the never-list (status surfaces, flags, privacy);
  8. comments and claims.
- **`.claude/skills/babysit/SKILL.md`.** The short watching posture: never punt, address every thread, a failing test is never assumed to be a flake, and a red head never waits. It defers to steward, which the agent harness prefers when both exist.

It adds no rule beyond those sources; on conflict they win. It installs no workflow, scheduler or register.

## Current disposition

- **Head:** `32b3aa9` on `claude/nice-maxwell-7i7ok0`. That branch was restarted from `main` `f9f59c2`; its earlier PRs are all merged.
- **Owner:** this session (author).
- **Review requested:** one non-Claude engineering/workflow read (OP-WORKFLOW's "Engineering, custody or workflow" tier). Check that each statement matches its cited source and that nothing contradicts `AGENTS.md` or OP-WORKFLOW. Please post a pickup naming the head first; the head stays frozen while the read is open.
- **Unresolved finding IDs:** none.
- **Validation:** I checked each referenced path and check name against the repository at `f9f59c2`:
  - `math-downstream-gates` needs `downstream-replay` and `formal / formal-evidence` (`.github/workflows/downstream-gate.yml`);
  - `certify-and-reject-mutants` and `kernel-evidence` are existing packet jobs;
  - the relative links resolve;
  - no check polices `.claude/`.
- **Next action:** wait for the read, then for a non-Claude integrator.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
