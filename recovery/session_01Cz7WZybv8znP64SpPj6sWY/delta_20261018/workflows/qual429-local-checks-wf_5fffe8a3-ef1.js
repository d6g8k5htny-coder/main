export const meta = {
  name: 'qual429-local-checks',
  description: 'Run the four triggered workflows\' steps locally on the main#429 current-Main composition (local commit 400391ce) with baseline comparison on failures',
  phases: [{ title: 'Run', detail: 'one agent per workflow, isolated worktree copies' }],
}
const C = '400391ce659969baf94db317da9fe572145130e7'
const MAIN = '9e7bacb8f04a1abba1fef0a77a7e0e666b064c76'
const BASE = '<SCRATCH>/q429'
const RES = {
  type: 'object',
  properties: {
    workflow: { type: 'string' },
    steps: { type: 'array', items: { type: 'object', properties: {
      job: { type: 'string' }, step: { type: 'string' }, command: { type: 'string' },
      outcome: { type: 'string', enum: ['PASS', 'FAIL', 'NOT_RUN'] }, seconds: { type: 'number' },
      detail: { type: 'string', description: 'key output lines (e.g. Ran N tests OK), or failure excerpt, or why NOT_RUN' },
      baseline_on_main: { type: 'string', description: 'if FAIL: same step on main 9e7bacb8 PASS/FAIL/NOT_RUN' },
    }, required: ['job', 'step', 'command', 'outcome', 'detail'] } },
    packet_specific_coverage: { type: 'string', description: 'which steps actually exercise experiments/universality/fold_isolation_diagonal files, and how' },
    reads_changed_docs: { type: 'string', description: 'whether any step reads docs/FORMAL_VERIFICATION.md, docs/PUBLIC_MATHEMATICS.md, docs/agents/LEAN_AUDIT_OWNER.md, docs/agents/UI_UX_OWNER.md, governance/OP-ACCESS-20261007.md, governance/OP-WORKFLOW-20260930.md' },
    summary: { type: 'string' },
  },
  required: ['workflow', 'steps', 'packet_specific_coverage', 'reads_changed_docs', 'summary'],
}
const common = (w, i) => `You are running LOCAL compatibility checks only. Repo clone: /home/user/main (git 2.43, partial clone; origin = github d6g8k5htny-coder/main). Do NOT push, do NOT create branches on the remote, do NOT comment on GitHub, do NOT dispatch/rerun workflows, do NOT pip-install into system Python unless the workflow itself does so (use a venv under your work dir if the workflow installs pinned tooling; if network install fails, mark NOT_RUN). Never open paths containing 99_DO_NOT_OPEN. Never print secrets.
Setup: create your own worktree: cd /home/user/main && git worktree add --detach ${BASE}/wt_${i} ${C} (commit ${C} is a LOCAL-ONLY composition: current main ${MAIN} plus 83 added files under experiments/universality/fold_isolation_diagonal/). Work only inside ${BASE}/wt_${i}. When done, remove your worktree with: cd /home/user/main && git worktree remove --force ${BASE}/wt_${i} (and the baseline worktree if you made one).
Task: read .github/workflows/${w}.yml from the worktree and execute every job's run: steps in order, exactly as written (same commands, same Python-mode variants such as -O/-B/-S, same working directory), on this Linux box (Python 3.11.17; node/chromium may or may not exist — check: node --version; Chromium is at /opt/pw-browsers, PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers, do not run 'playwright install'). For steps that need GitHub context (GITHUB_* env, github.event, secrets, artifacts upload, needs.* outputs, reusable workflow calls), either emulate faithfully with obviously-local placeholders where the script only records identity, or mark NOT_RUN with the exact reason; never fake a PASS. For reusable workflows (uses: ./.github/workflows/X.yml), open X.yml and run its run: steps likewise. Record per step: PASS/FAIL/NOT_RUN, seconds, key output (e.g. 'Ran 37 tests ... OK'). For any FAIL, create a second worktree at ${MAIN} (${BASE}/wt_${i}_base) and run the same step there to classify it as pre-existing vs packet-introduced. Also state which steps actually touch the packet directory, and whether any step reads the six docs/governance files that differ between current main and the historically qualified tree.`
const W = ['universality-controls', 'public-shop', 'workspace-landing', 'public-intake']
phase('Run')
const out = await parallel(W.map((w, i) => () => agent(common(w, i) + (w === 'public-intake' ? `\nNOTE for public-intake: it is pull_request_target and reads PR data from the API; emulate by running the trusted checker (tools/public_intake_check.py from MAIN) against the composed change set (git diff ${MAIN} ${C}) if the checker supports a local/file input mode; read the checker and its tests (tests/test_public_intake.py) to determine the exact contract and expected verdict for an 83-file additive experiments/ change by the repo owner account. If no local mode exists, report precisely what it would evaluate and mark NOT_RUN.` : ''), { label: `run:${w}`, phase: 'Run', schema: RES })))
return out
