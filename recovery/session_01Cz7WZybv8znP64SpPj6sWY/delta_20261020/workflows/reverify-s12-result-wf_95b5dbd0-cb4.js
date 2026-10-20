export const meta = {
  name: 'reverify-s12-result',
  description: 'Focused re-check of the rewritten §12 result before posting',
  phases: [{ title: 'Recheck', detail: 'changed statements and protocol' }],
}
const S = '<SCRATCH>/s12'
const CTX = `READ-ONLY pre-post re-check. Do NOT post, edit or push anything; scratch only under ${S}/work/reverify_<lens>/. Never reproduce Google Drive ids or Drive/Docs URLs. Do not name model versions (say 'OpenAI', 'xAI', 'Anthropic').
The draft board comment ${S}/result.md (for d6g8k5htny-coder/main#419) and pointer ${S}/ptr275_result.md were rewritten after a four-lens check whose findings are in <SESSION_DIR>/subagents/workflows/wf_4b78af88-f05/journal.jsonl (lines with type=result). The 12 underlying comparison results are in ${S}/journal_results.json; sources in ${S}/src/; agent scratch in ${S}/work/; repository /home/user/main (use 'git show origin/main:<path>'); 'gh api' works for GitHub reads (e.g. gh api repos/d6g8k5htny-coder/Math-/issues/comments/5894512272 --jq .body). The literal placeholders CONTENDER_TIME / CONTENDER_TEXT will be filled just before posting; ignore them.
Report ONLY concrete remaining problems: exact draft text, why (evidence), severity (blocker/fix/nit), replacement. Default to flagging statements the evidence does not support.`
const LENSES = [
  { key: 'applied', p: 'LENS: were the four-lens findings applied correctly, and did the rewrite introduce new errors? Go through every blocker and fix in the four-lens journal and confirm the rewrite handles it accurately (or that ignoring it is justified). Then check every NEW or CHANGED statement in the draft against sources: the xAI Math-#141 verdict text and scope, the main STATUS.md D5 row and docs/STATUS_RECONCILIATION_20260929.md, the reconciler/continuum session identity, the Drive manifest line about FlatSaddle (audits/dropbox_import/2026-09-27/files/Github/, the Part 01 of 13 manifest transcript), Q0_LEDGER E-C101-3/E-C101-7/C011 registry, the crosswalk RESULT.md l.58-63, the typed/untyped axis numbers (work/prepost_math/, work/lsder024_advmath/), the Monte Carlo values at t = 0.2 and 0.4 (work/lsder024/bfcheck_A.log), and the (b) section wording.' },
  { key: 'protocol', p: 'LENS: protocol and coherence. (1) No Drive ids/URLs, secrets, personal data, private host paths, or model version names in the draft or pointer. (2) No closing verb (close/fix/resolve and variants) before a #N. (3) Links well formed. (4) The draft stays inside the pickup scope (pickup2.md): no status/lemma/prize/PROOF_INDEX disposition, no review of P2 beyond the map. (5) Internal consistency: verdict vs §3, table vs §2, pointer vs draft, independence section vs corrections. (6) Readability: is anything confusing or self-contradictory for a board reader? Keep findings concrete.' },
]
const SCHEMA = { type: 'object', properties: {
  problems: { type: 'array', items: { type: 'object', properties: {
    draft_text: { type: 'string' }, why: { type: 'string' }, severity: { type: 'string', enum: ['blocker', 'fix', 'nit'] }, replacement: { type: 'string' } },
    required: ['draft_text', 'why', 'severity', 'replacement'] } },
  verdict: { type: 'string', enum: ['READY', 'READY_AFTER_FIXES', 'NOT_READY'] } },
  required: ['problems', 'verdict'] }
phase('Recheck')
return await parallel(LENSES.map(l => () => agent(`${CTX}\n\n${l.p}`, { label: `recheck:${l.key}`, phase: 'Recheck', schema: SCHEMA })))
