export const meta = {
  name: 's18-result-prepost-rules',
  description: 'Rules and clarity pre-post check of the §18 result draft',
  phases: [{ title: 'Prepost' }],
}
const S = '<SCRATCH>'
const SCH = { type: 'object', properties: { problems: { type: 'array', items: { type: 'object', properties: { sentence: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' }, severity: { type: 'string', enum: ['must', 'should', 'nit'] } }, required: ['sentence', 'problem', 'fix', 'severity'] } }, verdict: { type: 'string' } }, required: ['problems', 'verdict'] }
return await agent(`A draft board comment (a custody RESULT + PERSONAL RELEASE record for recovery row §18) is at ${S}/s18/result.md. Its last line holds placeholders CONTENDER_TIME / CONTENDER_TEXT that will be filled at posting time; ignore them. Read-only; never post or push; scratch only under ${S}/s18/work/prepost-rules/.
Check it against board posting rules and clarity:
(1) no Drive identifiers (tokens of 25-44 chars from [A-Za-z0-9_-] mixing upper, lower and a digit) and no drive/docs.google links; paths under drive/mirrors must not be reproduced beyond top folders;
(2) no model names or model identifiers (e.g. '<model>', '<model>', '<model>', '<model>', '<product>', version-numbered model names); providers like 'Anthropic', 'xAI' and document ids like 'AO48-AUD-047', 'KIMI-AUD-006' are fine; flag anything borderline;
(3) no cross-repo shorthand like 'Math-#40' (autolinks to main#40); Math- references must be explicit URLs or 'Math- issue 40' style; same-repo 'PR #2', 'main#117' are fine but check they mean main's own PRs/issues;
(4) no closing verbs before #N (close/fix/resolve #N);
(5) no claim of independence credit; scientific effect NONE and no status change stated; no claim that anything was reviewed mathematically;
(6) links: the pickup 6282065819, the checkpoint 6282579128 and the §5 result 6281985456 point at issues/419 — verify with gh api repos/d6g8k5htny-coder/main/issues/comments/<id> --jq .html_url;
(7) clarity for a reader who has not seen the session: undefined jargon, ambiguous references, tables that would render badly in GitHub markdown (check the pipe table has 7 columns on every row), overlong sentences.
Return each problem with the exact sentence and a minimal fix.`, { label: 'prepost:rules', phase: 'Prepost', schema: SCH })
