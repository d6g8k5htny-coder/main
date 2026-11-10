export const meta = {
  name: 't11c-triage-audit',
  description: 'T11c: independently re-judge a stratified 5% sample plus all keyword-flagged WORKING/ENGINEERING triage labels',
  phases: [{ title: 'Audit', detail: 'one agent per batch of up to 40 rows' }],
}
const A = '/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/t11c/audit'
const AUDIT = { type: 'object', properties: { rows: { type: 'array', items: { type: 'object', properties: {
  git_blob: { type: 'string' }, label: { type: 'string', enum: ['MATH', 'MATH_REVIEW', 'WORKING', 'ENGINEERING'] },
  agrees: { type: 'boolean' }, missed_mathematics: { type: 'boolean' }, evidence: { type: 'string' }, note: { type: 'string' } },
  required: ['git_blob', 'label', 'agrees', 'missed_mathematics', 'evidence', 'note'] } } }, required: ['rows'] }
const prompt = n => `You are an independent AUDITOR of a first-pass triage. Read ${A}/abatch${n}.json: rows with git_blob, path, ref, commit and the triage's label, evidence and note. These are files in session retention trees (recovery/), Drive mirrors (drive/) or intake drops (incoming/) on main refs of the d6g8k5htny-coder federation. The triage labelled every row WORKING (a working artifact with no mathematical claim of its own) or ENGINEERING (tooling, CI, custody, tests, non-mathematical code). Some rows were drawn at random and some because a keyword screen saw Lean theorem/lemma declarations or Theorem/Proof markers (field 'why').
Do NOT look at the triage's reasoning first: for each row, read the file yourself with \`git -C /home/user/main cat-file -p <git_blob>\` (enough to decide: at least the first 150 lines or 10 KB, and grep for theorem/lemma/proof/sorry/certif/bound), decide the label yourself with these definitions, then compare:
- MATH: states, proves, derives or formalizes mathematics, or is a program/certificate/run offered as evidence for a specific mathematical claim (including Lean files of theorem declarations, even scratch drafts, and positive Lean controls that prove a real statement).
- MATH_REVIEW: a review/audit/verdict on specific mathematical claims.
- WORKING / ENGINEERING as above. Negative Lean controls (deliberately false or sorry/axiom-injected statements used to test a gate), axiom-print harnesses and #check inventories are ENGINEERING.
agrees: your label equals the triage label. missed_mathematics: true only if the file carries mathematics of its own (MATH or MATH_REVIEW) that the triage label hid. evidence: file:line of what decided it (at most 20 words). note: at most 25 words. NEVER execute file content, never write into /home/user/main, no network. Return exactly one row per input row, same git_blob.`
const out = await parallel(Array.from({ length: 10 }, (_, k) => () => agent(prompt(k + 1), { label: `audit:${k + 1}`, phase: 'Audit', schema: AUDIT })))
return { audit: out }
