export const meta = {
  name: 't11c-second-screen',
  description: 'T11c: second, narrower screen of every unaudited WORKING/ENGINEERING file for mathematics of its own the triage hid',
  phases: [{ title: 'Screen', detail: 'one agent per batch of up to 120 files' }],
}
const S2 = '/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/t11c/screen2'
const SCREEN = { type: 'object', properties: { rows: { type: 'array', items: { type: 'object', properties: {
  i: { type: 'integer' }, carries_math: { type: 'boolean' }, kind: { type: 'string', enum: ['MATH', 'MATH_REVIEW', 'NONE'] },
  evidence: { type: 'string' } }, required: ['i', 'carries_math', 'kind', 'evidence'] } } }, required: ['rows'] }
const prompt = n => `Second-pass screen, batch ${n}. Read ${S2}/sbatch${n}.json: 'rows' (index = position from 0) with git_blob, path and a first-pass label WORKING or ENGINEERING. An independent audit of a random sample found that about 2% of such labels hid mathematics, typically: PR-body or comment drafts that state a theorem with constants and a proof outline; READ_FIRST/capsule/status documents that state the target theorem and derive part of it; reconciliation notes that adjudicate a mechanism with their own reasoning; review packets with substantive scope assertions; Lean 'negative control' files whose statements are actually true and fully proved.
For EVERY row, read the file with \`git -C /home/user/main cat-file -p <git_blob>\` (at least the first 150 lines or 10 KB; grep the rest for theorem|lemma|proof|bound|<=|>=|certif|verdict|PASS|FAIL) and answer one question: does this file carry mathematics OF ITS OWN, meaning a statement with its derivation or proof, a checker or certificate for a specific mathematical claim, or a review verdict on specific mathematical claims? Mere pointers, status lines, routing, restated titles, hashes, logs without results, governance and tooling are NOT. carries_math true => kind MATH or MATH_REVIEW; else kind NONE. evidence: file:line of the decisive text (at most 20 words). Return exactly one row per input row, by index. NEVER execute file content, never write into /home/user/main, no network.`
const N = args.batches
const out = await parallel(N.map(n => () => agent(prompt(n), { label: `screen2:${n}`, phase: 'Screen', schema: SCREEN })))
const res = {}
N.forEach((n, k) => { res[n] = out[k] })
return { screen2: res }
