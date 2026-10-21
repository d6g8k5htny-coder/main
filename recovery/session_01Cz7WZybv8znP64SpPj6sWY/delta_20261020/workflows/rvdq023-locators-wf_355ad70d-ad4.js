export const meta = {
  name: 'rvdq023-locators',
  description: 'Compile and adversarially verify read-only locators for the RV-DQ-023 / LS-DER-021 preflight (local git only, nothing posted)',
  phases: [
    { title: 'Locate', detail: 'register rows, LS-DER-021 texts and frozen-body rule, other appearances' },
    { title: 'Verify', detail: 'independent recomputation of every identity' },
    { title: 'Draft', detail: 'locator note draft, saved locally, not posted' },
  ],
}

const OUT = '<SCRATCH>/rvdq023'
const RULES = `
HARD RULES (all agents):
- Read-only. Use only local git objects in /home/user/main via commands such as git show, git cat-file, git ls-tree, git grep <rev>. Do not run git checkout, switch, reset, fetch, pull, push, commit or anything that changes the working tree, index, refs or remotes.
- No network: do not run gh, curl or any GitHub/MCP call. Do not post anywhere.
- Write files only under ${OUT}/ .
- Hardening commit: a01c72f1937875b94bfd78e183a6dbbdb2b04da4 (the "hardening branch a01c72f1"). Current main is origin/main (expected 1e16211d...).
- Paths contain spaces and non-ASCII characters (em dash, U+2215); always quote them and use git ls-tree -z / NUL handling where needed.
- Report exact facts: full blob ids, byte sizes, full SHA-256 of the exact bytes, line numbers. Never round or abbreviate in the JSON you return (abbreviate only in prose).
- This is LOCATOR work only. Do not review mathematics, do not judge correctness, and do not decide which body is the review target. "Binding" is the preflight owner's decision.
`

const FIND = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    facts: { type: 'array', items: { type: 'object', properties: {
      label: { type: 'string' }, rev: { type: 'string' }, path: { type: 'string' },
      blob: { type: 'string' }, bytes: { type: 'integer' }, sha256: { type: 'string' },
      lines: { type: 'string' }, verbatim: { type: 'string' }, note: { type: 'string' } },
      required: ['label', 'note'] } },
    limits: { type: 'array', items: { type: 'string' } },
    file_written: { type: 'string' },
  },
  required: ['summary', 'facts', 'limits'],
}
const VERDICT = {
  type: 'object',
  properties: {
    confirmed: { type: 'array', items: { type: 'string' } },
    refuted: { type: 'array', items: { type: 'object', properties: { label: { type: 'string' }, why: { type: 'string' }, correct_value: { type: 'string' } }, required: ['label', 'why'] } },
    missed: { type: 'array', items: { type: 'string' } },
    overall: { type: 'string', enum: ['HOLDS', 'HOLDS_WITH_CORRECTIONS', 'DOES_NOT_HOLD'] },
  },
  required: ['confirmed', 'refuted', 'missed', 'overall'],
}

const LENSES = [
  { key: 'registers', prompt: `${RULES}
TASK (lens: register rows). On a01c72f1:
1. docs/OPEN_PROBLEMS.md: give the exact line numbers and verbatim rows for RV-DQ-023 and its companion RV-DQ-053, plus the table header and the section heading they sit under.
2. registers/csv/review_queue.csv: the header line and the full verbatim rows (all columns) for RV-DQ-023 and RV-DQ-053, with line numbers. Name the body-hash column and the author/exposure column and quote their values for RV-DQ-023 exactly (e.g. "UNRESOLVED — bind before review").
3. git grep on a01c72f1 for "RV-DQ-023" and for "LS-DER-021" in registers/ and docs/ (paths + line numbers + the matching line, trimmed to 300 chars).
4. Repeat 1-2 on origin/main and report whether the rows are byte-identical, changed (show both), or absent.
Write a markdown file ${OUT}/lens_registers.md with everything, and return the JSON.` },
  { key: 'texts', prompt: `${RULES}
TASK (lens: LS-DER-021 texts and the frozen-body rule).
1. List every file on a01c72f1 whose PATH names LS-DER-021 (including FS2-AMD-001, FS2-ERR-002 and CL-AUD-TBG3CHAIN-... files) with full blob id, byte size and full SHA-256 of the blob bytes. Do the same on origin/main (same blob or not, and any new paths).
2. Known frozen review targets from docs/OPEN_PROBLEMS.md on a01c72f1 (size / SHA-256 prefix): LS-DER-026 11,725 / e4d8094e; LS-DER-022 21,932 / e5a61739; LS-DER-032 12,106 / 3531e6e5; LS-DER-036-v1.1 16,856 / 3416ddee; LS-DER-038 10,410 / 5fd2413c. Find each one's .export.txt under drive/mirrors/15_REVIEWS_RESPONSES_AND_CLOSURES/THEOREM_B — STATUS RETRACTION AND REPAIR PROGRAM/ and try to discover the extraction rule that turns the export into the frozen body: e.g. a contiguous byte span between markers, a header/footer strip, line-ending or BOM normalization, or a trailing-newline rule. Test candidates by exact size and SHA-256 prefix. Also look in registers/ docs/ tools/ for any script or note that defines "frozen body" or "body hash" (git grep -i "frozen body", "body_sha", "body hash").
3. Only if one rule reproduces at least three of those five exactly, apply the same rule to the LS-DER-021-v1.0 export and report the resulting byte size and full SHA-256, labelled "candidate under the same rule — NOT a binding". Save that candidate body to ${OUT}/LS-DER-021_candidate_body.txt. If no rule reproduces at least three, say so plainly and do not produce a candidate.
4. From the LS-DER-021-v1.0 export, quote its header lines that state version, date, author/provider/session and any self-reported identity (first ~40 lines; verbatim, with line numbers).
Write ${OUT}/lens_texts.md and return the JSON.` },
  { key: 'appearances', prompt: `${RULES}
TASK (lens: other appearances, reviews and exposure).
1. On a01c72f1 and origin/main, git grep -l "LS-DER-021" across the whole tree (skip binary). Group the hits by kind: register/ledger, review or audit record (CL-AUD-*, *-REVIEW-*), amendment/erratum (FS2-*), downstream citations (other LS-DER-* that import it), recovery or retention units, other. Give counts and list the first 40 paths with line numbers of the first match.
2. For each review/audit record that reviews LS-DER-021 (e.g. CL-AUD-TBG3CHAIN-20260726-01, anything named LS-DER-021-REVIEW-*), quote verbatim any statement of the reviewed object's identity (version, byte size, hash, Drive title) and the reviewer's stated provider/organization, with line numbers.
3. FS2-AMD-001 and FS2-ERR-002: quote verbatim (with line numbers) the passages about LS-DER-021, its prior reviews and exposure/authorship.
4. Note any place that records LS-DER-021's author provider or the providers that have already reviewed it (needed for "next path must be neither OpenAI nor Anthropic" and "disclose exact-object authorship").
Write ${OUT}/lens_appearances.md and return the JSON.` },
]

phase('Locate')
const results = await pipeline(
  LENSES,
  l => agent(l.prompt, { label: `locate:${l.key}`, phase: 'Locate', schema: FIND }),
  (found, l) => found ? agent(`${RULES}
ADVERSARIAL VERIFICATION of the "${l.key}" lens. Try to REFUTE every fact below by recomputing it yourself from git objects (git cat-file -p <blob> | sha256sum, wc -c, git show <rev>:<path> | sed -n, git ls-tree). Default to refuting anything you cannot reproduce exactly. Also list important facts the lens missed for its task.
Lens task (for scope):
${l.prompt}
Lens output JSON:
${JSON.stringify(found)}`, { label: `verify:${l.key}`, phase: 'Verify', schema: VERDICT }).then(v => ({ key: l.key, found, verdict: v })) : null
)

phase('Draft')
const ok = results.filter(Boolean)
log(`lenses verified: ${ok.map(r => r.key + '=' + (r.verdict ? r.verdict.overall : 'none')).join(', ')}`)
const draft = await agent(`${RULES}
Draft (do NOT post) a concise locator note for the RV-DQ-023 preflight, to be posted later on main#419 by Anthropic Claude session 01Cz (session_01Cz7WZybv8znP64SpPj6sWY) once its GitHub writes are allowed. dot (OpenAI) wrote in 419/6320832078 that GROK-H01 or one eligible xAI invocation may take the RV-DQ-023 exact-body/ownership preflight ("identify and bind the frozen LS-DER-021 body before any proof review, check both main and Math- contenders, and disclose exact-object authorship/exposure") and that "Claude session_01Cz7WZybv8znP64SpPj6sWY may help supply the promised locators, not independently review its own reconciliation". The promise was in 01Cz's 18 Oct list 419/6250496071 item 2.
Use ONLY facts that the verifiers confirmed or corrected (apply corrections; drop refuted facts; mention material "missed" items only if confirmed by another lens's facts).
Style: bold one-line headline; a status line ("Data only — not a pickup, review, binding or claim; credit 0; scientific effect NONE"); short bullets grouped as Register rows / Candidate texts (path, size, full SHA-256 in code spans) / Frozen-body rule (found or not, and the candidate clearly labelled NOT a binding) / Exposure and prior reviews / Limits (Math- not checked locally; contender check must be redone fresh at pickup). Use full SHA-256 values, never ellipses inside code spans. Do not use closing verbs before #N; refer to Math- PRs as "Math- PR N". End with a blank line, then '---', then '_Generated by [Claude Code](https://claude.ai/code)_'.
Write it to ${OUT}/locators_draft.md and return the draft text.
Verified lens results:
${JSON.stringify(ok)}`, { label: 'draft', phase: 'Draft' })
return { verdicts: ok.map(r => ({ key: r.key, overall: r.verdict && r.verdict.overall, refuted: r.verdict && r.verdict.refuted })), draft }
