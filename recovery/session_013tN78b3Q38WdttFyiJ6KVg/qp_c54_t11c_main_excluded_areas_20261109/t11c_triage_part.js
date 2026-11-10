export const meta = {
  name: 't11c-triage-part',
  description: 'T11c triage (a range of batches): label recovery/drive/incoming residual files on main refs; optional Drive-id metadata sample',
  phases: [
    { title: 'Triage', detail: 'one agent per batch labels every file with evidence' },
    { title: 'Repair', detail: 'rerun a batch once for any file left unlabelled' },
    { title: 'Drive', detail: 'metadata-only check of sampled Drive ids from mirror manifests' },
  ],
}
const T = '/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/t11c'
const N = 87
const LABELS = ['MATH', 'MATH_REVIEW', 'WORKING', 'ENGINEERING']
const TRIAGE = { type: 'object', properties: {
  files: { type: 'array', items: { type: 'object', properties: {
    i: { type: 'integer' }, label: { type: 'string', enum: LABELS }, copy_of: { type: 'string' },
    evidence: { type: 'string' }, note: { type: 'string' } }, required: ['i', 'label', 'copy_of', 'evidence', 'note'] } },
  packets: { type: 'array', items: { type: 'object', properties: {
    title: { type: 'string' }, files: { type: 'array', items: { type: 'integer' } }, summary: { type: 'string' },
    already_accounted_hint: { type: 'string' } }, required: ['title', 'files', 'summary', 'already_accounted_hint'] } } },
  required: ['files', 'packets'] }
const RULES = `Labels (exactly one per file):
- MATH: the file states, proves, derives or formalizes mathematics, or is a program/certificate/numerical run offered as evidence for a specific mathematical claim (Lean declarations, proof notes, theorem/lemma/bound statements, symbolic or interval checkers, certificate JSON with mathematical content, run outputs that report a mathematical result). A Drive mirror (drive/mirrors/...) of a Drive object with such content is MATH; the Drive object is its source.
- MATH_REVIEW: a review, audit, readback or verdict about specific mathematical claims (with findings on them).
- WORKING: a session working artifact with no mathematical claim of its own: logs without results, drafts of GitHub comments or PR bodies, commit messages, prompts, plans, inventories, manifests, environment notes, timing, transcripts, raw API dumps, process notes, governance prose.
- ENGINEERING: tooling, CI, custody or test infrastructure, renderers, non-mathematical code.
copy_of: if the file is recognisably a copy, extract, excerpt or another version of a source held elsewhere (a Math- or main path, a PR, a zip member, a GitHub comment, a Drive object), name it as precisely as you can; otherwise ''. For drive/ files, put the manifest Drive id here (e.g. 'Drive object <id>, exact=<true|false>').
evidence: for MATH and MATH_REVIEW, the file:line range of the key statement or verdict (e.g. 'L12-L30: Theorem 2 ...', at most 20 words); for other labels, a short reason ('L1-L5: git commit message').
note: at most 25 words.
packets: group the MATH and MATH_REVIEW files of this batch into objects for a later deep pass (one packet = one coherent piece of work, normally one directory). For each, give a precise title, the file indices, a 1-3 sentence summary of what is claimed, and already_accounted_hint: any record ids, comment-census rows or catalog ids the files cite or that the batch entry lists (comment_census_rows, impl_sourcing_rows), else ''.`
const tri = n => `You are triaging tranche-11c batch ${n} of mathematical accounting for the d6g8k5htny-coder federation. Read ${T}/tb/tbatch${n}.json: a list 'files' of entries (index = position in the list, starting at 0) with git_blob, path, bytes, ref, commit, cluster, and sometimes comment_census_rows, impl_sourcing_rows (citations by Claude S's comment-math census/sourcing, meaning the file was identified as an implementation of comment-delivered mathematics) or drive_manifest_row. These are files on main pull-request heads and branch tips (session retention trees under recovery/, Drive mirrors under drive/, incoming/) whose bytes occur nowhere else in git. Read each file with \`git -C /home/user/main cat-file -p <git_blob>\` (pipe through head -c or sed -n for large files; read enough to decide, at least the first 120 lines or 8 KB, and grep for theorem/lemma/proof/sorry/assert/certif when unsure). NEVER execute any file content, never write into /home/user/main, and do not use the network. Label EVERY file in the batch exactly once, by index.\n\n${RULES}`
const LO = args.lo, HI = args.hi
const IDS = Array.from({ length: HI - LO + 1 }, (_, k) => LO + k)
const results = await pipeline(IDS,
  n => agent(tri(n), { label: `triage:${n}`, phase: 'Triage', schema: TRIAGE }),
  (r, n) => r)
const out = {}
IDS.forEach((n, k) => { out[n] = results[k] })
phase('Repair')
const missing = IDS.filter(n => !out[n])
log(`batches ${LO}-${HI} with no result: ${missing.length}`)
const repaired = await parallel(missing.map(n => () => agent(tri(n), { label: `triage-retry:${n}`, phase: 'Repair', schema: TRIAGE })))
missing.forEach((n, j) => { out[n] = repaired[j] })
if (!args.drive) return { lo: LO, hi: HI, triage: out }
phase('Drive')
const DRIVE = { type: 'object', properties: { checked: { type: 'array', items: { type: 'object', properties: {
  drive_id: { type: 'string' }, found: { type: 'boolean' }, trashed: { type: 'boolean' }, title_matches_path: { type: 'boolean' },
  size_matches_manifest: { type: 'string' }, note: { type: 'string' } }, required: ['drive_id', 'found', 'trashed', 'title_matches_path', 'size_matches_manifest', 'note'] } } }, required: ['checked'] }
const half = (a, b) => `Read ${T}/drive_sample.json ('sample' list; take entries with index ${a} to ${b - 1} inclusive of the 0-based list). For each entry, look up its drive_id on Google Drive by METADATA ONLY using the Google Drive MCP tool for file metadata (load it with ToolSearch, e.g. query 'google drive get_file_metadata'). Never download, preview or read file content. Record only: found (the id resolves), trashed (if reported), title_matches_path (the Drive title equals the last component of 'path' after removing a trailing '.export.txt' or similar export suffix, or equals it outright), size_matches_manifest ('yes', 'no' or 'n/a' for native Google files with no size; the manifest bytes are in ${T}/tb/*.json entries' drive_manifest_row for that path, which you may grep), and a short note. Do NOT record any owner, email, sharing, permission or user information anywhere in your output.`
const drv = await parallel([0, 1].map(h => () => agent(half(h * 58, (h + 1) * 58), { label: `drive:${h + 1}`, phase: 'Drive', schema: DRIVE })))
return { lo: LO, hi: HI, triage: out, drive: drv }
