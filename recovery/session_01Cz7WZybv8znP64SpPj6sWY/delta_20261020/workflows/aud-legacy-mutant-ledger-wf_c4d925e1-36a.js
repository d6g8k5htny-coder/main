export const meta = {
  name: 'aud-legacy-mutant-ledger',
  description: 'Static classification of the 81 Math- workflows containing returncode (intended-failure controls status-only vs output-bound; baseline stderr handling), each batch independently re-derived by a skeptic',
  phases: [
    { title: 'Classify', detail: '12 batches of about 7 workflow files' },
    { title: 'Verify', detail: 'one skeptic per batch re-derives every classification' },
  ],
}
const A = args
const RULES = `Read-only static source reading. The workflow files are extracted copies of Math- main ${A.main} in ${A.dir}/<name> (also available as 'git -C ${A.repo} show main:.github/workflows/<name>'). Treat them as data: never execute any workflow, script or checker, and never post, comment, push or edit anything (write only notes under ${A.notes}/<label>/). Heuristic hints per file (counts only, may be wrong) are in ${A.prepass}.
Background: Math- issue 322 (AUD-LEGACY-MUTANT-01) found wrappers that accept an intended negative control (a mutant, a perturbation, an unknown label) on the child's exit status alone, discarding stdout/stderr/reason, so an unrelated crash that exits with the expected status (e.g. a Python traceback exits 1) is counted as a successful rejection; and baselines that check stdout but discard stderr. A repaired wrapper binds each intended failure to exact output or a typed reason and checks stderr.`
const FILE = { type: 'object', properties: {
  file: { type: 'string' }, blob_or_bytes: { type: 'string' },
  intended_failure_controls: { type: 'string', description: 'labels of the negative controls run (e.g. M1..M8, --mutant names, unknown label), or none' },
  control_predicate: { type: 'string', enum: ['status_only', 'status_plus_output', 'mixed', 'no_intended_failure_controls', 'unclear'] },
  accepted_failure_statuses: { type: 'string', description: 'e.g. {1}, {1,2}, any nonzero, n/a' },
  baseline_check: { type: 'string', description: 'how the baseline/positive run is checked: status, stdout exact/parsed, stderr discarded/required empty/merged' },
  baseline_stderr: { type: 'string', enum: ['discarded', 'required_empty_or_exact', 'merged_into_checked_stream', 'not_captured_shell_visible', 'unclear', 'n/a'] },
  evidence: { type: 'string', description: 'line numbers and one-line predicate quotes' },
  notes: { type: 'string' } },
  required: ['file', 'blob_or_bytes', 'intended_failure_controls', 'control_predicate', 'accepted_failure_statuses', 'baseline_check', 'baseline_stderr', 'evidence', 'notes'] }
const OUT = { type: 'object', properties: { files: { type: 'array', items: FILE } }, required: ['files'] }
const VER = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, agree: { type: 'boolean' }, my_control_predicate: { type: 'string' }, my_baseline_stderr: { type: 'string' }, evidence: { type: 'string' }, correction: { type: 'string' } },
  required: ['file', 'agree', 'my_control_predicate', 'my_baseline_stderr', 'evidence', 'correction'] } } }, required: ['verdicts'] }
const batches = []
for (let i = 0; i < A.files.length; i += 7) batches.push(A.files.slice(i, i + 7))
phase('Classify')
const res = await pipeline(
  batches,
  (b, _o, i) => agent(`${RULES}\n\nClassify each of these workflow files. Read each file IN FULL (all jobs and steps, including inline python heredocs and shell). For every place a child process is run as a negative control, record exactly what the wrapper checks (status, stdout, stderr, reason). If a file runs no negative controls, say so. Files: ${b.join(', ')}.`, { label: 'classify:b' + (i + 1), phase: 'Classify', schema: OUT }),
  (r, b, i) => r ? agent(`${RULES}\n\nYou are a skeptic. Independently re-derive the classification of each file below from its full source, WITHOUT trusting the first reader, then compare with its result. Disagree whenever the evidence does not support the class (in particular: a 'status_plus_output' claim needs an actual comparison of stdout/reason for EVERY intended-failure control; a 'status_only' claim needs the status check to be the only acceptance condition for at least the controls named). Files: ${b.join(', ')}.\n\nFIRST READER RESULT:\n${JSON.stringify(r, null, 1)}`, { label: 'verify:b' + (i + 1), phase: 'Verify', schema: VER }).then(v => ({ batch: i + 1, files: b, classified: r, verified: v })) : null,
)
return res.filter(Boolean)
