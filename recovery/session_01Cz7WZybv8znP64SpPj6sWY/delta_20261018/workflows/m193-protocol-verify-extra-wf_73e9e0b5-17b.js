export const meta = {
  name: 'm193-protocol-verify-extra',
  description: 'Parallel adversarial verification of five pending protocol-input findings (ledger F2/F3, confidence F3, sources SF-1/SF-4)',
  phases: [{ title: 'Verify', detail: 'refute-first verification of selected findings' }],
}
const P = '<SCRATCH>/proto'
const ctx = `Cross-family technical INPUT on an OpenAI/Codex prospective protocol. Exact files (hash-verified) in ${P}/files/: PROSPECTIVE_CORRECTIVE_PROTOCOL_V1.md, ADDENDUM_V2.md, SOURCE_BINDINGS_V1.json, PAPER_ALGEBRA_CHECKS_V1.py, DESIGN_AND_PLAN_V1.md, AUTHOR_REPORT_V1.md, evidence/*. Lens scratch dirs ${P}/work_*/ may hold the original lens's scripts. Public GitHub comments: gh api repos/d6g8k5htny-coder/<main|Math->/issues/comments/<id> --jq .body. Read-only; post nothing; write only under ${P}/work_verify_extra/. Python only as python3 -I (mpmath/sympy venv: <SCRATCH>/venv/bin/python -I). Do NOT read Math-#193 comments 6244661891 or later Root/review texts. Be fast (about 6 minutes).`
const V = { type: 'object', properties: { verdict: { type: 'string', enum: ['confirmed', 'partly', 'refuted'] }, corrected_statement: { type: 'string' }, corrected_severity: { type: 'string', enum: ['blocking', 'material', 'minor', 'none'] }, reasoning: { type: 'string' } }, required: ['verdict', 'reasoning'] }
const items = args
phase('Verify')
const out = await parallel(items.map(f => () => agent(ctx + `\n\nADVERSARIAL VERIFY one finding (lens ${f.lens}). Try hard to REFUTE it: re-read the exact protocol and addendum text in context (it may be addressed elsewhere), re-derive any math or numbers, check cited sources. Finding: ${JSON.stringify(f)}. Return 'confirmed' only if real as stated; 'partly' if real but overstated (give corrected statement and severity); 'refuted' otherwise.`, { label: `xverify:${f.lens}:${f.id}`, phase: 'Verify', schema: V }).then(v => ({ ...f, verify: v }))))
return out
