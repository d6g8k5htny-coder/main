export const meta = {
  name: 'm193-original-equality',
  description: 'Check every ORIGINAL(k) equality claimed by the frozen GAD proof against the exact ORIGINAL bytes, incl. C_alg (1) and the 2^(-1/3) vs 2^(2/3) bookkeeping',
  phases: [{ title: 'Compare', detail: 'equation-by-equation against ORIGINAL' }, { title: 'Verify', detail: 'adversarial recheck of any mismatch' }],
}
const DIR = '<SCRATCH>/m193'
const R = {
  type: 'object',
  properties: {
    references: { type: 'array', items: { type: 'object', properties: {
      proof_location: { type: 'string' }, original_location: { type: 'string' },
      claim: { type: 'string' }, result: { type: 'string', enum: ['EQUAL', 'EQUIVALENT_AFTER_STATED_STEP', 'MISMATCH', 'NOT_FOUND'] },
      detail: { type: 'string' },
    }, required: ['proof_location', 'original_location', 'claim', 'result', 'detail'] } },
    calg_equal: { type: 'boolean', description: 'Is the proof §1 (1) C_alg expression exactly ORIGINAL (1)?' },
    factor_two_trace: { type: 'string', description: 'Explain precisely how ORIGINAL line ~348 2^(-1/3) relates to proof (19) 2^(2/3) and whether the final constants agree; show the factor bookkeeping on both sides.' },
    final_constant_agreement: { type: 'string', enum: ['AGREE', 'DISAGREE', 'CANNOT_DETERMINE'] },
    notes: { type: 'string' },
  },
  required: ['references', 'calg_equal', 'factor_two_trace', 'final_constant_agreement', 'notes'],
}
phase('Compare')
const cmp = await agent(`Read in full: ${DIR}/proof.md (frozen proof; its first 22,163 B are the frozen source) and ${DIR}/original.md (exact ORIGINAL, 24,027 B, sha256 f38c085a...). Read-only; do not post anything; scripts only under ${DIR}/work_orig/, python3 -I (sympy is available in ${DIR}/venv if needed: ${DIR}/venv/bin/python).
Task: find EVERY place where the proof claims equality/identity with ORIGINAL (e.g. 'ORIGINAL(1)', 'ORIGINAL(4)', 'as in ORIGINAL', 'unchanged ORIGINAL expression', '(6)', '(8)', '(11)', moments a,b,c,B,V, F_{d-1}, Gamma_e, C_alg) and compare against the exact ORIGINAL text, equation by equation, symbolically where needed (normalize notation; LaTeX vs ASCII). In particular: (i) is the proof's §1 expression (1) for C_alg literally ORIGINAL's (1) (full expression, all factors, including what follows Gamma(7/6)/[24^(1/3) sqrt(pi)(2pi)^d a^(d/2)] ...)? (ii) Trace the factor bookkeeping: ORIGINAL near line 348 has '=2^{-1/3} Gamma(7/6)/sqrt(pi) V_L(e)^{2/3}' while the proof's (19) has '2^(2/3) Gamma(7/6) V^(2/3)/sqrt(pi)' and then '2^(2/3)/(2*12^(1/3))=24^(-1/3)'. Determine exactly where each side places the antipodal 1/2, sphere area, 12^(-1/3) and volume factors, and whether both arrive at the identical final constant C_alg. (iii) Check ORIGINAL's units remark (~line 380-382: no additional L^-d; lengths 3d, -4(d-1), -4) against the proof's volume-cancellation claim. Report every reference with EQUAL / EQUIVALENT_AFTER_STATED_STEP / MISMATCH / NOT_FOUND and details.`, { label: 'compare-original', phase: 'Compare', schema: R })
phase('Verify')
const mism = (cmp?.references || []).filter(r => r.result === 'MISMATCH' || r.result === 'NOT_FOUND')
let verify = null
if (mism.length || (cmp && (cmp.final_constant_agreement !== 'AGREE' || !cmp.calg_equal))) {
  verify = await agent(`Adversarially re-verify these claimed mismatches between the frozen proof ${DIR}/proof.md and the exact ORIGINAL ${DIR}/original.md. Try hard to show each is NOT a real mismatch (notation, a stated intermediate step, a factor moved between equations). Keep only what survives. Items: ${JSON.stringify(mism)}; calg_equal=${cmp?.calg_equal}; final_constant_agreement=${cmp?.final_constant_agreement}; factor trace: ${cmp?.factor_two_trace}. Read-only; scripts only under ${DIR}/work_orig_verify/. Return plain text: for each item SURVIVES/REFUTED with reason, and the final verdict on whether proof C_alg equals ORIGINAL (1).`, { label: 'verify-mismatch', phase: 'Verify' })
} else { log('No mismatches; verify stage skipped') }
return { compare: cmp, verify }
