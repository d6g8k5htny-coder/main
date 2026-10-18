export const meta = {
  name: 'm193-gad-hypotheses-fidelity',
  description: 'Check that the frozen proof §2 states the consumed GAD theorem and its hypotheses faithfully, against the located GAD source',
  phases: [{ title: 'Check', detail: 'GAD text vs proof §2' }, { title: 'Verify', detail: 'adversarial recheck of any gap' }],
}
const DIR = '<SCRATCH>/m193'
const S = { type: 'object', properties: {
  hypotheses: { type: 'array', items: { type: 'object', properties: {
    gad_hypothesis: { type: 'string' }, gad_location: { type: 'string' }, stated_in_proof_s2: { type: 'string', enum: ['YES', 'PARAPHRASED_EQUIVALENT', 'WEAKER_OR_DIFFERENT', 'MISSING'] },
    verified_for_original_law_in_proof: { type: 'string', enum: ['YES', 'PARTIAL', 'NO', 'NOT_APPLICABLE'] }, detail: { type: 'string' } },
    required: ['gad_hypothesis', 'gad_location', 'stated_in_proof_s2', 'verified_for_original_law_in_proof', 'detail'] } },
  conclusion_matches: { type: 'boolean', description: 'GAD conclusion (density formula/coefficient (27)/(28), measure, representative) as used by the proof matches GAD text' },
  gaps: { type: 'string' }, verdict: { type: 'string', enum: ['FAITHFUL', 'FAITHFUL_WITH_NOTES', 'GAP'] } },
  required: ['hypotheses', 'conclusion_matches', 'gaps', 'verdict'] }
phase('Check')
const r = await agent(`Read in full: the GAD source text, which is the 40,231-byte range starting at byte offset 1326 of ${DIR}/gad/c6242006932.md (extract it with python: open(...,'rb').read()[1326:1326+40231]; its sha256 must be 8d992e194f0a898e563c4c3d05b8079cde292f9e87d7ff51030ad12ef5faf319 — verify), and the accepted GAD review at byte offset 651, length 38035 of ${DIR}/gad/c6242020976.md (sha256 6fedc43ba34408f23f8934166b3898e2cd19175c57e61e37227c85f3ca5d4c70). Then read the frozen proof ${DIR}/proof.md (§2 'Exact theorem imports' especially, and §§3,7,8 where GAD is applied). Read-only; nothing posted; scripts under ${DIR}/work_gadfid/ with python3 -I.
Task: list EVERY hypothesis of GAD's main theorem(s) the proof consumes (smooth-series admission: index set, amplitudes a_j>0, smooth functions phi_j, all-norm summability, dense real smooth span, compact connected manifold without boundary, dimension condition d>=2, any basis/orthonormality/independence/Gaussianity condition, any nondegeneracy condition on jets or two-point laws, any condition on the metric/volume, any genericity/Morse condition, the once-counting selector/representative statement, the exact density formula (26)/(27)/(28) and its measure dσ). For each: is it stated in the proof's §2 (YES / PARAPHRASED_EQUIVALENT / WEAKER_OR_DIFFERENT / MISSING), and does the proof verify it for the ORIGINAL law (YES/PARTIAL/NO/NA), with locations. Also check the conclusion the proof uses matches GAD exactly (including the representative/version statement and the 1/(2*12^(1/3)*V) prefactor with |T|^(4/3)). Report gaps precisely.`, { label: 'gad-fidelity', phase: 'Check', schema: S })
phase('Verify')
let v = null
if (r && r.verdict !== 'FAITHFUL') {
  v = await agent(`Adversarially re-check these claimed gaps between the GAD source (${DIR}/gad/c6242006932.md bytes 1326..41556) and the frozen proof §2 (${DIR}/proof.md). Try to show each is not a real gap (an equivalent statement elsewhere in the proof, an implied condition, or a hypothesis GAD does not actually have). Keep only survivors with exact quotes. Gaps: ${r.gaps}\nHypothesis table: ${JSON.stringify(r.hypotheses.filter(h => h.stated_in_proof_s2 !== 'YES' || h.verified_for_original_law_in_proof !== 'YES'))}. Read-only. Return plain text: each item SURVIVES/REFUTED with reason, then overall FAITHFUL / FAITHFUL_WITH_NOTES / GAP.`, { label: 'verify-gaps', phase: 'Verify' })
}
return { check: r, verify: v }
