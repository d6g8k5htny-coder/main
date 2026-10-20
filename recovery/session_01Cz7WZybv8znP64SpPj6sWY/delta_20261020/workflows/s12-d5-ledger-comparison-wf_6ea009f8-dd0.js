export const meta = {
  name: 's12-d5-ledger-comparison',
  description: 'Compare four July D5 historical carriers with the P2 punctured-pin proof ledger; adversarially verify',
  phases: [{ title: 'Lens', detail: 'one lens per carrier' }, { title: 'Verify', detail: 'two refuters per lens' }],
}
const S = '<SCRATCH>/s12'
const CTX = `READ-ONLY board-record support. Do NOT post, edit or push anything on GitHub or in any repository. Work from local files in ${S}/src/ and write scratch only under ${S}/work/<your-label>/. Python with sympy/mpmath: ${S}/../venv/bin/python (run with -I). Never reproduce Google Drive ids or Drive/Docs URLs, and do not name model versions in your output (say 'OpenAI', 'xAI', 'Anthropic').

TASK: recovery row §12 asks to "compare [the July D5 historical carriers'] ledgers with the P2 punctured-pin proof d8acf0bc". P2 = ${S}/src/P2.md (Math- reviews/d5_local_collar_20260928/PUNCTURED_PIN_PROOF.md, blob d8acf0bc, 23,771 B): an all-height punctured endpoint-disk bound (P2) with ledger items (P1)-(P20), Sections 2-10, regions I/II, power ledger at l.365. Its committed nonauthor records: ${S}/src/d5_punctured_pin_nonauthor_20260928.REVIEW.md (xAI; accepts algebraic skeleton P4/P9/P17/P20, HOLD on continuum) and ${S}/src/d5_punctured_pin_continuum_claude_20260929.REVIEW.md (Anthropic, another session; continuum steps). Context: ${S}/src/D5_RECONCILIATION.md (Math-#138; composes punctured pin disks, nested microdisks, collar, fixed annulus, window shells, remote window into D5-a..D5-f at planar existential scope; it supersedes the historical carriers at that scope, so their value now is an independent consistency check).
Carriers (July 2026, same-family author-side; exact bytes rebuilt from hardening a01c72f1 DG-MIGRATION objects): C100 Collar Cubic Closure ${S}/src/C100.bin (24eb9d55…); LS-DER-024 (docx; text extracted at ${S}/src/LS-DER-024.txt; focus §5 and §7; NOTE its AMEND notice by LS-DER-035 on Lemma 2.1 Case III); LS-DER-033 (docx; text ${S}/src/LS-DER-033.txt; exact endpoint determinants, envelope <= C r^4 t^-6); FlatSaddle Tail Lemma ${S}/src/FlatSaddle.bin (3875016e…; nested-disk O(r^5) under (U1)-(U4)).

For YOUR carrier: (1) state exactly what it claims (region, law/field, scale window, heights, marks, constants, status flags such as AMEND/withdrawn/conditional hypotheses), with line or equation loci; (2) map each of its ingredients onto the corresponding P2 item(s) (P1)-(P20) or, if the region is not P2's, onto the D5 reconciliation component it corresponds to (collar, nested microdisk, intermediate shells, ...), and classify each as AGREE (same bound/power with compatible hypotheses), SCOPE_DIFFERS (e.g., L=24 window vs all heights/any fixed T, compact marks vs k>=k_-, existential vs explicit, different law such as pair-Palm vs Q_r), or CONTRADICTS (a stated bound/power/identity that conflicts with P2 or the reconciliation); (3) re-derive any power ledger you compare (exponents of r, pin-collision cancellations, determinant orders) yourself, with sympy if useful; (4) flag anything in the carrier that is withdrawn or amended and must not be used as support. Be exact and cite loci.`
const LENSES = [
  { key: 'C100', p: 'Your carrier: C100 Collar Cubic Closure.' },
  { key: 'LSDER024', p: 'Your carrier: LS-DER-024 (§5 and §7 in particular; respect the AMEND notice).' },
  { key: 'LSDER033', p: 'Your carrier: LS-DER-033.' },
  { key: 'FlatSaddle', p: 'Your carrier: the FlatSaddle Tail Lemma.' },
]
const LENS_SCHEMA = { type: 'object', properties: {
  carrier_claim: { type: 'string' }, status_flags: { type: 'array', items: { type: 'string' } },
  rows: { type: 'array', items: { type: 'object', properties: { ingredient: { type: 'string' }, carrier_locus: { type: 'string' }, p2_or_component: { type: 'string' }, classification: { type: 'string' }, evidence: { type: 'string' } }, required: ['ingredient', 'carrier_locus', 'p2_or_component', 'classification', 'evidence'] } },
  summary: { type: 'string' }, caveats: { type: 'array', items: { type: 'string' } } },
  required: ['carrier_claim', 'status_flags', 'rows', 'summary', 'caveats'] }
const V_SCHEMA = { type: 'object', properties: {
  refuted_rows: { type: 'array', items: { type: 'object', properties: { ingredient: { type: 'string' }, why: { type: 'string' }, correction: { type: 'string' } }, required: ['ingredient', 'why'] } },
  upheld: { type: 'array', items: { type: 'string' } }, missed: { type: 'array', items: { type: 'string' } }, summary_upheld: { type: 'boolean' }, notes: { type: 'string' } },
  required: ['refuted_rows', 'upheld', 'missed', 'summary_upheld', 'notes'] }
return await pipeline(
  LENSES,
  l => agent(`${CTX}\n\n${l.p}`, { label: `lens:${l.key}`, phase: 'Lens', schema: LENS_SCHEMA }),
  (rep, l) => parallel(['source-fidelity (re-read every cited locus in the carrier, P2, the reviews and the reconciliation; is each quote/locus exact and does it say what is claimed?)',
                        'mathematics (re-derive each classified power ledger, cancellation or determinant order independently; try hard to find a row whose classification is wrong — an AGREE that is really SCOPE_DIFFERS, or a missed CONTRADICTS)'].map(angle => () =>
    agent(`${CTX}\n\nADVERSARIAL VERIFICATION, angle: ${angle}. The lens report for carrier '${l.key}' is below. Try to REFUTE each row; default to refuted if unsupported by exact text or your own derivation. List anything the lens missed that bears on the comparison.\n\nLENS REPORT:\n${JSON.stringify(rep)}`,
      { label: `verify:${l.key}:${angle.split(' ')[0]}`, phase: 'Verify', schema: V_SCHEMA })))
    .then(vs => ({ carrier: l.key, report: rep, verifications: vs })),
)
