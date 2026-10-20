export const meta = {
  name: 'verify-s12-result',
  description: 'Pre-post adversarial check of the §12 D5 ledger-comparison result before posting on main#419',
  phases: [{ title: 'Check', detail: 'four independent lenses on the draft post' }],
}
const S = '<SCRATCH>/s12'
const CTX = `READ-ONLY pre-post check. Do NOT post, edit, push, or modify any file outside ${S}/work/prepost_<your-lens>/ (scratch only). Never reproduce Google Drive ids or Drive/Docs URLs. Do not name model versions in your output (say 'OpenAI', 'xAI', 'Anthropic').

DRAFT POST to check: ${S}/result.md (a board comment for GitHub issue d6g8k5htny-coder/main#419). Companion pointer: ${S}/ptr275_result.md (RESULT_ID is a placeholder filled after posting).
Sources:
- carriers: ${S}/src/C100.bin, ${S}/src/LS-DER-024.txt (HTML-entity-escaped text layer of the docx; its line numbers are not frozen-body line numbers), ${S}/src/LS-DER-033.txt, ${S}/src/FlatSaddle.bin; docx originals beside them.
- P2: ${S}/src/P2.md (blob d8acf0bc); nonauthor records ${S}/src/d5_punctured_pin_nonauthor_20260928.REVIEW.md (xAI) and ${S}/src/d5_punctured_pin_continuum_claude_20260929.REVIEW.md (Anthropic); D5 reconciliation ${S}/src/D5_RECONCILIATION.md (Math- blob 4062d76b at Math- commit 79f9a735).
- the 12 underlying comparison/verification results: ${S}/journal_results.json (4 lens reports + 8 adversarial verifications; read them all).
- scratch scripts and outputs of those agents: ${S}/work/*/
- repository: /home/user/main (shallow clone; use 'git show origin/main:<path>' — origin/main is current). Q0 ledger at incoming/chatgpt-c022-q0-recovery-20260928/context/Q0_LEDGER.md; Q0 crosswalk at incoming/chatgpt-q0-recovery-crosswalk-20260928/RESULT.md; STATUS.md.
- the author's earlier posts it references: ${S}/pickup.md (posted as 6251517975), ${S}/pickup2.md (6279655509), ${S}/late_release.md (6279524549).
- Python with sympy/mpmath: ${S}/../venv/bin/python (run with -I).

Report ONLY concrete problems in the draft: a false or unsupported statement, a wrong locus/quote/number, an overclaim relative to the evidence, a missing caveat that a careful reader needs, or a protocol breach. For each, give the exact draft text, why it is wrong (with evidence), severity (blocker = must fix before posting; fix = should fix; nit), and replacement text. Default to flagging when the draft says more than the evidence shows.`
const LENSES = [
  { key: 'source', p: 'LENS: source fidelity. Check every quote, line locus, equation number, section number, hash, size, blob id and comment id in the draft against the sources. Quotes must be verbatim (modulo HTML entities). Check that each attributed statement is really in the cited carrier/P2/reconciliation/ledger.' },
  { key: 'math', p: 'LENS: mathematics. Independently re-check the numerical and symbolic claims in the draft: the intensity ledger (-3+6-2), the normalizer constant 3.2310 at b=6/5, C100\'s leading factor and its order vs (P16)/(P17), the A_4 ~ -72k/(r(2p-1)) axis claim and the r^3 rho^2 axis order, the typed outer-axis Monte Carlo 46.6 vs analytic 46.59, the t^6 growth of product/(r^6 rho^2) (about 20 at t=1), the c(X)/Y^2 suppression claim, the fixed-t positive limit (about 1.04 t) and the r^-4 ratio growth at t=0.4, the r^2 t^4 typed-weight claim, (25)\'s (cs)^-6 factor, the FlatSaddle ledger r^4 * r^-5 * r^8 * r^-2 = r^5 and its tail-set epsilon nu^2 r^3 defect. Use the agents\' scratch outputs where they exist, and your own sympy where cheap. Flag any number the evidence does not support as stated.' },
  { key: 'protocol', p: 'LENS: board protocol, privacy and consistency. Check: (1) no Google Drive ids or Drive/Docs URLs, no secrets, no personal data, no private host paths, no model version names anywhere in the draft or pointer; (2) no closing verb (close/closes/fix/fixes/resolve/resolves and variants) immediately before a #N reference; (3) every github link is well formed and points to the right repository/issue (comment ids 6279655509, 6251517975, 6279524549 on main#419); (4) the claim id, session id and credit/scientific-effect lines match the pickup ${S}/pickup2.md, and the result stays inside the pickup\'s stated scope and exclusions (no status change, no PROOF_INDEX edit, no review of P2 beyond the map); (5) the draft does not flip or imply any lemma/prize/status disposition; (6) the personal release is explicit; (7) self-consistency: numbers, labels and component names agree between sections and with the pointer; (8) the corrections section accurately describes the earlier posts.' },
  { key: 'completeness', p: 'LENS: completeness and faithfulness to the 12 underlying results. Read every lens report and verification in journal_results.json. (a) Does the draft represent their findings faithfully, without overclaiming beyond what the verifications upheld, and with every verifier correction applied? (b) Is any material finding omitted that a reader of the board record needs (e.g., caveats about the reconciliation being a proposal vs main STATUS.md now recording D5 as accepted, the independence caveat, window-definition caveats, any CONTRADICTS row, any withdrawn item)? (c) Does any verifier dispute a statement the draft makes? List only material omissions or misrepresentations.' },
]
const SCHEMA = { type: 'object', properties: {
  problems: { type: 'array', items: { type: 'object', properties: {
    draft_text: { type: 'string' }, why: { type: 'string' }, severity: { type: 'string', enum: ['blocker', 'fix', 'nit'] }, replacement: { type: 'string' } },
    required: ['draft_text', 'why', 'severity', 'replacement'] } },
  checked_ok: { type: 'array', items: { type: 'string' } },
  verdict: { type: 'string', enum: ['READY', 'READY_AFTER_FIXES', 'NOT_READY'] } },
  required: ['problems', 'checked_ok', 'verdict'] }
phase('Check')
return await parallel(LENSES.map(l => () => agent(`${CTX}\n\n${l.p}`, { label: `prepost:${l.key}`, phase: 'Check', schema: SCHEMA })))
