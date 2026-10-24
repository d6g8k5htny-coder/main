export const meta = {
  name: 's9-side24-crosswalk',
  description: 'Crosswalk the SIDE24 chain (LS-DER/CL-AUD/LS-CLS-077) to Math- side24_v1 PROOF.md and parent Eq. (15.2); adversarially verify',
  phases: [
    { title: 'Lens', detail: 'four independent lenses over local source copies' },
    { title: 'Verify', detail: 'two refuters per lens report' },
  ],
}
const S = '<SCRATCH>/s9'
const CTX = `You are helping produce a READ-ONLY, board-only source crosswalk record. Do NOT post anything to GitHub, do NOT edit any repository, do NOT push. Work only from the local files listed. Python with sympy/mpmath is at ${S}/../venv/bin/python (run with -I). Write any scratch scripts under ${S}/work/<your-lens-name>/ only.
Never reproduce Google Drive ids or Drive/Docs URLs in your output.

QUESTION: Do the two sources below define and evaluate the SAME number c_(3,24) (d=3, torus side 24), at formula level (field, normalization, counted object, conventions, factors), or does something fail? An 18-digit decimal match alone is NOT source equivalence.

SOURCE A (the chain, July 2026, Drive-mirror exports on hardening branch a01c72f1; each file is an export with a header, often a frozen body between BEGIN_..._FROZEN_BODY and END_..._FROZEN_BODY lines and a declared BODY_SHA256):
- ${S}/named/*.txt (index with blob ids, bytes, sha256 in ${S}/chain_index.tsv). Key: LS-CLS-077-v1.0.txt (terminal closure), LS-CLS-074-v1.0.txt, CL-AUD-LSDER062/071/072/073/073V13/075/076-*.txt (Anthropic verdicts), LS-DER-075-v1.0.txt (absolute normalization), LS-DER-076-v1.0.txt (directed-sphere / ordered-pair crosswalk), LS-DER-056-v1.0.txt (exact finite-torus d3 coefficient formula (24)-(25)), LS-DER-040/041/053/058 (theorem, field), LS-DER-059/061/062/063 (side-24 correction, P_3(24)), LS-DER-050/047 (closed forms), LS-DATA-075A/B.
SOURCE B (landing source): Math- coefficients/side24_v1 at e329fba1: ${S}/side24/PROOF.md (10272 B, sha256 c06daccc...), ENCLOSURE.json, coefficient.py, test_coefficient.py. Its parent P: ${S}/../s7/src/P.md (40261 B, sha256 9350ad6e...), especially Section 1 (statement, (1.3)), Sections 13-15, Eq. (15.2) at lines ~447-480.
ALREADY ESTABLISHED by the coordinator's own sympy runs (outputs in ${S}/work/sym_core.out and ${S}/work/sym_finiteL.out; scripts alongside): (i) LS-DER-075 eq (21), LS-CLS-077's closed form for c_(3,infinity), and PROOF.md formula (1) at d=3 are algebraically identical; (ii) P (15.2) evaluated with the reference covariance exp(-|x|^2/2) reduces to formula (1) for d=2,3; (iii) the LS-DER-056 (24)-(25) integrand equals the P (15.2) d=3 integrand pointwise, under the identifications p_G(0)<->p_{g_t}(0), p_Vu(0)<->p_{h_t}(0), D_u<->D_t, tau_u^2 = 144 sigma_t^2 (since u_t = -f_ttt/12), and the same sphere measure; (iv) c_ref lies inside PROOF's 20-digit d=3 enclosure; |P_3(24) e^-288| ~ 1.49e-118 < 1e-106.
Your job is to test and extend these within your lens, citing exact file + line numbers or equation numbers for every claim. Report honestly; if something does not match, say exactly what and where. Keep claims exact and scoped.`

const LENSES = [
  { key: 'objects', prompt: `LENS objects-and-conventions. Check, with line-level citations in both sources: (1) FIELD: is the chain's K_L (product of normalized 1D periodized kernels k_L, LS-DER-041/043/061/062) exactly PROOF.md's K_24 (3D image sum normalized by sum exp(-|24n|^2/2)), variance one, same scale exp(-|x|^2/2)? (2) COUNTED OBJECT: is the chain's nu_(3,L)(ell) (selected, finite nonessential superlevel H0 bars, first-moment short-lifetime density, 'specific' = per unit volume?) the same object as P's coefficient's density (per-unit-volume, finite superlevel H0 bars, essential class excluded, selected full-pair count = finite-bar count)? Check units: density in ell per unit volume in both? superlevel in both? (3) CONVENTIONS: ordered maximum/saddle roles vs typed ordered pairs; full directed S^2 vs ordinary sphere area over S^(d-1); any factor 1/2, antipodal, branch, unordered-pair multipliers; whether P and the chain both integrate the contact direction over the FULL sphere with the same integrand. (4) DEFINITIONS of the integrand objects: P's V_u, A_u, D_u, tau_u, p_G, p_Vu (P Section 15) vs LS-DER-056's h_t, q_t/Q_t, D_t, sigma_t, Sigma_g, Sigma_h. Are they literally the same Gaussian objects? (5) Any place where the two sources' conventions DIFFER (e.g. sign of u_t, k vs kappa, amplitude scaling) and whether it changes the number. Output a concise table-like list: item, chain locus, landing locus, MATCH/DIFFERS/UNCLEAR, note.` },
  { key: 'factors', prompt: `LENS factor-map. Independently (do your own algebra; you may use sympy) map every multiplicative factor of the chain's absolute normalization onto P's (15.2) and PROOF.md formula (1): the cubic-gap Kac-Rice factor 1/6 (LS-DER-075 (1)-(5)), the corrected-pin density r^-(d+3) and soft-endpoint r^2 factors, the lifetime Jacobian 6^(2/3)/3 (LS-DER-075 (6)-(11)), the kappa integral (LS-DER-056 (23)), B_3 (LS-DER-056 (25)); versus P's pin Jacobian |det T_r|=12 r^-(d+3) (P (3.2)), its contact density and z_0=(6k)^2 E[...] (P Section 15), the factor 144 and phi_tau(12k), and the gamma integral. Establish the change of variables between P's scaled gap mark k and the chain's kappa (is kappa = 6k? check P's definition of k and of the saddle value / lifetime in P Sections 1-6 and 13), and confirm that the two products agree as functions of the covariance (not only at the reference law). Also check LS-DER-075 Section 4 'no double counting' against P: does P contain the factor 1/6 once, and the lifetime factor once? Re-run or extend ${S}/work/sym_finiteL.py if useful. Flag any factor that appears in one source and not the other, or any mismatch in the definition of k/kappa/lifetime ell (e.g., ell = kappa r^3/6 vs P's lifetime definition).` },
  { key: 'custody', prompt: `LENS custody-and-verdict-scope. (1) For each of LS-CLS-077, LS-CLS-074, LS-DER-056, LS-DER-075, LS-DER-076, LS-DER-062, LS-DER-063 and each CL-AUD-LSDER062/071/072/073/073V13/075/076 file: if it declares a freeze policy, BODY_BYTES and BODY_SHA256, extract the frozen body exactly per its stated freeze policy (text strictly between the BEGIN and END lines, CRLF/CR->LF, keep leading whitespace, strip trailing LFs, append exactly one LF, UTF-8 without BOM; strip a leading UTF-8 BOM of the export if the body starts the file) and compute bytes and sha256; report MATCH/MISMATCH vs declared. Note these are rendered Drive exports so mismatch may be export artefacts: report exactly. (2) For each CL-AUD verdict: what object it reviewed (id/version), its disposition string, its exact scope, any conditions, caveats, or failures, and whether it reproduced exact hashes (LS-CLS-077 Section 3 says the reviews did not reproduce frozen-body hashes because the connector exposed rendered text). (3) Does anything in the chain (later errata, LS-DER-063's 'precision-corrected' evaluation superseding LS-DER-061, CL-AUD-062 findings) leave a FAIL/open item affecting c_(3,24)? (4) Ordering/dates: chain is July 2026; P and PROOF.md are 24 Sept 2026. Does PROOF.md or P cite the chain? Does the chain cite P? (grep). Report the provenance limits plainly (authorship families as stated in the files, Anthropic verdicts on OpenAI derivations; no model names needed beyond the family).` },
  { key: 'correction', prompt: `LENS side-24-correction-and-numerics. (1) From LS-DER-059/061/062/063 and CL-AUD-LSDER062, state exactly what the chain asserts about c_(3,24)/c_(3,infinity)-1 (P_3(24) = -620813376/35, the e^-288 factor, the epsilon_24 bound) and with what status (analytic interval? exact rational? reviewed?). (2) Compare with PROOF.md (2)-(4): |c_d,24/c_d,ref - 1| < 1e-106 and the outward 20-digit enclosure in ENCLOSURE.json; is the chain's statement consistent with, stronger than, or in conflict with PROOF's? Is PROOF's c_d,ref the chain's c_(3,infinity) (same K_infinity)? (3) Numerically at >= 60 digits (mpmath), compute c_3,ref, the chain's c_(3,24) = c_inf (1 + P_3(24) e^-288) and confirm both lie in PROOF's enclosure; report how many leading digits are shared and why that alone is not source equivalence. (4) OPTIONAL but valuable if tractable within ~20 minutes: independently sanity-check the SIGN and ORDER of magnitude of the first periodization correction for d=3, L=24 (e.g., via the leading image term exp(-288) in the jets m2, m4, m6 of k_L and how c depends on them per LS-DER-055/059), WITHOUT claiming a certificate; if not tractable, say so. (5) Note LS-DER-061 vs LS-DER-063 (a 'precision-corrected' evaluation): what changed and whether the final chain value relies on the corrected one. Do NOT attempt R-CERT (directed-rounding certification of the full quadrature).` },
]

const LENS_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', description: 'MATCH, MATCH_WITH_NOTES, DIFFERS, or HOLD for this lens' },
    findings: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, claim: { type: 'string' }, chain_locus: { type: 'string' }, landing_locus: { type: 'string' },
      status: { type: 'string' }, evidence: { type: 'string' } }, required: ['id', 'claim', 'status', 'evidence'] } },
    caveats: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdict', 'findings', 'caveats'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    refuted_findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, why: { type: 'string' }, correction: { type: 'string' } }, required: ['id', 'why'] } },
    upheld_findings: { type: 'array', items: { type: 'string' } },
    missed_issues: { type: 'array', items: { type: 'string' } },
    lens_verdict_upheld: { type: 'boolean' },
    notes: { type: 'string' },
  },
  required: ['refuted_findings', 'upheld_findings', 'missed_issues', 'lens_verdict_upheld', 'notes'],
}

const results = await pipeline(
  LENSES,
  l => agent(`${CTX}\n\n${l.prompt}`, { label: `lens:${l.key}`, phase: 'Lens', schema: LENS_SCHEMA }),
  (rep, l) => parallel(['source-fidelity (re-read every cited line; is each quote/locus exact and does it say what is claimed?)',
                        'mathematics (re-derive each claimed identity or mismatch independently; try hard to find a factor, convention, or definition that does NOT match)'].map(angle => () =>
    agent(`${CTX}\n\nADVERSARIAL VERIFICATION, angle: ${angle}. Another agent's lens report for lens '${l.key}' is below as JSON. Try to REFUTE each finding; default to refuted if a claim is not supported by the exact cited text or by your own re-derivation. Also list any issue the lens missed that bears on the crosswalk question.\n\nLENS REPORT:\n${JSON.stringify(rep)}`,
      { label: `verify:${l.key}:${angle.split(' ')[0]}`, phase: 'Verify', schema: VERIFY_SCHEMA })))
    .then(vs => ({ lens: l.key, report: rep, verifications: vs })),
)
return results
