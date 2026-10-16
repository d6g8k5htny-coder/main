export const meta = {
  name: 'historical-math-recovery',
  description: 'Forensic search of all historical sources for recoveries relevant to the 155 current gaps; adversarial verification of every strong candidate; synthesis of recovery matrix and assessment',
  phases: [
    { title: 'Search', detail: '15 read-only search agents (12 gap groups, artifact-ID leads, live Drive, live Dropbox)' },
    { title: 'Verify', detail: 'adversarial verifier per strong candidate: bytes, quote, chronology, contradictions' },
    { title: 'Synthesize', detail: 'recovery matrix, top candidates, dependency graph, assessment' },
  ],
}

const SP = '/tmp/claude-0/-home-user-main/790ebd9d-259f-58e4-b82b-54c1c6b107a2/scratchpad'
const CS = `python3 -B ${SP}/corpus/cs.py`

const RULES = `You are a HISTORICAL MATHEMATICAL RECOVERY agent for Dylan Roy's Universal Law research program. SCIENTIFIC EFFECT: NONE. You are READ-ONLY: do not push, commit, comment, edit, merge, trash or move anything anywhere (GitHub, Drive, Dropbox).

TASK: recover mathematical work that ALREADY exists in the historical record for the current gaps assigned to you. Do NOT invent mathematics. Do NOT derive a missing lemma and call it recovered. Do NOT infer a proof from its conclusion. Agreement between AI agents is not evidence of correctness. If fragments would allow reconstruction, label it "RECONSTRUCTED CANDIDATE — NOT ORIGINAL CARRIER" and never call it recovered.

CURRENT GAP LIST: ${SP}/recovery/gaps.json (155 rows G01..G155 with name, aliases, statement, current_status, status_sources, cited_artifacts, missing_carriers). Read the rows for YOUR gap ids first.

SOURCES and TOOLS:
1. CORPUS (primary): every blob from every ref (all branches, PR heads, closed PRs, unreachable objects) of main, Math-, trial, governance-, meta-framework, sandbox, query-, google-drive, Universal-Law-Workspace; the hardening branch's Drive mirrors (drive/mirrors, *.export.txt) and the content-addressed Drive custody store DG-MIGRATION-20260919 (3,618 Drive objects, docx/xlsx/pdf/zip members extracted); Dropbox texts recovered on 2026-09-27. Search it with:
   ${CS} -e 'REGEX' [-e 'REGEX2'] [-i] [--max-files 60] [--lines 6] [--src git|custody|dropbox|pdf]
   ${CS} --show <textfile> --range 1:200      (read a hit; prints provenance)
   ${CS} --prov <textfile>
   Provenance: [git:<repo>] id=<git blob sha1>; [custody:...] id=<sha256> names=<drive id; status; modified time>; [dropbox] id=<dropbox id>. Date a git blob: git -C <repo> log --all --format='%h %cI %an | %s' --find-object=<blob sha1> (repos: main=/home/user/main, others ${SP}/repos/<name>). Show a file at a ref: git -C <repo> show <commit>:<path>. Find where a path lived: git -C <repo> log --all --format='%h %cI %s' -- '<path glob>'.
   The corpus is large (8.6 GB): use specific regexes, --max-files, and --src to keep output small.
2. Hardening research branch checkout: ${SP}/wt/hardening (drive/source_map, drive/inventory.jsonl map Drive ids to titles/paths; registers/, docs/math_status/, claims/).
3. GitHub discussions (issues/PR comments, incl. closed): GitHub MCP tools via ToolSearch "select:mcp__github__issue_read,mcp__github__search_issues,mcp__github__pull_request_read,mcp__github__search_pull_requests". Owner d6g8k5htny-coder.
4. Google Drive (live): ToolSearch "select:mcp__Google_Drive__search_files,mcp__Google_Drive__read_file_content,mcp__Google_Drive__get_file_metadata" — only if your scope says so or a lead is Drive-only.
5. Dropbox (live): ToolSearch "select:mcp__Dropbox__search,mcp__Dropbox__fetch,mcp__Dropbox__get_file_metadata" — only if your scope says so.
Search SEMANTICALLY: older work uses other names (e.g. "inner chart", "near-axis region", "microdisk", "singular face", "local witness neighborhood", "inner window" for today's D5 pin neighborhood; "elder rule", "merging saddle", "pair Palm", "critical pair", "contact kernel", "Hessian envelope", "Gram determinant", "divided difference", "q(r,b)", "q(r,6/5)", "ell^-1/3", "24 jet", "Hermite remainder", "GOE cone", "first variation", "endpoint Hessian", "transverse face", "axial face", "Window Lemma", "Reduction Lemma", "triple witness", "self attachment"). Search equations, constants, characteristic phrases, artifact IDs, hash prefixes, theorem names.

CHRONOLOGY / CONTRADICTION: for every candidate, look for later AMEND, REJECT, HOLD, SUPERSEDED, COUNTEREXAMPLE, CORRECTION, ERRATUM, INVALID, QUARANTINE, WITHDRAWAL, NONCONTROLLING. The latest valid successor is the relevant object.

CLASSIFY each assigned gap as exactly one of:
A GENUINELY OPEN MATHEMATICS; B SOLVED PREVIOUSLY BUT ORIGINAL CARRIER NOT YET RECOVERED; C PARTIALLY SOLVED PREVIOUSLY; D SUPERSEDED BY A LATER ARGUMENT; E REFUTED/INVALIDATED; F DUPLICATED UNDER ANOTHER NAME/ID; G PRESENT IN GITHUB BUT NOT CORRECTLY CROSS-LINKED; H PRESENT IN DRIVE/DROPBOX/OLD THREADS BUT NOT GITHUB; I REFERENCED BY HISTORICAL CONTEXT BUT SOURCE STILL UNLOCATED; J RECONSTRUCTABLE FROM FRAGMENTS BUT NOT RECOVERED AS AN ORIGINAL CARRIER. Never treat J as B. "B" requires evidence that a complete solution existed (e.g. a review/ledger stating a proof carrier with identity existed) while the carrier is missing. "G"/"H" require that you actually located the carrier bytes (give the corpus id). If nothing relevant is found after a real search, say A (or I if history references a missing source) and list what you searched.

CONFIDENCE for each fragment: EXACT CARRIER | BYTE-IDENTICAL COPY | VERIFIED TRANSCRIPTION | STRONG HISTORICAL REFERENCE | PARTIAL FRAGMENT | RECONSTRUCTION CANDIDATE | WEAK MEMORY / UNSOURCED REFERENCE.
PRIORITY: P0 carrier could resolve a theorem-sized blocker; P1 substantially narrows a blocker; P2 missing review/provenance/supporting derivation; P3 duplicate/organizational.

Quote claims EXACTLY (<= 400 chars) and only from bytes you actually read. Every fragment must carry a locator a verifier can re-open (corpus text file path + provenance id, or commit:path, or issue/comment id, or Drive id). Keep the output compact but complete.`

const FRAG = {
  type: 'object',
  properties: {
    recovery_id: { type: 'string' }, current_gap: { type: 'string' }, historical_source: { type: 'string' },
    source_type: { type: 'string' }, date: { type: 'string' }, model_author: { type: 'string' },
    original_artifact_id: { type: 'string' }, original_location: { type: 'string' }, current_location: { type: 'string' },
    locator: { type: 'string', description: 'corpus text file + provenance id, or commit:path, or issue/comment id, or Drive id' },
    sha256: { type: 'string' }, exact_quoted_claim: { type: 'string' }, mathematical_content: { type: 'string' },
    dependencies: { type: 'string' }, review_status_at_time: { type: 'string' }, later_supersession: { type: 'string' },
    later_invalidation: { type: 'string' }, relation_to_current_gap: { type: 'string' },
    confidence: { type: 'string', enum: ['EXACT CARRIER', 'BYTE-IDENTICAL COPY', 'VERIFIED TRANSCRIPTION', 'STRONG HISTORICAL REFERENCE', 'PARTIAL FRAGMENT', 'RECONSTRUCTION CANDIDATE', 'WEAK MEMORY / UNSOURCED REFERENCE'] },
    recommended_action: { type: 'string' },
  },
  required: ['recovery_id', 'current_gap', 'historical_source', 'source_type', 'locator', 'exact_quoted_claim', 'mathematical_content', 'relation_to_current_gap', 'confidence', 'recommended_action'],
}
const SEARCH_SCHEMA = {
  type: 'object',
  properties: {
    gap_assessments: { type: 'array', items: { type: 'object', properties: {
      gap: { type: 'string' },
      classification: { type: 'string', enum: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] },
      priority: { type: 'string', enum: ['P0', 'P1', 'P2', 'P3'] },
      best_evidence: { type: 'string' }, historical_names: { type: 'array', items: { type: 'string' } },
      missing_piece: { type: 'string' }, recommended_next_action: { type: 'string' },
      searched: { type: 'string', description: 'queries/sources actually searched' },
      fragment_ids: { type: 'array', items: { type: 'string' } },
    }, required: ['gap', 'classification', 'priority', 'best_evidence', 'missing_piece', 'recommended_next_action', 'searched'] } },
    fragments: { type: 'array', items: FRAG },
    crosswalk_notes: { type: 'string', description: 'name changes / artifact lineage discovered (old name -> new name -> current object)' },
  },
  required: ['gap_assessments', 'fragments'],
}

const GROUPS = [
  { key: 'd0-carriers', gaps: 'G12 G13 G14 G05 G16 G17 G34 G38', focus: 'Exact-carrier hunt for the named missing D0 files: rnu_env.py, rnu_white.py, rnu_t4.py, rnu_spine.py, allcell_fdz_enclosures.json, CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip, CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip, eval_F / StationBox / F-map code, SIDE24 RN prep triad objects, R2 nine expected-byte targets. Search by filename AND by content signatures (Python def names, constants, JSON keys, comments these files would contain as cited by the current ledgers). Check zip members inside every archive in the corpus (member paths appear after "!" in names). Check whether any candidate bytes match cited sha256 prefixes.' },
  { key: 'd0-math', gaps: 'G01 G02 G03 G04 G06 G07 G08 G09 G10 G11 G15 G18 G19 G20 G21 G27 G28 G29 G30 G31 G32 G33', focus: 'D0 RN walls mathematics: RN-UNIF Piece 1/Piece 2 (annulus Riemann sum 0.1<=|y|<=5), ENV-RESCOV, LOGQ-TAIL, SYM-Fw jet, CH-LIFT / certified C_H, chi^2 floor, C_comp, env_form q=2..4, OBL-H5-JETMOD 24-jet band, H5 rungs/ZBAND/REMOTE-THRESHOLD, RN3 RN-JOINT-001, CL-RNU-001/003, RN5 near-moment. Look for historical proofs/certificates/derivations (not just ledgers restating status) and for whether an old claim was later refuted or superseded.' },
  { key: 'd1-parent', gaps: 'G39 G40 G42 G43 G44 G45 G22 G23 G96 G97', focus: 'D1 lifetime parent Theorem A Sections 2-7 (A1-A7 quantitative selection chain), A3 congruence erratum / type-indicator convergence, marked-cylinder cap / G_r => elder pairing, s7 r^5 skeleton, Theorem C ss11-12, Section 9 Borel elder-mark weighted Kac-Rice repair C1-C6, marked Kac-Rice / global elder / regression interfaces, Theorem D1 v2.2/v2.3, Q0-C101 qualitative 0<=1-q(r,6/5)<=C r^3, Q0-C103 typed pair-Palm cubic rate. Search older names: elder rule, elder selection, merging saddle, maximum-saddle pair, pair Palm, selection constant, "constant is one", 1 - q(r,b) = O(r^3), q_MS, matrix cap, regression, congruence.' },
  { key: 'd1-normalizer', gaps: 'G41 G98 G99 G100 G31 G139 G123 G118', focus: 'Normalizer floors and selection premises: Z_r >= z_* r^2 (compact marks), PREMISE-Z-LOWER on 0<r<=0.025 (b=6/5), PREMISE-BRANCH-MASS, B_miss cap {D(M)!=S} cubic envelope, H3 normalizer floor all-r/all-angle, V3.3 Palm normalizer c r^2<=Z_r<=C r^2 (G.3.1), P02-LM-006 exact typed normalizer, shared measurable selected branch / Borel adjacency event.' },
  { key: 'd5-inner', gaps: 'G67 G68 G69 G70 G71 G72 G73', focus: 'D5 pin neighborhoods, outer cone/steep strips three-determinant Hessian envelope and L^p assembly, inner microdisk and nested r^2 disk majorant, microdisk conditioned determinant repulsion h^2, microdisk quartic law of lambda / Q=0 blow-up, same-law transport of f_ss(S), det H_S power counting errata, collar from pin chart to reviewed annulus. Historical names: inner chart, near-axis region, microdisk, singular face, singular-near faces, local witness neighborhood, inner window, endpoint charts, axial/transverse faces, integration to O(r^3), angular-axis analysis, RP-S singular-near witness, RP-C collar, G.9/G.9.1/G.9.2.' },
  { key: 'd5-outer', gaps: 'G74 G75 G76 G77 G78 G79 G80 G81 G82 G83 G84 G85', focus: 'D5 intermediate scale r << |x| << rho, uniform growing-scaled-radius bridge, shrinking multiple-witness collision / factorial moments / triple-witness elimination / self-attachment exclusion, mesoscopic scaled annulus J_0 row list and determinant exponent ledger, thin tube R1-R8, inner axial belt, whole-annulus windowed saddle asymptotic, contact-kernel tails, angular tails/near-axis uniform integrability, fold-axis singularity, recovered TRANSVERSE_CONTACT_ASYMPTOTIC.md, PR9 chart rows. Historical names: remote witness, third witness, triple witness, annulus, collar, intermediate region, mesoscopic, contact kernel, transverse contact, first variation, Hermite remainder, midpoint secondary expansion.' },
  { key: 'side24', gaps: 'G63 G86 G134 G135 G136 G137 G138 G140 G141 G142 G143 G144 G145 G146 G147', focus: 'SIDE24 periodized Bargmann-Fock: planar BF six-pin identities transferred to SIDE24, six-linear-pin BF scalar and endpoint-Hessian covariance certificates, SIDE24 3D theorem package (T1,T2,|eps_24|<1e-180), RP-C/RP-S, RP-A/RP-L (G.7.1 Palm transfer), RP-F (G.8), pre-review dossier 12-item register, historical V3.3 eigenfloor tables (AO48-AUD-044 s.C(a)), Module I s3 extraction 9,700 B (eqs 25-32; b958c21d...), V3.4 replay archive, arithmetic_ledgers.py, three V3.4 diagnostic scripts, KIMI-AUD-006/006b raw audit carriers, SIDE24_AUDIT_EVIDENCE_2026-08-01.zip corrupt members, eigenfloor verifier raw bytes (blob 2c7fdde7), blind d^3/48 midpoint reflection vs Kimi Task-6. Exact-carrier hunt first (sizes, hashes), then mathematics.' },
  { key: 'lower-bounds', gaps: 'G46 G102 G104 G105 G106 G107 G108 G109 G110 G111 G112', focus: 'Lower bounds 1 - q(r,6/5) >= c r^3 (parent elder-defect lower bound; LPW qualitative lower bound 2D side-24; LPW explicit constant 6.238e-44; QC-RETURN03 fallback 1e-1235 r^3; K3 2D lower campaign K3-THM-001; KIMI-THM-023 c=0.9144; WP loss; P-NMZ-gamma tube clearance; W8 Lambda side / H-B3; C030/C031 base cycle set, LB-4 residue enumeration, KIMI-AUD-023) and matching 2D upper 1-q_2<=C r^3.' },
  { key: 'p01-p02', gaps: 'G113 G114 G115 G116 G117 G119 G120 G121 G122 G124 G125 G126 G127 G128 G129 G130 G131 G132 G133', focus: 'P0.1 selected-branch adjacency positivity, LS-CTL-003 independence/integrity predicates, upper-normalizer periodization C_Z=15, quantitative constants r_GR, p_B, C_D, r_star, shared Taylor transfer (Domain D + P02-LM-004), P0.2 adjacency-to-one 1-a_r=O(r^3) thirteen-interface chain, P02-LM-003..012, E1 pair-Palm to configured-process bridge, V4 typed Palm to fold-Palm, V3 outside rerun, PRCP practical-r bridge HB-024/025, law-specific q machine, GP-DATA-054. Older P0.1 material is a priority.' },
  { key: 'theorem-b', gaps: 'G47 G48 G49 G50 G51 G52 G53 G54 G55 G56 G24 G25 G26 G35 G36 G37', focus: 'Theorem B lineage: Condition (ND)/(ND\') repair and File-3 Table 4.1 (ABSENT), short-bar law nu_B(l)=C* l^-1/3 (Q0-C104), TB-G1..G7 gates, LS-DER-025/039..042 global-birth theorems and C_* normalization, T3 full lifetime law, PERC-DECAY/PD-CONN, OBL-B1-BRANCH, B4.loc/B4.rem, H5-RIM, UPPER2D STAGE_E, C026/C027.' },
  { key: 'd2-d3-d4', gaps: 'G57 G58 G59 G60 G61 G62 G64 G65 G66', focus: 'D2 lifetime successor Theorem R (c ell^-1/3 + O(1)) and its remainder constants; collision note Section A / density transfer theorem; D3 SIDE24 coefficient chain LS-DER-043..062 and analytic remainder LS-DER-064..069; D4 RN probability-to-count interface integrated N7 numerator, fixed-remote window selector crosswalk and the 24 open/partial selector x region cells.' },
  { key: 'misc', gaps: 'G87 G88 G89 G90 G91 G92 G93 G94 G95 G148 G149 G150 G151 G152 G153 G154 G155', focus: 'P15 (P15-B external review, restricted price theorem T1, P15-C 816-color certificate, P15-D/P14-E, P15-A/P14-A/B), engineering gates, formal alignment, Dropbox 26 not-imported files (audits/dropbox_import/2026-09-27/NOT_IMPORTED.csv on main), unrecoverable/candidate Drive carriers (the 15 UNRECOVERABLE + 1 CANDIDATE list), MPV3 state carriers. Lower priority: be efficient; for the Dropbox NOT_IMPORTED list and the Drive UNRECOVERABLE list, check whether any listed item is actually present in the corpus under another path (by name or sha256).' },
  { key: 'artifact-ids', gaps: 'ALL (lead-driven)', focus: 'Trace each historical lead to its definition, content, review status and successor, and map it to current gaps (by G id): GP-DER-118, GP-DER-206, GP-DATA-206, GP-DATA-212, CL-DER-204, AO48-DER-031, CL-REC-242, KIMI-AUD-005, KIMI-AUD-006, KIMI-AUD-006b, AO48-OPR-042, AO48-OPR-045, LS-CTL-003, C030, C031, G.9, G.9.1, G.9.2, endpoint-Hessian work, Window Lemma, Reduction Lemma, elder-rule selection constant, triple-witness elimination, contact covariance verification, arithmetic ledgers, GOE cone coefficient, first-variation derivations, transverse-face comparison, axial-face blow-up, midpoint secondary expansion, endpoint charts, singular-near faces, three-determinant Hessian envelope. For each lead produce at least one fragment (or state it was not found and what was searched). Use gap_assessments only for gaps where a lead changes the classification.' },
  { key: 'live-drive', gaps: 'priority gaps G39-G46 G67-G85 G134-G146 G12-G14 G05 G96-G100', focus: 'LIVE Google Drive search for material NOT in the 2026-09-19/20 hardening mirror or custody store (newer files, files outside mirrored folders, held folders): use mcp__Google_Drive__search_files with fullText/title queries for the key terms and artifact IDs (rnu_env, CL_ANTHROPIC_BUNDLE, allcell_fdz, eigenfloor, Module I, microdisk, pin neighborhood, elder, Borel, Kac-Rice, triple witness, collar, intermediate scale, GP-DER-118, GP-DER-206, CL-DER-204, AO48-DER-031, CL-REC-242, KIMI-AUD-006, G.9.1, Window Lemma, Reduction Lemma). For each promising hit, get_file_metadata (size, modifiedTime) and check whether its Drive id is already in the corpus (${CS} -F -e "<drive id>" --src custody --max-files 3); only read content (read_file_content) for items NOT already in the corpus. Record Drive id, title, size, modified time. Drive is private to the owner; never copy anything out.' },
  { key: 'live-dropbox', gaps: 'G12 G13 G14 G05 G138 G140 G141 G142 G144 G145 G151 plus any', focus: 'LIVE Dropbox search (mcp__Dropbox__search) for the missing carrier filenames (rnu_env.py, rnu_white.py, rnu_t4.py, rnu_spine.py, allcell_fdz_enclosures.json, CL_ANTHROPIC_BUNDLE, eigenfloor, V3.3, V3.4, Module I, arithmetic_ledgers, KIMI-AUD-006, SIDE24_AUDIT_EVIDENCE) and key mathematical terms (elder, microdisk, pin neighborhood, Borel, Kac-Rice, triple witness, collar). For hits, get_file_metadata (content_hash, size) and check whether already in the corpus (${CS} -F -e "<name or id>" --src dropbox). The Dropbox inventory of 5,105 files was classified on 2026-09-27: ${SP}/dbx/classified_exact.json (status GIT_EXACT/DRIVE_EXACT/.../MISSING) — use it to see whether a hit is already in git/Drive. Fetch (mcp__Dropbox__fetch) only items not already present. Dropbox is private to the owner; never copy anything out.' },
]

const VERDICT = {
  type: 'object',
  properties: {
    recovery_id: { type: 'string' },
    locator_reopened: { type: 'boolean' },
    quote_verified_verbatim: { type: 'boolean' },
    sha256_checked: { type: 'string', description: 'computed value or "n/a"' },
    confidence_after_check: { type: 'string', enum: ['EXACT CARRIER', 'BYTE-IDENTICAL COPY', 'VERIFIED TRANSCRIPTION', 'STRONG HISTORICAL REFERENCE', 'PARTIAL FRAGMENT', 'RECONSTRUCTION CANDIDATE', 'WEAK MEMORY / UNSOURCED REFERENCE', 'NOT FOUND / FABRICATED'] },
    classification_after_check: { type: 'string', enum: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] },
    later_contradictions: { type: 'string', description: 'later AMEND/REJECT/erratum/counterexample/supersession found, with locators; or "none found" + what was searched' },
    chronology: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'REFUTED'] },
    reason: { type: 'string' },
  },
  required: ['recovery_id', 'locator_reopened', 'quote_verified_verbatim', 'confidence_after_check', 'classification_after_check', 'later_contradictions', 'verdict', 'reason'],
}

const STRONG = new Set(['EXACT CARRIER', 'BYTE-IDENTICAL COPY', 'VERIFIED TRANSCRIPTION', 'STRONG HISTORICAL REFERENCE'])

phase('Search')
const results = await pipeline(
  GROUPS,
  g => agent(`${RULES}\n\nYOUR ASSIGNED GAPS: ${g.gaps}\nFOCUS: ${g.focus}`, { label: `search:${g.key}`, phase: 'Search', schema: SEARCH_SCHEMA }),
  (res, g) => {
    if (!res) return { group: g.key, error: 'search agent returned nothing' }
    const strong = (res.fragments || []).filter(f => STRONG.has(f.confidence))
    log(`${g.key}: ${res.gap_assessments.length} gaps assessed, ${res.fragments.length} fragments, ${strong.length} strong -> verify`)
    return parallel(strong.map(f => () => agent(`${RULES}\n\nYou are now an ADVERSARIAL VERIFIER. Default to DOWNGRADED/REFUTED unless the evidence survives. For the candidate below: (1) re-open its locator yourself and check the quoted claim appears VERBATIM (report quote_verified_verbatim); (2) recompute sha256 of the carrier bytes where a carrier is claimed (for git blobs: git -C <repo> cat-file blob <sha1> | sha256sum; for corpus custody ids the id IS the sha256); (3) establish the chronology (dates via git log --find-object, Drive modified time); (4) search actively for LATER AMEND/REJECT/HOLD/SUPERSEDED/COUNTEREXAMPLE/ERRATUM/WITHDRAWAL/QUARANTINE/NONCONTROLLING records on the same object or its successors; (5) challenge the confidence level and gap classification (J is never B; a reference is not a carrier; a transcription is not the original; a ledger restating status is not a proof). Report findings.\n\nCANDIDATE (JSON):\n${JSON.stringify(f)}`,
      { label: `verify:${g.key}:${(f.recovery_id || '').slice(0, 24)}`, phase: 'Verify', schema: VERDICT })))
      .then(vs => ({ group: g.key, search: res, verdicts: vs.filter(Boolean) }))
  },
)

phase('Synthesize')
const done = results.filter(Boolean)
const synthesis = await agent(`${RULES}\n\nYou are the SYNTHESIS agent. Below are all search results and verifier verdicts. Treat a fragment's confidence as the VERIFIER's confidence_after_check when a verdict exists; drop REFUTED fragments; mark unverified weaker fragments as such. Produce: (1) a RECOVERY MATRIX row for EVERY gap G01..G155 (use the search agents' gap_assessments; where several groups assessed the same gap, reconcile and say so; where a gap was not assessed, mark it NOT_ASSESSED — do not guess); (2) TOP RECOVERY CANDIDATES ranked by expected mathematical impact (gap, why it corresponds, whether the original carrier exists, whether it survived later criticism, what reintegration needs); (3) a HISTORICAL DEPENDENCY GRAPH for the main current theorems (Theorem A / D1 parent, D5 geometry, D0 RN-UNIF, SIDE24 3D package, P0.1/P0.2, lower bound) as indented text trees tagging each node recovered / reviewed / missing carrier / reconstructed candidate / genuinely open; (4) FINAL ASSESSMENT with numerator/denominator computed from the matrix: share of gaps whose classification is a recovery/custody problem (B, F, G, H, I, and C only where the missing part is custody), share genuinely unresolved (A, plus C/J where new mathematics is required), share superseded/refuted (D, E); the blocker most likely already solved somewhere; the blocker that appears genuinely open; and the five highest-value files/threads/artifacts still worth locating. Also list every fragment with confidence EXACT CARRIER / BYTE-IDENTICAL COPY / VERIFIED TRANSCRIPTION (after verification) as custody-package candidates with locator and sha256.\n\nDATA (JSON):\n${JSON.stringify(done)}`,
  { label: 'synthesis', phase: 'Synthesize', schema: {
    type: 'object',
    properties: {
      matrix: { type: 'array', items: { type: 'object', properties: {
        gap: { type: 'string' }, obligation: { type: 'string' }, classification: { type: 'string' }, priority: { type: 'string' },
        best_evidence: { type: 'string' }, current_github_location: { type: 'string' }, historical_source: { type: 'string' },
        confidence: { type: 'string' }, missing_piece: { type: 'string' }, next_action: { type: 'string' } },
        required: ['gap', 'obligation', 'classification', 'best_evidence', 'confidence', 'missing_piece', 'next_action'] } },
      top_candidates: { type: 'array', items: { type: 'object', properties: {
        rank: { type: 'integer' }, gap: { type: 'string' }, candidate: { type: 'string' }, why_corresponds: { type: 'string' },
        original_carrier_exists: { type: 'string' }, survived_criticism: { type: 'string' }, reintegration_needs: { type: 'string' } },
        required: ['rank', 'gap', 'candidate', 'why_corresponds', 'original_carrier_exists', 'survived_criticism', 'reintegration_needs'] } },
      dependency_graph: { type: 'string' },
      assessment: { type: 'object', properties: {
        denominator: { type: 'integer' }, recovery_custody_count: { type: 'integer' }, genuinely_open_count: { type: 'integer' },
        superseded_refuted_count: { type: 'integer' }, not_assessed_count: { type: 'integer' }, counting_rule: { type: 'string' },
        most_likely_already_solved: { type: 'string' }, genuinely_open_blocker: { type: 'string' },
        five_to_locate: { type: 'array', items: { type: 'string' } } },
        required: ['denominator', 'recovery_custody_count', 'genuinely_open_count', 'superseded_refuted_count', 'not_assessed_count', 'counting_rule', 'most_likely_already_solved', 'genuinely_open_blocker', 'five_to_locate'] },
      custody_candidates: { type: 'array', items: { type: 'object', properties: {
        recovery_id: { type: 'string' }, gap: { type: 'string' }, locator: { type: 'string' }, sha256: { type: 'string' }, confidence: { type: 'string' }, note: { type: 'string' } },
        required: ['recovery_id', 'gap', 'locator', 'confidence'] } },
    },
    required: ['matrix', 'top_candidates', 'dependency_graph', 'assessment', 'custody_candidates'],
  } })
return { synthesis, results: done }
