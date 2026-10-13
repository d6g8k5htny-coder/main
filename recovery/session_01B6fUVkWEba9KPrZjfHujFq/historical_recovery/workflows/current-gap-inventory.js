export const meta = {
  name: 'current-gap-inventory',
  description: 'Read-only: build the CURRENT GAP LIST of open/incomplete/absent mathematical obligations from the latest Math-, main and hardening research-branch state',
  phases: [
    { title: 'Inventory', detail: 'three readers over current state' },
    { title: 'Merge', detail: 'deduplicate into one numbered obligation list' },
  ],
}

const COMMON = `READ-ONLY historical-recovery preparation for Dylan Roy's Universal Law research program. Do NOT push, comment, edit, merge or change anything. SCIENTIFIC EFFECT: NONE.

Local sources (all read-only):
- Math- latest default branch: /home/user/math- (detached at origin/main a1fd679). Full-history mirror with all branches and PR heads: /tmp/claude-0/-home-user-main/790ebd9d-259f-58e4-b82b-54c1c6b107a2/scratchpad/repos/Math-full (use git -C <mirror> show <ref>:<path>, git log --all, git branch -r).
- main latest default branch: git -C /home/user/main show origin/main:<path> (origin/main = 2eaf083). All branches and PR heads are fetched in /home/user/main (refs/remotes/origin/*, origin/pr/*).
- Hardening research branch (the research tree with registers, claims graph and math_status): git -C /home/user/main show origin/chatgpt/drive-github-hardening-20260919:<path>; a checkout at a01c72f is at /tmp/claude-0/-home-user-main/790ebd9d-259f-58e4-b82b-54c1c6b107a2/scratchpad/wt/hardening (read files directly there; drive/ holds Drive mirrors, source maps and custody stores).
- GitHub issues/PR discussions via the GitHub MCP tools (load with ToolSearch: "select:mcp__github__issue_read,mcp__github__search_issues,mcp__github__pull_request_read,mcp__github__list_issues"). Many trackers were CLOSED at the owner's request on 2026-09-27 for list cleanup; closing did NOT discharge their obligations — include them.

GOAL: list every mathematical obligation that is currently OPEN / INCOMPLETE / ABSENT / AMEND / HOLD / BLOCKED_ABSENT / missing-carrier / unresolved, with enough identity to search history for it later. Pay special attention to: D1 lifetime parent theorem (Sections 2-7, elder-rule / selection chain, Borel elder-mark construction/repair, congruence/regression/matrix-cap interfaces, marked Kac-Rice), D2 lifetime successor, D3 coefficient, D4 RN count crosswalk, D5 (pin neighborhoods, inner microdisk, collar to reviewed annulus, intermediate scale r << |x| << rho, uniform growing-scaled-radius bridge, shrinking multiple-witness collision / factorial-moment bounds, endpoint charts, axial/transverse faces, singular-near faces, three-determinant Hessian envelopes, integration to O(r^3), angular-axis analysis), D6 P15, D0 historical RN walls (CH-LIFT / Piece-2 / 24-jet, JETMOD, certified_C_H, OBL-H5-JETMOD, rnu_env.py, CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip, allcell_fdz_enclosures.json), SIDE24 periodized Bargmann-Fock, 1 - q(r,b) = O(r^3), lower bound 1 - q(r,6/5) >= c r^3, P0.1, RP-A/RP-L/RP-F/RP-C/RP-S, G.7.1/G.8/G.9, V3.3 eigenfloor/Palm normalizer, Module I 9,700-byte extraction, and every theorem/review/certificate referenced by current files whose original carrier is missing. Record every artifact ID or hash the current files cite for each obligation (e.g. GP-DER-118, GP-DER-206, GP-DATA-206/212, CL-DER-204, AO48-DER-031, CL-REC-242, KIMI-AUD-005/006/006b, AO48-OPR-042/045, LS-CTL-003, C030/C031, AO48-AUD-044, blob/sha256 prefixes, Drive ids).

Be exhaustive within your scope but compact per row. Quote the exact status string and cite the exact file path + ref (commit or branch) or issue/comment id where it is stated.`

const OBL = {
  type: 'object',
  properties: {
    obligations: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          key: { type: 'string', description: 'short stable key, e.g. D5-PIN-MICRODISK' },
          lane: { type: 'string', description: 'D0..D7 or other lane' },
          name: { type: 'string' },
          aliases: { type: 'array', items: { type: 'string' } },
          statement: { type: 'string', description: 'what must be proved/recovered, one or two sentences' },
          current_status: { type: 'string', description: 'exact status string as quoted' },
          status_sources: { type: 'array', items: { type: 'string' }, description: 'path@ref or issue#/comment ids' },
          depends_on: { type: 'array', items: { type: 'string' } },
          blocks: { type: 'array', items: { type: 'string' } },
          cited_artifacts: { type: 'array', items: { type: 'string' }, description: 'artifact IDs, file names, blobs, sha256 prefixes, Drive ids the current files cite for this obligation' },
          missing_carriers: { type: 'array', items: { type: 'string' }, description: 'named sources the current files say are ABSENT or not recovered' },
          notes: { type: 'string' },
        },
        required: ['key', 'lane', 'name', 'statement', 'current_status', 'status_sources', 'cited_artifacts', 'missing_carriers'],
      },
    },
    scope_notes: { type: 'string' },
  },
  required: ['obligations'],
}

const SCOPES = [
  { key: 'math', text: 'Math- current default branch: PROOF_INDEX.md, claims/LANDING_CLAIMS.json, PUBLIC_READING_MAP.md, README.md, docs/, frontiers/*/ (PROOF.md, NOTE.md, OPEN/STATUS sections), reviews/*/ (review files, ledgers such as reviews/drive_hole_substitutes_20260926/LEDGER.md, reviews/contact_kernel_substitute_20260926, reviews/pin_neighborhood_recon_20260926), repairs/, imports/*/README.md, incoming/, certificates/, coefficients/. Also Math- issues #29, #39, #40, #62 (closed) and still-relevant closed PRs discussions only if the files point to them.' },
  { key: 'main', text: 'main current default branch: STATUS.md, docs/RESEARCH_INDEX.md, docs/QUEUE_RECONCILIATION*.md, docs/FORMAL_*.md, history/, reviews/, audits/ (the dropbox_import NOT_IMPORTED.csv lists project files not yet imported), and main issues #63, #67, #74, #76, #86, #94, #95, #113, #117 (all closed for cleanup) — read their bodies and last comments for the obligation state (D0-D7 in #86 especially; #117 source-chain audit for SIDE24 3D; #63/#67 D1/D2 reviews).' },
  { key: 'hardening', text: 'Hardening research branch origin/chatgpt/drive-github-hardening-20260919 (worktree at scratchpad/wt/hardening, may be slightly behind origin — check git log): claims/graph.json (or claims/*.json), docs/math_status/*.md (STATUS_RN_UNIF.md etc.), docs/OPEN_PROBLEMS.md, STATUS.md, registers/json/* and registers/source/GP-REG-032*.md (the coupled research registers: rows with OPEN/AMEND/HOLD/ABSENT), research/PINNED_SOURCES.md, drive/source_map and drive/README.md for ABSENT/HELD items, and any obligation ledgers. This is where GP-/CL-/AO48-/KIMI- artifact IDs are most likely named.' },
]

phase('Inventory')
const parts = await parallel(SCOPES.map(s => () =>
  agent(`${COMMON}\n\nYOUR SCOPE: ${s.text}`, { label: `inventory:${s.key}`, phase: 'Inventory', schema: OBL })))

phase('Merge')
const merged = await agent(`You are merging three obligation inventories for the Universal Law program into ONE deduplicated, numbered CURRENT GAP LIST. Same mathematics under different names must become one row (union the aliases, sources, cited artifacts and missing carriers). Keep genuinely distinct obligations separate. Do not invent obligations or statuses; only merge what is given. Assign each merged row an id G01, G02, ... ordered by lane (D0..D7, then others) and give it a cluster tag from: D1-PARENT, D1-ELDER-BOREL, D2-SUCCESSOR, D3-COEFF, D4-RN-COUNT, D5-GEOMETRY, D6-P15, D0-RN-WALLS, SIDE24-BF, CUSTODY-MISSING, OTHER.\n\nINPUT (JSON):\n${JSON.stringify(parts.filter(Boolean))}`,
  { label: 'merge-gaps', phase: 'Merge', schema: {
    type: 'object',
    properties: {
      gaps: { type: 'array', items: { type: 'object', properties: {
        id: { type: 'string' }, cluster: { type: 'string' }, key: { type: 'string' }, lane: { type: 'string' }, name: { type: 'string' },
        aliases: { type: 'array', items: { type: 'string' } }, statement: { type: 'string' }, current_status: { type: 'string' },
        status_sources: { type: 'array', items: { type: 'string' } }, depends_on: { type: 'array', items: { type: 'string' } },
        cited_artifacts: { type: 'array', items: { type: 'string' } }, missing_carriers: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } },
        required: ['id', 'cluster', 'key', 'name', 'statement', 'current_status', 'status_sources', 'cited_artifacts', 'missing_carriers'] } },
      merge_notes: { type: 'string' },
    }, required: ['gaps'] } })
return { merged, raw: parts }
