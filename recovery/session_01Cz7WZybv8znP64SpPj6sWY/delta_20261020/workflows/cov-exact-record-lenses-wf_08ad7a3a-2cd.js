export const meta = {
  name: 'cov-exact-record-lenses',
  description: 'Three extra lenses on the cov_exact unassessed-engines record (downstream reach, adversarial classification, archive-wide completeness), each finding adversarially verified',
  phases: [
    { title: 'Lenses', detail: 'downstream reach, adversarial classification, archive-wide completeness' },
    { title: 'Verify', detail: 'one skeptic per lens re-checks every finding' },
  ],
}

const A = args
const COMMON = `
Context. A draft data record (not yet posted) is at ${A.record}. It reports where the H2 library cov_exact.py (sha256 f08c1c5f...) is used by engines inside a reassembled archive, and classifies each interval-mode ("iv") call with a rule from the library-defect disposition. The rule is in ${A.rule}: class 1 = spectral even derivative order at argument s = 0 (moment_a(m,'spectral','iv'), k1(2m,0,'spectral','iv'), e_sum(2m,0,'iv')) - proven contained at any configuration; class 2 = upper majorants that divide a nonnegative sum plus tail by the interval normalizer's lower end and use only the upper endpoint - conservative; class 3 = every other iv value (spectral at s != 0 or odd order; k1_image; image impl) - unproven at the default configuration (dps 100, kmax 30, jmax 3) and observed to fail at (100,18), (100,20), (300,30). The library's iv branch is e_sum/s_sum in cov_exact.py: tot/Z1 + [-tau, tau] with Z1 the truncated lattice sum. cov(alpha,beta,dx,dy) = +/- k1(alpha0+beta0, dx) * k1(alpha1+beta1, dy).
Sources (read-only): extracted Python files under ${A.src} (paths like UPPER2D/H3_closure/h3_rung_floor.py are under it); the full archive ${A.zip} (1,794 members; read non-Python members with 'unzip -p ${A.zip} <member>' to stdout - never extract to disk, never execute anything from it); static scan output ${A.scan}; git clone /home/user/main with origin/main at ${A.main} (use 'git -C /home/user/main grep -n <pattern> origin/main' and 'git -C /home/user/main show origin/main:<path>'; never check out or modify). The prefix <v3> means incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/.
Rules: read only; post nothing anywhere; edit no files except your own notes under ${A.notes}/<your-lens>/; never run any engine from the archive; if you run Python of your own use 'python3 -I -B' with the script outside the extracted tree; never open any path containing 99_DO_NOT_OPEN. Every claim you return must carry file:line (or archive member + line) evidence you actually read. Say UNKNOWN rather than guess.`

const FINDINGS = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          evidence: { type: 'string' },
          affects_draft: { type: 'string', description: 'none | adds information | contradicts the draft | draft overclaims | draft underclaims' },
          proposed_wording: { type: 'string' },
        },
        required: ['claim', 'evidence', 'affects_draft', 'proposed_wording'],
      },
    },
  },
  required: ['summary', 'findings'],
}
const VERDICTS = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
          evidence: { type: 'string' },
          corrected_wording: { type: 'string' },
        },
        required: ['claim', 'verdict', 'evidence', 'corrected_wording'],
      },
    },
    missed: { type: 'string', description: 'anything important the lens missed in its own scope' },
  },
  required: ['verdicts', 'missed'],
}

const LENSES = [
  { key: 'downstream', task: `LENS downstream-reach. The draft says two engines reach a certified number through class-3 iv values used two-sided: UPPER2D/H3_closure/h3_rung_floor.py (the disposition says it feeds D3's Z_LO and D1's C_RN) and UPPER2D/H5_closure/h5_kernel.py (via its private Lattice twin). Map who consumes their outputs. (a) In the archive (all member types, incl. .md/.txt/.json transcripts and receipts): which files read, hard-code or cite the rung floor's certified numbers (its Z_0.05 lower/upper bounds, REQ_RUNG, mut_rung_floor_* transcripts, H3_RUNG_FLOOR.md) and the H5 drivers' certified outputs; give producer -> value -> consumer file:line, and whether each consumer uses the value as a lower bound, an upper bound or two-sided. (b) On origin/main: which records cite those outputs (e.g. D1 C_RN, D3 Z_LO, LPW v4 mut_rung_floor files, H5_ZBAND). Do not judge whether any theorem holds; report the dependency chain only.` },
  { key: 'classify', task: `LENS adversarial classification. Try to REFUTE the draft's classification headline. For each iv call site the draft discusses (h3_c0_certificate.py L84-86, L108-119; h3_rung_floor.py L143 and its callers L146-148; h5_kernel.py L104, L1196, L1272-1273 and every LAT.k1 / Lattice use incl. PinFrame L463-478 and KernelSeries L520-529; c2_probes.py L419; h3_band_ceil.py/h3_band_floor.py _LATT_IV reads), enumerate the derivative order and argument of every k1 factor (derive the points/separations from the source, including env-var overrides such as RUNG), and assign the class under the rule. Specifically try to find: (i) any load-bearing class-3 iv value in h3_c0_certificate.py (anything feeding CK4-CK6 / CERT_LO / CERT_HI), (ii) evidence that h3_rung_floor's or h5_kernel's certified path uses only class-1/class-2 values (which would mean the draft overstates), (iii) whether h5's Lattice twin really has the same shape as cov_exact's iv branch (compare tail formulas too), (iv) any other iv evaluator use in the 32 cov_exact-naming files (see ${A.scan}) the draft misses, e.g. direct reads of _LATT_IV or lattice().` },
  { key: 'complete', task: `LENS archive-wide completeness. The draft's scan covered only the 251 .py members (32 name cov_exact). Check the whole archive: (a) list members of every type (.py.txt, .txt, .sage, .ipynb, .jl, .m, .sh, .json, notebooks, any program text) that reference cov_exact, f08c1c5f, _LATT_IV, e_sum, s_sum, k1_image or ce.k1/ce.cov/ce.moment_a, and say whether any is program code that calls the evaluators in iv mode and is not one of the draft's files; (b) search the .py members for dynamic loading of H2 files that would evade a 'cov_exact' text search (importlib, runpy, exec(open(...)), spec_from_file_location, __import__, sys.modules tricks, path strings assembled at runtime); (c) confirm or refute the draft's counts and lists: 251 .py members, 32 naming cov_exact, BRANCH_dir/ (4 .py) and W8_lambda/ (3 .py) not naming cov_exact, the mpf-only list, the hash-pin-only list, pin_transform.pin_covariance uncalled by consumers, reg_lemmas reached only from its selftest, configure() never called with a non-default kmax/jmax, the H5 drivers setting dps 40 after import (check ordering in each h5_run*.py, and say which drivers instead run at 100 because cov_exact's import-time configure() resets dps); (d) any duplicate or variant copies (different sha256) of the named engines elsewhere in the archive.` },
]

phase('Lenses')
const results = await pipeline(
  LENSES,
  l => agent(`${COMMON}\n\n${l.task}\n\nReturn findings via the schema. Keep each claim to one sentence; put file:line in evidence.`, { label: `lens:${l.key}`, phase: 'Lenses', schema: FINDINGS }),
  (r, l) => r ? agent(`${COMMON}\n\nYou are a skeptic. Another agent (lens ${l.key}) returned the findings below. Re-check EVERY finding against the sources yourself; default to 'uncertain' if you cannot confirm it from what you read, and 'refuted' if the evidence contradicts it. Also say what the lens missed within its own scope.\n\nLENS TASK WAS:\n${l.task}\n\nFINDINGS:\n${JSON.stringify(r, null, 1)}`, { label: `verify:${l.key}`, phase: 'Verify', schema: VERDICTS }).then(v => ({ lens: l.key, found: r, verified: v })) : null,
)
return results.filter(Boolean)
