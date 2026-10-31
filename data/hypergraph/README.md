# Math artifact hypergraph — organization only, NOT A RESULT

credit 0 · OBL OPEN · no flags. This indexes existing records; it makes no correctness claim.

Built with Hypergraph-DB 0.4.0 by `tools/hypergraph/math_hypergraph.py`. Each coverage-ledger row is
**one hyperedge** whose `roles` map labels members as unit / source / review / impl / node / lane / record.

## Inputs (all in-repo)
- `math_coverage_ledger_20261023.md` — RETRO-MATH-BIND-02 ledger with the 24–25 October Main membership corrections and the 26 October Math- landing recheck (rows 318 and 345); the original 211/133 and subsequent 213/131 and 214/130 snapshots are retained in its history.
- `../../docs/site/dependency-source/GRAPH.json` — 49 graph nodes (#527).
- `graph_binding_report_20261023.json` — #527 binding report snapshot (21 bound nodes).

## Outputs
`math_hypergraph.json` (vertices + hyperedges), `query_results.json` (gaps, reach example, shared impls),
`bindings_hypergraph.png` (CI artifact only: the hyperedges that touch a graph node).

## Rebuild / query
Locally (from the repo root, Python 3.11/3.12):
```
pip install -r tools/hypergraph/requirements.txt
python -m tools.hypergraph.math_hypergraph build            # regenerate these JSON files (main only)
python -m tools.hypergraph.math_hypergraph build --check    # CI gate: exit 1 if stale
python -m tools.hypergraph.math_hypergraph reach main:node:math.d1-component.congruence-erratum 2
python -m tools.hypergraph.math_hypergraph gaps
python -m tools.hypergraph.math_hypergraph validate
python -m tools.hypergraph.math_hypergraph shared-impl 5
python -m tools.hypergraph.math_hypergraph export /tmp/hg-bundle --repos all --png
pytest tools/hypergraph
```
On GitHub: `.github/workflows/math-hypergraph.yml` runs the same steps on PRs/pushes touching these inputs or via
workflow_dispatch, with read-only checkouts of public Math- and query-, and uploads the all-repo bundle + PNG as the
`math-hypergraph` artifact. Vertex keys are repo-prefixed (`main:`, `Math-:`, `query-:`). See `tools/hypergraph/README.md`.
The PNG is a build artifact (optional, `--png`), not committed.

## Two explicit source paths (2026-10-30)

Rows 67 and 258 now spell out their repository-root source paths. Their former
unit-local `reviews/REVIEW_A_V2.md` and
`companions/COLLISION_TRANSFER_THEOREM.md` spellings collided with configured
repository-root prefixes, producing two incorrect derived source vertices.
The [bounded repair record](../../docs/superpowers/plans/2026-10-29-source-path-repair.md)
pins the actual Math `9f8dbb4d` blobs and SHA-256 identities. Neither former
repository-root path exists at that cut; both intended unit-local sources do.

Only those two source-member identities change. All 344 other rows, every
non-source cell, existing implementation memberships, graph nodes, bounds and
scientific qualifications remain. Counts stay 346 units / 1437 vertices /
346 edges, source 220/126, implemented 298/48, placeholder 30 and bound 14.
This is a path correction, not new proof coverage, implementation applicability
or scientific acceptance. The builder, configuration, tests and gates are
unchanged. Historical snapshots below retain their original source cuts.

## Determinant post-511 membership supplement (2026-10-28)

346 ledger rows /1437 vertices /346 ordered hyperedges. Existing row345
gains17 source memberships (ten added Lean modules/seven added SOURCE contracts)
and ten declared Lean implementation memberships. All old members remain.
Source220/126, implemented298/48, placeholder30 and bound14 are unchanged.
Earlier sections retain their frozen snapshot counts.

Frozen source: Math `4a0d51bbd06f0d9ac731a1d1df38f25999797135`,
container `frontiers/determinant_moment_20261025`, current folder tree
`ac09223bce4bd307ad9ccc8b781526c5a26d1a09` versus original#511
`8a6d93e67be8b233a4f35251188112ab3f44014a`. The [plan's exact source table](../../docs/superpowers/plans/2026-10-28-determinant-membership.md#new-memberships)
pins each new path, bytes and Git blob. All17 are new paths in that tree delta
and explicitly listed by manifest.source_modules/additional_source_contracts.
No new container or theorem unit is created.

Current modified README/SCOPE/root-import/manifest and unchanged SOURCE_MAP were
read as context, not added as new members. The current root770dec72 differs
from olddfe972cb; its changed closure is not qualified by the old#511receipt.
The three old proof modules retain their blobs. Historical text/reviews are
preserved, including their old pending-review wording. [AnthropicClaude01Cz's
actual scoped review6403917675](https://github.com/d6g8k5htny-coder/Math-/pull/512#issuecomment-6403917675)
binds the ten new modules at16f81b9, with no reviewer Lean/gate replay and all
N1–N4 qualifications intact. This inventory neither repeats nor broadens it.

The existing row346INPUT_BINDING's11determinant pins at16f81b9 include one
already-indexed DeterminantMoment and the ten additions. Its23-input custody
scope is reused as declared provenance, without new logical GRAPH edges or
proof applicability. Copied inputs/manifests/catalogs/evidence/tooling, old
SOURCE_MAP membership and later active#514 are excluded from this bounded delta.

All other345rawrows and every non-membership cell in345 remain exact; all1410
old vertices and346edge identities/non-role data remain. Only27 typed file
vertices and corresponding345source/implementation roles are new; five numeric
query statistics change, with gap/reach/shared/predicate results unchanged.
No source program, Lean/kernel, gate/checker/replay or CI request ran.
Scientific effectNONE; complete corpus/source applicability remainsOPEN.

## Landed source-derivative container (2026-10-28)

346ledger rows /1410vertices /346ordered hyperedges; source220/126,
implemented298/48, placeholder-only30 and bound14. The earlier snapshot counts
below retain their actual source cuts. New row346 records one landed Math-
container, `frontiers/source_derivative_gaussian_20261025`, at exact Math
`4a0d51bbd06f0d9ac731a1d1df38f25999797135`, folder tree
`aa776f39937e40b7aff96471aa097e7a566d8a9a`. It does not create18theorem units.

Bounded membership is39top-level mathematical sources:18Lean modules,
18`SOURCE_*.md` contracts and README/SCOPE/FIELD_CONDITIONING; the18Lean
files also have implementation roles. This deliberately excludes copied
inputs, manifests/catalogs, raw evidence/tooling and active unmerged#514.
The [plan's complete source table](../../docs/superpowers/plans/2026-10-28-source-derivative-membership.md#frozen-source-identities)
pins every included Git blob/byte identity. All39bodies were read inertly and
rehashed; author-side readers inspected Lean text, without executing it or
claiming mathematical proof/source applicability review.

Math-#513's [actual landed release](https://github.com/d6g8k5htny-coder/Math-/pull/513#issuecomment-6408484663)
binds merge180d6406 and the same18-module folder. Its four-module
[AnthropicClaude01Cz scoped review](https://github.com/d6g8k5htny-coder/Math-/pull/513#issuecomment-6405469445)
stays bound to37f8f01d; #512's fourteen-module reviews keep their original
cuts. Source weighting is not a geometric Palm identification, and the
derivative-tail object is the stated Fourier majorant. The source README's
old pending-review text is preserved as historical source; current native
review chronology is attributed separately, with no scientific-status effect.

INPUT_BINDING retains23declared dependencies:12smooth-source files at
e75cb81e toward row118 and11determinant files at16f81b9 toward row345
under receipt95e50247. No logical GRAPH edge or dependency applicability is
adjudicated here. The historical512/513-open-at-cut paragraph remains exact;
row345's later512expansion remains separately unaccounted. No Drive match
was performed for the new row.

The frozen comparison parent remains released #676 source1ab5b6ab/treeffbb;
all345parent rows,1350vertices and345orderededges stay exact. #676 actually
landed28October11:43:50UTC as Main `b95366c6651cb9c048a885cff61de8836da3c5b2`
and its branch was deleted. This initial source-only Draft targets Main, retaining
source parent1ab without author rebase or current-Main composition. Do not merge
into or recreate #676's branch. A separate eligible integrator must qualify
actual-current-Main composition and all current gates.
No mathematical program/checker/gate/Lean/kernel/replay/CI request ran.
Source/implementation membership supplies no theorem or parent acceptance;
scientific effectNONE and broader corpus accountingOPEN.

## Declared consumer memberships (2026-10-28)

345 ledger rows / 1350 vertices / 345 hyperedges. Source 219/126,
implementation-container 297/48, 30 placeholder-only and 14 bound counts are unchanged.
This additive supplement combines Capability Frontier's [rows 57/62 proposal](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6467446841)
with source-inventory's row 311 theta consumer under the [accepted single-writer handoff](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6467485158).
Inputs are frozen at Main `f20a971dbbdcca8e0b653e701a2877347daf1ad4`; prior
#659/#673 are already landed. All earlier snapshot counts below are historical.

### Declared direct replay-wrapper memberships (rows 57 and 62)

At Math- `4a0d51bbd06f0d9ac731a1d1df38f25999797135`, row 57's E3 packet declares its direct wrapper in `.github/workflows/elder-third-order.yml` (blob `80c5e010dd0df3b268628f6bb3f3dabe32dfe2a6`), while row 62's far-elder packet declares `.github/workflows/far-elder-rate.yml` (blob `3046c316c520a354b9ddb9f562855dadee5c46b6`). The unchanged checker identities are respectively `b5f04a87f2d2984280bf394bd4e5f3db560e1b33` and `58ff10f85cb89566433500855f385bc87f87e916`.

These are additive metadata memberships. Keep row 57's existing C2 workflow and row 62's existing E3 workflow: they have literal source-trigger/dependency roles. The [role census](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6425587613) distinguishes those roles from the declared direct replay routes. No workflow/checker execution, proof applicability, scientific acceptance or stronger implementation coverage follows from either co-membership. Original frozen-cut qualifications and every source/status/bound cell remain unchanged.

### Row 311 theta numerical companion

Existing row 311 additionally names `experiments/zeta_fold_exploration/theta/project_theta_checks.py`,
explicitly linked beside `dbn/lobe_identity.py` by [zeta README §4 item 4](https://github.com/d6g8k5htny-coder/main/blob/f20a971dbbdcca8e0b653e701a2877347daf1ad4/experiments/zeta_fold_exploration/README.md#L571).
README is 111858 B, blob `cf3cf4fd164216e2cad9418b2350d23ce1468776`, SHA256
`2e08a7d91dcb1c7f17e593a953046f3bf3eaf15ef9fbe96284c53f945d264be2`.
The program is 2674 B, blob `08dee18d35ff93ed9f77a539e55f2739bad4fcd6`, SHA256
`ce8eb3c6525bb34bf4ecb9b76cdf0055779c28ee423dedb3ccec2d9118ce36db`.

The script declares normalizer/covariance/normalization sanity checks and prints
the two lobe integrals at t=0 between the first two scaled zeta zeros. Its finite
sums, finite-u quadrature and rounded printouts contain no equality assertion or
certified error budget. This membership records that declared finite numerical
control, not a replay or proof of the general identity. No direct Bridge/C46/C47
edge is inferred. The unchanged §4 declaration and program also match pending
#482's head; its separate manifest-check repair is not adopted here.

All 342 other raw rows, all 1348 prior vertices and 345 ordered edge identities remain.
Only rows 57, 62 and 311 implementation memberships expand; no source, node, lane,
bound, status or Drive cell changes. Aggregate co-membership does not assert
that every program implements every source in a row. No project checker,
workflow, zeta program, replay or Lean ran. Scientific effect NONE; broader
corpus accounting remains OPEN.

## Row 311 lobe numerical-consumer membership (2026-10-26)

345 ledger rows / 1348 vertices / 345 hyperedges. Sourced containers remain
219 (126 missing); implemented 297 (48 missing), including 30 placeholder-only;
bound units 14. This child proposal is stacked on frozen #659 head
`b5931e6b30b220c8263f27bc891f178e7928cf57`; it does not expand that candidate.

Existing row 311 gains only `experiments/zeta_fold_exploration/dbn/lobe_identity.py`,
explicitly named by [zeta README §4 item 4](https://github.com/d6g8k5htny-coder/main/blob/b5931e6b30b220c8263f27bc891f178e7928cf57/experiments/zeta_fold_exploration/README.md#L571).
The README is 111858 bytes, blob `cf3cf4fd164216e2cad9418b2350d23ce1468776`,
SHA-256 `2e08a7d91dcb1c7f17e593a953046f3bf3eaf15ef9fbe96284c53f945d264be2`;
the program is 869 bytes, blob `023541dddbcd224d38df12fa2031487194cfa1c9`,
SHA-256 `3ae66f0ed35026f0d74561c98df360b90118a0163daa430cac69a095e48799d5`.
Both identities are unchanged at Main `336b6f8927f37e2a8d727c6bea845f874c52d8f5`.

This is a declared numerical check at t=0 between the first two scaled zeta
zeros, using `indep_ht.py`. It prints two rounded values without an equality
assertion or outward/tail error certificate. The existing
[source-bound evidence](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6425650646)
preserves #652's distinct frozen source, driver declaration and historical
job/output identities. No replay artifact is added as a theorem or blanket
implementation member, and no receipt is rebound to the whole current README.

All 1347 prior vertices and 345 ordered edges remain; only row 311 gains this
implementation membership. The other 344 raw rows and all source, node, lane,
bound, status and Drive cells are unchanged. No direct Bridge/C46/C47-to-program
edge is inferred; aggregate co-membership does not mean every program proves
every source statement. The theta companion and wider zeta inventory remain
separately scoped. No zeta program, mathematical replay or Lean ran, and no
general identity, persistence exponent, RH result or scientific acceptance is
established by this addition.

## Gap counts after canonical primary-source membership (2026-10-26)
345 ledger rows / 1347 vertices / 345 hyperedges. Sourced containers 219
(126 missing); implemented 297 (48 missing), including 30 placeholder-only.
Bound units 14; graph-node counts and qualifications are unchanged.

Ten omitted primary source paths join eight existing Math- containers. Only
rows 124, 143 and 181 gain source presence. All prior members and implementation,
node, bound, status and Drive cells remain, including the rows-318/345 landing
recheck. This reuses #638's source-document evidence for those three rows;
it does not adopt the full 83-row classification or claim its review complete.
The [implementation/evidence plan](../../docs/superpowers/plans/2026-10-26-canonical-source-membership.md)
retains exact source pins, two historical commit aliases and bounded consumer
roles. The tail README's LOW-P15-T-SOURCE indirection remains. No mathematical
program or Lean ran, and source presence is not proof-to-code verification.

## Gap counts after the Math- landing recheck (2026-10-26)
345 ledger rows / 1337 vertices / 345 hyperedges. Sourced containers 216 (129 missing). Implemented 297 (48 missing),
of which 30 are placeholder-only. Bound units 14. 28 of 49 graph nodes unbound per #527; 13 nodes touched by no unit.
Role problems 0. Row 318's lane now points at its Math-#510 landing (folder tree `a7ef0696`, not #477's head; its implementation and
status cells are unchanged by the ledger owner's ruling) and new row 345 records `frontiers/determinant_moment_20261025`
from Math-#511 (folder tree `8a6d93e6`); node, bound and Drive cells are unchanged.
Details are in the ledger's 26 October section. Open Math-#512/#513 stay `PENDING_OPEN_PR` with no row.

## Gap counts after the periodic-H0 consumer completion (2026-10-25)
344 ledger rows / 1325 vertices / 344 hyperedges. Sourced containers 215 (129 missing). Implemented 296 (48 missing),
of which 30 are placeholder-only ("in PR head" / "per census", no artifact). Bound units 14.
28 of 49 graph nodes unbound per #527; 13 nodes touched by no unit. Role problems 0.
Rows 307 and 310 now include the 56 tracked Markdown members under the existing
periodic-H0 and universality containers at Main
`32366ac0d4e2d1464eaac51866b9dce2128970b9`. These are proofs, reviews, notes,
protocols and historical reports, not 56 new theorem units or verified
proof-to-code relationships. That supplement had 1290 vertices and 213 sourced /
131 missing containers. The original 2026-10-23 graph had 1234 vertices and
211 sourced / 133 missing containers.

The later row-312 correction consumes the existing `formal/manifest.json`
source-to-target map at Main `9c03589b1a3f0c1fd6befd09abe6b95c99cff7d7`.
It adds the two pinned SIDE24 source copies and the omitted `Ledger.lean`
implementation beside the existing ConeMoments/Endpoints modules. The map
declares 27 proof-source arithmetic targets plus four published-number
endpoint targets; these do not establish the coefficient enclosure or parent
theorem. Statement alignment remains PENDING_INDEPENDENT_REVIEW; no Lean or
formal gate ran. Row 313 is import-only and retains MISSING source. That
correction produced 1293 vertices and 214/130 sourced/missing containers while
preserving all row identities and the other 343 raw rows.

Row 311's later zeta correction consumes committed README/CLAIMS and the
existing MANIFEST at Main `6c6c6b0ff649d5ac0413344c00b6cdbe95344f82`.
README's Proposition 1 is the primary proved-note carrier; CLAIMS is an audit.
The explicitly linked `test_exact.py` is a bounded exact algebraic-control
consumer beside the two existing numerical collision programs. This is not
verification of the full analytic proposition, every 46-script package claim,
a limiting exponent or RH progress. It reuses [Main548's](https://github.com/d6g8k5htny-coder/main/pull/548)
historical GB-8n proposal/pins and ENGINEERING_CONTROL/unbound meaning; no new
node or theorem unit is introduced. All original vertices and memberships,
343 other raw rows and all node/bound/status cells remain unchanged. Only row
311's source/implementation membership changes; no zeta program was executed.

Research-translation's later bounded membership addition reuses the existing
source manifests at Main `e016d4a445bf6ece2724f82dcf843040afc2157c`; its bytes
are unchanged in candidate base `af1845fc4f5f8b5eb7cabbdd1f624d365025fb60`.
Row 307 gains the EXPERIMENT protocol and MANUSCRIPT M4–M5 source interface,
plus the declared discrete pilot model/runner. The pilot stays inconclusive.
The spectral proof's historical manuscript90936297 pin is preserved; only its
unchanged 1506-byte consumed M4–M5 slice is reused, not a new parent proof.
Row 311 gains PERSISTENCE_BRIDGE only for §2.1's integration identity/kernel,
through the zeta note's explicit citation and existing finite exact controls.
These are scoped source/protocol/consumer relationships, not full manuscript,
C46/C47, bridge, analytic-zeta or continuum verification. Aggregate membership
does not assert that every program proves every co-member statement.

That addition preserves all 1296 prior vertices/memberships and all 344
ordered edges, changing only 307/311 memberships (342 other raw rows unchanged).
Five added vertices yield 1301; sourced/implemented container counts, all
node/bound/status cells and all original review/scientific qualifications
remain. No source manifest, historical pin, proof, program or production
builder changes. Full C46/C47 computational consumers remain unidentified in
the bounded search, alongside wider Main/history/cloud accounting gaps.

The periodic-H0 declared-consumer completion adds 24 source-named program
members to existing row 307, preserving its four previous programs. The
[49-artifact source/consumer map](periodic_h0_consumer_map_20261025.json)
records the 28 source carriers, 27 modules plus one named test, 21 additional
suite-selected support tests, 11 historical receipts and output roles. It is
an exact-source reference; the builder still consumes only its listed inputs.
Renderers and certificate callers keep their presentation/operational roles,
not theorem-proof labels. Other support tests are accounted without blanket
implementation membership. No source program or mathematical test was run.

All 1301 old vertices/memberships, all 344 ordered edges/row identities and
343 other raw rows remain; only row 307's implementation cell expands.
The graph now has 1325 vertices, while all container, node, bound and status
counts remain. Spectral historical pins and Gaussian-tail finite-domain versus
unbounded-asymptotic limits persist. The pilot32 receipt's older 11712-byte
runner differs from the current 12130-byte runner; the map preserves both
identities, and ten other receipt source inventories match current bytes.
No historical receipt is rebound, IID/coupling law authenticated, or full NUM
or continuum result verified by this membership correction.

This is a bounded membership correction, not complete Main/history/cloud
coverage. The exact-sample shrinking-bin comment sources have separately
recorded [archive custody in Main591](https://github.com/d6g8k5htny-coder/main/pull/591#issuecomment-6398496409);
this index change supplies no mathematical or execution qualification for
them. Separate storage crosswalks retain their own coverage/privacy limits;
this supplement ingests no Dropbox-only source bytes. No scientific acceptance
or original review outcome changes.
