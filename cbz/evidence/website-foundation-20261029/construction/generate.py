#!/usr/bin/env python3
"""Build a namespaced, evidence-bound proposal; never mutate Git or root docs."""
from pathlib import Path
import datetime as dt,hashlib,json,shutil
ROOT=Path('/workspace/universal-law/website-repair-20261028')
OUT=ROOT/'cbz/evidence/website-foundation-20261029'
WORK=Path('/workspace/cbz0')
PREFIX=str(OUT.relative_to(ROOT))
NOW=dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,j):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
def copy(src,rel):
 src=Path(src);p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);assert sha(src)==sha(p);return {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'original_path':str(src)}
def load(p):return json.loads(Path(p).read_text())
def canon(j):return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def binding(rel):return {'path':rel,'sha256':sha(ROOT/rel)}
evidence=[]
for name in ['MAIN_PROOF_INDEX.json','QUARANTINE_BOUNDARIES.json','STATEMENT_ALIGNMENT_GAPS.json','RESULT.json','REPORT.txt','TOOL_REGISTRY.json','portable-main-receipt.json','main-git-source-bindings.json','math-resource-boundary.json','portable-tests-green-final.log','portable-tests-optimized-final.log','main-tests-with-pinned-toolchain.log','main-tests-with-pinned-toolchain-optimized.log']:
 evidence.append(copy(WORK/'lean-foundation'/name,'sources/lean-foundation/'+name))
for folder,names in {'auditor':['G0.json','G1-main.json','TOOL_REGISTRY_ENTRY.json','REPORT.md','selftest-negative-controls.json'],'ref-tool':['registry.json','REPORT.txt','mutations.json','scale.json','test.log','test-optimized.log'],'storage':['TOOL_REGISTRY_ENTRY.json','REPORT.txt']}.items():
 for name in names:
  if (WORK/folder/name).exists():evidence.append(copy(WORK/folder/name,'sources/tooling/'+folder+'/'+name))
evidence.append(copy(WORK/'MISSION.md','sources/MISSION.md'))
evidence.append(copy(WORK/'DECISIONS.jsonl','sources/prior-decisions.jsonl'))
graph_source=WORK/'repos/Math-/frontiers/downstream_gate_20260925/GRAPH.json'
graph_binding=copy(graph_source,'sources/Math-/frontiers/downstream_gate_20260925/GRAPH.json');evidence.append(graph_binding)
main=load(WORK/'lean-foundation/MAIN_PROOF_INDEX.json');groups=load(WORK/'lean-foundation/QUARANTINE_BOUNDARIES.json');gaps=load(WORK/'lean-foundation/STATEMENT_ALIGNMENT_GAPS.json');graph=load(graph_source)
dataset_path=OUT/'sources/legacy-v20/engine/dataset.json';dataset=load(dataset_path)
dataset_binding=binding(str(dataset_path.relative_to(ROOT)))
source_manifest=[]
for group in groups['entries']:
 for source in group['source_files']:
  src=WORK/'repos/Math-'/source['path'];assert sha(src)==source['sha256'],str(src)
  b=copy(src,'sources/Math-/'+source['path']);source_manifest.append(b)
write(Path('sources/math-source-manifest.json'),{'scope':'all 162 exact current Math Lean source files named by worker boundaries','repository_commit':groups['Math_current_commit'],'files':source_manifest})
records=[];categories={k:[] for k in ['closed_main_arithmetic','legacy_v20_application_quarantines','current_math_application_quarantines','math_group_quarantines','alignment_gaps']}
receipt=binding(PREFIX+'/sources/lean-foundation/portable-main-receipt.json')
for p in main['proofs']:
 ident='MAIN:'+p['declaration'];path=str(Path(p['source_path']).relative_to(ROOT));assert sha(ROOT/path)==p['source_sha256']
 r={'id':ident,'status':'CLOSED','category':'literal_main_arithmetic','source_hashes':[binding(path),binding('formal/manifest.json')],'source_record_sha256':canon(p),'dependencies':[],'exact_boundary':{'declaration':p['declaration'],'module':p['module'],'statement':p['statement'],'prop_premise_binders':p['prop_premise_binders'],'nonvacuity':p['nonvacuity']},'scientific_closure_limits':p['does_not_claim']+'; informal/source alignment remains pending. No all-Main obligation census or analytic parent closure.','axioms':p['axioms'],'kernel_evidence':[receipt],'original_record':p}
 records.append(r);categories['closed_main_arithmetic'].append(ident)
qcommon={'status':'QUARANTINED','owner':'none','dependencies':[],'isolation':'Archival contribution source/application record only. Not imported by the closed dependency-free Main arithmetic modules; no CLOSED record depends on this record. No failing test is removed or disabled.','scientific_closure_limits':'Application-level kernel closure and exact source-statement alignment are not established. Historical classifications and source mathematics are preserved; quarantine does not assert falsity.'}
obligations={x['conclusion']:x for x in dataset['obligations']}
for p in dataset['records']:
 ident='V20:'+p['id'];r=dict(qcommon,id=ident,category='legacy_v20_application',source_hashes=[dataset_binding],source_record_sha256=canon(p),exact_boundary={'record_id':p['id'],'kind':p.get('kind'),'statement':p.get('statement'),'equations':p.get('equations'),'scope':p.get('scope'),'source':p.get('source'),'proof_ref':p.get('proof_ref'),'original_kind_preserved':True},reason='No aligned full-statement kernel certificate is attached for this legacy application record. Definitions and carriers remain exact definitions/carriers; their application/identification in a theorem is unclosed. Source-custody or local checks do not discharge imports, side conditions or nonvacuity.',attempts=['Preserved all exact source-record fields and original dataset/obligation/import/provenance files.','Read original full_theorem_checker_passed/theorem_certificate_status/obligation fields; no conversion of prior review labels into kernel closure.'],original_record=p,original_obligation=obligations.get(p['id']))
 records.append(r);categories['legacy_v20_application_quarantines'].append(ident)
for ident0,p in graph['nodes'].items():
 ident='MATH_GRAPH:'+ident0;r=dict(qcommon,id=ident,category='current_math_graph_application',source_hashes=[{k:graph_binding[k] for k in ['path','sha256']}],source_record_sha256=canon(p),exact_boundary={'node_id':ident0,'kind':p.get('kind'),'scope':p.get('scope'),'source':p.get('source'),'fingerprint':p.get('fingerprint'),'notes':p.get('notes'),'classification_preserved':p.get('classification'),'full_original_record_preserved':True},reason='The graph record carries historical/source classifications, including absent carriers and definitions. No current exact-application kernel certificate aligned to this entire record and its carrier/model has been established by this contribution.',attempts=['Preserved full exact 49-node GRAPH and all edges, source/reviewer fields, fingerprints, absent-carrier labels and notes.','Compared contribution scope against 16 source-bound Math runtime/semantic groups and seven statement-alignment gaps; no current whole-Math execution claimed.'],original_record=p)
 records.append(r);categories['current_math_application_quarantines'].append(ident)
for p in groups['entries']:
 ident='MATH_GROUP:'+p['id'];hashes=[binding(PREFIX+'/sources/Math-/'+s['path']) for s in p['source_files']]
 r=dict(qcommon,id=ident,category='current_math_source_group',source_hashes=hashes,source_record_sha256=canon(p),exact_boundary={'repository_commit':p['repository_commit'],'path_boundary':p['boundary'],'source_files':p['source_files']},reason=p['reason'],attempts=['Preserved worker build/cache/source-gate attempts and resource limits in sources/lean-foundation/REPORT.txt and RESULT.json.','No completed current Math project build, full axiom census or kernel replay was observed for this exact boundary.'],original_record=p)
 records.append(r);categories['math_group_quarantines'].append(ident)
for p in gaps['entries']:
 ident='ALIGNMENT:'+p['id'];src=Path(p['source_path']);assert sha(src)==p['source_sha256'];b=copy(src,'sources/alignment/'+p['id']+src.suffix);evidence.append(b)
 r=dict(qcommon,id=ident,category='statement_alignment_gap',source_hashes=[{k:b[k] for k in ['path','sha256']}],source_record_sha256=canon(p),exact_boundary={'declarations':p['declarations'],'source_path':p['source_path'],'source_sha256':p['source_sha256'],'gap':p['gap']},reason=p['gap'],attempts=['Source-exposed exact declaration/source comparison recorded by Lean foundation worker; no new scientific acceptance.'],minimal_repair=p['minimal_repair'],original_record=p)
 records.append(r);categories['alignment_gaps'].append(ident)
assert len(records)==207 and len({r['id'] for r in records})==207
index={'schema':'cbz.proof-index.v1','generated_utc':NOW,'baseline_status':'PARTIAL_BASELINE_CONTRIBUTION','scope':'Namespaced proposal only. 31 literal Main arithmetic closures; not an exhaustive Main obligation census. All104 v20 and49 current Math GRAPH applications preserved and quarantined;16 Math groups and7 alignment gaps. Canonical queue/other prose obligations are outside this contribution.','source_commit':main['source_commit'],'latest_portable_checked_commit':main['latest_portable_checked_commit'],'source_record_digest_format':'SHA256 of UTF-8 JSON with sort_keys=True,separators=(comma,colon),ensure_ascii=False; whole original source files also copied byte-for-byte.','records':records,**categories,'source_evidence':evidence,'original_math_graph_edges':graph['edges'],'legacy_obligations_exact':dataset['obligations'],'legacy_imports_exact':dataset['imports'],'coordination':{'canonical_queue':'https://github.com/d6g8k5htny-coder/main/issues/684','canonical_root_pr':'https://github.com/d6g8k5htny-coder/main/pull/685','root_message_observation':'New canonical Main tip81acc060 and #691 prose census reported by coordinating Root; not locally reauthenticated by this builder.','all_main_obligations_complete':False,'remote_cleanup_complete':False}}
write(Path('proof-index.json'),index)
# Registry originals remain exact; normalized active command entries add only explicit scope.
tools=[]
for path,pattern,purpose in [('tools/cbz_audit.py','test_cbz_audit.py','Finite static source inventory and lexical controls; not complete Layer A semantics.'),('tools/cbz_refs.py','test_cbz_refs.py','Exact Git ref snapshot and guarded archive/prune engineering; only Root executes real remote mutations.'),('tools/cbz_storage.py','test_cbz_storage.py','Exact immutable duplicate-storage preservation; no mathematical certification.'),('tools/cbz_lean_check.py','test_cbz_lean_check.py','31 literal arithmetic declarations, all project constants, named-module fresh kernel replay and finite nonvacuity witnesses.'),('tools/formal_gate_check.py','test_formal_gate.py','Existing source/build/kernel/axiom controls; analytic alignment is separate.')]:
  testpath='tests/'+pattern
  if not (ROOT/testpath).exists() and path.endswith('formal_gate_check.py'):testpath='tests/test_formal_gate_check.py';pattern='test_formal_gate_check.py'
  tools.append({'path':path,'sha256':sha(ROOT/path),'purpose':purpose,'self_test':['python','-B','-S','-m','unittest','discover','-s','tests','-p',pattern],'source_hashes':[binding(path)]+([binding(testpath)] if (ROOT/testpath).exists() else []),'status':'OBSERVED_WORKER_EVIDENCE; JOINT_FINAL_RUN_PENDING'})
write(Path('tool-registry.json'),{'schema':'cbz.tools.v1','generated_utc':NOW,'scope':'Namespaced contribution toolkit; original worker registries copied under sources/tooling and sources/lean-foundation. Complete canonical mission registry remains joint-run work.','tools':tools,'pending_tools':['tools/cbz_baseline.py (owner registers final source/selftests after implementation)','auditor coverage companion (owner supplies exact final registry)'],'original_registry_sources':[e for e in evidence if 'REGISTRY' in e['path'] or e['path'].endswith('/registry.json')]})
marker='<!-- cbz-gates: '+json.dumps({'G'+str(i):'INCOMPLETE' for i in range(12)},separators=(',',':'))+' -->'
intro='This is a namespaced contribution proposal for the single canonical queue, not a competing root status or a complete Main obligation census. '
docs={}
docs['STATUS']=f'''# Partial Baseline — proposed status

## Baseline state

{intro} No CBZ-0 or all-gates-pass claim is made. The 29 October owner directive supersedes older operational approval/review-family rules; it does not weaken mathematical statements, bounds, kernel trust or actual platform permissions. Canonical coordination is [#684](https://github.com/d6g8k5htny-coder/main/issues/684) and [Root PR #685](https://github.com/d6g8k5htny-coder/main/pull/685). This local proposal is based on source commit `{main['source_commit']}` and portable Main receipt at `{main['latest_portable_checked_commit']}`; later canonical Main and prose-obligation work must be reconciled by that queue.

## Scientific boundaries

Only **31 literal Nat/Int arithmetic/order declarations** are CLOSED here. All 45 original project constants have empty observed transitive axiom sets. Twenty-nine theorem statements have no Prop-premise binder; the two finite-domain cases have explicit witnesses for every admitted value (`q ∈ [0..6]`, `d ∈ [2,3]`). Four auditor witness constants use only the observed standard `propext`/`Quot.sound` axioms; no custom axiom is introduced. The receipt observes pinned build, named-module fresh kernel replay and finite adversarial controls. It does not identify Gaussian moments, prove analytic coefficient enclosures, discharge a parent theorem, or establish source-statement alignment.

All **104 v20 application records**, **49 current Math GRAPH records**, **16 exact Math source groups**, and **7 alignment gaps** are QUARANTINED in [proof-index.json](proof-index.json). Definitions, carriers, absent-carrier labels, historical mathematical classifications and full original fields are preserved verbatim inside `original_record`; application quarantine does not make a definition false. Quarantines have owner `none`, exact source hashes, attempts, reasons and isolation. No CLOSED record depends on a quarantine. This is 207 contribution records, not every Main obligation; known prose-obligation work in the canonical queue is outside the census.

## Cold-start commands

From a fresh checkout of the eventual contribution commit, install Python 3.10+, Git and the exact Lean toolchain declared by `formal/lean-toolchain`. Use the official Lean 4.34.1 release (commit `5045d0056413266e57c625dcd7c365b10e377c52`, archive SHA256 `47bf4bbd78f70c2e9670598ab7124d92b6efb7330ff33e5fbb4030f6fd72e4e4`); keep inherited proxy/CA trust and place that release's `bin` on PATH. Confirm `lean --version`, `lake --version` and `leanchecker` are available. Do not substitute an arbitrary Lean installation.

```sh
python -B -S tools/cbz_baseline.py inspect --root .
python -B -S tools/cbz_baseline.py run --root . --output /tmp/cbz-baseline-run
python -B -S tools/cbz_baseline.py cold-start --root .
python -B -S tools/cbz_lean_check.py
```

The baseline CLI defaults to this namespaced `baseline-config.json`; use `--config` for another explicit configuration. Outputs must be outside the source root. The integrated runner is under implementation and remains INCOMPLETE until its actual source-bound receipt exists; the commands above are proposed executable entry points, not recorded successes. The Lean command writes `formal/.lake/cbz-evidence/receipt.json` and can establish only its stated arithmetic scope. A nonzero exit, missing command/tool/dependency, missing receipt, source mismatch or incomplete coverage remains unmet.

## Evidence and invariants

Read this file first, run the commands, then use [PROOF_INDEX.proposed.md](PROOF_INDEX.proposed.md), [TOOLING.proposed.md](TOOLING.proposed.md), and [BASELINE.proposed.md](BASELINE.proposed.md) to interpret the exact result. Machine records bind copied source bytes and receipt hashes. The prior local STATUS is preserved exactly in [archive/STATUS.pre-cbz-20261029.md](archive/STATUS.pre-cbz-20261029.md), with its Git-blob/SHA256/source-link receipt; historical acceptance text remains history rather than kernel promotion.

Immutable A1–A8: no placeholders presented as closed; no unverified/custom axiom injection; no floating-point bound certification; no weakened bounds; no silent test removal; explicit vacuity/nonvacuity limits; actual observed statuses; auditors tested adversarially and limited to demonstrated scope. Source scans, hashes, CI success, reviews and merges do not replace a kernel certificate or exact statement/model alignment. Historical source/evidence copies are isolated data and are not active imports. Failed checks remain visible.

## Unmet criteria

All G0–G11 remain INCOMPLETE as a joint final baseline: prior worker successes are scoped observations, not a final federation receipt. Current Math project execution/kernel/axiom closure, full semantic Layer A coverage, complete Main prose-obligation census, final regression, pins, status truth, zero-context cold start, Git/tooling hygiene and final administrative integration remain joint work. The earlier remote interval inventory observed 2,115 branches, 69 open PRs and 26 tags; it does not describe subsequent cleanup. Classic branch-protection reads returned HTTP403; no green assumption follows. Root owns current remote receipts and serial integration. No cleanup completion is claimed here.
'''
docs['PROOF_INDEX']=f'''# Proof index — proposed contribution

{intro}

The authoritative contribution data is [proof-index.json](proof-index.json): 31 CLOSED literal Main arithmetic records and 176 QUARANTINED application/source/alignment records. Every record has a unique ID, exact source hashes, dependencies, exact boundary, closure limits and its full original record. All104 v20 dataset records and their104 obligations, imports and frozen/source-provenance JSON remain exact copies. Current Math GRAPH49 and55 original edges are retained; source classifications/reviewer provenance are not rewritten.

CLOSED means only the exact Lean declaration string in `exact_boundary.statement`, with the observed empty project axiom set and explicit premise-witness scope. It does not mean a paper theorem, Gaussian field model, moment identity, Palm identification, coefficient enclosure, geometric carrier or prose obligation is closed. See [STATUS.proposed.md](STATUS.proposed.md) for the31closure/nonvacuity boundary and actual receipt identity.

QUARANTINED means application closure is not established at this exact boundary. Definitions and carriers retain their original kinds and exact fields. An absent historical carrier stays absent; later constructions do not establish byte identity. The16group record source hashes cover all162 current Math Lean files at `{groups['Math_current_commit']}`. The7alignment records preserve exact declaration/path/hash gaps and minimal repairs. Resource-limited execution does not retract historical evidence or prove a theorem false.

Quarantine owner is `none`; no background maintenance, retries, approval queue or new research front is introduced. Reopening requires exact statement/model/carrier identity, missing nonvacuity/application evidence and an actual source-bound kernel/axiom result without weakened bounds. CLOSED dependencies are empty and import only the dependency-free Main formal package, so this contribution does not rely on quarantined Math sources. Full Main obligations outside this207recordcontribution remain explicitly uncounted.
'''
docs['GOVERNANCE']='''# Governance — proposed contribution

The current owner Crystalline Baseline directive controls this contribution. Prior operational approval freezes, review-family rules and repeated permission ceremonies are superseded where they conflict. Historical authorship, mathematical hypotheses, review provenance, unresolved findings and actual platform permissions remain intact. This proposal is submitted through the single canonical #684/#685 queue; it does not create another canonical root or authorize concurrent remote writers.

Layer A is immutable: A1 no logical placeholders called closed; A2 no unverified/new axioms; A3 exact arithmetic/certification only, no floating-point certification; A4 no weakened bounds; A5 no removed/disabled failing tests; A6 explicit vacuity and witnesses; A7 statuses match actual observed source/kernel/build/audit scope; A8 auditors have real adversarial controls and never exceed demonstrated scope. Historical informal acceptance is not rewritten as kernel proof.

Layer B is agent-owned procedure: perform authorized reversible work, minimize ceremony, use actual gates, preserve exact old refs/bytes before cleanup, keep one Root remote writer, and record explicit inaccessible/unknown items. No new research fronts or unrelated infrastructure expansion. A missing permission remains a tool gap, not an inferred green result. No permanent timers or maintenance owners are created for quarantines.

Layer C is append-only decisions. The contribution decision log records the owner directive, exact source preservation, literal arithmetic limit, application quarantine, and the later discovery of another canonical Root. Source/execution contributors may self-review engineering; no new scientific independence is fabricated. Canonical integration belongs to the existing Root queue.

The full terminal contract has12criteria: clean scoped Git/worktrees/stashes; no open PR/dangling branch/unreferenced unmerged tag; every obligation exactly CLOSED or QUARANTINED; complete Layer A accounting; full regression; Lean build/kernel/axioms; declared mutation threshold; consistent pins; documented self-verified tools; mutually truthful canonical surfaces; STATUS-only zero-context rerun; complete decisions/no administrative residue. This contribution asserts none of these federation-wide conjunctions.
'''
docs['TOOLING']='''# Tooling — proposed contribution

The machine registry is [tool-registry.json](tool-registry.json). Original worker registries/reports are preserved under sources/tooling/ and sources/lean-foundation/. Source hashes, exact commands and scope limits accompany active tools. The baseline runner and coverage companion remain pending final registration; no complete tool-hygiene claim is made.

| Tool | Command/self-verification | Demonstrated boundary |
|---|---|---|
| cbz_audit.py | `python -B tools/cbz_audit.py selftest`; `python -B tools/cbz_audit.py scan --root .` | finite lexical/source inventory;43normal and optimized tests,12/12finite mutants in original worker receipt; not semantic Layer A completion |
| cbz_lean_check.py | `python -B -S tools/cbz_lean_check.py`; unittest `test_cbz_lean_check.py` |31literal Main statements,45original constants, finite witness domains, named-module fresh kernel replay and6controls; analytic alignment excluded |
| formal_gate_check.py | see original lean-foundation registry; unittest `test_formal_gate_check.py` | existing build/kernel/target-axiom and source controls; no source parent theorem inference |
| cbz_refs.py | unittest `test_cbz_refs.py`; normal and optimized modes |22tests,6safety mutations,2215-tip disposable scale fixture; the fixture count is not the observed2115remote-branch count. Root alone uses mutation commands against real remotes |
| cbz_storage.py | unittest `test_cbz_storage.py`; normal and optimized modes | exact immutable duplicate bytes/paths/modes and verified archive preservation; no mathematical effect |
| cbz_baseline.py | `inspect`, `run`, `cold-start` commands in STATUS.proposed.md | namespaced configuration/receipt/source/doc consistency only; missing gates stay INCOMPLETE |

Use the pinned Lean4.34.1 release and default inherited proxy/CA. The worker verified restored runtime/header/static-library bytes against official archive members, preserved earlier failed attempts, and observed all9Math dependency revisions. Math cache/build residue remains explicit; a source-readiness/static validator is not a current project kernel execution.

Archive/source copies are data, not tools to execute blindly. The preservation/generation scripts and original registries retain their one-off scopes. No scanner, hash check, fixture, merged PR or green unit test substitutes for exact source-statement alignment, nonvacuity, full regression or scientific review.
'''
docs['BASELINE']='''# Partial Baseline — proposed contribution

This proposal is Partial Baseline evidence for the single canonical queue. It contains31literal arithmetic closures and exact preserved quarantines; it is not an exhaustive Main obligation inventory or a complete federation baseline.

The machine gate marker below starts every G0–G11 at INCOMPLETE. A scoped historical/worker PASS is retained in its own receipt and cannot set a final gate green. Actual source-bound runner receipts must agree with docs, source hashes, command argv, dependencies and live state. A final freeze must rerun all required gates after integration. The baseline runner must exit nonzero for missing semantics, permissions, proof coverage, tests, pins, evidence, cold-start answers or hygiene; an unknown value is unmet.

Gate order: G0 auditor self-test; G1 full Layer A accounting; G2 all project declarations/axioms; G3 Lean build; G4 kernel replay; G5 regression; G6 declared mutation threshold; G7 cross-repository pin consistency; G8 status truth; G9 cold start; G10 Git hygiene; G11 tooling hygiene. Main's saved31literal kernel receipt is evidence for that scope only, and cannot close all-Main/federation G2–G4.

Known residue includes current Math execution/cache limits and seven semantic alignment gaps; broader prose obligations in the canonical queue; final regression; all remote/local cleanup and exact disposition receipts; classic branch-protection403; coverage companion/gate registry integration; and final clone/rerun. No failed test is hidden by quarantine. Further exact repairs must preserve statements and bounds. If irreducible residue remains, retain this Partial Baseline label.
'''
docs['INVENTORY']='''# Inventory — proposed contribution

Scope is explicit and finite:31Main literal arithmetic records;104legacy v20 application records and their complete original dataset/obligation/import fields;49current Math GRAPH records and55edges;16Math source groups covering162Lean files;7source-hash statement-alignment gaps. These categories intentionally overlap at the source/obligation level and are not a count of all federation mathematics.

Source snapshots: Main original formal source4ef1867175dbe1072a54a02e95f80c4e2c70a4ff; portable Main receipt f20cc54d4d136e854608853372249882eb4c949f; Math current source8151f592918a845bf19eb8eafaf8444540a32734. Exact legacy dataset, frozen graph, provenance and historical ledgers plus needed upstream sources are under sources/legacy-v20. The full legacy333-file/13.61MB tree was byte-verified externally before the proposal was reduced to needed73files/2.40MB; preservation receipts record the full prior copy. No historical bytes were silently destroyed.

The separate remote worker snapshot observed12repositories,2115branches,69openPRs,26tags and310workflow records over2026-10-29T16:55:11Z–17:04:08Z. Three repositories were archived. All scoped collections reached terminal pages. Universal-Law-Workspace's8exactgitlinks resolved and matched its pins.json but differed from target default tips.36unique exactpin targets resolved at that interval. Classic branch protection was inaccessible403. Native outputs remain in the external federation-inventory archive; they are not a claim about later state.

Coordinating Root subsequently reported another canonical Root queue (#684/#685), later Main81acc060 and #691 prose-obligation work. These later observations require canonical reconciliation; this proposal does not claim a current remote freeze, completed cleanup, clean Git, or complete Main obligation closure. Root owns new machine receipts and serial integration.
'''
decision={'timestamp':NOW,'layer':'C','action':'Namespace website-foundation contribution after live canonical Root discovery','rationale':'Avoid competing canonical root docs/refs; preserve exact31arithmetic kernel scope and104legacy/49GRAPH application boundaries while canonical queue reconciles wider Main prose obligations.','reversible':True,'evidence':['Current coordinating Root instruction','#684','#685','#691'],'no_git_mutations_by_builder':True}
write(Path('decisions.json'),{'schema':'cbz.decisions.v1','entries':[decision],'prior_decisions_source':'sources/prior-decisions.jsonl'})
docs['DECISIONS']=f'''# Decisions — proposed contribution

This append-only contribution log preserves the earlier owner mission/decisions in sources/MISSION.md and sources/prior-decisions.jsonl. It does not replace the canonical queue's ledger.

- Adopt the current owner baseline directive: immutable A1–A8, agent-owned operational simplification, actual observed gates, no approval loops or new research front.
- Preserve old STATUS bytes before replacement. The exact local pre-CBZ file, Git commit/blob, SHA256 and source URL are retained under archive/. Root STATUS was not replaced by this contribution.
- Preserve all104legacy records and49current GRAPH records with full original fields. Add separate application QUARANTINED overlays; definitions/carriers, historical classifications and provenance remain unchanged.
- Accept only the31literal Main arithmetic declarations as CLOSED within their exact observed kernel/nonvacuity scope; broader Main prose obligations and all analytic/model alignment remain outside that closure.
- Preserve16Math source/hash groups and7alignment gaps with owner none and zero maintenance. No weakening, axiom injection, false refutation claim, test removal or hidden dependency.
- At {NOW}, change delivery to a namespaced proposal after Root reported the live #684/#685 canonical queue and later Main/prose work. This resolves coordination ambiguity prospectively; no competing canonical docs/ref/PR were created by this builder. Root retains serial integration and real remote mutations.

Machine decision entry: [decisions.json](decisions.json). Remaining administrative work, remote receipts, final integrated source/command checks and all-Main census belong to the canonical queue and remain explicit unmet items.
'''
for name,text in docs.items():
 # Keep prose readable without altering machine IDs or original records.
 for a,b in [('All104','All 104'),('GRAPH49','GRAPH 49'),('GRAPH49','GRAPH 49'),('all162','all 162'),('the31','the 31 '),('this207recordcontribution','this 207-record contribution'),('full terminal contract has12criteria','full terminal contract has 12 criteria')]:text=text.replace(a,b)
 (OUT/(name+'.proposed.md')).write_text(text.strip()+'\n\n'+marker+'\n')
write(Path('source-manifest.json'),{'schema':'cbz.source-manifest.v1','generated_utc':NOW,'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((OUT/'sources').rglob('*')) if p.is_file()]})
write(Path('CONTRIBUTION.json'),{'state':'PARTIAL_BASELINE_PROPOSAL','canonical_queue':'https://github.com/d6g8k5htny-coder/main/issues/684','counts':{k:len(v) for k,v in categories.items()},'all_main_obligations_complete':False,'root_docs_modified':False,'remote_writes':False,'git_mutations':False,'source_manifest':'source-manifest.json','proof_index':'proof-index.json','tool_registry':'tool-registry.json','gate_config_owner':'cbz_ref_snapshot_tool','final_joint_receipt':'NOT_OBSERVED'})
print(json.dumps({'out':str(OUT),'records':len(records),'categories':{k:len(v) for k,v in categories.items()},'source_files':len(source_manifest)}))
