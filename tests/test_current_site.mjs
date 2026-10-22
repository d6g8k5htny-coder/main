import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import './test_current_site_shell.mjs';
import './test_current_observation_coherence.mjs';
import './test_current_inventory_acquisition.mjs';
import './test_current_site_resume.mjs';
let m = {};
try { m = await import('../docs/site/current-model.mjs'); } catch (e) { if (e.code !== 'ERR_MODULE_NOT_FOUND') throw e; }
const commit='a'.repeat(40), digest='b'.repeat(64);
const fixture=()=>({schema_version:1,observed_at:'2026-10-13T12:00:00Z',scientific_effect:'NONE',repositories:[{repository:'d6g8k5htny-coder/main',commit,tree:commit},{repository:'d6g8k5htny-coder/Math-',commit,tree:commit}],sources:[{repository:'d6g8k5htny-coder/Math-',commit,path:'proof/PROOF.md',blob:commit,bytes:20}],graph:{identity:{repository:'d6g8k5htny-coder/Math-',commit,path:'frontiers/x/GRAPH.json',blob:commit,bytes:20,sha256:digest},data:{nodes:{a:{kind:'candidate',layer:'D1',classification:'OPEN_ACTIVE',source:'proof/PROOF.md'},b:{kind:'axiom',layer:'D0',classification:'HOLD'},c:{kind:'predicate',layer:'D1',classification:'OPEN_ACTIVE'}},edges:[{from:'a',to:'b',required:true,relation:'uses'},{from:'c',to:'a',required:false,relation:'context'}]}},equations:[{id:'e1',text:'<script>x</script> ell^(-1/3)',repository:'d6g8k5htny-coder/main',commit,path:'experiments/a/PROOF.md',line_start:1,line_end:2,sha256:digest}],references:[]});
test('current snapshot rejects invalid identities and preserves literal source values',()=>{
 assert.equal(typeof m.validateSnapshot,'function');
 const f=fixture(); assert.equal(m.validateSnapshot(f),f);
 for(const mutate of [x=>x.repositories[0].commit='main',x=>x.sources[0].path='../secret',x=>x.observed_at='bad',x=>x.graph.data.edges[0].to='missing',x=>x.equations[0].line_end=0,x=>x.repositories[0].repository='evil/private',x=>x.sources[0].commit='c'.repeat(40)]){const bad=fixture();mutate(bad);assert.throws(()=>m.validateSnapshot(bad));}
});
test('freshness separates future, stale and current observations',()=>{
 assert.equal(typeof m.freshness,'function');
 assert.equal(m.freshness('2026-10-13T12:00:00Z',Date.parse('2026-10-13T13:00:00Z')),'current');
 assert.equal(m.freshness('2026-10-13T12:00:00Z',Date.parse('2026-10-13T15:00:00Z')),'stale');
 assert.equal(m.freshness('2026-10-13T12:00:00Z',Date.parse('2026-10-13T11:00:00Z')),'clock disagreement');
});
test('source links bind safe paths to pinned commits and reject executable targets',()=>{
 assert.equal(typeof m.sourceURL,'function');
 assert.equal(m.sourceURL(fixture().sources[0]),`https://github.com/d6g8k5htny-coder/Math-/blob/${commit}/proof/PROOF.md`);
 for(const path of ['../secret','javascript:alert(1)','a\\b','a//b','/abs']) assert.equal(m.sourceURL({...fixture().sources[0],path}),null);
});
test('focus expansion deduplicates multi-parent nodes and enforces render ceiling',()=>{
 assert.equal(typeof m.neighborhood,'function');const g=fixture().graph.data;
 const n=m.neighborhood(g,['a'],150);assert.deepEqual(n.ids,['a','b','c']);assert.equal(n.edges.length,2);assert.equal(g.edges[0].required,true);assert.equal(g.edges[1].required,false);
 assert.equal(m.neighborhood(g,['a'],2).ids.length,2);
 assert.deepEqual(m.neighborhood(g,['unknown'],150).ids,[]);
});
test('five metadata axes and explicit mapping gaps never infer kernel or review approval',()=>{
 assert.equal(typeof m.evidenceAxes,'function');const axes=m.evidenceAxes({classification:'PROVED_REVIEWED',source:'proof/PROOF.md',review_disposition:'FAIL'});
 assert.deepEqual(axes.map(a=>a.axis),['source','review','kernel','computation','alignment']);
 assert.deepEqual(axes.map(a=>a.state),['recorded','recorded','not_recorded','not_recorded','not_recorded']);
 assert.equal(m.evidenceAxes({kind:'axiom'})[0].state,'not_recorded');
});
test('search and independent URL state preserve exact identifiers and reject unknown node',()=>{
 assert.equal(typeof m.readState,'function');const f=fixture();
 const state=m.readState('?atlas-node=a&equation=e1&source-q=universality&atlas-view=list',f);assert.equal(state.node,'a');assert.equal(state.equation,'e1');assert.equal(state.view,'list');assert.equal(state.sourceQuery,'universality');
 assert.match(m.readState('?atlas-node=missing',f).error,/Unknown/);
 assert.equal(m.filterSources(f.sources,'PROOF').length,1);
});
test('hash mismatch and malformed payload refuse current rendering',async()=>{
 assert.equal(typeof m.parseVerifiedSnapshot,'function');
 const raw=new TextEncoder().encode(JSON.stringify(fixture()));const sha=Buffer.from(await webcrypto.subtle.digest('SHA-256',raw)).toString('hex');
 assert.equal((await m.parseVerifiedSnapshot(raw,{sha256:sha,bytes:raw.length},webcrypto)).schema_version,1);
 await assert.rejects(m.parseVerifiedSnapshot(raw,{sha256:digest,bytes:raw.length},webcrypto),/digest/);
 await assert.rejects(m.parseVerifiedSnapshot(raw,{sha256:sha,bytes:1},webcrypto),/byte/);
});
test('global graph search finds unloaded nodes and bounds large catalogs',()=>{
 assert.equal(typeof m.searchGraph,'function');const g=fixture().graph.data;
 g.nodes.hidden={kind:'candidate',classification:'OPEN_ACTIVE',notes:'distinctive new theorem'};
 assert.deepEqual(m.searchGraph(g,'distinctive').ids,['hidden']);
 const large={nodes:{},edges:[]};for(let i=0;i<1000;i++)large.nodes['node'+i]={kind:'candidate',classification:'OPEN_ACTIVE'};
 const result=m.searchGraph(large,'candidate');assert.equal(result.ids.length,150);assert.equal(result.matches,1000);
});
test('equation and graph share bookmarks refuse changed main source instead of substituting',()=>{
 assert.match(m.readState('?main-snapshot='+('c'.repeat(40)),fixture()).error,/another snapshot/);
});
test('graph search filter survives source-bound URL state',()=>{assert.equal(m.readState('?atlas-q=witness-collision',fixture()).query,'witness-collision');});
test('prototype property names cannot become fabricated graph nodes',()=>{
 for(const id of ['constructor','toString','__proto__']){const f=fixture();assert.match(m.readState('?atlas-node='+id,f).error,/Unknown/);assert.deepEqual(m.neighborhood(f.graph.data,[id]).ids,[]);f.graph.data.edges[0].to=id;assert.throws(()=>m.validateSnapshot(f));}
});

const topology=(edges,metadata={})=>({nodes:Object.fromEntries([...new Set(edges.flatMap(e=>[e.from,e.to]))].map(id=>[id,{kind:'candidate',classification:'OPEN_ACTIVE',layer:'D1',...metadata[id]}])),edges:edges.map(e=>({relation:'uses',...e}))});
const diamond=()=>topology([{from:'A',to:'B',required:true},{from:'A',to:'C',required:true},{from:'B',to:'D',required:true},{from:'C',to:'D',required:true}],Object.fromEntries(['A','B','C'].map(id=>[id,{classification:'PROVED_REVIEWED'}])));
test('current reverse reach counts distinct nodes in a required diamond and excludes root',()=>{
 assert.equal(typeof m.downstreamReach,'function');const g=diamond(),before=structuredClone(g);
 const reach=m.downstreamReach(g,'D');assert.equal(reach.allEdge,3);assert.equal(reach.requiredOnly,3);assert.deepEqual(reach.allEdgeIds,['A','B','C']);
 assert.deepEqual(m.downstreamReach(g,'A').allEdgeIds,[]);assert.deepEqual(g,before);
 assert.match(reach.labels.allEdge,/context/);assert.match(reach.labels.requiredOnly,/Required/);
});
test('current required routes return one shortest route per unresolved research target',()=>{
 assert.equal(typeof m.requiredDependencyPaths,'function');const g=diamond();
 const routes=m.requiredDependencyPaths(g,'A');assert.deepEqual(routes.map(p=>p.ids),[['A','B','D']]);
 assert.equal(routes[0].edges[0],g.edges[0]);assert.equal(routes[0].edges[1],g.edges[2]);
 assert.deepEqual(m.requiredDependencyPaths(g,'D'),[]);assert.match(m.REQUIRED_ROUTE_SEMANTICS,/shortest/);
});
test('context-only unresolved targets affect all-edge reach but earn no required route',()=>{
 const g=topology([{from:'A',to:'U',required:false,relation:'context_only'}],{A:{classification:'PROVED_REVIEWED'}});
 assert.equal(typeof m.downstreamReach,'function');assert.deepEqual([m.downstreamReach(g,'U').allEdge,m.downstreamReach(g,'U').requiredOnly],[1,0]);
 assert.equal(typeof m.requiredDependencyPaths,'function');assert.deepEqual(m.requiredDependencyPaths(g,'A'),[]);
});
test('context shortcuts do not erase longer required routes and parallel edges count once',()=>{
 assert.equal(typeof m.downstreamReach,'function');assert.equal(typeof m.requiredDependencyPaths,'function');
 const g=topology([{from:'X',to:'Z',required:false},{from:'X',to:'Y',required:true},{from:'Y',to:'Z',required:true},{from:'Y',to:'Z',required:false,relation:'context'}],{X:{classification:'PROVED_REVIEWED'},Y:{classification:'PROVED_REVIEWED'}});
 assert.deepEqual([m.downstreamReach(g,'Z').allEdge,m.downstreamReach(g,'Z').requiredOnly],[2,2]);
 assert.deepEqual(m.requiredDependencyPaths(g,'X').map(p=>p.ids),[['X','Y','Z']]);
 const cycle=topology([{from:'A',to:'B',required:true},{from:'B',to:'A',required:false}]);
 assert.deepEqual([m.downstreamReach(cycle,'A').allEdge,m.downstreamReach(cycle,'A').requiredOnly],[1,0]);
 assert.deepEqual([m.downstreamReach(cycle,'B').allEdge,m.downstreamReach(cycle,'B').requiredOnly],[1,1]);
});
test('unresolved target kinds and classifications match the existing source vocabulary',()=>{
 assert.equal(typeof m.unresolvedTargets,'function');const g={nodes:{},edges:[]};
 for(const [id,classification,kind] of [['a','OPEN_ACTIVE','candidate'],['b','OPEN_HISTORICAL','predicate'],['c','AUTHOR_SIDE_CANDIDATE','proof'],['d','AUTHOR_SIDE_REDUCTION','reduction'],['e','OPEN_ACTIVE','engineering'],['f','OPEN_ACTIVE','integrity_control'],['g','OPEN_ACTIVE','inventory'],['h','OPEN_ACTIVE','register'],['i','BLOCKED_ABSENT','candidate'],['j','REFUTED','candidate']])g.nodes[id]={classification,kind};
 assert.deepEqual(m.unresolvedTargets(g).map(r=>r.id),['a','b','c','d']);
 assert.ok(m.unresolvedTargets(g).every(r=>r.allEdge===0&&r.requiredOnly===0&&r.node===g.nodes[r.id]));
});
test('actual graph retains the p15 boundary 1/0 and CH-LIFT 3/0 regressions',async()=>{
 assert.equal(typeof m.downstreamReach,'function');const g=JSON.parse(await readFile(new URL('../docs/site/dependency-source/GRAPH.json',import.meta.url),'utf8')),before=structuredClone(g);
 for(const [id,expected] of [['math.p15-price-boundary',[1,0]],['hist.CH-LIFT',[3,0]]]){const r=m.downstreamReach(g,id);assert.deepEqual([r.allEdge,r.requiredOnly],expected);}
 assert.equal(Object.keys(g.nodes).filter(id=>{const r=m.downstreamReach(g,id);return r.allEdge!==r.requiredOnly;}).length,19);
 assert.deepEqual(g,before);
});
test('full reach and required routes do not depend on filtered or capped viewport',()=>{
 assert.equal(typeof m.downstreamReach,'function');const g=diamond(),before=m.downstreamReach(g,'D');
 m.neighborhood(g,['D'],1);m.searchGraph(g,'unmatched',{limit:1,classification:'REFUTED'});
 assert.deepEqual(m.downstreamReach(g,'D'),before);assert.deepEqual(m.requiredDependencyPaths(g,'A').map(p=>p.ids),[['A','B','D']]);
 for(const id of ['unknown','constructor','__proto__']){assert.equal(m.downstreamReach(g,id).known,false);assert.deepEqual(m.requiredDependencyPaths(g,id),[]);}
});
test('bounded graph windows expose complete match and frontier counts with paging',()=>{
 const g={nodes:{root:{kind:'candidate',classification:'OPEN_ACTIVE'}},edges:[]};
 for(let i=0;i<999;i++){const id='leaf'+i;g.nodes[id]={kind:'candidate',classification:'OPEN_ACTIVE'};g.edges.push({from:id,to:'root',required:true,relation:'uses'});}
 const first=m.neighborhood(g,['root'],{limit:150});assert.equal(first.ids.length,150);assert.equal(first.matches,1000);assert.equal(first.rootCount,1);assert.equal(first.frontierCount,999);assert.equal(first.omitted,850);assert.equal(first.window.nextOffset,150);
 const second=m.neighborhood(g,['root'],{limit:150,offset:first.window.nextOffset});assert.equal(second.ids.length,150);assert.equal(second.matches,1000);assert.equal(second.frontierCount,999);assert.equal(second.window.previousOffset,0);assert.equal(second.window.nextOffset,300);assert.equal(first.ids.some(id=>second.ids.includes(id)),false);
 const search=m.searchGraph(g,'candidate',{limit:500});assert.equal(search.ids.length,150);assert.equal(search.matches,1000);assert.equal(search.window.nextOffset,150);
 const last=m.searchGraph(g,'candidate',{limit:150,offset:900});assert.equal(last.ids.length,100);assert.equal(last.window.nextOffset,null);
});
test('graph filters apply to complete matches without altering raw node labels or edges',()=>{
 const g=fixture().graph.data;const found=m.searchGraph(g,'',{limit:1,classification:'OPEN_ACTIVE',kind:'candidate',layer:'D1'});
 assert.deepEqual(found.ids,['a']);assert.equal(found.matches,1);assert.equal(g.nodes.a.classification,'OPEN_ACTIVE');
 assert.deepEqual(m.searchGraph(g,'',{classification:'HOLD',kind:'axiom'}).ids,['b']);
 assert.equal(m.neighborhood(g,['a'],{classification:'OPEN_ACTIVE'}).matches,2);
});
test('equation text filtering retains inert literal content and bounded complete counts',()=>{
 assert.equal(typeof m.searchEquations,'function');const equations=Array.from({length:170},(_,i)=>({...fixture().equations[0],id:'equation'+i}));
 const result=m.searchEquations(equations,'<script> ell',{limit:150});assert.equal(result.matches,170);assert.equal(result.equations.length,150);assert.equal(result.equations[0],equations[0]);assert.equal(result.window.nextOffset,150);
 assert.equal(m.searchEquations(equations,'<script> ell',{limit:150,offset:150}).equations.length,20);assert.equal(m.searchEquations(equations,'unknown').matches,0);
});
test('atlas offset equation search and exact metadata filters survive URL state',()=>{
 const state=m.readState('?atlas-node=a&atlas-offset=2&equation-q=ell%5E&atlas-classification=OPEN_ACTIVE&atlas-kind=candidate&atlas-layer=D1',fixture());
 assert.equal(state.offset,2);assert.equal(state.equationQuery,'ell^');assert.equal(state.classification,'OPEN_ACTIVE');assert.equal(state.kind,'candidate');assert.equal(state.layer,'D1');
 for(const raw of ['-1','abc','1.5','Infinity'])assert.equal(m.readState('?atlas-offset='+raw,fixture()).offset,0);
 assert.ok(Number.isSafeInteger(m.readState('?atlas-offset='+('9'.repeat(30)),fixture()).offset));
});
test('node identifiers and recorded source paths require own literal string fields',()=>{
 const f=fixture();assert.equal(m.hasNode(f.graph.data,{toString:()=> 'a'}),false);
 assert.equal(m.sourceForNode(f,Object.create({source:'proof/PROOF.md'})),null);
 assert.equal(m.sourceForNode(f,{source:['proof/PROOF.md']}),null);
 assert.equal(m.nodePage({toString:()=> 'a'}),null);
});
