import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
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
