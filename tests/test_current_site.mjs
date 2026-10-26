import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';
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

// Exercise the real atlas controller, model and shell. This small Node DOM
// implements only their DOM/event boundary, not browser layout, tabbing or AT.
async function atlasController(search='',data=fixture()) {
 const renderer=await import('../docs/site/current.mjs'),shell=await import('../docs/site/atlas-shell.mjs'),summary=await import('../docs/site/atlas-claim-summary.mjs');
 const source=await readFile(new URL('../docs/site/atlas.mjs',import.meta.url),'utf8');
 class Element extends EventTarget {
  constructor(tag,doc){super();this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.attributes=new Map();this.style={};this.dataset={};this.hidden=false;this._text='';this._value='';}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>typeof child==='string'?child:child.textContent).join('');}
  get id(){return this.getAttribute('id')||'';}set id(value){this.setAttribute('id',value);}
  get parentElement(){return this.parentNode;}
  setAttribute(key,value){this.attributes.set(key,String(value));}
  getAttribute(key){return this.attributes.get(key)??null;}
  get href(){return this.getAttribute('href')||'';}set href(value){this.setAttribute('href',value);}
  remove(){if(this.parentNode){const siblings=this.parentNode.children;siblings.splice(siblings.indexOf(this),1);this.parentNode=null;}}
  append(...children){for(const child of children){if(typeof child!=='string'){child.remove();child.parentNode=this;}this.children.push(child);}}
  replaceChildren(...children){for(const child of this.children)if(typeof child!=='string')child.parentNode=null;this.children=[];this._text='';this.append(...children);}
  insertBefore(child,before){child.remove();child.parentNode=this;this.children.splice(this.children.indexOf(before),0,child);}
  contains(child){return this===child||this.children.some(node=>typeof node!=='string'&&node.contains(child));}
  querySelectorAll(selector){const matches=[];for(const child of this.children)if(typeof child!=='string'){if(selector==='a[href]'?child.tagName==='A'&&!!child.getAttribute('href'):child.tagName===selector.toUpperCase())matches.push(child);matches.push(...child.querySelectorAll(selector));}return matches;}
  get firstChild(){return this.children[0];}
  get offsetHeight(){return 140;}
  get options(){return this.children;}
  get value(){return this.tagName==='SELECT'?(this.children.find(child=>child.value===this._value)||this.firstChild)?.value||'':this._value;}
  set value(value){this._value=String(value);}
  get selectedIndex(){return this.children.findIndex(child=>child.value===this.value);}
  set selectedIndex(index){this.value=this.children[index]?.value||'';}
  focus(){this.ownerDocument.activeElement=this;}
  scrollIntoView(){}
  getClientRects(){return this.hidden?[]:[{}];}
 }
 const doc={activeElement:null,createElement(tag){return new Element(tag,this);},createElementNS(ns,tag){return this.createElement(tag);}};
 const nodes=new Map();const add=(id,tag='div',parent)=>{const node=doc.createElement(tag);node.id=id;nodes.set(id,node);parent?.append(node);return node;};
 doc.getElementById=id=>nodes.get(id);const root=add('current-atlas'),layout=add('atlas-layout','div',root),browse=add('atlas-browse','div',layout);
 add('atlas-inspector','article',layout);add('atlas-diagnostics','ul',add('atlas-diagnostics-details','details',root));
 const html=await readFile(new URL('../docs/site/dependencies.html',import.meta.url),'utf8');
 for(const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"/g))if(!nodes.has(match[2]))add(match[2],match[1],browse);
 for(const value of ['graph','list']){const option=doc.createElement('option');option.value=value;nodes.get('atlas-view').append(option);}
 const window=new EventTarget(),media=new EventTarget();media.matches=false;
 const location=new URL('https://example.test/main/site/dependencies.html'+search),entries=[location.href];let cursor=0;
 const history={replaceState(state,title,url){location.href=new URL(url,location.href).href;entries[cursor]=location.href;},pushState(state,title,url){location.href=new URL(url,location.href).href;entries.splice(++cursor);entries.push(location.href);},back(){if(cursor>0){location.href=entries[--cursor];window.dispatchEvent(new Event('popstate'));}},forward(){if(cursor+1<entries.length){location.href=entries[++cursor];window.dispatchEvent(new Event('popstate'));}}};
 const saved=Object.getOwnPropertyDescriptor(globalThis,'document');globalThis.document=doc;
 const bindings={...m,...shell,...summary,currentSnapshot:Promise.resolve(data),element:renderer.element,anchor:renderer.anchor,document:doc,window,history,URL,URLSearchParams,matchMedia:()=>media,fetch:async()=>({ok:false})};
 Object.defineProperty(bindings,'location',{get:()=>location});
 try{
  // Imports use the actual modules above; awaiting the real start lifecycle
  // avoids testing a copied render function or a source-text pattern.
  const program=source.replace(/^import .*;\n/gm,'').replace(/^if\(typeof document[^\n]+\n?$/m,'');
  await runInNewContext(program+'\nstart();',bindings);
  const graphIDs=()=>nodes.get('atlas-canvas').querySelectorAll('button').map(node=>node.textContent.split(/candidate|axiom|predicate/)[0]);
  const listIDs=()=>nodes.get('atlas-list').querySelectorAll('button').map(node=>node.textContent.split(/candidate|axiom|predicate/)[0]);
  return {get:id=>nodes.get(id),graphIDs,listIDs,history,window,get url(){return location.href;},event(id,type){nodes.get(id).dispatchEvent(new Event(type));},async reload(){return atlasController(location.search,data);},close(){if(saved)Object.defineProperty(globalThis,'document',saved);else delete globalThis.document;}};
 }catch(error){if(saved)Object.defineProperty(globalThis,'document',saved);else delete globalThis.document;throw error;}
}

test('the real admitted inspector exposes the exact source-linked arithmetic companion without parent coverage',async()=>{
 const data={...fixture(),...JSON.parse(await readFile(new URL('./fixtures/atlas_formal_projection.json',import.meta.url),'utf8')),equations:[],references:[]};
 const f=await atlasController('?atlas-node=math.side24-coefficient',data);
 try{
  const panel=f.get('atlas-inspector');
  assert.match(panel.textContent,/UniversalLaw\.Side24\.pairing_sum_six/);
  assert.match(panel.textContent,/PENDING_INDEPENDENT_REVIEW/);
  assert.match(panel.textContent,/REJECTED/);
  assert.match(panel.textContent,/does not prove the coefficient or parent theorem/);
  assert.ok(panel.querySelectorAll('a[href]').some(a=>a.getAttribute('href').includes('receipt.json')));
  f.get('atlas-view').value='list';f.event('atlas-view','change');
  assert.match(panel.textContent,/UniversalLaw\.Side24\.pairing_sum_six/);
 }finally{f.close();}
});

test('claim summary puts exact scope and pinned source before continuation metadata',async()=>{
 const data=fixture();data.graph.data.nodes.a.scope='fixed d>=2; literal <hypothesis> & exact 1/3';const f=await atlasController('?atlas-node=a',data);
 try{const panel=f.get('atlas-inspector'),text=panel.textContent;assert.ok(text.indexOf('Immutable source reference')<text.indexOf('Resume this node'),'primary source must precede operational metadata');assert.ok(text.includes(data.graph.data.nodes.a.scope));assert.equal(panel.querySelectorAll('h3')[0].textContent,'a');assert.equal(panel.querySelectorAll('hypothesis').length,0);assert.match(text,/candidate.*OPEN_ACTIVE/);assert.ok(panel.querySelectorAll('a[href]').some(a=>a.getAttribute('href')===m.sourceURL(data.sources[0])));}
 finally{f.close();}
});
test('controller owns an explicit formal gap even without the spatial rendering platform',async()=>{
 const f=await atlasController('?atlas-node=a');
 try{const panel=f.get('atlas-inspector'),formal=panel.querySelectorAll('section').find(n=>n.id==='atlas-formal-evidence');assert.ok(formal,'formal summary must be in the admitted claim projection');assert.match(formal.textContent,/No verified graph-to-formal-artifact mapping/);assert.match(formal.textContent,/No mapped declaration or hypothesis record/);assert.match(formal.textContent,/No mapped trusted kernel-execution receipt/);assert.match(formal.textContent,/No mapped exact-source alignment review/);const details=formal.querySelectorAll('details').find(n=>n.id==='atlas-formal-details');assert.ok(details);assert.equal(details.open,false);}
 finally{f.close();}
});
test('same-claim layout preserves disclosures while changed and refused claims reset their projection',async()=>{
 const data=fixture(),f=await atlasController('?atlas-node=a',data),details=()=>f.get('atlas-inspector').querySelectorAll('details');
 try{const review=details().find(n=>n.id==='atlas-review-details'),formal=details().find(n=>n.id==='atlas-formal-details');assert.ok(review);assert.ok(formal);review.open=true;formal.open=true;f.get('atlas-view').value='list';f.event('atlas-view','change');assert.equal(details().find(n=>n.id==='atlas-review-details').open,true);assert.equal(details().find(n=>n.id==='atlas-formal-details').open,true);f.get('atlas-diagnostics').querySelectorAll('button').find(n=>n.textContent.startsWith('b:')).dispatchEvent(new Event('click'));assert.equal(details().find(n=>n.id==='atlas-review-details').open,false);assert.equal(details().find(n=>n.id==='atlas-formal-details').open,false);f.history.pushState(null,'','?snapshot='+'c'.repeat(40));f.window.dispatchEvent(new Event('popstate'));assert.equal(f.get('atlas-inspector').querySelectorAll('section').some(n=>n.id==='atlas-claim-summary'),false);assert.match(f.get('atlas-inspector').textContent,/snapshot/);}
 finally{f.close();}
});
test('missing source and scope stay explicit and recorded notes are not labelled hypotheses',async()=>{
 const data=fixture();data.graph.data.nodes.b.notes='A note, not an extracted statement.';const f=await atlasController('?atlas-node=b',data);
 try{const panel=f.get('atlas-inspector');assert.match(panel.textContent,/Recorded note/);assert.match(panel.textContent,/Source reference not recorded/);assert.equal(panel.querySelectorAll('a[href]').some(a=>a.textContent.includes('Immutable source reference')),false);assert.match(panel.textContent,/No mapped declaration or hypothesis record/);}
 finally{f.close();}
});

test('atlas Graph and List apply each bookmarked exact metadata filter without changing source evidence',async()=>{
 for(const [query,want]of [['atlas-classification=HOLD',['b']],['atlas-kind=predicate',['c']],['atlas-layer=D0',['b']],['atlas-classification=OPEN_ACTIVE&atlas-kind=candidate&atlas-layer=D1',['a']],['atlas-layer=unrecorded',[]],['atlas-classification=hold',[]]]){
  const data=fixture(),before=structuredClone(data),f=await atlasController('?atlas-node=a&'+query,data);
  try{assert.deepEqual(f.graphIDs(),want);assert.deepEqual(f.listIDs(),want);f.event('atlas-toggle-view','click');assert.equal(f.get('atlas-stage').hidden,true);assert.deepEqual(f.listIDs(),want);assert.deepEqual(data,before);}
  finally{f.close();}
 }
});

test('atlas metadata filters page the complete matching set and disclose the exact bookmark constraints',async()=>{
 const data=fixture();data.graph.data={nodes:{},edges:[]};
 for(let i=0;i<305;i++)data.graph.data.nodes['node'+i]={kind:i%2?'axiom':'candidate',classification:'OPEN_ACTIVE',layer:'D1'};
 const f=await atlasController('?atlas-node=node0&atlas-q=OPEN_ACTIVE&atlas-kind=candidate&atlas-view=list',data);
 try{
  assert.equal(f.listIDs().length,150);assert.match(f.get('atlas-count').textContent,/153 matching source nodes of 305 indexed source nodes/);assert.match(f.get('atlas-count').textContent,/kind: candidate/);
  assert.equal(f.get('atlas-next').disabled,false);f.event('atlas-next','click');assert.deepEqual(f.listIDs(),['node300','node302','node304']);assert.equal(new URL(f.get('atlas-share').href).searchParams.get('atlas-offset'),'150');assert.equal(f.get('atlas-next').disabled,true);
  f.event('atlas-previous','click');assert.equal(f.listIDs().length,150);assert.equal(new URL(f.get('atlas-share').href).searchParams.get('atlas-offset'),null);
 }finally{f.close();}
});

test('atlas filter bookmarks persist through search, node selection, reload and Back',async()=>{
 const f=await atlasController('?atlas-node=a&atlas-kind=predicate&atlas-layer=D1&source-q=independent');
 try{
  f.get('atlas-search').value='OPEN_ACTIVE';f.event('atlas-search','input');assert.deepEqual(f.listIDs(),['c']);
  const saved=new URL(f.get('atlas-share').href);assert.equal(saved.searchParams.get('atlas-kind'),'predicate');assert.equal(saved.searchParams.get('atlas-layer'),'D1');assert.equal(saved.searchParams.get('source-q'),'independent');assert.equal(saved.searchParams.get('snapshot'),commit);assert.equal(saved.searchParams.get('main-snapshot'),commit);
  const reloaded=await f.reload();try{assert.deepEqual(reloaded.graphIDs(),['c']);assert.equal(reloaded.get('atlas-search').value,'OPEN_ACTIVE');}finally{reloaded.close();}
  f.get('atlas-list').querySelectorAll('button')[0].dispatchEvent(new Event('click'));assert.equal(new URL(f.url).searchParams.get('atlas-node'),'c');assert.deepEqual(f.graphIDs(),['c']);
  // A same-document history entry with a different exact filter must restore
  // both its parsed state and real rendered window when Back/Forward runs.
  f.history.pushState(null,'','?atlas-node=a&atlas-classification=HOLD&atlas-view=list&snapshot='+commit+'&main-snapshot='+commit);f.window.dispatchEvent(new Event('popstate'));assert.deepEqual(f.listIDs(),['b']);
  f.history.back();assert.deepEqual(f.graphIDs(),['c']);assert.equal(new URL(f.get('atlas-share').href).searchParams.get('atlas-kind'),'predicate');
  f.history.forward();assert.deepEqual(f.listIDs(),['b']);assert.equal(f.get('atlas-view').value,'list');
 }finally{f.close();}
});

test('atlas refuses mismatched source bookmarks before applying metadata filters',async()=>{
 const f=await atlasController('?atlas-node=a&atlas-classification=HOLD&snapshot='+('c'.repeat(40)));
 try{assert.deepEqual(f.graphIDs(),[]);assert.deepEqual(f.listIDs(),[]);assert.match(f.get('atlas-count').textContent,/bookmark refused/);assert.match(f.get('atlas-inspector').textContent,/another snapshot/);assert.equal(new URL(f.url).searchParams.get('snapshot'),'c'.repeat(40));}
 finally{f.close();}
});
test('node identifiers and recorded source paths require own literal string fields',()=>{
 const f=fixture();assert.equal(m.hasNode(f.graph.data,{toString:()=> 'a'}),false);
 assert.equal(m.sourceForNode(f,Object.create({source:'proof/PROOF.md'})),null);
 assert.equal(m.sourceForNode(f,{source:['proof/PROOF.md']}),null);
 assert.equal(m.nodePage({toString:()=> 'a'}),null);
});

// Contract breaks caught here: losing the advertised immutable destination,
// treating ordinary source anchors as ambiguity, or letting untrusted URL parts
// choose a path outside the current page's site directory.
test('static reading links retain one advertised material archive among source links',()=>{
 const page='https://example.test/main/site/dependencies.html';
 const node='math.lifetime-remainder', archive='a'.repeat(64);
 const href=`snapshots/${archive}/nodes/index.html`;
 const links=Object.freeze(['https://github.com/example/research/blob/main/PROOF.md',href,href,'nodes/index.html','#current-research']);
 assert.equal(typeof m.staticNodeLinks,'function');
 assert.deepEqual(m.staticNodeLinks(node,links,page),{
  current:'nodes/math.lifetime-remainder.html',
  preserved:`snapshots/${archive}/nodes/math.lifetime-remainder.html`
 });
 assert.deepEqual([...links],['https://github.com/example/research/blob/main/PROOF.md',href,href,'nodes/index.html','#current-research']);
 for(const advertised of [href,`/main/site/${href}`,`https://example.test/main/site/${href}`]){
  assert.deepEqual(m.staticNodeLinks('math.node /?#é',[advertised],page),{
   current:'nodes/math.node%20%2F%3F%23%C3%A9.html',
   preserved:`snapshots/${archive}/nodes/math.node%20%2F%3F%23%C3%A9.html`
  });
 }
 // A sole well-formed generated pointer is trusted by producer/history admission;
 // this URL projection makes no independent archive-association claim.
 assert.equal(m.staticNodeLinks(node,[`snapshots/${'b'.repeat(64)}/nodes/index.html`],page).preserved,`snapshots/${'b'.repeat(64)}/nodes/math.lifetime-remainder.html`);
});
test('static reading links refuse missing ambiguous and unsafe archive addresses',()=>{
 const page='https://example.test/main/site/dependencies.html', node='math.lifetime-remainder';
 const archive='a'.repeat(64), href=`snapshots/${archive}/nodes/index.html`;
 assert.equal(typeof m.staticNodeLinks,'function');
 const unsafe=[
  [],['https://github.com/example/research/blob/main/PROOF.md'],
  [href,`snapshots/${'b'.repeat(64)}/nodes/index.html`],
  [`snapshots/${'b'.repeat(64)}/nodes/index.html`,href],
  [`https://elsewhere.test/main/site/${href}`],
  [`https://user:pass@example.test/main/site/${href}`],
  [`https://example.test:444/main/site/${href}`],
  [`https://example.test/elsewhere/site/${href}`],
  [`/main/site-other/${href}`],
  [`${href}?`],[`${href}?capture=wrong`],[`${href}#`],[`${href}#wrong`],
  [`snapshots/${archive.toUpperCase()}/nodes/index.html`],
  [`snapshots/${archive.slice(1)}/nodes/index.html`],
  [`snapshots/${archive}/nodes/not-index.html`],
  [`snapshots/${archive}/nodes/index.html/extra`],
  [`snapshots/%61${archive.slice(1)}/nodes/index.html`],
  [null,undefined,42,{},true],
 ];
 for(const hrefs of unsafe)assert.deepEqual(m.staticNodeLinks(node,hrefs,page),{current:'nodes/math.lifetime-remainder.html',preserved:null},JSON.stringify(hrefs));
 for(const invalid of [null,undefined,'',42,{},true])assert.deepEqual(m.staticNodeLinks(invalid,[href],page),{current:null,preserved:null});
 for(const badPage of ['not a URL','javascript:alert(1)',`https://user:pass@example.test/main/site/dependencies.html`])assert.deepEqual(m.staticNodeLinks(node,[href],badPage),{current:'nodes/math.lifetime-remainder.html',preserved:null});
 assert.deepEqual(m.staticNodeLinks(node,null,page),{current:'nodes/math.lifetime-remainder.html',preserved:null});
});


// A refused bookmark must not acquire current-source defaults through any
// serialization caller. Selecting a current diagnostic node is the exception.
test('refused source identity survives view changes Share reload and history',async()=>{
 for(const search of [
  '?snapshot='+('0'.repeat(40))+'&main-snapshot='+commit+'&atlas-node=a',
  '?snapshot='+commit+'&main-snapshot='+('0'.repeat(40))+'&atlas-node=a',
  '?snapshot='+('0'.repeat(40))+'&snapshot='+commit+'&main-snapshot='+('0'.repeat(40))+'&atlas-node=a&atlas-node=b',
  '?snapshot=malformed&main-snapshot='+commit+'&atlas-node=a',
  '?snapshot='+commit+'&main-snapshot='+commit+'&atlas-node=not-a-node'
 ]){
  const f=await atlasController(search+'&unrelated=keep#current-atlas');
  const identity=url=>{const q=new URL(url).searchParams;return ['snapshot','main-snapshot','atlas-node'].map(key=>q.getAll(key));};
  const original=identity(f.url);
  try{
   assert.match(f.get('atlas-count').textContent,/refused/);
   for(const view of ['list','graph']){
    f.get('atlas-view').value=view;f.event('atlas-view','change');
    assert.deepEqual(identity(f.url),original,'view change adopted current source');
    assert.deepEqual(identity(f.get('atlas-share').href),original,'Share adopted current source');
    assert.equal(new URL(f.url).searchParams.get('atlas-view'),view,'address lost chosen layout');
    assert.equal(new URL(f.url).searchParams.get('unrelated'),'keep');
    const reload=await f.reload();try{assert.match(reload.get('atlas-count').textContent,/refused/);assert.equal(reload.get('atlas-view').value,view);}finally{reload.close();}
   }
   f.history.pushState(null,'',new URL(f.url));f.event('atlas-toggle-view','click');f.history.back();assert.deepEqual(identity(f.url),original);assert.match(f.get('atlas-count').textContent,/refused/);f.history.forward();assert.deepEqual(identity(f.get('atlas-share').href),original);
  }finally{f.close();}
 }
});

test('initial Share and every non-adopting control preserve a refused identity',async()=>{
 const f=await atlasController('?snapshot='+('0'.repeat(40))+'&main-snapshot='+commit+'&atlas-node=a');
 const pins=url=>{const q=new URL(url).searchParams;return ['snapshot','main-snapshot','atlas-node'].map(k=>q.getAll(k));},original=pins(f.url);
 try{
  assert.equal(typeof f.get('atlas-share').href,'string','initial Share must be initialized for a refused bookmark');
  assert.deepEqual(pins(f.get('atlas-share').href),original,'initial Share did not preserve refused bookmark');
  for(const [id,event]of [['atlas-toggle-view','click'],['atlas-expand','click'],['atlas-collapse','click'],['atlas-reset','click'],['atlas-search','input'],['equation-search','input'],['equation-select','change']]){
   f.event(id,event);assert.deepEqual(pins(f.url),original,id+' adopted source defaults');assert.deepEqual(pins(f.get('atlas-share').href),original);assert.match(f.get('atlas-count').textContent,/refused/);
  }
 }finally{f.close();}
});

test('explicit current diagnostic selection alone may adopt current source pins',async()=>{
 const f=await atlasController('?snapshot='+('0'.repeat(40))+'&atlas-node=a');
 try{
  f.get('atlas-diagnostics').querySelectorAll('button')[0].dispatchEvent(new Event('click'));
  const q=new URL(f.url).searchParams;assert.equal(q.get('snapshot'),commit);assert.equal(q.get('main-snapshot'),commit);assert.equal(f.get('atlas-error').hidden,true);assert.ok(f.graphIDs().length>0);
  f.history.back();assert.match(f.get('atlas-count').textContent,/refused/);assert.equal(new URL(f.url).searchParams.get('snapshot'),'0'.repeat(40));f.history.forward();assert.equal(f.get('atlas-error').hidden,true);
 }finally{f.close();}
});
