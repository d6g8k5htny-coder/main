// Browser startup path of the conditional route: verified config/manifest reuse,
// claim selection, proof loading and host replacement. Engineering test only.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {digest,pin,fixture,documentFromHTML,isLive} from './fixtures/museum_fixture.mjs';

// Load the same entry URLs as the browser; a bare test import would create a
// second module instance beside the release-qualified transitive import.
const htmlURL=new URL('../docs/site/museum.html',import.meta.url);
const html=fs.readFileSync(htmlURL,'utf8');
const entry=name=>{
  const src=[...html.matchAll(/<script\b[^>]*src="([^"]+)"/g)].map(m=>m[1]).find(src=>new URL(src,htmlURL).pathname.endsWith('/'+name));
  assert.ok(src,`Missing entry ${name}`);return new URL(src,htmlURL);
};
const museum=await import(entry('museum.mjs'));
const conditionals=await import(entry('conditionals.mjs'));
const raw=new Uint8Array(fs.readFileSync(new URL('./fixtures/cumulative_transfer_source.txt',import.meta.url)));
const repo='d6g8k5htny-coder/Math-',commit='d6628da09384728992dcbe6e921cc28ba85aebb0';
const path='reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md';
const audited={repository:repo,commit,path,blob:'044ac5fdaf403a38e33983e31f0ad69f8e76d6d5',bytes:3272,
  sha256:'83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6',
  url:`https://raw.githubusercontent.com/${repo}/${commit}/${path}`,
  html_url:`https://github.com/${repo}/blob/${commit}/${path}`};
const CLAIM='cumulative-transfer-correction';
const VERIFYING='Verifying the conditional route source…';
const VERIFIED='Conditional route source bytes verified; the route is shown below. This establishes byte identity, not mathematical acceptance.';
const REFUSED=/^Unavailable: .+\. No conditional result inferred\.$/;
// The route's outcome lives in the page's one static status line; the host holds only the route.
const statusLine=document=>document.getElementById('conditional-route-status');
const hostOf=document=>document.getElementById('conditional-route');
function assertRefused(document,pattern=REFUSED){
  assert.match(statusLine(document).textContent,pattern);assert.match(statusLine(document).textContent,REFUSED);
  assert.equal(statusLine(document).className,'conditional-route-refusal error');
  assert.equal(hostOf(document).children.length,0,'a refused route leaves the host empty');assert.equal(hostOf(document).textContent,'');
}

function scene(proof=audited){
  const f=fixture({commit,proofs:{[CLAIM]:proof}});
  const manifestRaw=JSON.stringify(f.manifest);
  const config=JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(manifestRaw),sha256:digest(manifestRaw)}});
  const bodies=new Map([['config.json',config],['museum.json',manifestRaw],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,'Pinned proof.'],[audited.url,raw]]);
  const requests=[];
  const fetcher=async url=>{const key=String(url);requests.push(key);assert.ok(bodies.has(key),`Unexpected URL ${key}`);return new Response(bodies.get(key));};
  return {f,bodies,requests,fetcher,document:documentFromHTML()};
}
const count=(requests,url)=>requests.filter(r=>r===url).length;

test('a verified startup renders the route from the selected claim proof bytes',async()=>{
  const {requests,fetcher,document}=scene();
  const ok=await conditionals.startConditionals({document,fetcher});
  assert.equal(ok,true);
  const text=hostOf(document).textContent;
  for(const expected of ['Conditional route: cumulative transfer','H1','H2','H3','H4','ALL','NOT_EVALUATED',commit,audited.sha256,'lim_(ell->0)'])assert.ok(text.includes(expected),expected);
  assert.ok(!text.includes('Verifying the conditional route source'));
  assert.deepEqual(hostOf(document).children.map(n=>n.tagName),['ARTICLE']);
  assert.equal(statusLine(document).textContent,VERIFIED);assert.equal(statusLine(document).className,'');
  assert.deepEqual(requests.slice(0,2),['config.json','museum.json']);
  for(const url of ['config.json','museum.json',audited.url])assert.equal(count(requests,url),1,url);
});
test('the museum page and the conditional route share one verified startup and byte cache',async()=>{
  const {requests,fetcher,document}=scene();
  const [,ok]=await Promise.all([museum.startMuseum({document,fetcher,search:''}),conditionals.startConditionals({document,fetcher})]);
  assert.equal(ok,true);
  assert.equal(new Set(requests).size,requests.length,`repeated request: ${requests.join(', ')}`);
  assert.equal(count(requests,'config.json'),1);assert.equal(count(requests,'museum.json'),1);assert.equal(count(requests,audited.url),1);
  assert.match(document.getElementById('museum-state').textContent,/verified/i);
  assert.match(document.getElementById('claim-cards').textContent,/Object 8/);
  assert.match(hostOf(document).textContent,/NOT_EVALUATED/);
  assert.equal(statusLine(document).textContent,VERIFIED);
});
test('a rejected startup is not retained for a later caller',async()=>{
  const {bodies,requests,fetcher,document}=scene();
  const good=bodies.get('museum.json');bodies.set('museum.json',good+' ');
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assertRefused(document,/^Unavailable: /);
  bodies.set('museum.json',good);
  const {log}=observe(document);
  assert.equal(await conditionals.startConditionals({document,fetcher}),true);
  assert.equal(count(requests,'config.json'),2);
  assert.equal(statusLine(document).textContent,VERIFIED);assert.equal(statusLine(document).className,'','a later success clears the refusal colour');
  assert.deepEqual(log.filter(e=>e.kind==='status').map(e=>[e.prop,e.value]),[['textContent',VERIFYING],['className',''],['textContent',VERIFIED]],'a later run says Verifying… without the refusal colour, then its outcome');
});
test('changed proof bytes clear an earlier view and infer nothing',async()=>{
  const {bodies,fetcher,document}=scene();
  const altered=raw.slice();altered[30]^=1;bodies.set(audited.url,altered);
  const host=hostOf(document);host.textContent='old ready result';
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assertRefused(document,/^Unavailable: Source SHA-256 mismatch/);
  assert.ok(!host.textContent.includes('old ready'));assert.ok(!host.textContent.includes('H1'));
});
test('a manifest proof that is not the audited descriptor is refused before its bytes are fetched',async()=>{
  const other=pin(path,'A different text at the same path.',commit);
  const {requests,fetcher,document}=scene(other);
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assertRefused(document,/^Unavailable: Conditional source identity mismatch/);
  assert.equal(count(requests,other.url),0);
});
test('a stale manifest stops before any pinned remote request',async()=>{
  const {bodies,requests,fetcher,document}=scene();
  bodies.set('museum.json',bodies.get('museum.json').replace('"schema_version":1','"schema_version":2'));
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.deepEqual(requests,['config.json','museum.json']);
  assertRefused(document);
});
test('a page without the route container is left untouched',async()=>{
  const {requests,fetcher,document}=scene();document.nodes.delete('conditional-route');
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.deepEqual(requests,[]);
});

// Operation order (WCAG technique ARIA22): the status container exists with its role before
// its message changes. Every write to the route's status line is recorded with whether, at that
// moment, the line is connected to the page and carries role="status"; every node ever inserted
// under #conditional-route is collected. This is operation-order evidence on a DOM shim that
// models connection, not a screen-reader test.
const walk=node=>node&&typeof node==='object'?[node,...(node.children??[]).flatMap(walk)]:[];
function observe(document){
  const status=statusLine(document),host=hostOf(document),log=[],inserted=[];
  const record=(prop,value)=>log.push({kind:'status',prop,value:String(value),
    attached:status.isConnected,role:status.getAttribute('role')});
  const text=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(status),'textContent');
  Object.defineProperty(status,'textContent',{configurable:true,get(){return text.get.call(this);},set(value){record('textContent',value);text.set.call(this,value);}});
  let className=status.className;
  Object.defineProperty(status,'className',{configurable:true,get(){return className;},set(value){record('className',value);className=value;}});
  for(const method of ['append','replaceChildren']){
    const original=host[method];
    host[method]=function(...nodes){inserted.push(...nodes.flatMap(walk));log.push({kind:'host',method,tags:nodes.map(n=>n.tagName)});return original.apply(this,nodes);};
  }
  // Any role at all, or a live region by the shared definition (live role or any aria-live).
  const flagged=()=>[...new Set([...inserted,...host.children.flatMap(walk)])].filter(n=>isLive(n)||(typeof n.getAttribute==='function'&&n.getAttribute('role')!==null));
  return {log,flagged};
}
const changedProof=()=>{const altered=raw.slice();altered[30]^=1;return altered;};
const outcomes=[
  ['a verified route',{expect:true}],
  ['a changed museum.json at startup',{bodies:{'museum.json':b=>b+' '}}],
  ['a stale manifest',{bodies:{'museum.json':b=>b.replace('"schema_version":1','"schema_version":2')}}],
  ['a manifest proof that is not the audited descriptor',{proof:pin(path,'A different text at the same path.',commit)}],
  ['changed proof bytes',{bodies:{[audited.url]:changedProof}}],
  ['an unavailable museum module',{loadServices:async()=>{throw Error('Museum module unavailable');}}],
  ['a manifest without the route claim',{loadServices:async()=>({verifiedMuseum:async()=>({manifest:{claims:[]},cachedFetch:null}),verifyClaim:museum.verifyClaim})}],
];
async function runOutcome({expect=false,bodies:changes={},proof,loadServices}){
  const {bodies,fetcher,document}=scene(proof);
  for(const [url,change] of Object.entries(changes))bodies.set(url,change(bodies.get(url)));
  hostOf(document).textContent='old ready result';
  const watch=observe(document);
  const ok=await conditionals.startConditionals({document,fetcher,...(loadServices?{loadServices}:{})});
  assert.equal(ok,expect);
  return {document,...watch};
}
for(const [name,options] of outcomes){
  test(`${name}: the outcome is written to the attached role="status" line after the host changes`,async()=>{
    const {document,log}=await runOutcome(options);
    const writes=log.filter(e=>e.kind==='status');
    assert.ok(writes.length>0,'the outcome is written to #conditional-route-status');
    for(const write of writes)assert.deepEqual([write.attached,write.role],[true,'status'],`${write.prop} write while detached or without its role: ${write.value}`);
    // The static line already says "Verifying…", so a first run writes only its outcome.
    const outcome=options.expect?[['textContent',VERIFIED]]:[['textContent',statusLine(document).textContent],['className','conditional-route-refusal error']];
    assert.deepEqual(writes.map(e=>[e.prop,e.value]),outcome,'the line receives exactly one outcome and no earlier write');
    assert.ok(log.findLastIndex(e=>e.kind==='host')<log.indexOf(writes[0]),'the host has its final content before the outcome is announced');
    const host=hostOf(document);
    assert.deepEqual([host.getAttribute('role'),isLive(host)],[null,false],'the route host itself has no role and is not a live region');
    if(options.expect){
      assert.equal(statusLine(document).textContent,VERIFIED);assert.equal(statusLine(document).className,'');
      assert.deepEqual(hostOf(document).children.map(n=>n.tagName),['ARTICLE']);
    }else assertRefused(document);
  });
  test(`${name}: nothing inserted into the route host carries a role or is live`,async()=>{
    const {flagged}=await runOutcome(options);
    assert.deepEqual(flagged().map(n=>`${n.tagName}[role=${n.getAttribute?.('role')}][aria-live=${n.getAttribute?.('aria-live')}] ${n.textContent.slice(0,60)}`),[]);
  });
}
test('museum.html has exactly one route status line: static, role="status", directly before an empty host',()=>{
  const section=html.match(/<section id="conditionals">[\s\S]*?<\/section>/)?.[0]??'';
  assert.deepEqual([...section.matchAll(/<[^>]*\brole="status"[^>]*>/g)].map(m=>m[0]),['<p id="conditional-route-status" role="status">']);
  assert.ok(section.includes(`<p id="conditional-route-status" role="status">${VERIFYING}</p>\n<div id="conditional-route"></div>`));
  assert.ok(!section.includes('aria-live'),'no second live region in the route section');
});
test('on a first run the static Verifying… line receives no write while the route is verified',async()=>{
  const {fetcher,document}=scene();let release;const gate=new Promise(resolve=>release=resolve);
  const {log}=observe(document);
  const running=conditionals.startConditionals({document,fetcher,loadServices:async()=>{await gate;return museum;}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(log.filter(e=>e.kind==='status'),[],'no write while verifying');assert.equal(statusLine(document).textContent,VERIFYING);
  release();assert.equal(await running,true);
  assert.deepEqual(log.filter(e=>e.kind==='status').map(e=>[e.prop,e.value]),[['textContent',VERIFIED]]);
});
test('an older page without the status line shows one plain Verifying… paragraph in the host until the refusal replaces it',async()=>{
  const {fetcher,document}=scene();document.nodes.delete('conditional-route-status');
  // A className write would give the paragraph class="" in a browser; record any such write.
  const classWrites=[],create=document.createElement;
  document.createElement=tag=>{const node=create(tag);let value=node.className;Object.defineProperty(node,'className',{get:()=>value,set:v=>{classWrites.push([node.tagName,v]);value=v;}});return node;};
  let fail;const gate=new Promise((_,reject)=>fail=reject);
  const running=conditionals.startConditionals({document,fetcher,loadServices:()=>gate});
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(hostOf(document).children.map(n=>[n.tagName,n.textContent]),[['P',VERIFYING]],'one Verifying… paragraph while verifying');
  const [verifying]=hostOf(document).children;
  assert.deepEqual([verifying.attributes,verifying.className,classWrites],[{},'',[]],'a plain paragraph: no attributes and no class write');
  fail(Error('Museum module unavailable'));assert.equal(await running,false);
  assert.equal(hostOf(document).children.length,1,'one refusal paragraph');
  const [line]=hostOf(document).children;
  assert.notEqual(line,verifying);assert.equal(verifying.isConnected,false,'the refusal replaces the Verifying… paragraph');
  assert.match(line.textContent,REFUSED);assert.equal(line.className,'conditional-route-refusal error');
  assert.deepEqual([line.getAttribute('role'),isLive(line)],[null,false]);
  assert.deepEqual([hostOf(document).getAttribute('role'),isLive(hostOf(document))],[null,false],'the host is not live');
});
test('an older page without the status line still shows the refusal as a plain paragraph in the host',async()=>{
  const {bodies,fetcher,document}=scene();document.nodes.delete('conditional-route-status');
  bodies.set('museum.json',bodies.get('museum.json')+' ');
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  const [line,...rest]=hostOf(document).children;
  assert.equal(rest.length,0);assert.equal(line.tagName,'P');assert.match(line.textContent,REFUSED);
  assert.equal(line.className,'conditional-route-refusal error');assert.deepEqual([line.getAttribute('role'),isLive(line)],[null,false]);
  assert.deepEqual([hostOf(document).getAttribute('role'),isLive(hostOf(document))],[null,false],'the host is not live');
});
test('an older page without the status line draws only the route on success',async()=>{
  const {fetcher,document}=scene();document.nodes.delete('conditional-route-status');
  assert.equal(await conditionals.startConditionals({document,fetcher}),true);
  const host=hostOf(document);
  assert.deepEqual(host.children.map(n=>n.tagName),['ARTICLE'],'the route replaces the Verifying… paragraph; no outcome paragraph is added');
  assert.deepEqual([host.getAttribute('role'),isLive(host)],[null,false],'the host is not live');
});
