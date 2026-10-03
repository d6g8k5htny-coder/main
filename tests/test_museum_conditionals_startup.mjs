// Browser startup path of the conditional route: verified config/manifest reuse,
// claim selection, proof loading and host replacement. Engineering test only.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {digest,pin,fixture,documentFromHTML} from './fixtures/museum_fixture.mjs';

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
  const text=document.getElementById('conditional-route').textContent;
  for(const expected of ['Conditional route: cumulative transfer','H1','H2','H3','H4','ALL','NOT_EVALUATED',commit,audited.sha256,'lim_(ell->0)'])assert.ok(text.includes(expected),expected);
  assert.ok(!text.includes('Verifying the conditional route source'));
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
  assert.match(document.getElementById('conditional-route').textContent,/NOT_EVALUATED/);
});
test('a rejected startup is not retained for a later caller',async()=>{
  const {bodies,requests,fetcher,document}=scene();
  const good=bodies.get('museum.json');bodies.set('museum.json',good+' ');
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.match(document.getElementById('conditional-route').textContent,/Unavailable/);
  bodies.set('museum.json',good);
  assert.equal(await conditionals.startConditionals({document,fetcher}),true);
  assert.equal(count(requests,'config.json'),2);
});
test('changed proof bytes clear an earlier view and infer nothing',async()=>{
  const {bodies,fetcher,document}=scene();
  const altered=raw.slice();altered[30]^=1;bodies.set(audited.url,altered);
  const host=document.getElementById('conditional-route');host.textContent='old ready result';
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.match(host.textContent,/Unavailable: Source SHA-256 mismatch/);
  assert.ok(!host.textContent.includes('old ready'));assert.ok(!host.textContent.includes('H1'));
});
test('a manifest proof that is not the audited descriptor is refused before its bytes are fetched',async()=>{
  const other=pin(path,'A different text at the same path.',commit);
  const {requests,fetcher,document}=scene(other);
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.match(document.getElementById('conditional-route').textContent,/Conditional source identity mismatch/);
  assert.equal(count(requests,other.url),0);
});
test('a stale manifest stops before any pinned remote request',async()=>{
  const {bodies,requests,fetcher,document}=scene();
  bodies.set('museum.json',bodies.get('museum.json').replace('"schema_version":1','"schema_version":2'));
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.deepEqual(requests,['config.json','museum.json']);
  assert.match(document.getElementById('conditional-route').textContent,/Unavailable/);
});
test('a page without the route container is left untouched',async()=>{
  const {requests,fetcher,document}=scene();document.nodes.delete('conditional-route');
  assert.equal(await conditionals.startConditionals({document,fetcher}),false);
  assert.deepEqual(requests,[]);
});
