import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {webcrypto} from 'node:crypto';

const moduleURL=new URL('../docs/site/conditionals.mjs',import.meta.url);
const raw=new Uint8Array(fs.readFileSync(new URL('./fixtures/cumulative_transfer_source.txt',import.meta.url)));
const path='reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md';
const repo='d6g8k5htny-coder/Math-';
const commit='d6628da09384728992dcbe6e921cc28ba85aebb0';
const pin={repository:repo,commit,path,blob:'044ac5fdaf403a38e33983e31f0ad69f8e76d6d5',bytes:3272,
  sha256:'83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6',
  url:`https://raw.githubusercontent.com/${repo}/${commit}/${path}`,
  html_url:`https://github.com/${repo}/blob/${commit}/${path}`};
async function load(){assert.ok(fs.existsSync(moduleURL),'conditional projection implementation is absent');return import(moduleURL);}
async function project(){const m=await load();return m.projectVerified(raw,pin,webcrypto);}

class Node {
  constructor(tag){this.tagName=tag;this.children=[];this.text='';this.attributes={};}
  set textContent(v){this.text=String(v);this.children=[];}
  get textContent(){return this.text+this.children.map(x=>x.textContent).join(' ');}
  set innerHTML(v){throw Error('innerHTML must not be used');}
  append(...v){this.children.push(...v);}
  replaceChildren(...v){this.text='';this.children=[...v];}
  setAttribute(k,v){this.attributes[k]=v;}
}
const documentFor=host=>({createElement:tag=>new Node(tag),getElementById:id=>id==='conditional-route'?host:null});

test('six complete inputs feed exactly one conditional conclusion',async()=>{
  const p=await project();assert.equal(p.nodes.length,7);
  assert.deepEqual(p.routes,[{id:'cumulative-limit',logic:'ALL',requires:['setup','H1','H2','H3','H4','definition'],concludes:'conclusion'}]);
  assert.equal(p.scientific_status_authority,false);assert.equal(p.application_evaluation,'NOT_EVALUATED');
});
test('the normalization and fixed lower comparison are retained',async()=>{
  const p=await project();assert.match(p.nodes.find(n=>n.id==='H3').quote,/h\(r,z\)\/\(kappa\(z\)\*r\^m\) -> 1/);
  assert.match(p.nodes.find(n=>n.id==='H4').quote,/Uniformly in r,z/);assert.match(p.nodes.find(n=>n.id==='H4').quote,/fixed c0>0/);
});
test('measurability and almost-everywhere quantifiers are retained',async()=>{
  const q=(await project()).nodes[0].quote;
  for(const word of ['sigma-finite','alpha>-1','m>0','jointly measurable','lambda-almost every'])assert.ok(q.includes(word));
});
test('positive and zero cases and the density non-implication remain explicit',async()=>{
  const p=await project();assert.match(p.case_split,/If it is zero/);assert.match(p.case_split,/o\(ell\^beta\)/);
  assert.match(p.boundary,/not an excuse to infer a density asymptotic/);
});
test('changed source bytes are refused before projection',async()=>{
  const m=await load(),bad=raw.slice();bad[30]^=1;await assert.rejects(m.projectVerified(bad,pin,webcrypto),/identity/);
});
test('integer and mutable commit identities are refused',async()=>{
  const m=await load();for(const commit of [1111111111111111111111111111111111111111n,'main',true])
    await assert.rejects(m.projectVerified(raw,{...pin,commit},webcrypto),/identity/);
});
test('a numeric hash cannot be coerced into a string',async()=>{
  const m=await load();await assert.rejects(m.projectVerified(raw,{...pin,sha256:3333},webcrypto),/identity/);
});
test('private repository and changed proof paths are refused',async()=>{
  const m=await load();for(const changes of [{repository:'d6g8k5htny-coder/sandbox'},{path:'../proof.txt'},{path:'another/PROOF.md'}])
    await assert.rejects(m.projectVerified(raw,{...pin,...changes},webcrypto),/identity/);
});
test('redirected URL and changed blob or size are refused',async()=>{
  const m=await load();for(const changes of [{url:'https://example.com/proof'},{blob:'a'.repeat(40)},{bytes:'3272'}])
    await assert.rejects(m.projectVerified(raw,{...pin,...changes},webcrypto),/identity/);
});
test('deleting H3 cannot silently weaken the route',async()=>{
  const m=await load(),p=await project();p.nodes=p.nodes.filter(n=>n.id!=='H3');
  await assert.rejects(m.verifyProjection(p,raw,pin,webcrypto),/projection/);
});
test('ALL cannot become ANY',async()=>{
  const m=await load(),p=await project();p.routes[0].logic='ANY';
  await assert.rejects(m.verifyProjection(p,raw,pin,webcrypto),/projection/);
});
test('self dependency or unknown nodes are refused',async()=>{
  const m=await load();for(const node of ['conclusion','imaginary']){const p=await project();p.routes[0].requires.push(node);
    await assert.rejects(m.verifyProjection(p,raw,pin,webcrypto),/projection/);}
});
test('invented acceptance and independence fields are refused',async()=>{
  const m=await load();for(const changes of [{scientific_status_authority:true},{status:'ACCEPT'},{independence_credit:1}])
    await assert.rejects(m.verifyProjection({...await project(),...changes},raw,pin,webcrypto),/projection/);
});
test('rendering exposes hypotheses, exact identity, and application boundary',async()=>{
  const m=await load(),host=new Node('div');m.renderProjection(documentFor(host),host,await project(),pin);
  for(const text of ['H1','H2','H3','H4','ALL','NOT_EVALUATED',pin.commit,pin.sha256])assert.ok(host.textContent.includes(text),text);
  assert.ok(!host.textContent.includes('all hypotheses discharged'));
});
test('a failed load clears any earlier ready view',async()=>{
  const m=await load(),host=new Node('div');host.textContent='old ready result';
  const ok=await m.startConditionals({document:documentFor(host),loadServices:async()=>{throw Error('source unavailable');}});
  assert.equal(ok,false);assert.match(host.textContent,/Unavailable/);assert.ok(!host.textContent.includes('old ready'));
});
test('the source fixture is complete and immutable',async()=>{
  await project();assert.equal(raw.length,3272);assert.match(new TextDecoder().decode(raw),/## Proof/);
});
