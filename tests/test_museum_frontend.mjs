import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';

const moduleURL = new URL('../docs/site/museum.mjs', import.meta.url);
test('museum exposes source-bound rendering rather than silently omitting the page', () => {
  assert.ok(fs.existsSync(moduleURL), 'The museum implementation is missing');
});
const museum = fs.existsSync(moduleURL) ? await import(moduleURL) : null;
const available = {skip: !museum};
const digest = value => createHash('sha256').update(value).digest('hex');
const pin = (path, value) => ({repository:'d6g8k5htny-coder/Math-', path, commit:'a'.repeat(40), blob:'b'.repeat(40), bytes:Buffer.byteLength(value), sha256:digest(value), url:`https://raw.githubusercontent.com/d6g8k5htny-coder/Math-/${'a'.repeat(40)}/${path}`, html_url:`https://github.com/d6g8k5htny-coder/Math-/blob/${'a'.repeat(40)}/${path}`});
const ids = ['d2-lifetime-remainder','d3-side24-coefficient','d4-fixed-remote-rn','d5-all-height-annulus','d5-height-window-annulus','d5-two-scale','d5-inner-belt-density','d5-fixed-transverse','cumulative-transfer-correction','p15-demand-one-counterexample','d6-p15-full-price'];
function fixture() {
  const proof = pin('proof.md','Pinned proof.');
  const claims = ids.map((id,i) => ({id,title:`Object ${i}`,scope_quote:`- Object ${i}: [proof](proof.md). Scope ${i}.`,source_label:i===9?'EXACT_COUNTEREXAMPLE':'ACCEPT',class:i===9?'engineering-only':'ACCEPT-scoped',proof,review:null,replay:{command:null,url:null,notice:'No executable replay fixture is supplied.'},status_quote:null}));
  for(const i of [0,1,2,10])claims[i].status_quote=`Accepted scope ${i}.`;
  const open = ['d1-parent-selection-open','d5-pin-neighborhoods-open','sard-g-a1-a6-open'].map((id,i)=>({id,title:`Open ${i}`,scope_quote:`| **Open ${i}** | Still open ${i}. | [Proof](proof.md) |`,source_label:'AMEND / open',class:'AMEND/open',proof,review:null,replay:{command:null,url:null,notice:'No executable replay fixture is supplied.'},status_quote:`Still open ${i}.`}));
  const index = `# Index\n\n## Reviewed scoped results\n\n${claims.map(c=>c.scope_quote).join('\n')}\n\n## Open or conditional results with complete proof text in GitHub\n- D1 parent lifetime theorem: full proof [proof](proof.md).\n`;
  const acceptedRows=[[0,'D2'],[1,'D3'],[2,'D4'],[10,'D6']].map(([i,id])=>`| **${id} — scope** | Accepted scope ${i}. | Review | Limits |`).join('\n');
  const status = `# Status\n\n## ACCEPT — scoped\n\n| Object | Accepted scope | Source and review | Explicit limits |\n|---|---|---|---|\n${acceptedRows}\n\n## AMEND / open\n\n| Object | Current reason | Source |\n|---|---|---|\n${open.map(c=>c.scope_quote).join('\n')}\n\n## Engineering only\n`;
  const indexSource=pin('PROOF_INDEX.md',index),statusSource=pin('STATUS.md',status);open[1].proof=indexSource;open[2].proof=statusSource;
  return {index,status,manifest:{schema_version:1,scientific_status_authority:false,index_source:indexSource,status_source:statusSource,claims:[...claims,...open],exhibits:{ec014:proof,remote:proof,annulus:proof,p15:proof,lifetime:proof},packets:[]}};
}

test('full reviewed bullets and all three AMEND rows retain their classes and source wording',available,()=>{
  const {manifest,index,status}=fixture();
  assert.equal(museum.validateBoundManifest(manifest,index,status).claims.length,14);
  const changed=structuredClone(manifest);changed.claims[0].scope_quote+=' Broader conclusion.';
  assert.throws(()=>museum.validateBoundManifest(changed,index,status),/projection|quote/i);
  const omitted=structuredClone(manifest);omitted.claims.splice(2,1);
  assert.throws(()=>museum.validateBoundManifest(omitted,index,status),/projection|count|order/i);
});
test('source display cannot promote a counterexample or an AMEND row',available,()=>{
  for(const index of [9,11]) {const f=fixture();f.manifest.claims[index].class='ACCEPT-scoped';assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/class|promotion/i);}
  const f=fixture();f.manifest.scientific_status_authority=true;assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/authority/i);
});
test('a genuine source elsewhere in the index cannot replace this claim proof or scope',available,()=>{
  const f=fixture();f.index+='\n[Other proof](other.md)\n';f.manifest.index_source=pin('PROOF_INDEX.md',f.index);
  f.manifest.claims[3].proof=pin('other.md','A genuine different proof.');
  assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/proof|own.*quote/i);
  const swapped=fixture();swapped.manifest.claims[3].status_quote=swapped.manifest.claims[10].status_quote;
  assert.throws(()=>museum.validateBoundManifest(swapped.manifest,swapped.index,swapped.status),/STATUS.*scope/i);
});
test('replay command must occur exactly in the verified replay source',available,async()=>{
  const f=fixture(),claim=f.manifest.claims[0];claim.replay.command='python invented-solver.py';claim.replay.source=claim.proof;
  await assert.rejects(()=>museum.verifyClaim(claim,async()=>new Response('Pinned proof.')),/replay command/i);
});
test('lifetime source identity must match D2 and packets must use the pinned main snapshot',available,()=>{
  const f=fixture();f.manifest.exhibits.lifetime=pin('other.md','Different source.');
  assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/lifetime|D2/i);
  const g=fixture(),result=pin('incoming/side24-identity-replay-20260926/RESULT.md','Packet');result.repository='d6g8k5htny-coder/main';result.commit='c'.repeat(40);result.url=`https://raw.githubusercontent.com/${result.repository}/${result.commit}/${result.path}`;result.html_url=`https://github.com/${result.repository}/blob/${result.commit}/${result.path}`;
  g.manifest.packets=[{id:'side24-identity-replay-20260926',issue:null,result,scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'}];
  assert.throws(()=>museum.validateBoundManifest(g.manifest,g.index,g.status),/Packet.*snapshot|Packet.*commit/i);
});
test('malformed, mutable and falsely displayed source identities are refused before fetch',available,()=>{
  const source=pin('proof.md','Pinned proof.');
  for(const change of [{commit:'a'.repeat(8)},{sha256:'0'.repeat(63)},{path:'../proof.md'},{repository:'private'},{url:source.url.replace('/'+source.commit+'/','/main/')},{html_url:source.html_url.replace('/proof.md','/other.md')}]) assert.throws(()=>museum.validateDescriptor({...source,...change}),/identity|source|path|URL/i);
});
test('verified source bytes are required before a claim can be presented',available,async()=>{
  const f=fixture(),claim=f.manifest.claims[0];
  const badFetch=async()=>new Response('Changed proof.');
  await assert.rejects(()=>museum.verifyClaim(claim,badFetch),/byte count|SHA-256/i);
  const result=await museum.verifyClaim(claim,async()=>new Response('Pinned proof.'));
  assert.equal(result.proofText,'Pinned proof.');
});
test('streamed responses stop at the byte cap and cancel the reader',available,async()=>{
  let cancelled=false;
  const response=new Response(new ReadableStream({pull(controller){controller.enqueue(new Uint8Array(8));},cancel(){cancelled=true;}}));
  await assert.rejects(()=>museum.boundedFetch('source',async()=>response,12,1000),/byte limit/i);
  assert.equal(cancelled,true);
});
test('a stalled response body is aborted within its request deadline',available,async()=>{
  let cancelled=false,signal;
  const response=new Response(new ReadableStream({pull(){return new Promise(()=>{});},cancel(){cancelled=true;}}));
  await assert.rejects(()=>museum.boundedFetch('source',async(_url,options)=>{signal=options.signal;return response;},32,15),/timed out/i);
  assert.equal(signal.aborted,true);assert.equal(cancelled,true);
});
test('review comment pointers cannot masquerade as byte-frozen comment content',available,()=>{
  const f=fixture();f.manifest.claims[0].review={...f.manifest.index_source,pointer_only:true,review_url:'https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-123',review_notice:'Pointer only.'};
  assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/review pointer/i);
});
test('AMEND pointers bind their declared STATUS row or the specific open-obligation index entry',available,()=>{
  const f=fixture(),claim=f.manifest.claims[11],url='https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-123';
  const row=claim.scope_quote.replace('[Proof](proof.md)',`[Review](${url})`);
  f.status=f.status.replace(claim.scope_quote,row);claim.scope_quote=row;f.manifest.status_source=pin('STATUS.md',f.status);f.manifest.claims[13].proof=f.manifest.status_source;
  claim.review={...f.manifest.status_source,pointer_only:true,review_url:url,review_notice:'Pointer only; linked comment is not byte-frozen.'};
  assert.equal(museum.validateBoundManifest(f.manifest,f.index,f.status).claims[11].class,'AMEND/open');
  const wrong=structuredClone(f.manifest);wrong.claims[11].review={...wrong.index_source,pointer_only:true,review_url:url,review_notice:'Pointer only.'};
  assert.throws(()=>museum.validateBoundManifest(wrong,f.index,f.status),/review pointer/i);
});
test('packet index refuses a foreign packet, changed scientific effect and mismatched result path',available,()=>{
  const f=fixture(),result={...pin('incoming/side24-identity-replay-20260926/RESULT.md','scientific_effect: NONE\nreview_status: REVIEW_REQUIRED'),repository:'d6g8k5htny-coder/main'};
  result.url=result.url.replace('/Math-/','/main/');result.html_url=result.html_url.replace('/Math-/','/main/');
  const packet={id:'side24-identity-replay-20260926',issue:null,result,scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'};
  f.manifest.packets=[packet];assert.equal(museum.validateBoundManifest(f.manifest,f.index,f.status).packets.length,1);
  for(const change of [{id:'unlanded-draft'},{scientific_effect:'ACCEPT'},{result:{...result,path:'incoming/elsewhere/RESULT.md'}}]){const changed=structuredClone(f.manifest);Object.assign(changed.packets[0],change);assert.throws(()=>museum.validateBoundManifest(changed,f.index,f.status),/Packet|source URL/i);}
});

class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.listeners={};this.hidden=false;this._text='';}
  set textContent(value){this._text=String(value);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join('');}
  append(...children){this.children.push(...children);} replaceChildren(...children){this._text='';this.children=children;}
  setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(k,f){this.listeners[k]=f;}
}
function documentFromHTML(){
  const html=fs.readFileSync(new URL('../docs/site/museum.html',import.meta.url),'utf8');
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m=>[m[2],new Element(m[1])]));
  return {nodes,createElement:tag=>new Element(tag),createTextNode:value=>({textContent:value}),getElementById:id=>nodes.get(id)??null};
}
test('rendered claims use declared HTML containers, complete identities and literal disclaimer',available,async()=>{
  const f=fixture(),document=documentFromHTML();
  const mapping=new Map([['museum.json',JSON.stringify(f.manifest)],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,'Pinned proof.']]);
  await museum.startMuseum({document,search:'',fetcher:async url=>{assert.ok(mapping.has(String(url)),`Unexpected URL ${url}`);return new Response(mapping.get(String(url)));}});
  const text=document.getElementById('claim-cards').textContent;
  assert.match(text,/Object 0/);assert.match(text,/Claim and scope/);assert.match(text,/Source and review/);assert.match(text,/Replay/);assert.match(text,/Engineering — not acceptance/);
  for(const field of ['d2-lifetime-remainder','ACCEPT-scoped','d6g8k5htny-coder/Math-','proof.md','a'.repeat(40),digest('Pinned proof.')]) assert.ok(text.includes(field),field);
  assert.ok(text.includes('This canvas explains the pinned source. It is not a proof and does not change status.'));
  assert.match(document.getElementById('lifetime-fixture').textContent,/fixture absent/i);
  assert.equal(document.getElementById('undeclared-id'),null);
  document.nodes.delete('claim-cards');
  await assert.rejects(()=>museum.startMuseum({document,search:'',fetcher:async()=>new Response(JSON.stringify(f.manifest))}),/Missing museum container/);
});

test('actual pinned museum renders all cards and rejects cross-claim source, scope and replay substitutions',{skip:!museum||(!process.env.MUSEUM_MATH_ROOT&&!process.env.MUSEUM_FIXTURE)},async()=>{
  const manifest=JSON.parse(fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url)));
  const root=new URL('..',import.meta.url).pathname;
  const fixtures=process.env.MUSEUM_FIXTURE?JSON.parse(fs.readFileSync(process.env.MUSEUM_FIXTURE,'utf8')):null;
  const read=source=>{if(fixtures){assert.ok(Object.hasOwn(fixtures,source.url),`Missing pinned source fixture: ${source.url}`);return Buffer.from(fixtures[source.url],'base64');}return execFileSync('git',['-C',source.repository.endsWith('/Math-')?process.env.MUSEUM_MATH_ROOT:root,'show',`${source.commit}:${source.path}`],{maxBuffer:3*1024*1024});};
  const index=read(manifest.index_source).toString('utf8'),status=read(manifest.status_source).toString('utf8');
  const fetcher=async url=>{
    if(url==='museum.json')return new Response(JSON.stringify(manifest));
    if(fixtures){assert.ok(Object.hasOwn(fixtures,String(url)),`Unexpected request: ${url}`);return new Response(Buffer.from(fixtures[String(url)],'base64'));}
    const match=String(url).match(/^https:\/\/raw\.githubusercontent\.com\/(d6g8k5htny-coder\/(?:Math-|main))\/([0-9a-f]{40})\/(.+)$/);
    assert.ok(match,`Unexpected request: ${url}`);
    return new Response(read({repository:match[1],commit:match[2],path:decodeURIComponent(match[3])}));
  };
  const document=documentFromHTML();
  await museum.startMuseum({document,fetcher,search:'',geometryLoader:()=>{throw Error('Home must not load a graphics context');}});
  assert.match(document.getElementById('museum-state').textContent,/displayed source bytes verified/);
  assert.equal(document.getElementById('claim-cards').children.length,14);
  assert.match(document.getElementById('packet-cards').textContent,/packet — not STATUS/);
  const proof=structuredClone(manifest);proof.claims[3].proof=proof.claims[10].proof;
  assert.throws(()=>museum.validateBoundManifest(proof,index,status),/own pinned source quote/);
  const scope=structuredClone(manifest);scope.claims[3].status_quote=scope.claims[10].status_quote;
  assert.throws(()=>museum.validateBoundManifest(scope,index,status),/STATUS scope/);
  const replay=structuredClone(manifest.claims[0]);replay.replay.command='python invented-solver.py';
  await assert.rejects(()=>museum.verifyClaim(replay,fetcher),/Replay command.*recorded exactly/);
});
