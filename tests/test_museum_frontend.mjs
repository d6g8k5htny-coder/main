import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {digest,pin,fixture,documentFromHTML,isLive} from './fixtures/museum_fixture.mjs';

const moduleURL = new URL('../docs/site/museum.mjs', import.meta.url);
test('museum exposes source-bound rendering rather than silently omitting the page', () => {
  assert.ok(fs.existsSync(moduleURL), 'The museum implementation is missing');
});
const museum = fs.existsSync(moduleURL) ? await import(moduleURL) : null;
const available = {skip: !museum};

async function delayedClaimPage(interaction) {
  const f=fixture(),document=documentFromHTML(),window=new EventTarget(),events=[];
  window.location={hash:'#d3-side24-coefficient'};
  const frames=[];window.requestAnimationFrame=callback=>frames.push(callback);
  const create=document.createElement;
  document.createElement=tag=>{
    const node=create(tag);
    node.focus=()=>{document.activeElement=node;events.push('focus:'+node.id);};
    node.scrollIntoView=()=>events.push('scroll:'+node.id);
    return node;
  };
  const find=(node,id)=>node.id===id?node:node.children?.map(child=>find(child,id)).find(Boolean);
  document.getElementById=id=>document.nodes.get(id)||[...document.nodes.values()].map(node=>find(node,id)).find(Boolean);
  const raw=JSON.stringify(f.manifest),config=JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(raw),sha256:digest(raw)}});
  const mapping=new Map([['config.json',config],['museum.json',raw],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,'Pinned proof.'],[f.manifest.claims[11].review.url,'Pinned reconciliation.']]);
  let release,started;const gate=new Promise(resolve=>release=resolve),requested=new Promise(resolve=>started=resolve);
  const loading=museum.startMuseum({document,window,search:'',currentHash:()=>window.location.hash,fetcher:async url=>{
    if(url===f.manifest.claims[0].proof.url){started();await gate;}
    assert.ok(mapping.has(url),'Unexpected museum source '+url);return new Response(mapping.get(url));
  }});
  await requested;
  if(interaction)window.dispatchEvent(new Event(interaction));
  release();await loading;
  for(const frame of frames)frame();
  return {events,document};
}
test('a delayed initial museum bookmark still reaches its verified claim when untouched',async()=>{
  const page=await delayedClaimPage();
  assert.deepEqual(page.events,['scroll:d3-side24-coefficient','focus:d3-side24-coefficient']);
});
for(const interaction of ['wheel','keydown','pointerdown','focusin','hashchange','popstate','pagehide']) {
  test(`museum ${interaction} while verifying sources cancels delayed focus and scroll`,async()=>{
    const page=await delayedClaimPage(interaction);assert.deepEqual(page.events,[]);
  });
}
test('full reviewed bullets, the reconciled D1 row and both AMEND rows retain their classes and source wording',available,()=>{
  const {manifest,index,status}=fixture();
  assert.equal(museum.validateBoundManifest(manifest,index,status).claims.length,14);
  const changed=structuredClone(manifest);changed.claims[0].scope_quote+=' Broader conclusion.';
  assert.throws(()=>museum.validateBoundManifest(changed,index,status),/projection|quote/i);
  const omitted=structuredClone(manifest);omitted.claims.splice(2,1);
  assert.throws(()=>museum.validateBoundManifest(omitted,index,status),/projection|count|order/i);
});
test('source display cannot promote a counterexample or an AMEND row',available,()=>{
  for(const index of [9,12,13]) {const f=fixture();f.manifest.claims[index].class='ACCEPT-scoped';assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/class|promotion/i);}
  {const f=fixture();f.manifest.claims[11].class='AMEND/open';assert.throws(()=>museum.validateBoundManifest(f.manifest,f.index,f.status),/class|promotion/i);}
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
  g.manifest.packets=[{id:'side24-identity-replay-20260926',issue:null,result,identity:{...result,path:'incoming/side24-identity-replay-20260926/IDENTITY.json',url:result.url.replace('RESULT.md','IDENTITY.json'),html_url:result.html_url.replace('RESULT.md','IDENTITY.json')},output:{...result,path:'incoming/side24-identity-replay-20260926/output.json',url:result.url.replace('RESULT.md','output.json'),html_url:result.html_url.replace('RESULT.md','output.json')},scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'}];
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
test('the accepted D1 row binds its STATUS quote, index-pinned proof and byte-bound review',available,()=>{
  const f=fixture();
  const d1=museum.validateBoundManifest(f.manifest,f.index,f.status).claims[11];
  assert.equal(d1.id,'d1-parent-lifetime');assert.equal(d1.class,'ACCEPT-scoped');
  const pointer=fixture();pointer.manifest.claims[11].review={...pointer.manifest.status_source,pointer_only:true,review_url:'https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-123',review_notice:'Pointer only.'};
  assert.throws(()=>museum.validateBoundManifest(pointer.manifest,pointer.index,pointer.status),/byte-bound|review pointer/i);
  const other=fixture();other.manifest.claims[11].review=pin('other.md','Different record.');
  assert.throws(()=>museum.validateBoundManifest(other.manifest,other.index,other.status),/review file/i);
  const scope=fixture();scope.manifest.claims[11].status_quote='Broader D1 scope.';
  assert.throws(()=>museum.validateBoundManifest(scope.manifest,scope.index,scope.status),/STATUS.*scope/i);
  const proof=fixture();proof.manifest.claims[11].proof=pin('other.md','A genuine different proof.');
  assert.throws(()=>museum.validateBoundManifest(proof.manifest,proof.index,proof.status),/D1 proof/i);
});
test('packet index refuses a foreign packet, changed scientific effect and mismatched result path',available,()=>{
  const f=fixture(),result={...pin('incoming/side24-identity-replay-20260926/RESULT.md','scientific_effect: NONE\nreview_status: REVIEW_REQUIRED'),repository:'d6g8k5htny-coder/main'};
  result.url=result.url.replace('/Math-/','/main/');result.html_url=result.html_url.replace('/Math-/','/main/');
  const oldCommit='71400b94f6cb354a8cf7aba73ffede2138a64efa';
  for(const source of [result,f.manifest.status_source]){source.url=source.url.replace('a'.repeat(40),oldCommit);source.html_url=source.html_url.replace('a'.repeat(40),oldCommit);source.commit=oldCommit;}
  const packet={id:'side24-identity-replay-20260926',issue:null,result,identity:{...result,path:'incoming/side24-identity-replay-20260926/IDENTITY.json',url:result.url.replace('RESULT.md','IDENTITY.json'),html_url:result.html_url.replace('RESULT.md','IDENTITY.json')},output:{...result,path:'incoming/side24-identity-replay-20260926/output.json',url:result.url.replace('RESULT.md','output.json'),html_url:result.html_url.replace('RESULT.md','output.json')},scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'};
  f.manifest.packets=[packet];assert.equal(museum.validateBoundManifest(f.manifest,f.index,f.status).packets.length,1);
  for(const change of [{id:'unlanded-draft'},{scientific_effect:'ACCEPT'},{result:{...result,path:'incoming/elsewhere/RESULT.md'}}]){const changed=structuredClone(f.manifest);Object.assign(changed.packets[0],change);assert.throws(()=>museum.validateBoundManifest(changed,f.index,f.status),/Packet|source URL/i);}
});

test('new landed chart packet requires its exact separate main commit and remains engineering-only',available,()=>{
  const f=fixture(),root='incoming/side24-chart-claude-20260926/',commit='a12c178c0f857a130cf434e9efd44233a038195b';
  const source=(name)=>({...pin(root+name,'Packet bytes'),repository:'d6g8k5htny-coder/main',commit,
    url:`https://raw.githubusercontent.com/d6g8k5htny-coder/main/${commit}/${root+name}`,
    html_url:`https://github.com/d6g8k5htny-coder/main/blob/${commit}/${root+name}`});
  const oldRoot='incoming/side24-identity-replay-20260926/',oldCommit='71400b94f6cb354a8cf7aba73ffede2138a64efa';
  const old=(name)=>({...source(name),path:oldRoot+name,commit:oldCommit,
    url:`https://raw.githubusercontent.com/d6g8k5htny-coder/main/${oldCommit}/${oldRoot+name}`,
    html_url:`https://github.com/d6g8k5htny-coder/main/blob/${oldCommit}/${oldRoot+name}`});
  f.manifest.status_source={...f.manifest.status_source,commit:oldCommit,url:f.manifest.status_source.url.replace('a'.repeat(40),oldCommit),html_url:f.manifest.status_source.html_url.replace('a'.repeat(40),oldCommit)};
  f.manifest.claims[13].proof=f.manifest.status_source;
  f.manifest.packets=[{id:'side24-identity-replay-20260926',issue:null,result:old('RESULT.md'),identity:old('IDENTITY.json'),output:old('output.json'),scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'},
    {id:'side24-chart-claude-20260926',issue:141,result:source('RESULT.md'),identity:source('IDENTITY.json'),output:source('output.json'),scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'}];
  assert.equal(museum.validateBoundManifest(f.manifest,f.index,f.status).packets.length,2);
  const changed=structuredClone(f.manifest);changed.packets[1].result={...changed.packets[1].result,commit:oldCommit,url:old('RESULT.md').url,html_url:old('RESULT.md').html_url};
  assert.throws(()=>museum.validateBoundManifest(changed,f.index,f.status),/Packet commit|source URL/i);
  const promoted=structuredClone(f.manifest);promoted.packets[1].scientific_effect='ACCEPT';
  assert.throws(()=>museum.validateBoundManifest(promoted,f.index,f.status),/scientific effect/i);
});

test('rendered claims use declared HTML containers, complete identities and literal disclaimer',available,async()=>{
  const f=fixture(),document=documentFromHTML();
  const raw=JSON.stringify(f.manifest),config=JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(raw),sha256:digest(raw)}});
  const mapping=new Map([['config.json',config],['museum.json',raw],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,'Pinned proof.']]);
  await museum.startMuseum({document,search:'',fetcher:async url=>{assert.ok(mapping.has(String(url)),`Unexpected URL ${url}`);return new Response(mapping.get(String(url)));}});
  const text=document.getElementById('claim-cards').textContent;
  assert.match(text,/Object 0/);assert.match(text,/Claim and scope/);assert.match(text,/Source and review/);assert.match(text,/Replay/);assert.match(text,/Engineering — not acceptance/);
  for(const field of ['d2-lifetime-remainder','ACCEPT-scoped','d6g8k5htny-coder/Math-','proof.md','a'.repeat(40),digest('Pinned proof.')]) assert.ok(text.includes(field),field);
  assert.ok(text.includes('This canvas explains the pinned source. It is not a proof and does not change status.'));
  assert.match(document.getElementById('lifetime-fixture').textContent,/fixture absent/i);
  if(f.manifest.claims.some(claim=>claim.id==='d2-lifetime-remainder')){
    const link=descendants(document.getElementById('claim-cards'),n=>n.tagName==='A'&&n.textContent==='Dependency record for these exact bytes (Math 7858329): math.lifetime-remainder')[0];
    assert.ok(link,'The d2 card links its dependency-graph record');
    assert.equal(link.href,'dependencies.html?node=math.lifetime-remainder#node-detail');
    assert.match(text,/Same path and SHA-256 as this card's proof\. The graph's classification is the gate's own register, not this card's label\./);
  }
  assert.equal(document.getElementById('undeclared-id'),null);
  document.nodes.delete('claim-cards');
  await assert.rejects(()=>museum.startMuseum({document,search:'',fetcher:async()=>new Response(JSON.stringify(f.manifest))}),/Missing museum container/);
});

test('museum manifest must match the config byte count and SHA-256 before source projection',available,async()=>{
  const f=fixture(),raw=JSON.stringify(f.manifest),pin={url:'museum.json',bytes:Buffer.byteLength(raw),sha256:digest(raw)};
  for(const altered of [raw+' ',raw.replace('"schema_version":1','"schema_version":2')]){
    const requested=[];
    const document=documentFromHTML();
    await museum.startMuseum({document,fetcher:async url=>{requested.push(String(url));return new Response(url==='config.json'?JSON.stringify({museum_json:pin}):altered);}});
    assert.deepEqual(requested,['config.json','museum.json']);
    assert.match(document.getElementById('museum-state').textContent,/unavailable/i);
    assert.equal(document.getElementById('museum-state').className,'error');
    for(const id of ['claim-cards','packet-cards','lifetime-fixture']){const children=document.getElementById(id).children;assert.equal(children.length,1,id);assert.match(children[0].textContent,/^Not shown: .*No result inferred\. Read the pinned Math proof index at d6628da and the STATUS snapshot at f2e432e directly\.$/);assert.deepEqual([isLive(document.getElementById(id)),isLive(children[0])],[false,false],`${id} and its Not shown note are not live`);}
    assert.equal(document.getElementById('active-exhibit').children.length,0,'no exhibit was requested');
  }
  const exhibit=documentFromHTML();
  await museum.startMuseum({document:exhibit,search:'?view=ec014',fetcher:async url=>new Response(url==='config.json'?JSON.stringify({museum_json:pin}):raw+' ')});
  const hosts=['claim-cards','packet-cards','lifetime-fixture','active-exhibit'].map(id=>exhibit.getElementById(id).children);
  assert.ok(hosts.every(children=>children.length===1&&/Not shown/.test(children[0].textContent)),'a requested exhibit host carries the same sentence');
  const links=descendants(hosts[0][0],n=>n.tagName==='A').map(n=>[n.textContent,n.href]);
  assert.deepEqual(links,[['Math proof index at d6628da','https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/PROOF_INDEX.md'],['STATUS snapshot at f2e432e','https://github.com/d6g8k5htny-coder/main/blob/f2e432ea5c86742e480c66624775bc9103343314/STATUS.md']]);
});

// #museum-state is the page's one static role="status" line for cards. A refused card is not a
// second live region; each refused view is counted in that line's final text instead. The only
// other static live region is the conditional route's status line; neither is hidden.
test('museum.html declares exactly two live regions, #museum-state and #conditional-route-status, neither hidden',()=>{
  const html=fs.readFileSync(new URL('../docs/site/museum.html',import.meta.url),'utf8');
  const live=[...html.matchAll(/<[a-z][a-z0-9]*\b[^>]*>/gi)].map(m=>m[0]).filter(tag=>/\brole="(status|alert|log|marquee|timer)"|\baria-live\b/i.test(tag));
  assert.deepEqual(live,['<p id="museum-state" role="status">','<p id="conditional-route-status" role="status">']);
  const state=documentFromHTML().getElementById('museum-state');
  assert.deepEqual([state.getAttribute('role'),state.getAttribute('aria-live')],['status',null]);
});
// The route's status line stays visible, and only a line that has not refused is recoloured, so a
// refusal keeps the error colour.
test('brand.css keeps the route status line visible and in the error colour when it refuses',()=>{
  const css=fs.readFileSync(new URL('../docs/site/brand.css',import.meta.url),'utf8').replace(/\/\*[\s\S]*?\*\//g,'');
  const rules=[...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(([,selector,body])=>[selector.trim(),body]);
  const route=rules.filter(([selector])=>selector.includes('conditional-route-status'));
  assert.ok(route.length>0,'the route status line is styled');
  for(const [selector,body] of route){
    assert.doesNotMatch(body,/display\s*:\s*none|visibility\s*:\s*(hidden|collapse)|\bclip\b|opacity\s*:\s*0(?![.\d]*[1-9])/,selector);
    if(/(^|;)\s*color\s*:/.test(body))assert.equal(selector,'#conditional-route-status:not(.error)','only a line that has not refused is recoloured');
  }
  assert.ok(rules.some(([selector,body])=>selector.split(',').map(s=>s.trim()).includes('.conditional-route-refusal')&&/(^|;)\s*color\s*:\s*var\(--ul-error\)/.test(body)),'a refused line takes the error colour');
});
async function renderRefusals(search,{proofText='Pinned proof.',geometryLoader,packet=false}={}){
  const f=fixture(),document=documentFromHTML();
  if(packet){// A landed packet whose pinned result cannot be fetched.
    const id='side24-identity-replay-20260926',commit='71400b94f6cb354a8cf7aba73ffede2138a64efa',at=`d6g8k5htny-coder/main/${commit}/incoming/${id}/`;
    const source=name=>({...pin(`incoming/${id}/${name}`,'Unfetched packet bytes.',commit),repository:'d6g8k5htny-coder/main',url:`https://raw.githubusercontent.com/${at}${name}`,html_url:`https://github.com/${at.replace('/main/','/main/blob/')}${name}`});
    f.manifest.packets=[{id,issue:null,result:source('RESULT.md'),identity:source('IDENTITY.json'),output:source('output.json'),scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'}];
  }
  const raw=JSON.stringify(f.manifest);
  const mapping=new Map([['config.json',JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(raw),sha256:digest(raw)}})],['museum.json',raw],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,proofText],[f.manifest.claims[11].review.url,'Pinned reconciliation.']]);
  await museum.startMuseum({document,search,...(geometryLoader?{geometryLoader}:{}),fetcher:async url=>new Response(mapping.get(String(url))??'',{status:mapping.has(String(url))?200:404})});
  const views=['claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(id=>document.getElementById(id));
  const refused=views.flatMap(view=>descendants(view,n=>n.className==='museum-card refused'));
  const flagged=n=>isLive(n)||(typeof n.getAttribute==='function'&&n.getAttribute('role')!==null);
  const roled=views.flatMap(view=>[...(flagged(view)?[view]:[]),...descendants(view,flagged)]);
  const line=document.getElementById('museum-state');
  return {state:line.textContent,stateLive:[line.getAttribute('role'),line.getAttribute('aria-live')],refused,roled};
}
const countedViews=n=>`Pinned source projections verified. ${n} source view(s) unavailable; no result inferred for those views.`;
test('refused cards carry no role and are not live; the museum status line counts every one of them',available,async()=>{
  const page=await renderRefusals('',{proofText:'Changed proof.',packet:true});
  assert.ok(page.refused.length>2,'claim cards, the D2 fixture and the packet are refused');
  assert.equal(page.refused.filter(n=>n.children[0].textContent==='Packet source').length,1,'the packet whose source fails is refused');
  const packetCard=page.refused.find(n=>n.children[0].textContent==='Packet source');
  assert.equal(packetCard.id,'side24-identity-replay-20260926','the refused packet card carries its packet id');
  assert.ok(packetCard.children.some(n=>n.tagName==='P'&&n.textContent==='Packet ID: side24-identity-replay-20260926'),'the refused packet card names its packet');
  assert.deepEqual(page.roled.map(n=>n.textContent),[],'refused cards and their notes are not live regions');
  for(const card of page.refused)assert.ok(card.children.some(n=>n.tagName==='P'&&n.className==='error'&&/^Unavailable: .+\. No result inferred\.$/.test(n.textContent)));
  assert.equal(page.state,countedViews(page.refused.length));
  assert.deepEqual(page.stateLive,['status',null],'the counting line is still the static status line');
});
test('an unknown exhibit is counted in the museum status line like any other unavailable view',available,async()=>{
  const page=await renderRefusals('?view=unknown');
  assert.deepEqual(page.refused.map(n=>n.children[0].textContent),['Unknown exhibit']);
  assert.equal(page.state,countedViews(1));
  assert.deepEqual(page.stateLive,['status',null],'the counting line is still the static status line');
  assert.deepEqual(page.roled.map(n=>n.textContent),[],'the refusal note is not a live region');
});
test('a declared exhibit that cannot be shown is counted in the museum status line',available,async()=>{
  const page=await renderRefusals('?view=remote',{geometryLoader:()=>{throw Error('Graphics unavailable');}});
  assert.deepEqual(page.refused.map(n=>n.children[0].textContent),['Source illustration']);
  assert.equal(page.state,countedViews(1));
  assert.deepEqual(page.stateLive,['status',null],'the counting line is still the static status line');
  assert.deepEqual(page.roled.map(n=>n.textContent),[],'the refusal note is not a live region');
});

test('the dependency-record join map binds each card to its graph node by exact path and SHA-256, never by title',available,()=>{
  const manifest=JSON.parse(fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url),'utf8'));
  const graph=JSON.parse(fs.readFileSync(new URL('../docs/site/dependency-source/GRAPH.json',import.meta.url),'utf8'));
  const claims=new Map(manifest.claims.map(claim=>[claim.id,claim]));
  const entries=Object.entries(museum.DEPENDENCY_RECORDS);
  assert.equal(entries.length,6);assert.equal(entries.flatMap(([,ids])=>ids).length,7);
  for(const [claimId,ids] of entries){
    const claim=claims.get(claimId);assert.ok(claim,`unknown card ${claimId}`);
    for(const id of ids){
      const node=graph.nodes[id];assert.ok(node,`unknown node ${id}`);
      assert.equal(node.source,claim.proof.path,id);
      assert.equal(node.fingerprint,claim.proof.sha256,id);
    }
  }
  // Path-only coincidences stay out of the map: the graph node's fingerprint is not this card's proof bytes.
  for(const absent of ['d5-height-window-annulus','d6-p15-full-price']){
    assert.ok(!Object.hasOwn(museum.DEPENDENCY_RECORDS,absent),absent);
    const samePath=Object.values(graph.nodes).filter(node=>node.source===claims.get(absent).proof.path);
    assert.ok(samePath.length>0&&samePath.every(node=>node.fingerprint!==claims.get(absent).proof.sha256),absent);
  }
});

test('a coherent old browser cache cannot hide the newly pinned packet',{skip:!museum||!process.env.MUSEUM_FIXTURE},async()=>{
  const current=fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url),'utf8');
  const latest=JSON.parse(current),old=structuredClone(latest);old.packets.pop();
  const oldRaw=JSON.stringify(old),oldConfig=JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(oldRaw),sha256:digest(oldRaw)}});
  const newConfig=fs.readFileSync(new URL('../docs/site/config.json',import.meta.url),'utf8');
  const fixtures=JSON.parse(fs.readFileSync(process.env.MUSEUM_FIXTURE,'utf8'));
  const document=documentFromHTML(),calls=[];
  await museum.startMuseum({document,fetcher:async(url,options)=>{
    calls.push({url:String(url),cache:options.cache});
    if(url==='config.json')return new Response(options.cache==='no-store'?newConfig:oldConfig);
    if(url==='museum.json')return new Response(options.cache==='no-store'?current:oldRaw);
    assert.ok(Object.hasOwn(fixtures,String(url)),`Unexpected source request: ${url}`);
    return new Response(Buffer.from(fixtures[String(url)],'base64'));
  }});
  assert.equal(document.getElementById('packet-cards').children.length,2);
  assert.match(document.getElementById('packet-cards').textContent,/side24-chart-claude-20260926/);
  assert.deepEqual(calls.slice(0,2),[{url:'config.json',cache:'no-store'},{url:'museum.json',cache:'no-store'}]);
  assert.ok(calls.slice(2).every(call=>call.cache===undefined),'Pinned remote source requests retain their cache policy');
});

test('a mixed new config and stale museum manifest fails closed',{skip:!museum},async()=>{
  const current=JSON.parse(fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url)));
  const old=structuredClone(current);old.packets.pop();
  const document=documentFromHTML(),calls=[];
  await museum.startMuseum({document,fetcher:async(url,options)=>{
    calls.push(String(url));
    return new Response(url==='config.json'?fs.readFileSync(new URL('../docs/site/config.json',import.meta.url)):JSON.stringify(old));
  }});
  assert.deepEqual(calls,['config.json','museum.json']);
  assert.match(document.getElementById('museum-state').textContent,/unavailable.*byte count|unavailable.*SHA-256/i);
  for(const id of ['packet-cards','claim-cards']){const children=document.getElementById(id).children;assert.equal(children.length,1,id);assert.match(children[0].textContent,/Not shown/);assert.ok(!/Claim and scope|packet — not STATUS/.test(children[0].textContent),'no card content is drawn from unverified bytes');}
});

test('actual pinned museum renders all cards and rejects cross-claim source, scope and replay substitutions',{skip:!museum||(!process.env.MUSEUM_MATH_ROOT&&!process.env.MUSEUM_FIXTURE)},async()=>{
  const manifest=JSON.parse(fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url)));
  const root=new URL('..',import.meta.url).pathname;
  const fixtures=process.env.MUSEUM_FIXTURE?JSON.parse(fs.readFileSync(process.env.MUSEUM_FIXTURE,'utf8')):null;
  const read=source=>{if(fixtures){assert.ok(Object.hasOwn(fixtures,source.url),`Missing pinned source fixture: ${source.url}`);return Buffer.from(fixtures[source.url],'base64');}return execFileSync('git',['-C',source.repository.endsWith('/Math-')?process.env.MUSEUM_MATH_ROOT:root,'show',`${source.commit}:${source.path}`],{maxBuffer:3*1024*1024});};
  const index=read(manifest.index_source).toString('utf8'),status=read(manifest.status_source).toString('utf8');
  const fetcher=async url=>{
    if(url==='config.json')return new Response(fs.readFileSync(new URL('../docs/site/config.json',import.meta.url)));
    if(url==='museum.json')return new Response(fs.readFileSync(new URL('../docs/site/museum.json',import.meta.url)));
    if(fixtures){assert.ok(Object.hasOwn(fixtures,String(url)),`Unexpected request: ${url}`);return new Response(Buffer.from(fixtures[String(url)],'base64'));}
    const match=String(url).match(/^https:\/\/raw\.githubusercontent\.com\/(d6g8k5htny-coder\/(?:Math-|main))\/([0-9a-f]{40})\/(.+)$/);
    assert.ok(match,`Unexpected request: ${url}`);
    return new Response(read({repository:match[1],commit:match[2],path:decodeURIComponent(match[3])}));
  };
  const document=documentFromHTML();
  await museum.startMuseum({document,fetcher,search:'',geometryLoader:()=>{throw Error('Home must not load a graphics context');}});
  assert.match(document.getElementById('museum-state').textContent,/displayed source bytes verified/,document.getElementById('packet-cards').textContent);
  assert.equal(document.getElementById('claim-cards').children.length,14);
  assert.match(document.getElementById('packet-cards').textContent,/packet — not STATUS/);
  const proof=structuredClone(manifest);proof.claims[3].proof=proof.claims[10].proof;
  assert.throws(()=>museum.validateBoundManifest(proof,index,status),/own pinned source quote/);
  const scope=structuredClone(manifest);scope.claims[3].status_quote=scope.claims[10].status_quote;
  assert.throws(()=>museum.validateBoundManifest(scope,index,status),/STATUS scope/);
  const replay=structuredClone(manifest.claims[0]);replay.replay.command='python invented-solver.py';
  await assert.rejects(()=>museum.verifyClaim(replay,fetcher),/Replay command.*recorded exactly/);
});

function descendants(node, predicate) {
  return (node.children??[]).flatMap(child => typeof child === 'object'
    ? [...(predicate(child) ? [child] : []), ...descendants(child, predicate)] : []);
}
async function renderQuotationFixture() {
  const f=fixture();
  const previous=f.manifest.claims[0].scope_quote;
  const quote='- Object 0: [proof](proof.md); [first review](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-123) records **ACCEPT** only for `O(1)`. No numerical constant.';
  f.manifest.claims[0].scope_quote=quote;
  f.index=f.index.replace(previous,quote);f.manifest.index_source=pin('PROOF_INDEX.md',f.index);
  f.manifest.claims[12].proof=f.manifest.index_source;
  const document=documentFromHTML(),raw=JSON.stringify(f.manifest);
  const mapping=new Map([['config.json',JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(raw),sha256:digest(raw)}})],['museum.json',raw],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,'Pinned proof.'],[f.manifest.claims[11].review.url,'Pinned reconciliation.']]);
  await museum.startMuseum({document,search:'',fetcher:async url=>new Response(mapping.get(String(url))??'',{status:mapping.has(String(url))?200:404})});
  const cards=descendants(document.getElementById('claim-cards'),n=>n.tagName==='ARTICLE');
  return {f,quote,cards};
}
test('verified claim quotes expose readable source and exact review citations',available,async()=>{
  const {cards}=await renderQuotationFixture();
  const readable=descendants(cards[0],n=>n.className==='source-quote source-quote-readable')[0];
  assert.ok(readable,'The claim must have a readable quote projection');
  assert.equal(readable.textContent,'- Object 0: proof; first review records ACCEPT only for O(1). No numerical constant.');
  const links=descendants(readable,n=>n.tagName==='A');
  assert.deepEqual(links.map(n=>[n.textContent,n.href]),[
    ['proof',`https://github.com/d6g8k5htny-coder/Math-/blob/${'a'.repeat(40)}/proof.md`],
    ['first review','https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-123']]);
  assert.equal(descendants(readable,n=>n.tagName==='CODE')[0].textContent,'O(1)');
  assert.equal(descendants(readable,n=>n.tagName==='STRONG')[0].textContent,'ACCEPT');
});
test('readable presentation retains the exact original source quote in a disclosure',available,async()=>{
  const {quote,cards}=await renderQuotationFixture();
  const originals=descendants(cards[0],n=>n.className==='source-quote source-quote-original');
  assert.equal(originals.length,1,'The original quote must remain available once');
  assert.equal(originals[0].textContent,quote);
  const details=descendants(cards[0],n=>n.tagName==='DETAILS'&&n.children.some(c=>c.textContent==='Exact source quote'));
  assert.equal(details.length,1,'The literal source quote has a named native disclosure');
  assert.ok(details[0].children.includes(originals[0]));
});

// The museum CI integration suite includes source-quote projection controls.
import "./test_source_quote.mjs";

// Packet cards keep manifest order whatever order their RESULT.md bytes arrive in: the first
// declared packet is held until the second has rendered, and a refusal keeps its own slot.
async function packetOrderPage({failFirst=false,failSecond=false}={}){
  const f=fixture(),ids=['side24-identity-replay-20260926','side24-chart-claude-20260926'],commits=['71400b94f6cb354a8cf7aba73ffede2138a64efa','a12c178c0f857a130cf434e9efd44233a038195b'];
  const text='Scientific effect: **NONE**\nReview status: **REVIEW_REQUIRED**\n';
  const source=(i,name)=>{const path=`incoming/${ids[i]}/${name}`;return {...pin(path,text,commits[i]),repository:'d6g8k5htny-coder/main',url:`https://raw.githubusercontent.com/d6g8k5htny-coder/main/${commits[i]}/${path}`,html_url:`https://github.com/d6g8k5htny-coder/main/blob/${commits[i]}/${path}`};};
  f.manifest.packets=ids.map((id,i)=>({id,issue:i?141:null,result:source(i,'RESULT.md'),identity:source(i,'IDENTITY.json'),output:source(i,'output.json'),scientific_effect:'NONE',review_status:'REVIEW_REQUIRED'}));
  const raw=JSON.stringify(f.manifest),document=documentFromHTML();
  const mapping=new Map([['config.json',JSON.stringify({museum_json:{url:'museum.json',bytes:Buffer.byteLength(raw),sha256:digest(raw)}})],['museum.json',raw],[f.manifest.index_source.url,f.index],[f.manifest.status_source.url,f.status],[f.manifest.claims[0].proof.url,'Pinned proof.'],[f.manifest.claims[11].review.url,'Pinned reconciliation.'],[f.manifest.packets[0].result.url,failFirst?text+'changed':text],[f.manifest.packets[1].result.url,failSecond?text+'changed':text]]);
  let release,secondServed;const gate=new Promise(resolve=>release=resolve),served=new Promise(resolve=>secondServed=resolve);
  const loading=museum.startMuseum({document,search:'',fetcher:async url=>{
    url=String(url);
    if(url===f.manifest.packets[0].result.url){await served;await new Promise(resolve=>setTimeout(resolve,0));await gate;}
    const response=new Response(mapping.get(url)??'',{status:mapping.has(url)?200:404});
    if(url===f.manifest.packets[1].result.url)secondServed();
    return response;
  }});
  // Release the first packet only once the second packet's card (rendered or refused) is in the page.
  const host=document.getElementById('packet-cards'),deadline=Date.now()+5000;
  await served;while(!descendants(host,n=>n.tagName==='ARTICLE'&&n.id===ids[1]).length&&Date.now()<deadline)await new Promise(resolve=>setTimeout(resolve,5));
  const article=child=>descendants(child,n=>n.tagName==='ARTICLE')[0]??(child.tagName==='ARTICLE'?child:undefined);
  const during=host.children.map(child=>child.textContent),duringIds=host.children.map(child=>article(child)?.id);
  release();await loading;
  const slots=host.children.map(child=>article(child)??child);
  return {state:document.getElementById('museum-state').textContent,settled:host.children.map(child=>child.children.length),leftover:/Verifying/.test(host.textContent),during,duringIds,ids,slots:slots.map(node=>/refused/.test(node.className)?'refused':node.id||node.textContent),refused:slots.filter(node=>/refused/.test(node.className))};
}
test('packet cards render in manifest order when the first packet arrives last',available,async()=>{
  const page=await packetOrderPage();
  assert.deepEqual(page.duringIds,[undefined,page.ids[1]],'the second packet rendered before the first was released');
  assert.match(page.during[0],/side24-identity-replay-20260926/,'the held packet keeps its first slot while it verifies');
  assert.deepEqual(page.slots,page.ids);
});
test('a refused packet keeps its manifest slot when the other packet arrives later',available,async()=>{
  const page=await packetOrderPage({failSecond:true});
  assert.deepEqual(page.duringIds,[undefined,page.ids[1]],'the second packet was refused before the first was released');
  assert.deepEqual(page.slots,[page.ids[0],'refused']);
  assert.deepEqual(page.refused.map(node=>node.id),[page.ids[1]],'the refused card carries its packet id');
  assert.equal(page.refused[0].children[0].textContent,'Packet source');
  assert.ok(page.refused[0].children.some(n=>n.tagName==='P'&&n.className==='error'&&/^Unavailable: .+\. No result inferred\.$/.test(n.textContent)));
  assert.ok(page.refused[0].children.some(n=>n.tagName==='P'&&n.textContent==='Packet ID: side24-chart-claude-20260926'),'the refused card names its packet');
});
test('every refused packet is counted in the museum status line',available,async()=>{
  const page=await packetOrderPage({failFirst:true,failSecond:true});
  assert.deepEqual(page.slots,['refused','refused']);
  assert.deepEqual(page.refused.map(node=>node.id),page.ids);
  assert.equal(page.state,countedViews(2));
});
test('a rendered packet replaces its Verifying placeholder',available,async()=>{
  const page=await packetOrderPage();
  assert.deepEqual(page.settled,[1,1]);
  assert.equal(page.leftover,false);
});
