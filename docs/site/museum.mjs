import {renderSourceQuote} from './source-quote.mjs?site-release=e4be6f0add344bc5b2fe108a98a40ff20be9136d9eb72f92c1d978f441177d69';
import {verifiedBytes, hex40, hex64} from './core.mjs?site-release=e4be6f0add344bc5b2fe108a98a40ff20be9136d9eb72f92c1d978f441177d69';

export const DISCLAIMER='This canvas explains the pinned source. It is not a proof and does not change status.';
const REVIEWED_IDS=['d2-lifetime-remainder','d3-side24-coefficient','d4-fixed-remote-rn','d5-all-height-annulus','d5-height-window-annulus','d5-two-scale','d5-inner-belt-density','d5-fixed-transverse','cumulative-transfer-correction','p15-demand-one-counterexample','d6-p15-full-price'];
const D1_ID='d1-parent-lifetime';
const OPEN_IDS=['d5-pin-neighborhoods-open','sard-g-a1-a6-open'];
const EXHIBITS={ec014:['EC-014 pair frame','EC-014'],remote:['Fixed-remote region','D4 fixed-remote RN'],annulus:['Fixed annulus','D5 all-height fixed annulus'],p15:['P15 discrete palette','D6 P15 full price']};
const VIEW_LINKS={'d3-side24-coefficient':'workspace.html#coefficient','d4-fixed-remote-rn':'museum.html?view=remote#active-exhibit','d5-all-height-annulus':'museum.html?view=annulus#active-exhibit','d5-height-window-annulus':'museum.html?view=annulus#active-exhibit','d6-p15-full-price':'museum.html?view=p15#active-exhibit'};
// Card → dependency-graph node, joined by exact bytes: each node's source equals the card's
// proof path and its fingerprint equals the proof SHA-256 in the pinned GRAPH.json snapshot
// (Math 7858329); tests/test_museum_frontend.mjs pins the map to both payloads. The two
// path-only pairs (d5-height-window-annulus, d6-p15-full-price) are deliberately absent.
export const DEPENDENCY_RECORDS={'d2-lifetime-remainder':['math.lifetime-remainder'],'d3-side24-coefficient':['math.side24-coefficient'],'d4-fixed-remote-rn':['math.rn-fixed-remote-window','math.d5-component.remote-window-proof'],'d5-all-height-annulus':['math.d5-component.annulus-proof'],'p15-demand-one-counterexample':['math.p15-price-boundary'],'d1-parent-lifetime':['math.uniform-matrix-cap-lifetime']};
const DEPENDENCY_SNAPSHOT='Math 7858329';
const INDEX_PIN_URL='https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/PROOF_INDEX.md';
const STATUS_PIN_URL='https://github.com/d6g8k5htny-coder/main/blob/f2e432ea5c86742e480c66624775bc9103343314/STATUS.md';
const D5_OPEN_REVIEW='https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md';
const CURRENT_STATUS_URL='https://github.com/d6g8k5htny-coder/main/blob/main/STATUS.md';
const PACKET_PINS=[
  {id:'side24-identity-replay-20260926',commit:'71400b94f6cb354a8cf7aba73ffede2138a64efa',issue:null},
  {id:'side24-chart-claude-20260926',commit:'a12c178c0f857a130cf434e9efd44233a038195b',issue:141}
];
const decoder=new TextDecoder('utf-8',{fatal:true});
const nonempty=value=>typeof value==='string'&&value.trim().length>0;
const sameIdentity=(left,right)=>['repository','path','commit','blob','bytes','sha256','url','html_url'].every(key=>left[key]===right[key]);

// The deadline covers both headers and body. Responses are capped while streaming,
// before allocating a complete buffer or handing source bytes to the hash verifier.
export async function boundedFetch(url,fetcher=fetch,maxBytes=2*1024*1024,timeoutMs=20000){
  if(!Number.isSafeInteger(maxBytes)||maxBytes<1||!Number.isFinite(timeoutMs)||timeoutMs<=0)throw Error('Invalid request bound');
  const controller=new AbortController();let reader,timer;
  const cancel=()=>{controller.abort();if(reader)void reader.cancel().catch(()=>{});};
  const deadline=new Promise((_,reject)=>{timer=setTimeout(()=>{cancel();reject(Error('Source request timed out'));},timeoutMs);});
  const read=async()=>{
    const response=await fetcher(url,{credentials:'omit',redirect:'error',signal:controller.signal});
    if(!response.ok)return {ok:false,status:response.status,bytes:new Uint8Array()};
    const declared=response.headers?.get('content-length');
    if(declared!==null&&declared!==undefined&&Number(declared)>maxBytes){cancel();throw Error('Response exceeds byte limit');}
    if(!response.body)return {ok:true,status:response.status,bytes:new Uint8Array()};
    reader=response.body.getReader();const chunks=[];let total=0;
    while(true){const {done,value}=await reader.read();if(done)break;total+=value.byteLength;if(total>maxBytes){cancel();throw Error('Response exceeds byte limit');}chunks.push(value);}
    const bytes=new Uint8Array(total);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.byteLength;}
    return {ok:true,status:response.status,bytes};
  };
  try{return await Promise.race([read(),deadline]);}finally{clearTimeout(timer);}
}

export function validateDescriptor(source){
  if(!source||!/^d6g8k5htny-coder\/(main|Math-|query-)$/.test(source.repository)||!hex40.test(source.commit)||!hex40.test(source.blob)||!hex64.test(source.sha256)||!Number.isSafeInteger(source.bytes)||source.bytes<1||source.bytes>2*1024*1024||!nonempty(source.path)||source.path.split('/').some(p=>!p||p==='.'||p==='..')||/[\\\x00-\x1f\x7f]/.test(source.path))throw Error('Invalid museum source identity or path');
  const path=source.path.split('/').map(encodeURIComponent).join('/');
  const raw=`https://raw.githubusercontent.com/${source.repository}/${source.commit}/${path}`;
  const html=`https://github.com/${source.repository}/blob/${source.commit}/${path}`;
  if(source.url!==raw||source.html_url!==html)throw Error('Museum source URL does not match displayed identity');
  if(source.pointer_only===true&&((!/^https:\/\/github\.com\/d6g8k5htny-coder\/(main|Math-)\/issues\/\d+(?:#issuecomment-\d+)?$/.test(source.review_url)&&source.review_url!==D5_OPEN_REVIEW)||!nonempty(source.review_notice)))throw Error('Invalid review pointer');
  return source;
}
function section(text,heading){
  const marker=`## ${heading}\n`;
  const pieces=text.replaceAll('\r\n','\n').split(marker);
  if(pieces.length!==2)throw Error('Missing or repeated source section');
  return pieces[1].split('\n## ')[0];
}
function statusCells(row){return row.slice(1,-1).split('|').map(value=>value.trim());}
function plainTitle(value){return value.replaceAll('**','').trim();}
function sourceContainsPath(text,source){
  return text.includes(`](${source.path})`)||text.includes(source.html_url)||text.includes(`/${source.path})`);
}
function firstSourceMatches(text,source){const first=text.match(/\[[^\]]+\]\(([^)]+)\)/)?.[1];return first===source.path||first===source.html_url;}
export function validateBoundManifest(manifest,indexText,statusText){
  if(manifest?.schema_version!==1||manifest.scientific_status_authority!==false)throw Error('Invalid museum schema or status authority');
  validateDescriptor(manifest.index_source);validateDescriptor(manifest.status_source);
  if(!Array.isArray(manifest.claims)||manifest.claims.length!==14)throw Error('Claim projection count mismatch');
  const reviewed=section(indexText,'Reviewed scoped results').split('\n').filter(line=>line.startsWith('- '));
  const open=section(statusText,'AMEND / open').split('\n').filter(line=>line.startsWith('|')).slice(2);
  if(reviewed.length!==11||open.length!==2)throw Error('Pinned source projection count mismatch');
  const acceptedRows=section(statusText,'ACCEPT — scoped').split('\n').filter(line=>line.startsWith('|')).slice(2);
  const d1Rows=acceptedRows.filter(row=>plainTitle(statusCells(row)[0]).startsWith('D1 —'));
  if(d1Rows.length!==1)throw Error('Pinned STATUS must have exactly one accepted D1 row');
  const expectedIds=[...REVIEWED_IDS,D1_ID,...OPEN_IDS];
  manifest.claims.forEach((claim,i)=>{
    if(claim.id!==expectedIds[i])throw Error('Claim projection order or object identity mismatch');
    const isD1=i===11,isOpen=i>=12,quote=isOpen?open[i-12]:isD1?d1Rows[0]:reviewed[i];
    if(claim.scope_quote!==quote)throw Error('Claim quote differs from exact source projection');
    const title=isOpen||isD1?plainTitle(statusCells(quote)[0]):quote.slice(2).split(':')[0];
    if(claim.title!==title)throw Error('Claim title differs from source projection');
    const expectedClass=isOpen?'AMEND/open':i===9?'engineering-only':'ACCEPT-scoped';
    const expectedLabel=isOpen?'AMEND / open':i===9?'EXACT_COUNTEREXAMPLE':'ACCEPT';
    if(claim.class!==expectedClass||claim.source_label!==expectedLabel)throw Error('Claim class promotion or source label mismatch');
    validateDescriptor(claim.proof);
    if(isD1){
      // The accepted D1 row links the immutable parent; its descriptor is the index-pinned mirror of the same bytes.
      const parent=section(indexText,'Open or conditional results with complete proof text in GitHub').split('\n').find(line=>line.startsWith('- D1 parent lifetime theorem:'))||'';
      if(!firstSourceMatches(parent,claim.proof)||!sourceContainsPath(quote,claim.proof)||claim.proof.repository!==manifest.index_source.repository||claim.proof.commit!==manifest.index_source.commit)throw Error('D1 proof differs from its exact index pointer and STATUS row');
      if(!claim.review||claim.review.pointer_only)throw Error('D1 review pointer must be a byte-bound record');
    }else if(!isOpen){
      if(!firstSourceMatches(quote,claim.proof)||claim.proof.repository!==manifest.index_source.repository||claim.proof.commit!==manifest.index_source.commit)throw Error('Claim proof is not bound by its own pinned source quote');
    }else if(!sameIdentity(claim.proof,claim.id==='d5-pin-neighborhoods-open'?manifest.index_source:manifest.status_source))throw Error('Open claim source pointer identity differs');
    if(claim.review!==null){
      validateDescriptor(claim.review);
      if(claim.review.pointer_only){
        const pointerSource=manifest.index_source;
        const pointerText=claim.id==='d5-pin-neighborhoods-open'?section(indexText,'Open obligations without a complete proof yet').split('\n').find(line=>line.startsWith('- D5 pin neighborhoods:'))||'':quote;
        if(!pointerText.includes(claim.review.review_url)||claim.review.sha256!==pointerSource.sha256||claim.review.url!==pointerSource.url||(claim.id==='d5-pin-neighborhoods-open'&&claim.review.review_url!==D5_OPEN_REVIEW))throw Error('Review pointer is not bound by its pinned source quote');
      }else if(!sourceContainsPath(quote,claim.review))throw Error('Review file differs from the source quote');
    }
    if(isOpen&&claim.status_quote!==statusCells(quote)[1])throw Error('AMEND reason differs from source quote');
    if(isD1&&claim.status_quote!==statusCells(quote)[1])throw Error('STATUS scope quote does not match this claim');
    if(!isOpen&&!isD1){
      const statusID={0:'D2',1:'D3',2:'D4',10:'D6'}[i];
      const matching=statusID?acceptedRows.filter(row=>plainTitle(statusCells(row)[0]).startsWith(`${statusID} —`)):[];
      const expected=statusID&&matching.length===1?statusCells(matching[0])[1]:null;
      if((statusID&&matching.length!==1)||claim.status_quote!==expected)throw Error('STATUS scope quote does not match this claim');
    }
    const replay=claim.replay;
    if(!replay||!(replay.command===null||nonempty(replay.command))||!(replay.url===null||/^https:\/\/github\.com\/d6g8k5htny-coder\/(main|Math-|query-)\/blob\/[0-9a-f]{40}\//.test(replay.url))||!nonempty(replay.notice))throw Error('Invalid replay declaration');
    if(replay.source){validateDescriptor(replay.source);if(replay.url!==null&&replay.url!==replay.source.html_url)throw Error('Replay source URL differs from its identity');}
    if(replay.command&&(!replay.source||replay.source.repository!==claim.proof.repository||replay.source.commit!==claim.proof.commit||replay.checkout!==claim.proof.commit))throw Error('Replay command lacks a matching pinned checkout/source');
  });
  for(const key of [...Object.keys(EXHIBITS),'lifetime'])validateDescriptor(manifest.exhibits?.[key]);
  if(!sameIdentity(manifest.exhibits.lifetime,manifest.claims[0].proof))throw Error('Lifetime source identity differs from D2');
  if(!Array.isArray(manifest.packets)||manifest.packets.length>PACKET_PINS.length)throw Error('Unexpected packet projection');
  for(const [index,packet] of manifest.packets.entries()){
    const expected=PACKET_PINS[index];
    if(packet.id!==expected.id||packet.issue!==expected.issue||packet.scientific_effect!=='NONE'||packet.review_status!=='REVIEW_REQUIRED')throw Error('Packet has unexpected identity or scientific effect');
    for(const key of ['result','identity','output'])validateDescriptor(packet[key]);
    if(packet.result.repository!=='d6g8k5htny-coder/main'||packet.result.path!==`incoming/${packet.id}/RESULT.md`)throw Error('Packet result path differs from its identity');
    if(packet.result.commit!==expected.commit
      ||!['identity','output'].every(key=>packet[key].repository==='d6g8k5htny-coder/main'&&packet[key].commit===expected.commit&&packet[key].path===`incoming/${packet.id}/${key==='identity'?'IDENTITY.json':'output.json'}`))throw Error('Packet commit differs from its pinned main snapshot');
  }
  return manifest;
}
export async function verifyClaim(claim,fetcher=fetch){
  validateDescriptor(claim.proof);
  const proofText=decoder.decode(await verifiedBytes(claim.proof,fetcher));
  if(claim.review){validateDescriptor(claim.review);await verifiedBytes(claim.review,fetcher);}
  if(claim.replay.command&&!claim.replay.source)throw Error('Replay command has no pinned source');
  if(claim.replay.source){
    validateDescriptor(claim.replay.source);const replayText=decoder.decode(await verifiedBytes(claim.replay.source,fetcher));
    if(claim.replay.command&&!replayText.replaceAll('\r\n','\n').split('\n').some(line=>line.trim()===claim.replay.command))throw Error('Replay command is not recorded exactly in its verified source');
  }
  return {claim,proofText};
}
function element(document,tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function anchor(document,text,url){const node=element(document,'a',text);node.href=url;return node;}
function identity(document,source){
  const list=element(document,'dl',undefined,'source-identity');
  for(const [name,value] of [['Repository',source.repository],['Path',source.path],['Commit',source.commit],['SHA-256',source.sha256]])list.append(element(document,'dt',name),element(document,'dd',value));
  return list;
}
function card(document,id,title,className,source){
  const article=element(document,'article',undefined,`museum-card ${className==='AMEND/open'?'amend-open':className==='ACCEPT-scoped'?'accept-scoped':className}`);article.id=id;
  article.setAttribute('data-object-id',id);article.setAttribute('data-object-class',className);
  const header=element(document,'div',undefined,'museum-card-header');
  const label=element(document,'p',id,'object-label');label.append(element(document,'span',className,'object-class'));
  const details=element(document,'details');details.append(element(document,'summary','Source identity and verification'),identity(document,source),element(document,'p',DISCLAIMER,'canvas-disclaimer'));
  header.append(label,element(document,'h3',title),details);
  article.append(header);return article;
}
function strip(document,text){const bar=element(document,'aside',undefined,'engineering-strip');bar.append(element(document,'strong','Engineering — not acceptance'),element(document,'p',text));return bar;}
function refusal(document,title,error){const article=element(document,'article',undefined,'museum-card refused');const note=element(document,'p',`Unavailable: ${error.message}. No result inferred.`,'error');note.setAttribute('role','status');article.append(element(document,'h3',title),note);return article;}
// Absence is a sentence with a route: when the local projection cannot be verified, each
// card container says so and points at the two pinned sources instead of staying blank.
function notShown(document){
  const note=element(document,'p',undefined,'boundary');
  note.append(document.createTextNode('Not shown: the pinned source projection could not be verified (see the status above). No result inferred. Read the pinned '),anchor(document,'Math proof index at d6628da',INDEX_PIN_URL),document.createTextNode(' and the '),anchor(document,'STATUS snapshot at f2e432e',STATUS_PIN_URL),document.createTextNode(' directly.'));
  return note;
}
function renderClaim(document,claim,quoteSource){
  const article=card(document,claim.id,claim.title,claim.class,claim.proof);
  if(claim.id==='d3-side24-coefficient'){const alias=element(document,'span');alias.id='d3-side24';article.append(alias);}
  const columns=element(document,'div',undefined,'claim-columns');
  const scope=element(document,'div',undefined,'claim-column');scope.append(element(document,'h4','Claim and scope'),element(document,'p',`Source label: ${claim.source_label}`),...renderSourceQuote(document,claim.scope_quote,quoteSource));
  if(claim.class==='AMEND/open'){const later=element(document,'p','STATUS.md was revised after this snapshot. ');later.append(anchor(document,'Read the current STATUS.md ↗',CURRENT_STATUS_URL));scope.append(element(document,'p','AMEND / open at the pinned STATUS snapshot (main f2e432e, 29 September 2026): the claimed closure was not accepted there.'),later);}
  if(claim.status_quote!==null){const detail=element(document,'details');detail.append(element(document,'summary','Separate STATUS scope / reason'),element(document,'blockquote',claim.status_quote,'source-quote'));scope.append(detail);}
  const source=element(document,'div',undefined,'claim-column');source.append(element(document,'h4','Source and review'),anchor(document,claim.proof.availability?'Open pinned source contract ↗':'Open pinned full proof ↗',claim.proof.html_url),element(document,'p'));
  source.append(anchor(document,'Open raw source ↗',claim.proof.url));
  if(claim.proof.availability)source.append(element(document,'p',claim.proof.availability),element(document,'p','No landed proof was supplied for this obligation in the pinned source.'));
  if(VIEW_LINKS[claim.id])source.append(element(document,'p'),anchor(document,'Open source illustration ↗',VIEW_LINKS[claim.id]));
  if(claim.review){
    source.append(element(document,'p',claim.review.pointer_only?'Review pointer in the verified index':'Byte-verified review'),anchor(document,'Open review ↗',claim.review.review_url||claim.review.html_url));
    if(claim.review.pointer_only)source.append(element(document,'p',claim.review.review_notice));
    if(claim.id==='d5-pin-neighborhoods-open')source.append(element(document,'p','Review pointer only. At the pinned Math snapshot it targeted an unmerged pull-request head (Math-#55). Math-#55 merged on 27 September 2026, and Math- main carries the same review file (Git blob 11a6b8d). Its target is not fetched or adopted by this viewer.'));
    const details=element(document,'details');details.append(element(document,'summary',claim.review.pointer_only?'Identity of the pointer source':'Review source identity'),identity(document,claim.review));source.append(details);
  }else source.append(element(document,'p','No separate byte-frozen review descriptor is supplied. Read the review links in the exact source quote.'));
  for(const id of Object.hasOwn(DEPENDENCY_RECORDS,claim.id)?DEPENDENCY_RECORDS[claim.id]:[])source.append(anchor(document,`Dependency record for these exact bytes (${DEPENDENCY_SNAPSHOT}): ${id}`,`dependencies.html?node=${encodeURIComponent(id)}#node-detail`),element(document,'p',"Same path and SHA-256 as this card's proof. The graph's classification is the gate's own register, not this card's label."));
  const replay=element(document,'div',undefined,'claim-column');replay.append(element(document,'h4','Replay'),element(document,'p',claim.replay.notice));
  if(claim.replay.command){const pre=element(document,'pre');pre.append(element(document,'code',claim.replay.command));replay.append(pre);}
  if(claim.replay.url)replay.append(anchor(document,'Open pinned replay source ↗',claim.replay.url));
  if(claim.replay.source){const details=element(document,'details');details.append(element(document,'summary','Replay source identity'),identity(document,claim.replay.source));replay.append(details);}
  columns.append(scope,source,replay);article.append(columns,strip(document,'Source byte count and SHA-256 verified. Review wording is quoted at its declared scope. Hashes, browser controls and replay output do not change status.'));return article;
}
async function renderLifetime(document,source,fetcher){
  await verifiedBytes(validateDescriptor(source),fetcher);
  const article=card(document,'d2-lifetime-fixture','D2 lifetime source contract','engineering-only',source);
  const body=element(document,'div',undefined,'fixture-body');
  body.append(element(document,'p','Fixture absent — no pinned numeric lifetime fixture is supplied.','fixture-absent'),element(document,'p','The source gives an existential O(1) remainder for sufficiently small lifetime. It supplies no numerical remainder constant or numerical lifetime cutoff. No lifetime curve or error band is inferred.'),anchor(document,'Read the pinned D2 proof ↗',source.html_url));
  article.append(body,strip(document,'Replay unavailable until a source-bound fixture is supplied and reviewed. This is a source-contract stub.'));return article;
}
async function renderPacket(document,packet,fetcher){
  const text=decoder.decode(await verifiedBytes(packet.result,fetcher));
  if(!/Scientific effect:\s*\*\*NONE\*\*/.test(text)||!/Review status:\s*(?:\*\*REVIEW_REQUIRED\*\*|`REVIEW_REQUIRED`)/.test(text))throw Error('Packet source declarations differ');
  const article=card(document,packet.id,'Packet — not STATUS','engineering-only',packet.result),body=element(document,'div',undefined,'packet-body');
  body.append(element(document,'p','packet — not STATUS'),element(document,'p',`Packet ID: ${packet.id}`),packet.issue===null?element(document,'p','Task issue: not stated in the packet'):anchor(document,`Task issue: #${packet.issue}`,`https://github.com/d6g8k5htny-coder/main/issues/${packet.issue}`),element(document,'p','scientific_effect: NONE · review_status: REVIEW_REQUIRED'),anchor(document,'Open pinned RESULT.md ↗',packet.result.html_url));
  article.append(body,strip(document,'Landed packet bytes are visible for review. The result is not an accepted claim and is not inserted into STATUS.'));return article;
}
async function verifyStartup(fetcher){
  const freshLocal=(url,options)=>fetcher(url,{...options,cache:'no-store'});
  const configResponse=await boundedFetch('config.json',freshLocal,32*1024);if(!configResponse.ok)throw Error(`Museum config unavailable (${configResponse.status})`);
  const config=JSON.parse(decoder.decode(configResponse.bytes));
  const pin=config.museum_json;
  if(pin?.url!=='museum.json'||!hex64.test(pin.sha256)||!Number.isSafeInteger(pin.bytes)||pin.bytes<1||pin.bytes>256*1024)throw Error('Invalid museum manifest descriptor');
  const localFetch=async url=>{const response=await boundedFetch(url,freshLocal,pin.bytes);return {ok:response.ok,status:response.status,arrayBuffer:async()=>response.bytes.buffer};};
  const raw=await verifiedBytes(pin,localFetch);
  const manifest=JSON.parse(decoder.decode(raw));validateDescriptor(manifest.index_source);validateDescriptor(manifest.status_source);
  const cache=new Map();
  const cachedFetch=url=>{const key=String(url);if(!cache.has(key))cache.set(key,boundedFetch(url,fetcher));return cache.get(key).then(result=>({ok:result.ok,status:result.status,arrayBuffer:async()=>result.bytes.buffer}));};
  const [index,status]=await Promise.all([verifiedBytes(manifest.index_source,cachedFetch),verifiedBytes(manifest.status_source,cachedFetch)]);
  validateBoundManifest(manifest,decoder.decode(index),decoder.decode(status));
  return {manifest,cachedFetch};
}
// One verified config→manifest→index/status startup per fetcher. Every page module
// that shares the browser fetch reuses the same verified manifest and byte cache
// instead of repeating the central fetches. Failures are not retained, so a later
// caller re-verifies rather than inheriting a stale refusal.
const startups=new WeakMap();
export function verifiedMuseum({fetcher=globalThis.fetch}={}){
  if(typeof fetcher!=='function')throw Error('Museum fetch is unavailable');
  if(!startups.has(fetcher))startups.set(fetcher,verifyStartup(fetcher).catch(error=>{startups.delete(fetcher);throw error;}));
  return startups.get(fetcher);
}
// The fragment may arrive before source-bound cards have been verified and mounted.
export function revealClaimFragment(document,hash,allowedIds){
  let id;try{id=decodeURIComponent(hash.replace(/^#/,''));}catch{return false;}
  if(!allowedIds.includes(id))return false;
  const target=document.getElementById(id);
  if(!target?.scrollIntoView)return false;
  target.tabIndex=-1;target.scrollIntoView({block:'start'});
  target.focus?.({preventScroll:true});return true;
}
function prepareClaimFragment(document,window,currentHash) {
  const hash=currentHash();
  let id;try{id=decodeURIComponent(hash.replace(/^#/,''));}catch{return ()=>false;}
  if(![...REVIEWED_IDS,D1_ID,...OPEN_IDS].includes(id))return ()=>false;
  if(!window?.addEventListener)return allowedIds=>revealClaimFragment(document,hash,allowedIds);
  const events=['wheel','touchstart','touchmove','keydown','pointerdown','pointermove','focusin','hashchange','popstate','pagehide'];
  const options={capture:true,passive:true};
  let cancelled=false,finished=false;
  const cleanup=()=>events.forEach(event=>window.removeEventListener(event,cancel,options));
  const cancel=event=>{
    if(event.type==='pointermove'&&!event.buttons)return;
    // Native fragment focus on the requested card is not a new reader action.
    if(event.type==='focusin'&&event.target?.id===id)return;
    cancelled=true;cleanup();
  };
  events.forEach(event=>window.addEventListener(event,cancel,options));
  return allowedIds=>{
    if(finished)return false;
    finished=true;
    window.requestAnimationFrame(()=>{
      cleanup();
      if(!cancelled&&currentHash()===hash)revealClaimFragment(document,hash,allowedIds);
    });
    return true;
  };
}
export async function startMuseum({document=globalThis.document,window=globalThis.window,fetcher=globalThis.fetch,search=globalThis.location?.search||'',geometryLoader=()=>import('./geometry.mjs?site-release=e4be6f0add344bc5b2fe108a98a40ff20be9136d9eb72f92c1d978f441177d69'),currentHash=()=>globalThis.location?.hash||''}={}){
  const ids=['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'];
  const containers=Object.fromEntries(ids.map(id=>{const node=document.getElementById(id);if(!node)throw Error(`Missing museum container: ${id}`);return [id,node];}));
  const finishFragment=prepareClaimFragment(document,window,currentHash);
  let claimIds=[];
  try{
    const {manifest,cachedFetch}=await verifiedMuseum({fetcher});
    containers['museum-state'].textContent='Pinned source projections verified: 11 reviewed index entries, the separately reconciled D1 row and 2 AMEND rows. Verifying each linked source before display…';
    const placeholders=manifest.claims.map(claim=>{const host=element(document,'div');host.append(element(document,'p',`Verifying ${claim.title}…`));containers['claim-cards'].append(host);return host;});
    const jobs=manifest.claims.map(async(claim,i)=>{try{await verifyClaim(claim,cachedFetch);placeholders[i].replaceChildren(renderClaim(document,claim,REVIEWED_IDS.includes(claim.id)?manifest.index_source:manifest.status_source));}catch(error){const unavailable=refusal(document,claim.title,error);unavailable.id=claim.id;placeholders[i].replaceChildren(unavailable);return false;}return true;});
    jobs.push((async()=>{try{containers['lifetime-fixture'].replaceChildren(await renderLifetime(document,manifest.exhibits.lifetime,cachedFetch));return true;}catch(error){containers['lifetime-fixture'].replaceChildren(refusal(document,'D2 lifetime fixture',error));return false;}})());
    if(!manifest.packets.length)containers['packet-cards'].append(element(document,'p','No packet appears in this pinned projection.'));
    for(const packet of manifest.packets)jobs.push((async()=>{try{containers['packet-cards'].append(await renderPacket(document,packet,cachedFetch));return true;}catch(error){containers['packet-cards'].append(refusal(document,'Packet source',error));return false;}})());
    const kind=new URLSearchParams(search).get('view');
    if(kind){
      if(!Object.hasOwn(EXHIBITS,kind))containers['active-exhibit'].append(refusal(document,'Unknown exhibit',Error('Select one of the four declared source illustrations')));
      else jobs.push((async()=>{try{
        const source=manifest.exhibits[kind],sourceText=decoder.decode(await verifiedBytes(source,cachedFetch));
        const quote=manifest.exhibit_quotes?.[kind];
        if(!nonempty(quote)||!sourceText.replaceAll('\r\n','\n').includes(quote))throw Error('Exhibit quote differs from the pinned source');
        const [title,id]=EXHIBITS[kind],article=card(document,id,title,'illustration',source),body=element(document,'div',undefined,'museum-visual'),host=element(document,'div',undefined,'geometry-host');
        body.append(anchor(document,'Open pinned source ↗',source.html_url),host,anchor(document,'Return to all claim cards','museum.html#claims'));article.append(body,strip(document,'Illustrative coordinates and camera motion explain the pinned source. They compute no field, certificate or scientific status.'));containers['active-exhibit'].append(article);
        const module=await geometryLoader();await module.mountGeometry(host,kind,source);
        const quoted=element(document,'details');quoted.append(element(document,'summary','Exact source excerpt'),element(document,'blockquote',quote,'source-quote'));body.append(quoted);return true;
      }catch(error){containers['active-exhibit'].replaceChildren(refusal(document,'Source illustration',error));return false;}})());
    }
    const outcomes=await Promise.all(jobs),failed=outcomes.filter(value=>!value).length;
    containers['museum-state'].textContent=failed?`Pinned source projections verified. ${failed} source view(s) unavailable; no result inferred for those views.`:'Source projections and displayed source bytes verified. These checks establish byte identity, not mathematical acceptance.';
    claimIds=manifest.claims.map(claim=>claim.id);
  }catch(error){
    containers['museum-state'].textContent=`Unavailable: ${error.message}. No result inferred.`;containers['museum-state'].className='error';
    const routed=['claim-cards','lifetime-fixture','packet-cards'];if(new URLSearchParams(search).has('view'))routed.push('active-exhibit');
    for(const id of ids.slice(1))containers[id].replaceChildren(...(routed.includes(id)?[notShown(document)]:[]));
  }finally{finishFragment(claimIds);}
}
if(typeof document!=='undefined')await startMuseum();
