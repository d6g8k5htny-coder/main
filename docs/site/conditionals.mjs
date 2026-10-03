// One dependency transcription of an already-indexed proof. No status engine.
const CLAIM_ID='cumulative-transfer-correction';
const SOURCE_COMMIT='d6628da09384728992dcbe6e921cc28ba85aebb0';
const SOURCE_PATH='reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md';
const SOURCE_SHA256='83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6';
const SOURCE_BLOB='044ac5fdaf403a38e33983e31f0ad69f8e76d6d5';
const encoder=new TextEncoder(),decoder=new TextDecoder('utf-8',{fatal:true});
const hex40=v=>typeof v==='string'&&/^[0-9a-f]{40}$/.test(v);

function validateSource(source){
  if(!source||source.repository!=='d6g8k5htny-coder/Math-'||!hex40(source.commit)||source.commit!==SOURCE_COMMIT
    ||source.path!==SOURCE_PATH||source.sha256!==SOURCE_SHA256||source.blob!==SOURCE_BLOB
    ||source.bytes!==3272)throw Error('Conditional source identity mismatch');
  const suffix=`${source.repository}/${source.commit}/${SOURCE_PATH}`;
  if(source.url!==`https://raw.githubusercontent.com/${suffix}`
    ||source.html_url!==`https://github.com/${source.repository}/blob/${source.commit}/${SOURCE_PATH}`)
    throw Error('Conditional source URL identity mismatch');
  return source;
}
async function digest(algorithm,raw,crypto){
  if(!crypto?.subtle)throw Error('Source identity hashing unavailable');
  return [...new Uint8Array(await crypto.subtle.digest(algorithm,raw))].map(v=>v.toString(16).padStart(2,'0')).join('');
}
function between(text,start,end){
  if(text.split(start).length!==2)throw Error('Conditional source boundary changed');
  const tail=text.split(start)[1];
  if(tail.split(end).length!==2)throw Error('Conditional source boundary changed');
  return tail.split(end)[0];
}
function projectText(text){
  const theorem=between(text,'## Corrected cumulative theorem\n\n','\n## Proof\n').trim();
  const nodes=[{id:'setup',role:'hypothesis',quote:theorem.split('\n1. ')[0].trim()}];
  for(let i=1;i<=4;i++){
    const start=`${i}. `,end=i<4?`\n${i+1}. `:'\n\nNeither a derivative';
    nodes.push({id:`H${i}`,role:'hypothesis',quote:(start+between(theorem,start,end)).trim()});
  }
  nodes.push({id:'definition',role:'definition',quote:('    N(ell)='+between(theorem,'    N(ell)=','\n\nThen')).trim()});
  nodes.push({id:'conclusion',role:'conditional-conclusion',quote:('    lim_(ell->0)'+between(theorem,'    lim_(ell->0)','\n\nIf the displayed')).trim()});
  return {
    schema_version:1,projection_kind:'conditional-proof-route',claim_id:CLAIM_ID,
    scientific_status_authority:false,scientific_effect:'NONE',application_evaluation:'NOT_EVALUATED',
    source_sha256:SOURCE_SHA256,nodes,
    routes:[{id:'cumulative-limit',logic:'ALL',requires:['setup','H1','H2','H3','H4','definition'],concludes:'conclusion'}],
    case_split:'If the displayed'+theorem.split('If the displayed')[1],
    boundary:text.slice(text.lastIndexOf('This corrects an underspecified theorem statement;')).trim(),
  };
}
export async function projectVerified(raw,source,crypto=globalThis.crypto){
  validateSource(source);
  if(!(raw instanceof Uint8Array)||raw.byteLength!==source.bytes
    ||await digest('SHA-256',raw,crypto)!==source.sha256)throw Error('Conditional source byte identity mismatch');
  const header=encoder.encode(`blob ${raw.byteLength}\0`),blob=new Uint8Array(header.length+raw.length);
  blob.set(header);blob.set(raw,header.length);
  if(await digest('SHA-1',blob,crypto)!==source.blob)throw Error('Conditional Git blob identity mismatch');
  const identity=Object.fromEntries(['repository','path','commit','blob','bytes','sha256','url','html_url'].map(key=>[key,source[key]]));
  return {...projectText(decoder.decode(raw)),source:identity};
}
function stable(value){
  if(value===null||typeof value==='string'||typeof value==='boolean')return JSON.stringify(value);
  if(typeof value==='number'&&Number.isSafeInteger(value))return String(value);
  if(Array.isArray(value))return '['+value.map(stable).join(',')+']';
  if(value&&Object.getPrototypeOf(value)===Object.prototype)
    return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+stable(value[k])).join(',')+'}';
  throw Error('Unsupported projection value');
}
export async function verifyProjection(data,raw,source,crypto=globalThis.crypto){
  const expected=await projectVerified(raw,source,crypto);
  if(stable(data)!==stable(expected))throw Error('Conditional projection differs from the complete pinned statement');
  return expected;
}
function el(document,tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
export function renderProjection(document,host,data,source){
  const article=el(document,'article');article.className='museum-card engineering-only';
  article.setAttribute('data-object-class','engineering-only');
  article.append(el(document,'h3','Conditional route: cumulative transfer'),
    el(document,'p','ALL six inputs are required. These are hypotheses and definitions, not newly discharged obligations.'),
    el(document,'p','Application evaluation: NOT_EVALUATED. A proved implication is not proof that its hypotheses hold for a particular field.'));
  const identity=el(document,'dl');identity.className='source-identity';
  for(const key of ['repository','path','commit','sha256'])identity.append(el(document,'dt',key),el(document,'dd',source[key]));
  article.append(identity);
  const list=el(document,'ol');
  for(const node of data.nodes.filter(n=>n.id!=='conclusion')){
    const item=el(document,'li'),details=el(document,'details');
    details.append(el(document,'summary',`${node.id} — ${node.role}`),el(document,'blockquote',node.quote));
    item.append(details);list.append(item);
  }
  article.append(list,el(document,'p','setup + H1 + H2 + H3 + H4 + definition → conclusion (ALL)'),
    el(document,'pre',data.nodes.find(n=>n.id==='conclusion').quote),
    el(document,'p',data.case_split),el(document,'p',data.boundary));
  const sourceLink=el(document,'a','Open the exact source');sourceLink.href=source.html_url;
  const reviewLink=el(document,'a','Read the existing source/review card');reviewLink.href='#'+CLAIM_ID;
  article.append(sourceLink,el(document,'p'),reviewLink,
    el(document,'p','Generated from verified source bytes. This graph is not formal verification, a proof review, or a scientific-status register.'));
  host.replaceChildren(article);
}
const services=()=>import('./museum.mjs?site-release=8b15350e744dac57c6fab662a1ce7e5bb6ea730f009649479e49a4786b240b10');
// The museum module already verifies config → manifest → index/status once per
// page fetch and caches every pinned byte request. This route consumes that same
// verified startup, so it adds only the audited-descriptor check and projection.
export async function startConditionals({document=globalThis.document,fetcher=globalThis.fetch,loadServices=services}={}){
  const host=document?.getElementById('conditional-route');if(!host)return false;
  host.replaceChildren(el(document,'p','Verifying the conditional route source…'));
  try{
    const s=await loadServices();
    const {manifest,cachedFetch}=await s.verifiedMuseum({fetcher});
    const selected=manifest.claims.filter(c=>c.id===CLAIM_ID);
    if(selected.length!==1)throw Error('Conditional source claim is missing or duplicated');
    const claim=selected[0];validateSource(claim.proof);
    const {proofText}=await s.verifyClaim(claim,cachedFetch);
    const data=await projectVerified(encoder.encode(proofText),claim.proof);
    renderProjection(document,host,data,claim.proof);return true;
  }catch(error){host.replaceChildren(el(document,'p',`Unavailable: ${error.message}. No conditional result inferred.`));return false;}
}
if(typeof document!=='undefined')void startConditionals();
