// Verifier 2 for Q1-3. Independent harness (does not import the finder's common.mjs).
// Loads docs/site/museum.mjs from the extracted cd32303 tree by path; global fetch throws;
// every served source byte string is checked here against the pin before being served.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const S='<SCRATCH>/museum';
const TREE=`${S}/cd32303`, FIX=`${S}/probes/substitution/fixture/Math-`;
globalThis.fetch=async u=>{throw Error('NETWORK BLOCKED '+u);};
const museum=await import(`${TREE}/docs/site/museum.mjs`);
const sha=b=>createHash('sha256').update(b).digest('hex');
const shippedRaw=fs.readFileSync(`${TREE}/docs/site/museum.json`);
console.log('museum.mjs sha256', sha(fs.readFileSync(`${TREE}/docs/site/museum.mjs`)));
console.log('museum.json sha256', sha(shippedRaw));
const shipped=()=>JSON.parse(shippedRaw.toString('utf8'));
const m0=shipped();
const indexB=fs.readFileSync(`${FIX}/PROOF_INDEX.md`), statusB=fs.readFileSync(`${TREE}/STATUS.md`);
console.log('PROOF_INDEX pin match', sha(indexB)===m0.index_source.sha256, '| STATUS pin match', sha(statusB)===m0.status_source.sha256);
const IDX=indexB.toString('utf8'), ST=statusB.toString('utf8');
const v=m=>{try{museum.validateBoundManifest(m,IDX,ST);return 'ACCEPT';}catch(e){return 'REFUSE: '+e.message;}};
console.log('baseline', v(shipped()));
const RE=/^https:\/\/github\.com\/d6g8k5htny-coder\/(main|Math-)\/issues\/\d+(?:#issuecomment-\d+)?$/;
const cases=[
 ['d2-lifetime-remainder','https://github.com/d6g8k5htny-coder/main/issues/6'],
 ['d3-side24-coefficient','https://github.com/d6g8k5htny-coder/main/issues/6'],
 ['d3-side24-coefficient','https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5'],
 ['d3-side24-coefficient','https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841779222'], // reconciliation comment, delimited in the same bullet
 ['d4-fixed-remote-rn','https://github.com/d6g8k5htny-coder/main/issues/7'],
 ['p15-demand-one-counterexample','https://github.com/d6g8k5htny-coder/main/issues/6'],
 ['d1-parent-selection-open','https://github.com/d6g8k5htny-coder/main/issues/6'],
 ['d1-parent-selection-open','https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-58'],
 ['d3-side24-coefficient','https://github.com/d6g8k5htny-coder/main/issues/99'],   // control: not a substring
 ['d1-parent-selection-open','https://github.com/d6g8k5htny-coder/main/issues/64'], // control: not a substring
];
for(const [id,url] of cases){
  const m=shipped(),c=m.claims.find(x=>x.id===id);c.review.review_url=url;
  console.log(`${id} -> ${url.replace('https://github.com/d6g8k5htny-coder/','')}: regex=${RE.test(url)} delimited=${c.scope_quote.includes('](' + url + ')')} substring=${c.scope_quote.includes(url)} => ${v(m)}`);
}
// pointer_only type confusion
for(const [label,mut] of [
  ['pointer_only:1, review_url:"" (no notice)',r=>({...r,pointer_only:1,review_url:''})],
  ['pointer_only:"yes", review_url:"reviews/" (non-URL, regex skipped)',r=>({...r,pointer_only:'yes',review_url:'reviews/'})],
  ['pointer_only:true, review_url:"" (control)',r=>({...r,pointer_only:true,review_url:'',review_notice:'x'})],
]){const m=shipped(),c=m.claims.find(x=>x.id==='d5-two-scale');c.review=mut(m.index_source);console.log(`d5-two-scale ${label} => ${v(m)}`);}

// Full render through startMuseum with a minimal DOM and a local, digest-checked fetcher.
class El{constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';this.href=undefined;this.className='';this.id='';}
 set textContent(x){this._t=String(x);this.children=[];} get textContent(){return this._t+this.children.map(c=>c.textContent).join('');}
 append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,x){this.attributes[k]=String(x);} getAttribute(k){return this.attributes[k]??null;} addEventListener(){} querySelector(){return null;}}
function doc(){const n=new Map(['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(i=>[i,new El('div')]));return {n,createElement:t=>new El(t),createElementNS:(_,t)=>new El(t),getElementById:i=>n.get(i)??null};}
function walk(x,f){f(x);for(const c of x.children||[])walk(c,f);}
function local(m){const all=[];const add=d=>{if(d&&d.url)all.push(d);};add(m.index_source);add(m.status_source);for(const c of m.claims){add(c.proof);add(c.review);add(c.replay?.source);}
 return async u=>{u=String(u);if(u==='museum.json')return new Response(JSON.stringify(m));const d=all.find(x=>x.url===u);if(!d)return new Response('',{status:404});
  const f=d.repository.endsWith('/Math-')?`${FIX}/${d.path}`:`${TREE}/${d.path}`;if(!fs.existsSync(f))return new Response('',{status:404});const b=fs.readFileSync(f);if(sha(b)!==d.sha256)return new Response('',{status:404});return new Response(b);};}
async function show(m,ids){const d=doc();await museum.startMuseum({document:d,fetcher:local(m),search:'',geometryLoader:()=>{throw Error('no geometry');}});
 for(const id of ids){let card=null;walk(d.n.get('claim-cards'),x=>{if(!card&&x.attributes?.['data-object-id']===id)card=x;});
  if(!card){let t='';walk(d.n.get('claim-cards'),x=>{if(x.className==='museum-card refused')t=x.textContent;});console.log(`  [${id}] NOT RENDERED`);continue;}
  const out=[];walk(card,x=>{if(x.tagName==='A'&&/review/i.test(x._t))out.push(`A(${x._t} -> ${x.href})`);else if(x.tagName==='P'&&/Review pointer|Byte-verified review|not byte-frozen|No separate/.test(x._t))out.push(`P(${x._t})`);else if(x.tagName==='P'&&x._t===''&&x.children.length===0)out.push('P(<empty>)');});
  console.log(`  [${id}] class=${card.attributes['data-object-class']} ::`, out.filter(s=>s!=='P(<empty>)'||true).join(' | '));}}
console.log('\nRENDER baseline');await show(shipped(),['d3-side24-coefficient','d1-parent-selection-open','d5-two-scale']);
{const m=shipped();m.claims.find(x=>x.id==='d3-side24-coefficient').review.review_url='https://github.com/d6g8k5htny-coder/main/issues/6';
 m.claims.find(x=>x.id==='d1-parent-selection-open').review.review_url='https://github.com/d6g8k5htny-coder/main/issues/6';
 const c=m.claims.find(x=>x.id==='d5-two-scale');c.review={...m.index_source,pointer_only:1,review_url:''};
 console.log('RENDER mutated (D3 -> issues/6, D1 -> issues/6, D5 two-scale -> pointer_only:1/"")');await show(m,['d3-side24-coefficient','d1-parent-selection-open','d5-two-scale']);}
