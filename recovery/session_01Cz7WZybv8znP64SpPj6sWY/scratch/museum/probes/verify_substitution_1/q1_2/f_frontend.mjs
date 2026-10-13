// Verifier-2 for Q1-2 (frontend half). Own harness: no network (fetch stubbed to throw), sources served
// from local bytes; validateBoundManifest and startMuseum are the unmodified cd32303 museum.mjs.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const SCR='<SCRATCH>/museum';
const TREE=`${SCR}/cd32303`, FIX=`${SCR}/probes/verify_substitution_1/fixture/Math-`;
const MATH='d6628da09384728992dcbe6e921cc28ba85aebb0', MAIN='71400b94f6cb354a8cf7aba73ffede2138a64efa';
globalThis.fetch=async u=>{throw Error('NETWORK BLOCKED '+u);};
const museum=await import(`${TREE}/docs/site/museum.mjs`);
const sha=b=>createHash('sha256').update(b).digest('hex');
const blob=b=>createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`),b])).digest('hex');
const desc=(repository,commit,path,b)=>{const e=path.split('/').map(encodeURIComponent).join('/');return {repository,commit,path,blob:blob(b),bytes:b.length,sha256:sha(b),url:`https://raw.githubusercontent.com/${repository}/${commit}/${e}`,html_url:`https://github.com/${repository}/blob/${commit}/${e}`};};
const shipped=()=>JSON.parse(fs.readFileSync(`${TREE}/docs/site/museum.json`,'utf8'));
const index=fs.readFileSync(`${FIX}/PROOF_INDEX.md`), status=fs.readFileSync(`${TREE}/STATUS.md`);
const m0=shipped();
console.log('index bytes match pin:',sha(index)===m0.index_source.sha256,'| status bytes match pin:',sha(status)===m0.status_source.sha256);
const local=(repo,commit,p)=>{const f=repo==='d6g8k5htny-coder/Math-'&&commit===MATH?`${FIX}/${p}`:repo==='d6g8k5htny-coder/main'&&commit===MAIN?`${TREE}/${p}`:null;return f&&fs.existsSync(f)?fs.readFileSync(f):null;};
const fetcherFor=(manifest,extra)=>async url=>{const u=String(url);if(u==='museum.json')return new Response(JSON.stringify(manifest));if(extra.has(u))return new Response(extra.get(u));
  const r=u.match(/^https:\/\/raw\.githubusercontent\.com\/(d6g8k5htny-coder\/[^/]+)\/([0-9a-f]{40})\/(.+)$/);const b=r&&local(r[1],r[2],decodeURIComponent(r[3]));return b?new Response(b):new Response('x',{status:404});};
class El{constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';this.style={};}set textContent(v){this._t=String(v);this.children=[];}get textContent(){return this._t+this.children.map(c=>c.textContent).join('');}
  append(...c){this.children.push(...c);}replaceChildren(...c){this._t='';this.children=c;}setAttribute(k,v){this.attributes[k]=String(v);}getAttribute(k){return this.attributes[k]??null;}addEventListener(){}}
const doc=()=>{const n=new Map(['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(i=>[i,new El('div')]));return {createElement:t=>new El(t),getElementById:i=>n.get(i)??null};};
const walk=(n,f)=>{if(!n||typeof n!=='object')return;f(n);(n.children||[]).forEach(c=>walk(c,f));};
const texts=n=>{const o=[];walk(n,x=>{if(x._t)o.push(x._t);});return o;};
const cardOf=(root,id)=>{let h=null;walk(root,x=>{if(!h&&x.attributes?.['data-object-id']===id)h=x;});return h;};
const RP='reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md';
const real=fs.readFileSync(`${FIX}/${RP}`);
console.log('pinned review bytes match shipped descriptor:',sha(real)===m0.claims[3].review.sha256,'| shipped review commit == index commit:',m0.claims[3].review.commit===m0.index_source.commit);
const alt=Buffer.from(real.toString('utf8').replace('| R7 crossover, absorption, and full cover (17)–(20), (2) | **ACCEPT** |','| R7 crossover, absorption, and full cover (17)–(20), (2) | **AMEND** |'));
console.log('synthetic alternate differs:',!alt.equals(real),'| has AMEND verdict row:',alt.includes('| **AMEND** |'));
const tryV=m=>{try{museum.validateBoundManifest(m,index.toString('utf8'),status.toString('utf8'));return 'validateBoundManifest ACCEPTS';}catch(e){return 'validateBoundManifest REFUSES: '+e.message;}};
async function run(label,i,mut){
  const m=shipped();const extra=new Map();mut(m,extra);
  console.log(`\n[${label}] ${tryV(m)}`);
  const d=doc();await museum.startMuseum({document:d,fetcher:fetcherFor(m,extra),search:'',geometryLoader:()=>{throw Error('no geometry');}});
  const c=cardOf(d.getElementById('claim-cards'),m.claims[i].id);
  if(!c){console.log('  card REFUSED:',texts(d.getElementById('claim-cards')).filter(t=>/Unavailable/.test(t)).join(' | '));return;}
  const t=texts(c);const k=t.indexOf('Review source identity');
  console.log('  card:',JSON.stringify(t.filter(x=>/Byte-verified review|^ACCEPT-scoped$|^AMEND\/open$|Source label/.test(x))));
  if(k>=0)console.log('  review identity shown:',t[t.indexOf('Repository',k)+1],'@',t[t.indexOf('Commit',k)+1].slice(0,12),'| index:',m.index_source.repository,'@',m.index_source.commit.slice(0,12));
  console.log('  museum-state:',d.getElementById('museum-state').textContent);
}
await run('baseline shipped manifest',3,()=>{});
for(const [label,repo,commit] of [['other Math- commit','d6g8k5htny-coder/Math-','1'.repeat(40)],['main repository','d6g8k5htny-coder/main','2'.repeat(40)],['query- repository','d6g8k5htny-coder/query-','3'.repeat(40)]])
  await run(`review ${label}`,3,(m,extra)=>{const r=desc(repo,commit,RP,alt);m.claims[3].review=r;extra.set(r.url,alt);});
await run('control: same commit change on the PROOF',3,(m,extra)=>{const p=m.claims[3].proof;const b=fs.readFileSync(`${FIX}/${p.path}`);const r=desc(p.repository,'1'.repeat(40),p.path,b);m.claims[3].proof=r;extra.set(r.url,b);});
// Same gap on the open sard-g row: its review is bound to STATUS by path only.
const SP='reviews/sard_g_successor_a1_a6_20260926/REVIEW.md';
await run('sard-g open: review at a Math- commit instead of status snapshot',13,(m,extra)=>{const b=Buffer.from('# not the cited review\n');const r=desc('d6g8k5htny-coder/Math-','5'.repeat(40),SP,b);m.claims[13].review=r;extra.set(r.url,b);});
