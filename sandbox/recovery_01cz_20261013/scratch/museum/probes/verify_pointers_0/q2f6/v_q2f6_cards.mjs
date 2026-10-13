// Verifier 1 (independent) for Q2-F6. Imports cd32303 museum.mjs read-only; serves only fixture files
// (served by URL from the cd32303 museum.json descriptors); all other fetches 404; global fetch disabled.
// A rendered (non-refused) card implies core.mjs verifiedBytes accepted size+sha256 of each served source.
import fs from 'node:fs'; import path from 'node:path'; import {pathToFileURL} from 'node:url';
const S='<SCRATCH>/museum';
const CD=S+'/cd32303', FX=S+'/probes/pointers/fixture';
globalThis.fetch=async u=>{throw Error('NETWORK DISABLED '+u)};
const museum=await import(pathToFileURL(CD+'/docs/site/museum.mjs'));
const manifest=JSON.parse(fs.readFileSync(CD+'/docs/site/museum.json','utf8'));
// Map every descriptor url in the manifest to a fixture file by repo/path.
const urlToFile=new Map();
(function collect(o){ if(o&&typeof o==='object'){ if(typeof o.url==='string'&&o.repository&&o.path) urlToFile.set(o.url, path.join(FX,o.repository.split('/')[1],o.path)); for(const v of Object.values(o)) collect(v);} })(manifest);
const served=[], missing=[];
const fetcher=async(u)=>{const k=String(u); if(k==='museum.json')return new Response(JSON.stringify(manifest)); const f=urlToFile.get(k); if(f&&fs.existsSync(f)){served.push(k);return new Response(fs.readFileSync(f));} missing.push(k); return new Response('nf',{status:404});};
class El{constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';this.className='';}
 set textContent(v){this._t=String(v);this.children=[];} get textContent(){return this._t+this.children.map(c=>c.textContent).join(' ');}
 append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){}}
const nodes=new Map(['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(i=>[i,new El('div')]));
const document={createElement:t=>new El(t),getElementById:i=>nodes.get(i)??null};
await museum.startMuseum({document,fetcher,search:'',geometryLoader:()=>{throw Error('no geometry')}});
console.log('STATE:',nodes.get('museum-state').textContent);
console.log('missing fetches:',JSON.stringify(missing));
function dump(n,depth,out){ if(!n||!n.tagName)return; const own=n._t; const tag=n.tagName.toLowerCase();
  if(tag==='a') out.push('  '.repeat(depth)+`<a href="${n.href.replace('https://github.com/d6g8k5htny-coder/','GH:')}">${own}`);
  else if(['p','h4','summary','dt','dd','strong'].includes(tag)&&own) out.push('  '.repeat(depth)+`<${tag}>${own.length>140?own.slice(0,140)+'…':own}`);
  else if(tag==='blockquote') out.push('  '.repeat(depth)+`<blockquote>[${own.length} chars]`);
  for(const c of n.children) dump(c,depth+1,out); }
const want=new Set(['d2-lifetime-remainder','d1-parent-selection-open','d5-pin-neighborhoods-open']);
let rendered=0;
for(const host of nodes.get('claim-cards').children){const art=host.children[0]; const id=art?.attributes?.['data-object-id']; if(id)rendered++; else {console.log('REFUSED CARD:',art?.textContent);continue;}
  if(!want.has(id))continue; const cols=art.children[1]; const srcCol=cols.children[1]; const out=[]; dump(srcCol,0,out); console.log('=== '+id+' : "Source and review" column'); console.log(out.join('\n'));
  // Is the binding PROOF_INDEX line text (for d5-pin) present anywhere on the card?
  if(id==='d5-pin-neighborhoods-open'){ const t=art.textContent; console.log('  card contains "AMEND review in PR55":',t.includes('AMEND review in PR55'),'| card contains review_url text:',t.includes('4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22'), '| review_url appears only as anchor href'); }
  if(id==='d1-parent-selection-open'){ const t=art.textContent; console.log('  card text mentions "verified index":',t.includes('verified index'),'| identity block path STATUS.md shown:',/STATUS\.md/.test(t)); }
}
console.log('rendered claim cards:',rendered);
