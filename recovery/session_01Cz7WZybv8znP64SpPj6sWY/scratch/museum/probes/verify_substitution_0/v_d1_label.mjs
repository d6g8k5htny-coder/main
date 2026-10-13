// verify_substitution_0: independent re-check of Q1-7 against the cd32303 bytes, shipped museum.json unmodified.
// No network: global fetch throws; the probe fetcher serves only three local files, each sha256-checked by core.mjs.
import fs from 'node:fs';
const S='<SCRATCH>/museum';
const T=`${S}/cd32303`, FX=`${S}/probes/verify_substitution_0/fx`;
globalThis.fetch=async u=>{throw Error('NETWORK BLOCKED '+u);};
const museum=await import(`${T}/docs/site/museum.mjs`);
const manifestText=fs.readFileSync(`${T}/docs/site/museum.json`,'utf8'); const m=JSON.parse(manifestText);
const d1=m.claims.find(c=>c.id==='d1-parent-selection-open');
const served=new Map([[m.index_source.url,`${FX}/PROOF_INDEX.md`],[m.status_source.url,`${FX}/STATUS.md`],[d1.proof.url,`${FX}/D1_PROOF.md`]]);
const fetcher=async u=>{u=String(u); if(u==='museum.json')return new Response(manifestText); const f=served.get(u); return f?new Response(fs.readFileSync(f)):new Response('x',{status:404});};
class El{constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';}
 set textContent(v){this._t=String(v);this.children=[];} get textContent(){return this._t+this.children.map(c=>c.textContent).join('');}
 append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);}}
const ids=['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'];
const nodes=new Map(ids.map(i=>[i,new El('div')]));
const document={createElement:t=>new El(t),getElementById:i=>nodes.get(i)??null};
await museum.startMuseum({document,fetcher,search:''});
const walk=(n,f)=>{f(n);for(const c of n.children||[])walk(c,f);};
let card=null; walk(nodes.get('claim-cards'),n=>{if(!card&&n.attributes?.['data-object-id']==='d1-parent-selection-open')card=n;});
if(!card){console.log('D1 card NOT rendered; state:',nodes.get('museum-state').textContent);process.exit(1);}
const seq=[]; walk(card,n=>{if(n._t)seq.push((n.tagName==='A'?`<a href=${n.href}>`:`<${n.tagName.toLowerCase()}>`)+n._t);});
const k=seq.findIndex(x=>x.includes('Review pointer'));
console.log('[1] D1 card rendered with class:',card.attributes['data-object-class']);
console.log('[2] Source-and-review sequence from the pointer label:');for(const x of seq.slice(k,k+13))console.log('    '+x);
console.log('[3] review descriptor === status_source identity:',['repository','path','commit','sha256','url'].every(f=>d1.review[f]===m.status_source[f]),'| === index_source identity:',['repository','path','commit','sha256','url'].every(f=>d1.review[f]===m.index_source[f]));
const status=fs.readFileSync(`${FX}/STATUS.md`,'utf8'), index=fs.readFileSync(`${FX}/PROOF_INDEX.md`,'utf8');
const lab=(t)=>[...t.matchAll(/\[([^\]]+)\]\(https:\/\/github\.com\/d6g8k5htny-coder\/main\/issues\/63#issuecomment-5841830743\)/g)].map(x=>x[1]);
console.log('[4] link text for issuecomment-5841830743 in STATUS.md:',JSON.stringify(lab(status)),'| in PROOF_INDEX.md:',JSON.stringify(lab(index)));
// [5] Does the validator bind D1 to STATUS (so the index label is not what is checked)? Swap the D1 review descriptor to the index identity.
const swapped=JSON.parse(manifestText); const r=swapped.claims[11].review; Object.assign(r,{repository:m.index_source.repository,path:m.index_source.path,commit:m.index_source.commit,blob:m.index_source.blob,bytes:m.index_source.bytes,sha256:m.index_source.sha256,url:m.index_source.url,html_url:m.index_source.html_url});
try{museum.validateBoundManifest(swapped,index,status);console.log('[5] D1 review re-pointed at PROOF_INDEX identity: ACCEPTED');}catch(e){console.log('[5] D1 review re-pointed at PROOF_INDEX identity: REFUSED:',e.message);}
try{museum.validateBoundManifest(JSON.parse(manifestText),index,status);console.log('[6] shipped manifest: ACCEPTED by validateBoundManifest');}catch(e){console.log('[6] shipped: REFUSED',e.message);}
console.log('[7] museum-state:',nodes.get('museum-state').textContent);
