// Shared synthetic museum projection for Node tests. Not a source of record.
import fs from 'node:fs';
import {createHash} from 'node:crypto';

export const digest = value => createHash('sha256').update(value).digest('hex');
export const pin = (path, value, commit='a'.repeat(40)) => ({repository:'d6g8k5htny-coder/Math-', path, commit, blob:'b'.repeat(40), bytes:Buffer.byteLength(value), sha256:digest(value), url:`https://raw.githubusercontent.com/d6g8k5htny-coder/Math-/${commit}/${path}`, html_url:`https://github.com/d6g8k5htny-coder/Math-/blob/${commit}/${path}`});
export const ids = ['d2-lifetime-remainder','d3-side24-coefficient','d4-fixed-remote-rn','d5-all-height-annulus','d5-height-window-annulus','d5-two-scale','d5-inner-belt-density','d5-fixed-transverse','cumulative-transfer-correction','p15-demand-one-counterexample','d6-p15-full-price'];
// `proofs` substitutes a real pinned descriptor for a reviewed claim; `commit` must then
// equal that descriptor's commit because claims bind to the index commit.
export function fixture({commit='a'.repeat(40),proofs={}}={}) {
  const proof = pin('proof.md','Pinned proof.',commit);
  const claims = ids.map((id,i) => {const p=proofs[id]??proof;return {id,title:`Object ${i}`,scope_quote:`- Object ${i}: [proof](${p.path}). Scope ${i}.`,source_label:i===9?'EXACT_COUNTEREXAMPLE':'ACCEPT',class:i===9?'engineering-only':'ACCEPT-scoped',proof:p,review:null,replay:{command:null,url:null,notice:'No executable replay fixture is supplied.'},status_quote:null};});
  for(const i of [0,1,2,10])claims[i].status_quote=`Accepted scope ${i}.`;
  const open = ['d1-parent-selection-open','d5-pin-neighborhoods-open','sard-g-a1-a6-open'].map((id,i)=>({id,title:`Open ${i}`,scope_quote:`| **Open ${i}** | Still open ${i}. | [Proof](proof.md) |`,source_label:'AMEND / open',class:'AMEND/open',proof,review:null,replay:{command:null,url:null,notice:'No executable replay fixture is supplied.'},status_quote:`Still open ${i}.`}));
  const index = `# Index\n\n## Reviewed scoped results\n\n${claims.map(c=>c.scope_quote).join('\n')}\n\n## Open or conditional results with complete proof text in GitHub\n- D1 parent lifetime theorem: full proof [proof](proof.md).\n`;
  const acceptedRows=[[0,'D2'],[1,'D3'],[2,'D4'],[10,'D6']].map(([i,id])=>`| **${id} — scope** | Accepted scope ${i}. | Review | Limits |`).join('\n');
  const status = `# Status\n\n## ACCEPT — scoped\n\n| Object | Accepted scope | Source and review | Explicit limits |\n|---|---|---|---|\n${acceptedRows}\n\n## AMEND / open\n\n| Object | Current reason | Source |\n|---|---|---|\n${open.map(c=>c.scope_quote).join('\n')}\n\n## Engineering only\n`;
  const indexSource=pin('PROOF_INDEX.md',index,commit),statusSource=pin('STATUS.md',status,commit);open[1].proof=indexSource;open[2].proof=statusSource;
  return {index,status,manifest:{schema_version:1,scientific_status_authority:false,index_source:indexSource,status_source:statusSource,claims:[...claims,...open],exhibits:{ec014:proof,remote:proof,annulus:proof,p15:proof,lifetime:proof},packets:[]}};
}

export class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.listeners={};this.hidden=false;this._text='';}
  set textContent(value){this._text=String(value);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join('');}
  append(...children){this.children.push(...children);} replaceChildren(...children){this._text='';this.children=children;}
  setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(k,f){this.listeners[k]=f;}
}
export function documentFromHTML(){
  const html=fs.readFileSync(new URL('../../docs/site/museum.html',import.meta.url),'utf8');
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m=>[m[2],new Element(m[1])]));
  return {nodes,createElement:tag=>new Element(tag),createTextNode:value=>({textContent:value}),getElementById:id=>nodes.get(id)??null};
}
