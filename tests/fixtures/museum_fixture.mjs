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
  // The reconciled D1 row is accepted in STATUS; its proof is the index-pinned parent and its review a byte-bound record.
  const review=pin('review.md','Pinned reconciliation.',commit);
  const d1Row='| **D1 — parent scope** | Accepted D1 scope. | [Proof](proof.md) · [Reconciliation](review.md) | Limits |';
  const d1={id:'d1-parent-lifetime',title:'D1 — parent scope',scope_quote:d1Row,source_label:'ACCEPT',class:'ACCEPT-scoped',proof,review,replay:{command:null,url:null,notice:'No executable replay fixture is supplied.'},status_quote:'Accepted D1 scope.'};
  const open = ['d5-pin-neighborhoods-open','sard-g-a1-a6-open'].map((id,i)=>({id,title:`Open ${i}`,scope_quote:`| **Open ${i}** | Still open ${i}. | [Proof](proof.md) |`,source_label:'AMEND / open',class:'AMEND/open',proof,review:null,replay:{command:null,url:null,notice:'No executable replay fixture is supplied.'},status_quote:`Still open ${i}.`}));
  const index = `# Index\n\n## Reviewed scoped results\n\n${claims.map(c=>c.scope_quote).join('\n')}\n\n## Open or conditional results with complete proof text in GitHub\n- D1 parent lifetime theorem: full proof [proof](proof.md).\n`;
  const acceptedRows=[d1Row,...[[0,'D2'],[1,'D3'],[2,'D4'],[10,'D6']].map(([i,id])=>`| **${id} — scope** | Accepted scope ${i}. | Review | Limits |`)].join('\n');
  const status = `# Status\n\n## ACCEPT — scoped\n\n| Object | Accepted scope | Source and review | Explicit limits |\n|---|---|---|---|\n${acceptedRows}\n\n## AMEND / open\n\n| Object | Current reason | Source |\n|---|---|---|\n${open.map(c=>c.scope_quote).join('\n')}\n\n## Engineering only\n`;
  const indexSource=pin('PROOF_INDEX.md',index,commit),statusSource=pin('STATUS.md',status,commit);open[0].proof=indexSource;open[1].proof=statusSource;
  return {index,status,manifest:{schema_version:1,scientific_status_authority:false,index_source:indexSource,status_source:statusSource,claims:[...claims,d1,...open],exhibits:{ec014:proof,remote:proof,annulus:proof,p15:proof,lifetime:proof},packets:[]}};
}

// A node is a live region if it has a live role or any aria-live attribute, whatever its value.
// Every "not a live region" assertion in the museum tests uses this one definition.
export const LIVE_ROLES=['status','alert','log','marquee','timer'];
const attributeOf=(node,name,property)=>(typeof node?.getAttribute==='function'?node.getAttribute(name):node?.attributes?.[name])??node?.[property]??null;
export const isLive=node=>LIVE_ROLES.includes(attributeOf(node,'role','role'))||attributeOf(node,'aria-live','ariaLive')!==null;

// Connection is modelled as in the DOM: inserting a node moves it out of its old parent, and a
// node is connected only while its parent chain reaches the page root.
const detach=node=>{const parent=node?.parentNode;if(parent){parent.children=parent.children.filter(child=>child!==node);node.parentNode=null;}};
const adopt=(parent,nodes)=>{for(const node of nodes)if(node&&typeof node==='object'){detach(node);node.parentNode=parent;}};
export class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.listeners={};this.hidden=false;this._text='';this.className='';this.parentNode=null;}
  get isConnected(){return Boolean(this.parentNode?.isConnected);}
  set textContent(value){for(const child of this.children)if(child?.parentNode===this)child.parentNode=null;this._text=String(value);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join('');}
  append(...children){adopt(this,children);this.children.push(...children);}
  replaceChildren(...children){for(const child of this.children)if(child?.parentNode===this)child.parentNode=null;adopt(this,children);this._text='';this.children=children;}
  insertBefore(node,reference){if(reference==null){this.append(node);return node;}if(!this.children.includes(reference))throw Error('NotFoundError: the reference node is not a child');adopt(this,[node]);this.children.splice(this.children.indexOf(reference),0,node);return node;}
  before(...nodes){const parent=this.parentNode;if(parent)for(const node of nodes)parent.insertBefore(node,this);}
  after(...nodes){const parent=this.parentNode;if(!parent)return;let previous=this;for(const node of nodes){adopt(parent,[node]);parent.children.splice(parent.children.indexOf(previous)+1,0,node);previous=node;}}
  remove(){detach(this);} replaceWith(...nodes){if(this.parentNode){this.after(...nodes.filter(node=>node!==this));if(!nodes.includes(this))this.remove();}}
  setAttribute(k,v){this.attributes[k]=String(v);} getAttribute(k){return Object.hasOwn(this.attributes,k)?this.attributes[k]:null;} addEventListener(k,f){this.listeners[k]=f;}
}
export function documentFromHTML(){
  const html=fs.readFileSync(new URL('../../docs/site/museum.html',import.meta.url),'utf8');
  // Each id'd element keeps its static attributes (for example role="status", or a valueless hidden) and plain static
  // text from the page, and starts connected as a child of one page root in document order
  // (nesting is not modelled). getElementById finds a node only while it is connected.
  const body=new Element('body');Object.defineProperty(body,'isConnected',{value:true});
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)].map(m=>{
    const node=new Element(m[1]);for(const [,k,v] of m[2].matchAll(/\s([a-z][a-z-]*)(?:="([^"]*)")?/gi))if(k!=='id')node.attributes[k]=v??'';
    const text=html.slice(m.index+m[0].length).match(new RegExp(`^([^<&]*)</${m[1]}>`))?.[1];if(text)node._text=text;
    body.append(node);return [m[3],node];}));
  return {nodes,body,createElement:tag=>new Element(tag),createTextNode:value=>({textContent:value}),getElementById:id=>{const node=nodes.get(id);return node?.isConnected?node:null;}};
}
