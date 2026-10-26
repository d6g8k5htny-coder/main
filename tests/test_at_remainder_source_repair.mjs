import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {runInNewContext} from 'node:vm';

const m=await import('../docs/site/current-model.mjs');

function fixture(){
  const commit='c'.repeat(40);
  return {
    observed_at:'2026-10-13T14:56:55Z',
    repositories:[{repository:'d6g8k5htny-coder/main',commit},{repository:'d6g8k5htny-coder/Math-',commit}],
    references:[],sources:[{repository:'d6g8k5htny-coder/Math-',path:'formal/side24.lean',commit,blob:'b'.repeat(40),bytes:12,sha256:'a'.repeat(64)}],
    equations:Array.from({length:12},(_,i)=>({id:'e'+i,text:'sigma '+i,path:'Math-/proofs/eq'+i+'.lean',repository:'d6g8k5htny-coder/Math-',commit,line_start:1,line_end:2,sha256:String(i).padStart(64,'0')})),
    graph:{identity:{repository:'d6g8k5htny-coder/Math-',commit,path:'frontiers/downstream_gate_20260925/GRAPH.json',sha256:'g'.repeat(64)},
      data:{nodes:{a:{kind:'candidate',classification:'OPEN_ACTIVE',layer:'D1',source:'formal/side24.lean'},b:{kind:'axiom',classification:'AXIOM',layer:'D0',source:'formal/side24.lean'}},
        edges:[{from:'a',to:'b',required:true,relation:'depends'}]}}
  };
}

test('dependencies.html wraps current-state and load-status live regions without role on writers',async()=>{
  const html=await readFile(new URL('../docs/site/dependencies.html',import.meta.url),'utf8');
  assert.match(html,/<div role="status"><p class="current-state">/);
  assert.doesNotMatch(html,/<p class="current-state"[^>]*\brole="/);
  assert.match(html,/<div role="status"><p id="load-status" class="muted">/);
  assert.doesNotMatch(html,/<p id="load-status"[^>]*\brole="/);
  assert.doesNotMatch(html,/<p id="load-status"[^>]*\baria-live="/);
});

test('brand and archive CSS keep visual child selectors after status wrappers',async()=>{
  const brand=await readFile(new URL('../docs/site/brand.css',import.meta.url),'utf8');
  const layout=await readFile(new URL('../docs/site/atlas-layout.css',import.meta.url),'utf8');
  assert.match(brand,/\.current-research-compact > \[role=status\] > \.current-state/);
  assert.doesNotMatch(brand,/\.current-research-compact > \.current-state\s*\{/);
  assert.match(layout,/\.atlas-archive-notices > \[role=status\] > p/);
  assert.match(layout,/\.atlas-archive-notices > p,/);
});

test('View enable rebuilds options and preserves a non-default selection',async()=>{
  const data=fixture();
  const f=await atlasController('?atlas-node=a&atlas-view=list',data);
  try{
    const view=f.get('atlas-view');
    assert.equal(view.disabled,false);
    assert.equal(view.value,'list');
    assert.deepEqual([...view.options].map(o=>[o.value,o.text]),[['graph','Graph'],['list','List']]);
    // Fresh Option nodes at enable: the pre-start options must not be retained.
    assert.equal(view.options.length,2);
  }finally{f.close();}
});

test('equation option reconcile reuses stable options and updates count without wholesale replace',async()=>{
  const data=fixture();
  const f=await atlasController('?atlas-node=a',data);
  try{
    const eq=f.get('equation-select'),count=f.get('equation-count'),search=f.get('equation-search');
    const before=[...eq.options];
    const firstCount=count.textContent;
    assert.match(firstCount,/^12 matching excerpts of 12 acquired excerpts/);
    // Typing that keeps the full match set must not churn option nodes.
    search.value='sig';search.dispatchEvent(new Event('input'));
    assert.equal(count.textContent,firstCount);
    assert.equal(eq.options.length,before.length);
    for(let i=0;i<before.length;i++)assert.equal(eq.options[i],before[i],'option node '+i+' must be reused');
    // Narrowing then widening updates count and prunes without replaceChildren.
    search.value='sigma 9';search.dispatchEvent(new Event('input'));
    assert.match(count.textContent,/^1 matching excerpts of 12 acquired excerpts/);
    assert.equal(eq.options.length,2);
    const narrow=[...eq.options];
    search.value='sigma';search.dispatchEvent(new Event('input'));
    assert.match(count.textContent,/^12 matching excerpts of 12 acquired excerpts/);
    assert.equal(eq.options[0],narrow[0]);
    assert.equal(eq.options[1],narrow[1]);
  }finally{f.close();}
});

test('formal-bridge call sites stay: createClaimSummary(selectedId) and release-key imports',async()=>{
  const source=await readFile(new URL('../docs/site/atlas.mjs',import.meta.url),'utf8');
  assert.match(source,/createClaimSummary\(data,n,selected\)/);
  assert.match(source,/from '\.\/atlas-claim-summary\.mjs\?site-release=[0-9a-f]{64}'/);
  assert.match(source,/from '\.\/current\.mjs\?site-release=[0-9a-f]{64}'/);
  assert.match(source,/from '\.\/current-model\.mjs\?site-release=[0-9a-f]{64}'/);
  assert.match(source,/from '\.\/atlas-shell\.mjs\?site-release=[0-9a-f]{64}'/);
  const layout=await readFile(new URL('../docs/site/atlas-layout.css',import.meta.url),'utf8');
  assert.match(layout,/#atlas-formal-evidence/);
  assert.match(layout,/#atlas-formal-connection/);
});

// Minimal controller harness (same boundary as test_current_site.mjs).
async function atlasController(search='',data=fixture()){
 const renderer=await import('../docs/site/current.mjs'),shell=await import('../docs/site/atlas-shell.mjs'),summary=await import('../docs/site/atlas-claim-summary.mjs');
 const source=await readFile(new URL('../docs/site/atlas.mjs',import.meta.url),'utf8');
 class Element extends EventTarget {
  constructor(tag,doc){super();this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.attributes=new Map();this.style={};this.dataset={};this.hidden=false;this.disabled=false;this._text='';this._value='';}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>typeof child==='string'?child:child.textContent).join('');}
  get id(){return this.getAttribute('id')||'';}set id(value){this.setAttribute('id',value);}
  get parentElement(){return this.parentNode;}
  setAttribute(key,value){this.attributes.set(key,String(value));}
  getAttribute(key){return this.attributes.get(key)??null;}
  get href(){return this.getAttribute('href')||'';}set href(value){this.setAttribute('href',value);}
  remove(...args){
   if(this.tagName==='SELECT'&&args.length===1&&Number.isInteger(args[0])){
    const child=this.children[args[0]];if(!child)return;child.parentNode=null;this.children.splice(args[0],1);return;
   }
   if(this.parentNode){const siblings=this.parentNode.children;siblings.splice(siblings.indexOf(this),1);this.parentNode=null;}
  }
  add(option){this.append(option);}
  append(...children){for(const child of children){if(typeof child!=='string'){child.remove();child.parentNode=this;}this.children.push(child);}}
  replaceChildren(...children){for(const child of this.children)if(typeof child!=='string')child.parentNode=null;this.children=[];this._text='';this.append(...children);}
  insertBefore(child,before){child.remove();child.parentNode=this;this.children.splice(this.children.indexOf(before),0,child);}
  contains(child){return this===child||this.children.some(node=>typeof node!=='string'&&node.contains(child));}
  querySelectorAll(selector){const matches=[];for(const child of this.children)if(typeof child!=='string'){if(selector==='a[href]'?child.tagName==='A'&&!!child.getAttribute('href'):child.tagName===selector.toUpperCase())matches.push(child);matches.push(...child.querySelectorAll(selector));}return matches;}
  get firstChild(){return this.children[0];}
  get offsetHeight(){return 140;}
  get options(){return this.children;}
  get text(){return this.textContent;}set text(value){this.textContent=value;}
  get value(){return this.tagName==='SELECT'?(this.children.find(child=>child.value===this._value)||this.firstChild)?.value||'':this._value;}
  set value(value){this._value=String(value);}
  get selectedIndex(){return this.children.findIndex(child=>child.value===this.value);}
  set selectedIndex(index){this.value=this.children[index]?.value||'';}
  focus(){this.ownerDocument.activeElement=this;}
  scrollIntoView(){}
  getClientRects(){return this.hidden?[]:[{}];}
 }
 const doc={activeElement:null,createElement(tag){return new Element(tag,this);},createElementNS(ns,tag){return this.createElement(tag);}};
 function Option(text='',value=''){const option=doc.createElement('option');option.text=text;option.value=value;return option;}
 const nodes=new Map();const add=(id,tag='div',parent)=>{const node=doc.createElement(tag);node.id=id;nodes.set(id,node);parent?.append(node);return node;};
 doc.getElementById=id=>nodes.get(id);const root=add('current-atlas'),layout=add('atlas-layout','div',root),browse=add('atlas-browse','div',layout);
 add('atlas-inspector','article',layout);add('atlas-diagnostics','ul',add('atlas-diagnostics-details','details',root));
 const html=await readFile(new URL('../docs/site/dependencies.html',import.meta.url),'utf8');
 for(const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"/g))if(!nodes.has(match[2]))add(match[2],match[1],browse);
 for(const value of ['graph','list']){const option=doc.createElement('option');option.value=value;option.text=value==='graph'?'Graph':'List';nodes.get('atlas-view').append(option);}
 for(const id of ['atlas-view','equation-select','equation-search','atlas-search'])nodes.get(id).disabled=true;
 const window=new EventTarget(),media=new EventTarget();media.matches=false;
 const location=new URL('https://example.test/main/site/dependencies.html'+search),entries=[location.href];let cursor=0;
 const history={replaceState(state,title,url){location.href=new URL(url,location.href).href;entries[cursor]=location.href;},pushState(state,title,url){location.href=new URL(url,location.href).href;entries.splice(++cursor);entries.push(location.href);},back(){if(cursor>0){location.href=entries[--cursor];window.dispatchEvent(new Event('popstate'));}},forward(){if(cursor+1<entries.length){location.href=entries[++cursor];window.dispatchEvent(new Event('popstate'));}}};
 const saved=Object.getOwnPropertyDescriptor(globalThis,'document');globalThis.document=doc;
 const bindings={...m,...shell,...summary,currentSnapshot:Promise.resolve(data),element:renderer.element,anchor:renderer.anchor,document:doc,window,history,URL,URLSearchParams,Option,matchMedia:()=>media,fetch:async()=>({ok:false})};
 Object.defineProperty(bindings,'location',{get:()=>location});
 try{
  const program=source.replace(/^import .*;\n/gm,'').replace(/^if\(typeof document[^\n]+\n?$/m,'');
  await runInNewContext(program+'\nstart();',bindings);
  return {get:id=>nodes.get(id),close(){if(saved)Object.defineProperty(globalThis,'document',saved);else delete globalThis.document;}};
 }catch(error){if(saved)Object.defineProperty(globalThis,'document',saved);else delete globalThis.document;throw error;}
}
