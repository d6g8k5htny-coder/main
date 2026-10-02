import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

// Run the actual app against original hash-verified catalog bytes. These browser
// and fetch boundary doubles do not attest to layout or full accessibility.
const root=new URL('../',import.meta.url);
const html=fs.readFileSync(new URL('docs/site/workspace.html',root),'utf8');
const config=JSON.parse(fs.readFileSync(new URL('docs/site/config.json',root)));
let run=0;
class Element extends EventTarget {
  constructor(tag,attributes){super();this.tag=tag;this.children=[];this.value='';this._text='';this.disabled=attributes.includes(' disabled');this.hidden=attributes.includes(' hidden');}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>typeof child==='string'?child:child.textContent).join('');}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this._text='';this.children=children;}
  setAttribute(name,value){this[name]=String(value);}
  removeAttribute(name){delete this[name];}
}
async function page(t,{query='',hash='#inventory',historyFailure=false,inventoryFailure=false,delay=false}={}){
  const nodes=new Map();
  for(const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi))nodes.set(match[3],new Element(match[1],match[2]));
  const window=new EventTarget(),writes=[],focus=[];
  window.location=new URL('https://example.test/site/workspace.html'+query+hash);
  window.history={state:{reader:'keep'},replaceState(state,title,url){if(historyFailure)throw new Error('History refused');writes.push({state,url});window.location=new URL(url,window.location);}};
  window.requestAnimationFrame=fn=>{fn();return 1;};
  const document={getElementById:id=>nodes.get(id)||null,createElement:tag=>new Element(tag,''),createElementNS:(_,tag)=>new Element(tag,''),createTextNode:text=>({textContent:text})};
  for(const [id,node] of nodes){node.scrollIntoView=()=>{};node.focus=()=>focus.push(id);}
  nodes.get('dimension').value='2';
  const mapping=new Map([
    ['status.json','docs/site/status.json'],[config.status.url,'tests/fixtures/status_f2e432e.md'],
    [config.inventory.url,'docs/public-math/sources.json'],...config.inventory.pages.map(pin=>[pin.url,pin.path])
  ]);
  let release;const pending=new Promise(r=>release=r);
  const saved=new Map(['window','document','fetch'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  t.after(()=>{for(const [name,value] of saved)if(value)Object.defineProperty(globalThis,name,value);else delete globalThis[name];});
  globalThis.window=window;globalThis.document=document;
  globalThis.fetch=async url=>{
    if(url==='config.json')return new Response(JSON.stringify(config));
    if(url===config.coefficient.url||url===config.imports.url)return new Response('',{status:503});
    if(url===config.inventory.url){if(delay)await pending;if(inventoryFailure)return new Response('',{status:503});}
    assert.ok(mapping.has(url),`Unexpected fetch ${url}`);return new Response(fs.readFileSync(new URL(mapping.get(url),root)));
  };
  const loading=import(`../docs/site/app.js?catalog-query-test=${++run}`);
  if(!delay)await loading;
  const change=(id,value,event='input')=>{nodes.get(id).value=value;nodes.get(id).dispatchEvent(new Event(event));};
  return {window,writes,focus,nodes,loading,release,change};
}
const text=(p,id)=>p.nodes.get(id).textContent;
const count=p=>p.nodes.get('catalog').children.length;

test('a bookmarked combined search restores the one exact proof and source link',async t=>{
  const p=await page(t,{query:'?q=PROOF.md&repository=Math-&path=coefficients%2Fside24_v1%2F'});
  assert.equal(p.nodes.get('search').value,'PROOF.md');assert.equal(count(p),1);
  assert.match(p.nodes.get('catalog').children[0].textContent,/coefficients\/side24_v1\/PROOF.md/);
  assert.match(p.nodes.get('catalog').children[0].children[0].children[1].href,/\/blob\/[0-9a-f]{40}\//);
  assert.equal(p.writes.length,0);
});
test('typing preserves unrelated parameters, fragment and history state; share link only includes filters',async t=>{
  const p=await page(t,{query:'?from=research',hash:'#coefficient'});
  p.change('search','SIDE24');assert.match(text(p,'inventory-state'),/^96 matches/);
  assert.equal(p.window.location.search,'?from=research&q=SIDE24');assert.equal(p.window.location.hash,'#coefficient');
  assert.deepEqual(p.writes[0].state,{reader:'keep'});
  const link=new URL(p.nodes.get('catalog-link').href);
  assert.equal(link.search,'?q=SIDE24');assert.equal(link.hash,'#inventory');
});
test('popstate restores filters and pagination without rewriting history or stealing focus',async t=>{
  const p=await page(t);p.nodes.get('more').dispatchEvent(new Event('click'));assert.equal(count(p),100);
  const focusBefore=p.focus.length;
  p.window.location=new URL('https://example.test/site/workspace.html?q=SIDE24&repository=Math-#board');
  p.window.dispatchEvent(new Event('popstate'));
  assert.equal(p.nodes.get('search').value,'SIDE24');assert.equal(p.nodes.get('repository-filter').value,'Math-');assert.equal(count(p),1);
  assert.equal(p.writes.length,0);assert.equal(p.focus.length,focusBefore);assert.equal(p.window.location.hash,'#board');
});
test('clear filters recovers from no matches and resets pagination with search focus',async t=>{
  const p=await page(t,{query:'?q=NO-MATCH&repository=main&path=missing&keep=1'});
  assert.equal(count(p),0);assert.match(text(p,'inventory-state'),/clear filters/i);
  p.nodes.get('catalog-clear').dispatchEvent(new Event('click'));
  assert.equal(count(p),50);assert.equal(p.nodes.get('search').value,'');assert.equal(p.nodes.get('repository-filter').value,'');assert.equal(p.nodes.get('path-filter').value,'');
  assert.equal(p.window.location.search,'?keep=1');assert.equal(p.focus.at(-1),'search');
  p.nodes.get('more').dispatchEvent(new Event('click'));assert.equal(count(p),100);
  p.nodes.get('catalog-clear').dispatchEvent(new Event('click'));assert.equal(count(p),50);
});
test('unknown repository and duplicate parameters use documented first-value rules',async t=>{
  const p=await page(t,{query:'?q=SIDE24&q=absent&repository=unknown&repository=Math-&path='});
  assert.equal(p.nodes.get('search').value,'SIDE24');assert.equal(p.nodes.get('repository-filter').value,'');assert.equal(count(p),50);
  assert.match(text(p,'inventory-state'),/^96 matches/);assert.equal(p.writes.length,0);
});
test('encoded punctuation stays inert search text',async t=>{
  const p=await page(t,{query:'?q=%3Cscript%3E%26%23%2B%25E0%25A4%25A&path=a%2Fb'});
  assert.equal(p.nodes.get('search').value,'<script>&#+%E0%A4%A');assert.equal(count(p),0);
  const url=new URL(p.nodes.get('catalog-link').href);assert.equal(url.searchParams.get('q'),'<script>&#+%E0%A4%A');assert.equal(url.searchParams.get('path'),'a/b');
});
test('catalog readiness uses the latest location while sources are still loading',async t=>{
  const p=await page(t,{query:'?q=first',delay:true});
  assert.equal(p.nodes.get('catalog-clear').disabled,true);assert.ok(!p.nodes.get('catalog-link').href);
  p.window.location=new URL('https://example.test/site/workspace.html?q=SIDE24#inventory');p.release();await p.loading;
  assert.equal(p.nodes.get('search').value,'SIDE24');assert.match(text(p,'inventory-state'),/^96 matches/);
});
test('an unavailable inventory keeps new controls unavailable and infers no results',async t=>{
  const p=await page(t,{inventoryFailure:true});
  assert.equal(p.nodes.get('catalog-clear').disabled,true);assert.ok(!p.nodes.get('catalog-link').href);
  assert.equal(p.nodes.get('search').disabled,true);assert.match(text(p,'inventory-state'),/Unavailable:.*No result inferred/);
});
test('a refused history update leaves filtering and the explicit search link usable',async t=>{
  const p=await page(t,{historyFailure:true});p.change('search','SIDE24');
  assert.match(text(p,'inventory-state'),/^96 matches/);assert.equal(new URL(p.nodes.get('catalog-link').href).searchParams.get('q'),'SIDE24');
  assert.match(text(p,'catalog-query-note'),/could not|cannot/i);
});
test('restored empty filters preserve the ordinary initial 50-row catalog',async t=>{
  const p=await page(t,{query:'?q=%20%20&repository=&path='});assert.equal(count(p),50);assert.match(text(p,'inventory-state'),/^2,138 matches/);
  assert.equal(new URL(p.nodes.get('catalog-link').href).search,'');
});

test('malformed UTF-8 query escapes remain inert and are canonically encoded in the link',async t=>{
  const p=await page(t,{query:'?q=%E0%A4%A'});
  assert.equal(p.nodes.get('search').value,'\uFFFD%A');assert.equal(count(p),0);
  assert.equal(new URL(p.nodes.get('catalog-link').href).searchParams.get('q'),'\uFFFD%A');
});
