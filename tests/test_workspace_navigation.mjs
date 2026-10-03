import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';

// Exercise actual app startup and source validation. Only the browser and
// network boundaries are replaced; this is not a browser layout test.
const root=new URL('../',import.meta.url);
const html=fs.readFileSync(new URL('docs/site/workspace.html',root),'utf8');
const originalConfig=JSON.parse(fs.readFileSync(new URL('docs/site/config.json',root)));
const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r;});return {promise,resolve};};
let run=0;

class Element extends EventTarget {
  constructor(tag){super();this.tag=tag;this.children=[];this.attributes={};this.value='';this._text='';}
  set textContent(text){this._text=String(text);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>typeof child==='string'?child:child.textContent).join('');}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this._text='';this.children=children;}
  setAttribute(name,value){this.attributes[name]=String(value);}
}

function page(hash='#inventory',{configFailure=false,importSuccess=false,querySuccess=false}={}) {
  const config=structuredClone(originalConfig);
  const importBytes=JSON.stringify({scientific_effect:'NONE',byte_copies:Array.from({length:config.imports.count},()=>({kind:'BYTE_COPY',source_label_adopted:false}))});
  const queryBytes=JSON.stringify({math_tip:config.query.math_pin,scientific_status_authority:false});
  for(const [key,bytes] of [['imports',importBytes],['query',queryBytes]]){
    config[key].bytes=Buffer.byteLength(bytes);config[key].sha256=createHash('sha256').update(bytes).digest('hex');
  }
  const events=[],nodes=new Map(),window=new EventTarget(),listeners=new Set(),frames=[];
  window.location={hash};
  window.requestAnimationFrame=callback=>{frames.push(callback);return frames.length;};
  const flushFrames=()=>{for(const callback of frames.splice(0))callback();};
  const add=window.addEventListener.bind(window),remove=window.removeEventListener.bind(window);
  window.addEventListener=(name,callback,options)=>{listeners.add(name);add(name,callback,options);};
  window.removeEventListener=(name,callback,options)=>{listeners.delete(name);remove(name,callback,options);};
  for(const match of html.matchAll(/<([a-z][a-z0-9-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi))nodes.set(match[2],new Element(match[1]));
  const document={getElementById:id=>nodes.get(id)||null,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>({textContent:text})};
  document.activeElement=null;
  for(const [id,node] of nodes){
    node.scrollIntoView=options=>events.push({kind:'scroll',id,options,cards:nodes.get('counts').children.length});
    node.focus=options=>{document.activeElement=node;events.push({kind:'focus',id,options});};
  }
  nodes.get('dimension').value='2';
  const requested=deferred(),configuration=deferred(),statusRequested=deferred(),status=deferred(),statusRendered=deferred(),custodyRequested=deferred(),custody=deferred();
  const counts=nodes.get('counts');
  counts.append=(...children)=>{Element.prototype.append.call(counts,...children);if(counts.children.length===3)statusRendered.resolve();};
  const mapping=new Map([
    ['status.json','docs/site/status.json'],
    [config.status.url,'tests/fixtures/status_f2e432e.md'],
    [config.inventory.url,'docs/public-math/sources.json'],
    ...config.inventory.pages.map(pin=>[pin.url,pin.path]),
  ]);
  const saved=new Map(['window','document','fetch'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  globalThis.window=window;globalThis.document=document;
  const requests=[];
  globalThis.fetch=async (url,options)=>{
    requests.push({url,options});
    if(url==='config.json'){requested.resolve();await configuration.promise;return configFailure?new Response('',{status:503}):new Response(JSON.stringify(config));}
    if(url===config.status.url){statusRequested.resolve();await status.promise;}
    if(url===config.imports.url){custodyRequested.resolve();await custody.promise;return importSuccess?new Response(importBytes):new Response('',{status:503});}
    if(url===config.coefficient.url)return new Response('',{status:503});
    if(url===config.query.url)return querySuccess?new Response(queryBytes):new Response('',{status:503});
    assert.ok(mapping.has(url),`Unexpected fetch ${url}`);
    return new Response(fs.readFileSync(new URL(mapping.get(url),root)));
  };
  const loading=import(`../docs/site/app.js?workspace-navigation-test=${++run}`).finally(()=>{
    for(const [name,descriptor] of saved)if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];
  });
  return {window,document,nodes,events,listeners,frames,flushFrames,loading,requested,configuration,statusRequested,status,statusRendered,custodyRequested,custody,requests};
}

async function complete(page) {
  page.configuration.resolve();page.status.resolve();page.custody.resolve();
  await page.loading;
  page.flushFrames();
}

test('workspace refreshes configuration before validating current source pins',async()=>{
  const p=page();await complete(p);
  assert.equal(p.requests.find(row=>row.url==='config.json').options.cache,'no-store');
  assert.match(p.nodes.get('inventory-state').textContent,/2,138 matches/);
});

test('initial Library navigation is restored only after delayed layout and failed sources settle',async()=>{
  const p=page();
  await p.requested.promise;
  assert.deepEqual(p.events,[]);
  p.configuration.resolve();
  await Promise.all([p.statusRequested.promise,p.custodyRequested.promise]);
  assert.deepEqual(p.events,[],'Do not reveal before upper-page status content exists');
  p.status.resolve();
  await p.statusRendered.promise;
  assert.deepEqual(p.events,[],'A still-pending failed source can also change layout');
  p.custody.resolve();
  await p.loading;
  p.flushFrames();
  assert.equal(p.nodes.get('counts').children.length,3);
  assert.match(p.nodes.get('inventory-state').textContent,/2,138 matches/);
  assert.match(p.nodes.get('custody-note').textContent,/Unavailable:.*No result inferred/);
  assert.deepEqual(p.events.filter(event=>event.kind==='scroll'),[{kind:'scroll',id:'inventory',options:{block:'start',behavior:'instant'},cards:3}]);
  assert.equal(p.document.activeElement,p.nodes.get('inventory'));
  assert.equal(p.nodes.get('inventory').tabIndex,-1);
  assert.deepEqual(p.events.filter(event=>event.kind==='focus'),[{kind:'focus',id:'inventory',options:{preventScroll:true}}]);
  assert.equal(p.listeners.size,0,'Remove temporary observers when rendering finishes');
});

test('existing section bookmarks all receive keyboard focus after rendering',async()=>{
  for(const id of ['board','coefficient','contribute']){
    const p=page('#'+id);await complete(p);
    assert.equal(p.document.activeElement,p.nodes.get(id));
    assert.equal(p.events.filter(event=>event.kind==='scroll')[0].id,id);
  }
});

test('empty, unknown and malformed fragments never trigger a correction',async()=>{
  for(const hash of ['', '#main-content', '#other', '#%E0%A4%A', '#inventory/../', '#https://evil.test']){
    const p=page(hash);await complete(p);
    assert.deepEqual(p.events,[]);assert.equal(p.listeners.size,0);
  }
});

for(const event of ['wheel','touchstart','touchmove','keydown','pointerdown','pointermove','focusin','hashchange','popstate','pagehide']){
  test(`reader ${event} during loading prevents delayed scrolling and focus stealing`,async()=>{
    const p=page();await p.requested.promise;
    const interaction=new Event(event);if(event==='pointermove')interaction.buttons=1;
    p.window.dispatchEvent(interaction);
    await complete(p);
    assert.deepEqual(p.events,[]);assert.equal(p.listeners.size,0);
  });
}

test('unpressed pointer movement while sources load does not cancel Library navigation',async()=>{
  const p=page();await p.requested.promise;
  const hover=new Event('pointermove');hover.buttons=0;p.window.dispatchEvent(hover);
  await complete(p);assert.equal(p.document.activeElement,p.nodes.get('inventory'));
});

test('a changed fragment is not restored even without a hashchange event',async()=>{
  const p=page();await p.requested.promise;p.window.location.hash='#coefficient';
  await complete(p);assert.deepEqual(p.events,[]);
});

test('changing a fragment and returning to it leaves the pending correction cancelled',async()=>{
  const p=page();await p.requested.promise;
  p.window.location.hash='#coefficient';p.window.dispatchEvent(new Event('hashchange'));
  p.window.location.hash='#inventory';p.window.dispatchEvent(new Event('hashchange'));
  await complete(p);assert.deepEqual(p.events,[]);
});

test('browser fragment scrolling and layout shifts do not count as reader intent',async()=>{
  const p=page();await p.requested.promise;p.window.dispatchEvent(new Event('scroll'));
  await complete(p);assert.equal(p.document.activeElement,p.nodes.get('inventory'));
});

test('a refused configuration still reveals the requested unavailable section',async()=>{
  const p=page('#inventory',{configFailure:true});await complete(p);
  assert.match(p.nodes.get('inventory-state').textContent,/Unavailable: Shop config unavailable/);
  assert.equal(p.document.activeElement,p.nodes.get('inventory'));
  assert.equal(p.listeners.size,0);
});

test('a missing target is harmless after loading and removes temporary observers',async()=>{
  const p=page();await p.requested.promise;p.nodes.delete('inventory');
  await complete(p);assert.deepEqual(p.events,[]);assert.equal(p.listeners.size,0);
});

test('reader movement between source settlement and final rendering prevents correction',async()=>{
  const p=page();
  p.configuration.resolve();p.status.resolve();p.custody.resolve();await p.loading;
  p.window.dispatchEvent(new Event('wheel'));p.flushFrames();
  assert.deepEqual(p.events,[]);assert.equal(p.listeners.size,0);
});

test('an import refusal does not leave the separate query source loading forever',async()=>{
  const p=page();await complete(p);
  assert.match(p.nodes.get('query-note').textContent,/Unavailable:.*503.*No result inferred/);
  assert.doesNotMatch(p.nodes.get('query-note').textContent,/Loading/);
  assert.equal(p.nodes.get('query-identity').children.length,0);
});

test('a config refusal settles every pending source note',async()=>{
  const p=page('',{configFailure:true});await complete(p);
  assert.match(p.nodes.get('query-note').textContent,/Unavailable: Shop config unavailable/);
});

test('a query refusal preserves verified import custody information',async()=>{
  const p=page('',{importSuccess:true});await complete(p);
  assert.match(p.nodes.get('custody-note').textContent,/9 byte-copy imports landed/);
  assert.match(p.nodes.get('query-note').textContent,/Unavailable:.*503/);
});

test('a successful query remains usable when the unrelated import source fails',async()=>{
  const p=page('',{querySuccess:true});await complete(p);
  assert.match(p.nodes.get('custody-note').textContent,/Unavailable:.*503/);
  assert.match(p.nodes.get('query-note').textContent,/performs no live tip check/);
  assert.match(p.nodes.get('query-identity').textContent,/c88768bb11efd1f7d6bda188f13064bedec54a06/);
});
