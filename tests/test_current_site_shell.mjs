import test from 'node:test';
import assert from 'node:assert/strict';
// Node has no DOM. This bounded fixture exercises the real shell's state,
// attributes and event handlers; it does not model CSS, native tabbing or AT.
let createAtlasShell,viewForViewport;
try {({createAtlasShell,viewForViewport}=await import('../docs/site/atlas-shell.mjs'));}
catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
class Node extends EventTarget {
  constructor(tag,doc){super();this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.attributes=new Map();this.hidden=false;this.textContent='';this.tabIndex=-1;}
  get parentElement(){return this.parentNode;}
  get id(){return this.getAttribute('id')||'';}set id(value){this.setAttribute('id',value);}
  setAttribute(key,value){this.attributes.set(key,String(value));}
  getAttribute(key){return this.attributes.get(key)??null;}
  append(...nodes){for(const node of nodes){node.remove();node.parentNode=this;this.children.push(node);}}
  remove(){if(this.parentNode){const siblings=this.parentNode.children;siblings.splice(siblings.indexOf(this),1);this.parentNode=null;}}
  insertBefore(node,before){node.remove();node.parentNode=this;const i=this.children.indexOf(before);assert.ok(i>=0,'insertBefore anchor must belong to parent');this.children.splice(i,0,node);}
  contains(node){return this===node||this.children.some(child=>child.contains(node));}
  focus(){this.ownerDocument.activeElement=this;}
  scrollIntoView(){this.scrolled=true;}
  getClientRects(){return this.hidden?[]:[{}];}
  find(id){return this.id===id?this:this.children.map(child=>child.find(id)).find(Boolean);}
}
function fixture(search='',afterChange=()=>{}) {
  assert.equal(typeof createAtlasShell,'function','Atlas shell behavior has not been implemented');
  const doc={activeElement:null,createElement(tag){return new Node(tag,this);}};
  const root=doc.createElement('section');root.id='current-atlas';
  const layout=doc.createElement('div'),browse=doc.createElement('div'),inspector=doc.createElement('article'),diagnostics=doc.createElement('details');
  layout.setAttribute('class','atlas-layout');inspector.id='atlas-inspector';inspector.tabIndex=-1;inspector.textContent='Pinned source and five actual axes';
  diagnostics.textContent='Actual mapping gaps';root.append(layout,diagnostics);layout.append(browse,inspector);
  const media=new EventTarget();media.matches=false;
  const location={href:'https://example.invalid/site/dependencies.html'+search};
  const writes=[],moves=[];
  const history={back(){moves.push('back');},forward(){moves.push('forward');}};
  const shell=createAtlasShell({root,inspector,diagnostics,media,location,history,onChange(change){if(change?.write!==false)writes.push('change');afterChange({root,inspector,browse,change});}});
  return {root,layout,browse,inspector,diagnostics,media,location,history,writes,moves,shell,doc,get:id=>root.find(id)};
}
function key(node,value){const event=new Event('keydown',{cancelable:true});Object.defineProperty(event,'key',{value});node.dispatchEvent(event);return event;}
function click(node,extra={}){const event=new Event('click',{cancelable:true});for(const [key,value]of Object.entries({button:0,...extra}))Object.defineProperty(event,key,{value});node.dispatchEvent(event);return event;}

test('narrow startup and history fallback use List while an explicit Graph choice survives',()=>{
  assert.equal(typeof viewForViewport,'function');
  for(const [search,parsed,narrow,want]of [['','graph',true,'list'],['?q=old','graph',true,'list'],['?atlas-view=graph','graph',true,'graph'],['?atlas-view=list','list',false,'list'],['','graph',false,'graph']])assert.equal(viewForViewport(search,parsed,narrow),want);
});
test('implicit viewport defaults adapt in both directions and unknown view values are not preferences',()=>{
  for(const [search,previous,narrow,want]of [['','graph',true,'list'],['','list',false,'graph'],['?atlas-view=unknown','graph',true,'list'],['?atlas-view=unknown','list',false,'graph'],['?atlas-view=graph','graph',true,'graph'],['?atlas-view=list','list',false,'list']])assert.equal(viewForViewport(search,previous,narrow),want);
});
test('enhancement moves the exact admitted DOM into stable labelled panels without a history write',()=>{
  const f=fixture('?atlas-panel=diagnostics');
  assert.equal(f.diagnostics.parentElement,f.get('atlas-panel-diagnostics'));
  assert.equal(f.inspector.parentElement,f.get('atlas-panel-evidence'));
  assert.equal(f.inspector.textContent,'Pinned source and five actual axes');
  assert.equal(f.diagnostics.textContent,'Actual mapping gaps');
  for(const name of ['evidence','diagnostics']){const tab=f.get('atlas-tab-'+name),panel=f.get('atlas-panel-'+name);assert.equal(tab.getAttribute('role'),'tab');assert.equal(tab.getAttribute('aria-controls'),panel.id);assert.equal(panel.getAttribute('role'),'tabpanel');assert.equal(panel.getAttribute('aria-labelledby'),tab.id);assert.equal(panel.tabIndex,0);}
  assert.deepEqual(f.writes,[]);assert.equal(f.get('atlas-tab-diagnostics').tabIndex,0);assert.equal(f.get('atlas-panel-evidence').hidden,true);
});
test('horizontal arrow keys wrap focus without changing the selected panel or URL',()=>{
  const f=fixture(),evidence=f.get('atlas-tab-evidence'),diagnostics=f.get('atlas-tab-diagnostics');evidence.focus();
  assert.equal(key(evidence,'ArrowRight').defaultPrevented,true);assert.equal(f.doc.activeElement,diagnostics);
  assert.equal(diagnostics.getAttribute('aria-selected'),'false');assert.equal(evidence.getAttribute('aria-selected'),'true');assert.deepEqual(f.writes,[]);
  key(diagnostics,'ArrowRight');assert.equal(f.doc.activeElement,evidence);key(evidence,'End');assert.equal(f.doc.activeElement,diagnostics);key(diagnostics,'Home');assert.equal(f.doc.activeElement,evidence);
  assert.equal(key(evidence,'ArrowDown').defaultPrevented,false);
});
test('activating a tab changes the visible panel exactly once and preserves one tab stop',()=>{
  const f=fixture();click(f.get('atlas-tab-diagnostics'));
  assert.equal(f.get('atlas-panel-evidence').hidden,true);assert.equal(f.get('atlas-panel-diagnostics').hidden,false);assert.equal(f.get('atlas-tab-diagnostics').tabIndex,0);assert.equal(f.get('atlas-tab-evidence').tabIndex,-1);assert.deepEqual(f.writes,['change']);
  click(f.get('atlas-tab-diagnostics'));assert.deepEqual(f.writes,['change']);
});
test('node selection reveals Evidence before focus even when Diagnostics was selected',()=>{
  const f=fixture('?atlas-panel=diagnostics');f.media.matches=true;f.shell.selectNode();f.shell.focusReading();
  assert.equal(f.get('atlas-panel-evidence').hidden,false);assert.equal(f.root.getAttribute('data-atlas-panel'),'evidence');assert.equal(f.doc.activeElement,f.inspector);assert.deepEqual(f.writes,[]);
});
test('restoring absent or unknown panel values returns to browse without rewriting the address',()=>{
  const f=fixture('?atlas-panel=diagnostics');
  for(const search of ['', '?atlas-panel=constructor','?atlas-panel=__proto__']){f.shell.restore('https://example.invalid/'+search);assert.equal(f.root.getAttribute('data-atlas-panel'),'browse');assert.equal(f.get('atlas-tab-evidence').getAttribute('aria-selected'),'true');}
  assert.deepEqual(f.writes,[]);
});
test('layout URLs preserve all scientific pins, historical filters and independent searches',()=>{
  const f=fixture('?atlas-panel=diagnostics');const source='https://example.invalid/dependencies.html?snapshot=wrong&main-snapshot=refused&atlas-node=constructor&q=old&classification=OPEN&source-q=source&atlas-q=graph&equation-q=ell&equation=e#search-heading';
  const u=new URL(source);f.shell.decorateURL(u);const expected=new URL(source);expected.searchParams.set('atlas-panel','diagnostics');assert.equal(u.href,expected.href);
  f.shell.updateLinks(u);const browse=new URL(f.get('atlas-browse-link').href);assert.equal(browse.href,new URL(source).href);assert.deepEqual(f.writes,[]);
});
test('unmodified direct browse activation exits reading and focuses the browse entry point',()=>{
  const f=fixture('?atlas-panel=evidence');f.media.matches=true;
  const event=click(f.get('atlas-browse-link'));assert.equal(event.defaultPrevented,true);assert.equal(f.root.getAttribute('data-atlas-panel'),'browse');assert.equal(f.doc.activeElement,f.get('atlas-open-reader'));assert.deepEqual(f.writes,['change']);
  const u=new URL('https://example.invalid/?atlas-node=n&atlas-view=graph&atlas-panel=evidence');f.shell.decorateURL(u);assert.equal(u.searchParams.get('atlas-panel'),null);assert.equal(u.searchParams.get('atlas-view'),'graph');
});
test('modified browse clicks retain native open-link behavior',()=>{
  const f=fixture('?atlas-panel=diagnostics');for(const extra of [{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{button:1}])assert.equal(click(f.get('atlas-browse-link'),extra).defaultPrevented,false);
  assert.deepEqual(f.writes,[]);assert.equal(f.root.getAttribute('data-atlas-panel'),'diagnostics');
});
test('reader Back and Forward use ordinary browser history without creating entries',()=>{
  const f=fixture();click(f.get('atlas-reader-back'));click(f.get('atlas-reader-forward'));assert.deepEqual(f.moves,['back','forward']);assert.deepEqual(f.writes,[]);
});
test('restoring history moves focus out of the now hidden reading or browse region',()=>{
  const f=fixture('?atlas-panel=evidence');f.media.matches=true;f.inspector.focus();f.shell.restore('https://example.invalid/',{focus:true});assert.equal(f.doc.activeElement,f.get('atlas-open-reader'));
  f.shell.restore('https://example.invalid/?atlas-panel=diagnostics',{focus:true});assert.equal(f.doc.activeElement,f.get('atlas-panel-diagnostics'));assert.deepEqual(f.writes,[]);
});
test('changing viewport repairs focus only when the currently focused browse content is hidden',()=>{
  const f=fixture('?atlas-panel=evidence');const button=f.doc.createElement('button');f.browse.append(button);button.focus();f.media.matches=true;f.media.dispatchEvent(new Event('change'));assert.equal(f.doc.activeElement,f.inspector);assert.deepEqual(f.writes,[]);
  const outside=f.doc.createElement('button');outside.focus();f.media.dispatchEvent(new Event('change'));assert.equal(f.doc.activeElement,outside);
});
test('the browse reading button enters Evidence and reset removes only presentation state',()=>{
  const f=fixture();f.media.matches=true;click(f.get('atlas-open-reader'));assert.equal(f.root.getAttribute('data-atlas-panel'),'evidence');assert.equal(f.doc.activeElement,f.inspector);assert.deepEqual(f.writes,['change']);f.shell.reset();assert.equal(f.root.getAttribute('data-atlas-panel'),'browse');assert.deepEqual(f.writes,['change']);
});
test('widening the viewport moves focus off mobile-only reader and browse controls',()=>{
  for(const id of ['atlas-reader-back','atlas-reader-forward','atlas-browse-link','atlas-open-reader']){
    const f=fixture(id==='atlas-open-reader'?'':'?atlas-panel=evidence');f.media.matches=true;f.get(id).focus();f.media.matches=false;f.media.dispatchEvent(new Event('change'));
    assert.equal(f.doc.activeElement.id,'atlas-tab-evidence',id+' retained hidden focus');assert.deepEqual(f.writes,[]);
  }
});
test('resize repairs focus after the renderer removes a focused source or graph child',()=>{
  for(const [panel,region,want]of [['evidence','inspector','atlas-inspector'],['browse','browse','atlas-tab-evidence']]){
    let button;const f=fixture('?atlas-panel='+panel,({change})=>{if(change?.write===false)button.remove();});button=f.doc.createElement('button');f[region].append(button);button.focus();f.media.matches=false;f.media.dispatchEvent(new Event('change'));
    assert.equal(f.doc.activeElement.id,want);assert.equal(f.root.contains(f.doc.activeElement),true);assert.deepEqual(f.writes,[]);
  }
});
test('resize recovers a mobile control blurred by CSS before the media change event',()=>{
  for(const [hidden,related,want]of [[true,null,'atlas-tab-evidence'],[false,null,'fixture-body'],[true,'outside','outside']]){
    const f=fixture('?atlas-panel=evidence');f.media.matches=true;
    const back=f.get('atlas-reader-back');back.getClientRects=()=>hidden?[]:[{}];
    f.doc.body=f.doc.createElement('body');f.doc.body.id='fixture-body';
    const outside=f.doc.createElement('button');outside.id='outside';
    back.focus();f.doc.activeElement=related?outside:f.doc.body;
    const event=new Event('focusout');Object.defineProperty(event,'target',{value:back});Object.defineProperty(event,'relatedTarget',{value:related?outside:null});f.root.dispatchEvent(event);
    f.media.matches=false;f.media.dispatchEvent(new Event('change'));
    assert.equal(f.doc.activeElement.id,want);assert.deepEqual(f.writes,[]);
  }
});
test('implicit graph-to-list resize repairs a connected hidden stage and preserves outside focus',()=>{
  for(const outsideFocus of [false,true]){
    let stage;const f=fixture('',({change})=>{if(change?.write===false)stage.hidden=true;});
    stage=f.doc.createElement('div');stage.id='atlas-stage';f.browse.append(stage);
    const outside=f.doc.createElement('button');outside.id='outside';
    (outsideFocus?outside:stage).focus();f.media.matches=true;f.media.dispatchEvent(new Event('change'));
    assert.equal(f.doc.activeElement,outsideFocus?outside:f.get('atlas-open-reader'));
    assert.ok(f.doc.activeElement.getClientRects().length);assert.deepEqual(f.writes,[]);
  }
});
