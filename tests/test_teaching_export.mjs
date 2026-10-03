import test from 'node:test';
import assert from 'node:assert/strict';

let feature = {};
try { feature = await import('../docs/site/teaching-export.mjs'); } catch {}
const api = name => { assert.equal(typeof feature[name], 'function', `${name} is available`); return feature[name]; };
const ns = 'http://www.w3.org/2000/svg';
const styles = {fill:'rgb(239, 245, 253)',stroke:'none','fill-opacity':'1','stroke-opacity':'1',opacity:'1','stroke-width':'1px','stroke-dasharray':'none','stroke-linecap':'butt','stroke-linejoin':'miter','font-family':'system-ui, sans-serif','font-size':'21px','font-weight':'400','font-style':'normal','text-anchor':'start'};
const expected = [
  {tag:'title',attributes:{id:'sample-title'},text:'Current <figure> & title'},
  {tag:'desc',attributes:{id:'sample-desc'},text:'Current description'},
  {tag:'circle',attributes:{cx:'200',cy:'120',r:'27',class:'diagram-point'}},
  {tag:'text',attributes:{x:'200',y:'160',class:'','text-anchor':'middle'},text:'Current label'}
];
const options = () => ({id:'sample',viewBox:'0 0 600 300',titleId:'sample-title',descriptionId:'sample-desc',expectedNodes:structuredClone(expected),metadata:{source:'<source> & "identity"'},footerLines:['Teaching figure · non-certifying','Only current displayed geometry.']});
const record = () => ({attributes:{id:'sample',viewBox:'0 0 600 300',role:'img','aria-labelledby':'sample-title sample-desc'},background:'rgb(7, 17, 31)',nodes:expected.map(node=>({...structuredClone(node),style:['title','desc'].includes(node.tag)?{}:{...styles}}))});
function fakeSVG() {
  const root = record();
  const node = value => ({localName:value.tag,namespaceURI:ns,attributes:Object.entries(value.attributes).map(([name,value])=>({name,value})),textContent:value.text||'',childNodes:value.text?[{nodeType:3,textContent:value.text}]:[],style:{...styles,...value.style,display:'block',visibility:'visible'}});
  return {...node({tag:'svg',attributes:root.attributes}),children:root.nodes.map(node),childNodes:[],style:{...styles,display:'block',visibility:'visible','background-color':root.background},getClientRects:()=>[{width:600,height:300}]};
}
const computedStyle = node => ({getPropertyValue:key => node.style[key] || 'none'});

test('sanitized export keeps current labels, concrete styles, opaque background and safely escaped metadata', () => {
  const svg = api('serializeTeachingSVG')(record(),options());
  assert.match(svg,/<svg[^>]*viewBox="0 0 600 370"/);
  assert.match(svg,/<rect[^>]*width="600"[^>]*height="370"[^>]*fill="rgb\(7, 17, 31\)"/);
  assert.match(svg,/<circle[^>]*cx="200"[^>]*cy="120"[^>]*r="27"[^>]*fill="rgb\(239, 245, 253\)"/);
  assert.match(svg,/Current &lt;figure&gt; &amp; title/);
  assert.match(svg,/&lt;source&gt; &amp; \\&quot;identity\\&quot;/);
  assert.match(svg,/Teaching figure · non-certifying/);
  assert.doesNotMatch(svg,/\sclass=|\sstyle=|var\(|<script|foreignObject|\shref=|url\(/i);
  const light=record(); light.background='#ffffff'; light.nodes[3].style.fill='#102c4d';
  assert.match(api('serializeTeachingSVG')(light,options()),/fill="#102c4d"/);
});

test('exact expected nodes reject missing geometry, stale coordinates, labels, classes and root state', () => {
  const serialize = api('serializeTeachingSVG');
  for (const change of [r=>r.nodes.pop(),r=>r.nodes[2].attributes.cx='201',r=>r.nodes[3].text='Old label',r=>r.nodes[2].attributes.class='diagram-grid',r=>r.attributes.viewBox='0 0 600 350',r=>r.attributes.hidden='']) {
    const bad=record(); change(bad); assert.throws(()=>serialize(bad,options()),/refused/i);
  }
});

test('unsupported resources and unbounded or invisible styles refuse serialization', () => {
  const serialize = api('serializeTeachingSVG');
  for (const change of [r=>r.nodes.push({tag:'script',attributes:{},style:{},text:'evil()'}),r=>r.nodes[2].attributes.onload='evil()',r=>r.nodes[2].attributes.href='https://evil.test',r=>r.nodes[2].style.fill='url(https://evil.test)',r=>r.nodes[2].style['font-family']='url(font)',r=>r.nodes[2].style.opacity='0',r=>r.nodes[2].style['stroke-width']='601px',r=>r.nodes[2].style.filter='blur(2px)',r=>r.background='rgba(0, 0, 0, 0)',r=>r.nodes[3].style['font-size']='2px']) {
    const bad=record(); change(bad); assert.throws(()=>serialize(bad,options()),/refused/i);
  }
  const unsafe=options(); unsafe.expectedNodes[2].attributes.r='Infinity'; const bad=record(); bad.nodes[2].attributes.r='Infinity';
  assert.throws(()=>serialize(bad,unsafe),/refused/i);
});

test('capture reads displayed primitives and their concrete computed styles', () => {
  const capture = api('captureTeachingDiagram'); const svg=fakeSVG();
  const actual=capture(svg,computedStyle);
  assert.deepEqual(actual.attributes,record().attributes);
  assert.equal(actual.background,'rgb(7, 17, 31)');
  assert.equal(actual.nodes[2].style.fill,'rgb(239, 245, 253)');
  assert.equal(actual.nodes[3].text,'Current label');
  assert.match(api('serializeTeachingSVG')(actual,options()),/Current label/);
});

test('capture refuses hidden root or primitives, nested content, effects and non-SVG children', () => {
  const capture=api('captureTeachingDiagram');
  for(const change of [s=>s.style.display='none',s=>s.style.visibility='hidden',s=>s.style.opacity='0',s=>s.getClientRects=()=>[],s=>s.children[2].style.display='none',s=>s.children[2].style.visibility='hidden',s=>s.children[2].style.filter='url(effect)',s=>s.children[2].childNodes.push({nodeType:1}),s=>s.children[2].namespaceURI='http://www.w3.org/1999/xhtml',s=>s.childNodes.push({nodeType:8,textContent:'comment'})]) {
    const svg=fakeSVG(); change(svg); assert.throws(()=>capture(svg,computedStyle),/refused/i);
  }
});

test('capture refuses CSS numeric geometry that overrides current circle and rectangle attributes',()=>{
  const capture=api('captureTeachingDiagram');
  const current=fakeSVG(); Object.assign(current.children[2].style,{cx:'200px',cy:'120px',r:'27px'});
  assert.equal(capture(current,computedStyle).nodes[2].attributes.r,'27');
  for(const [property,value] of [['cx','201px'],['cy','121px'],['r','28px']]){
    const svg=fakeSVG();svg.children[2].style[property]=value;assert.throws(()=>capture(svg,computedStyle),/geometry|refused/i);
  }
  for(const [property,value] of [['x','201px'],['y','121px'],['width','51px'],['height','61px'],['rx','6px'],['ry','2px']]){
    const svg=fakeSVG(),rect=svg.children[2];rect.localName='rect';rect.attributes=Object.entries({x:'200',y:'120',width:'50',height:'60',rx:'5',class:'diagram-box'}).map(([name,value])=>({name,value}));rect.style[property]=value;
    assert.throws(()=>capture(svg,computedStyle),/geometry|refused/i);
  }
});

test('capture token-compares computed path geometry and refuses hidden or overridden paths',()=>{
  const capture=api('captureTeachingDiagram');
  const d='M24 40H380V302H24Z M202 55a117 117 0 1 0 0 234a117 117 0 1 0 0 -234Z';
  function pathSVG(computed){const svg=fakeSVG(),path=svg.children[2];path.localName='path';path.attributes=Object.entries({d,'fill-rule':'evenodd',class:'diagram-fill'}).map(([name,value])=>({name,value}));path.style.d=computed;return svg;}
  assert.equal(capture(pathSVG('path("M 24 40 H 380 V 302 H 24 Z M 202 55 a 117 117 0 1 0 0 234 a 117 117 0 1 0 0 -234 Z")'),computedStyle).nodes[2].attributes.d,d);
  assert.equal(capture(pathSVG('path("M 24 40 H 380 V 302 H 24 Z M 202 55 A 117 117 0 1 0 202 289 A 117 117 0 1 0 202 55 Z")'),computedStyle).nodes[2].attributes.d,d);
  for(const computed of ['path("M0 0L2 2")','none'])assert.throws(()=>capture(pathSVG(computed),computedStyle),/geometry|refused/i);
});

test('manual fallback exposes complete current text and blocks unavailable clipboard writes', async () => {
  const controller=api('createFigureCitationController')({buildCitation:p=>`Current ${p.r}`});
  controller.setFigure({r:.5}); assert.equal(controller.state().text,'Current 0.5');
  assert.equal(controller.state().manualFallback,true); assert.equal(controller.state().canCopy,false);
  assert.equal(await controller.copy(),false); assert.match(controller.state().status,/manual/i);
  controller.clear(); assert.equal(controller.state().text,''); assert.equal(await controller.copy(),false); assert.match(controller.state().status,/unavailable/i);
});

test('pending copies serialize writes and never claim success after control changes', async () => {
  let finish; const writes=[];
  const controller=api('createFigureCitationController')({buildCitation:p=>`Current ${p.r}`,writeText:value=>{writes.push(value);return new Promise(resolve=>{finish=resolve;});}});
  controller.setFigure({r:.5}); const pending=controller.copy();
  controller.setFigure({r:.25}); assert.equal(await controller.copy(),false); assert.deepEqual(writes,['Current 0.5']);
  finish(); assert.equal(await pending,false); assert.equal(controller.state().text,'Current 0.25'); assert.match(controller.state().status,/changed.*copy again/i); assert.equal(controller.state().canCopy,true);
});

test('refusal and failed citation builds invalidate pending writes without reviving cleared fallback text', async () => {
  for (const rejects of [false,true]) {
    let finish;
    const controller=api('createFigureCitationController')({buildCitation:p=>{if(p.bad)throw Error('refused');return `Current ${p.r}`;},writeText:()=>new Promise((resolve,reject)=>{finish=rejects?()=>reject(Error('denied')):resolve;})});
    controller.setFigure({r:.5}); const pending=controller.copy();
    assert.throws(()=>controller.setFigure({bad:true}),/refused/);
    assert.equal(controller.state().text,''); finish(); assert.equal(await pending,false);
    assert.equal(controller.state().pending,false); assert.equal(controller.state().canCopy,false); assert.match(controller.state().status,/unavailable/i); assert.doesNotMatch(controller.state().status,/copied|manually below/i);
  }
});

test('current clipboard success and denial report only the current figure state', async () => {
  const changes=[];
  const controller=api('createFigureCitationController')({buildCitation:p=>`Current ${p.r}`,writeText:async()=>{},onChange:s=>changes.push(s)});
  controller.setFigure({r:.25}); assert.equal(await controller.copy(),true); assert.match(controller.state().status,/copied figure citation/i);
  const denied=api('createFigureCitationController')({buildCitation:p=>`Current ${p.r}`,writeText:async()=>{throw Error('denied');}});
  denied.setFigure({r:.25}); assert.equal(await denied.copy(),false); assert.equal(denied.state().manualFallback,true); assert.equal(denied.state().text,'Current 0.25'); assert.match(denied.state().status,/denied|could not/i);
  assert.equal(changes.at(-1).pending,false);
});
