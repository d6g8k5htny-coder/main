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

function rectSVG(attributes,geometry) {
  const svg=fakeSVG(),rect=svg.children[2];
  rect.localName='rect';rect.attributes=Object.entries(attributes).map(([name,value])=>({name,value}));
  Object.assign(rect.style,geometry);return svg;
}

// A controlled native-geometry witness exercises the guard, not browser layout.
// The hosted browser harness separately records real SVGLength/getBBox values.
function heightSVG(height='11.888639999999995',displayed='11.8886px') {
  const svg=rectSVG({x:'448',y:'78',width:'113',height,class:'diagram-fill'},
    {x:'448px',y:'78px',width:'113px',height:displayed,rx:'auto',ry:'auto'});
  // The first tuple is measured Chromium parsing, which differs from JS fround.
  // Other fixture values remain explicitly simulated, not browser evidence.
  const parse=value=>value==='11.888639999999995'?11.888639450073242:Math.fround(Number(value));
  const rect=svg.children[2],native=parse(height);
  rect.height={baseVal:{value:native}};
  rect.getBBox=()=>({x:448,y:78,width:113,height:native});
  rect.ownerDocument={createElementNS(namespace,tag){
    assert.equal(namespace,ns);assert.equal(tag,'rect');
    return {height:{baseVal:{value:0}},setAttribute(key,value){assert.equal(key,'height');this.height.baseVal.value=parse(value);}};
  }};
  return svg;
}

test('rounded rectangle height requires a matching native used-height witness',()=>{
  const capture=api('captureTeachingDiagram');
  const svg=heightSVG();
  assert.equal(svg.children[2].height.baseVal.value,11.888639450073242);
  assert.equal(capture(svg,computedStyle).nodes[2].attributes.height,'11.888639999999995');
  // The original comparison still suffices without any native witness.
  const exact=heightSVG('13.4375','13.4375px');delete exact.children[2].getBBox;
  assert.equal(capture(exact,computedStyle).nodes[2].attributes.height,'13.4375');
});

test('captured Chromium attribute parsing need not equal JavaScript Float32 conversion',()=>{
  const svg=heightSVG(),rect=svg.children[2];
  assert.notEqual(rect.height.baseVal.value,Math.fround(Number(rect.attributes.find(a=>a.name==='height').value)));
  assert.equal(api('captureTeachingDiagram')(svg,computedStyle).nodes[2].attributes.height,'11.888639999999995');
});

test('native height equality cannot admit real CSS overrides sharing a rounded string',()=>{
  const capture=api('captureTeachingDiagram');
  for(const actual of [11.88861,11.88862,11.88864,11.88865,12,0]) {
    const svg=heightSVG();svg.children[2].getBBox=()=>({height:Math.fround(actual)});
    assert.throws(()=>capture(svg,computedStyle),/geometry|refused/i);
  }
  // Equal reflected/used values cannot disguise a changed source attribute.
  const wrong=heightSVG();wrong.children[2].height.baseVal.value=Math.fround(11.88861);
  wrong.children[2].getBBox=()=>({height:Math.fround(11.88861)});
  assert.throws(()=>capture(wrong,computedStyle),/geometry|refused/i);
});

test('rectangle height witness is finite, bounded and fail-closed',()=>{
  const capture=api('captureTeachingDiagram');
  for(const change of [
    r=>delete r.getBBox,r=>r.getBBox=()=>{throw Error('unavailable');},
    r=>r.getBBox=()=>null,r=>r.getBBox=()=>({}),r=>delete r.height,
    r=>r.height={baseVal:null},r=>Object.defineProperty(r,'height',{get(){throw Error('unavailable');}}),
    r=>delete r.ownerDocument,r=>r.ownerDocument.createElementNS=()=>{throw Error('unavailable');},
    r=>delete r.ownerDocument.createElementNS,
    r=>r.ownerDocument.createElementNS=()=>null,
    r=>r.ownerDocument.createElementNS=()=>({setAttribute(){throw Error('unavailable');}}),
    r=>r.ownerDocument.createElementNS=()=>({setAttribute(){}}),
    r=>r.ownerDocument.createElementNS=()=>({setAttribute(){},get height(){throw Error('unavailable');}}),
    r=>r.ownerDocument.createElementNS=()=>({setAttribute(){},height:{baseVal:{value:11.88861}}}),
    ...[NaN,Infinity,-Infinity,-1,0,601,'11.888640403747559',11.888639999999995].flatMap(value=>[
      r=>r.getBBox=()=>({height:value}),r=>r.height.baseVal.value=value,
      r=>r.ownerDocument.createElementNS=()=>({setAttribute(){},height:{baseVal:{value}}})
    ])
  ]) {
    const svg=heightSVG();change(svg.children[2]);
    assert.throws(()=>capture(svg,computedStyle),/geometry|refused/i);
  }
});

test('native representation check is bounded to positive rectangle height roundoff',()=>{
  const capture=api('captureTeachingDiagram');
  for(const [height,displayed] of [['15.99999','16px'],['16.00001','16px'],['40.12416','40.1242px'],['143.08250000000004','143.083px'],['599.99999','600px']]) {
    assert.equal(capture(heightSVG(height,displayed),computedStyle).nodes[2].attributes.height,height);
  }
  // No broad epsilon increase, arbitrary computed string, or other-axis fallback.
  for(const displayed of ['11.888638px','11.889px','12px','0px','-1px','NaN','Infinity','auto'])
    assert.throws(()=>capture(heightSVG(undefined,displayed),computedStyle),/geometry|refused/i);
  for(const height of ['0','-1','601','NaN','Infinity'])
    assert.throws(()=>capture(heightSVG(height,'11.8886px'),computedStyle),/geometry|refused/i);
  const nonnative=heightSVG();nonnative.children[2].height.baseVal.value=11.888639999999995;
  nonnative.children[2].getBBox=()=>({height:11.888639999999995});
  assert.throws(()=>capture(nonnative,computedStyle),/geometry|refused/i);
  const otherAxis=heightSVG();otherAxis.children[2].style.width='112.999px';
  assert.throws(()=>capture(otherAxis,computedStyle),/geometry|refused/i);
  assert.equal(capture(heightSVG('0','0px'),computedStyle).nodes[2].attributes.height,'0');
  assert.equal(capture(heightSVG('600','600px'),computedStyle).nodes[2].attributes.height,'600');
});

test('capture refuses auto rectangle dimensions that suppress otherwise current native geometry',()=>{
  const capture=api('captureTeachingDiagram');
  const attributes={x:'36',y:'157',width:'178',height:'90',rx:'12',class:'diagram-box'};
  for(const property of ['width','height']) {
    const svg=rectSVG(attributes,{x:'36px',y:'157px',width:'178px',height:'90px',rx:'12px',ry:'auto',[property]:'auto'});
    assert.throws(()=>capture(svg,computedStyle),/geometry|refused/i);
  }
});

test('capture compares effective auto corner radii and preserves native rounded and square rectangles',()=>{
  const capture=api('captureTeachingDiagram');
  const rounded={x:'36',y:'157',width:'178',height:'90',rx:'12',class:'diagram-box'};
  const native={x:'36px',y:'157px',width:'178px',height:'90px',rx:'12px',ry:'auto'};
  assert.equal(capture(rectSVG(rounded,native),computedStyle).nodes[2].attributes.rx,'12');
  assert.throws(()=>capture(rectSVG(rounded,{...native,rx:'auto',ry:'auto'}),computedStyle),/geometry|refused/i);
  // One auto radius uses the other numeric radius; both native and CSS forms
  // have hand-checked effective radii (12, 12) in this rectangle.
  assert.equal(capture(rectSVG(rounded,{...native,rx:'auto',ry:'12px'}),computedStyle).nodes[2].attributes.rx,'12');
  const square={x:'448',y:'78',width:'113',height:'13.4375',class:'diagram-fill'};
  const actual=capture(rectSVG(square,{x:'448px',y:'78px',width:'113px',height:'13.4375px',rx:'auto',ry:'auto'}),computedStyle);
  assert.deepEqual(actual.nodes[2].attributes,square);
});

test('capture refuses root opacity that would be lost from the opaque standalone export',()=>{
  const capture=api('captureTeachingDiagram');
  for(const opacity of ['0.5','0.999','0']) {
    const svg=fakeSVG();svg.style.opacity=opacity;
    assert.throws(()=>capture(svg,computedStyle),/opacity|hidden|refused/i);
  }
  assert.equal(capture(fakeSVG(),computedStyle).background,'rgb(7, 17, 31)');
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
