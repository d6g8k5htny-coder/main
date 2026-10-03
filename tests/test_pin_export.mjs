import test from 'node:test';
import assert from 'node:assert/strict';

let feature = {};
try { feature = await import('../docs/site/pin-export.mjs'); } catch {}
const api = name => {assert.equal(typeof feature[name],'function',`${name} is available`);return feature[name];};
const timestamp='2026-10-03T12:34:56.000Z';
const canonical='https://d6g8k5htny-coder.github.io/main/site/explore.html?r=0.5&region=annulus#distance';
const textStyle={fill:'rgb(239, 245, 253)',stroke:'none','fill-opacity':'1','stroke-opacity':'1',opacity:'1','stroke-width':'1px','stroke-dasharray':'none','stroke-linecap':'butt','stroke-linejoin':'miter','font-family':'system-ui, sans-serif','font-size':'18px','font-weight':'400','font-style':'normal','text-anchor':'start'};
function diagram(r=.5,region='annulus') {
  // Hand-checked literal geometry at the two comparison settings; never use the exporter to make its own fixture.
  const values=r===.5?{ring:'1.00 to 2.00',inner:39,outer:78,mid:58.5,width:39,left:192.25,right:211.75,rectHeight:107.5,bottom:185.5,labelY:215.5,gap:'0.125000'}:{ring:'0.50 to 1.00',inner:19.5,outer:39,mid:29.25,width:19.5,left:197.125,right:206.875,rectHeight:13.4375,bottom:91.4375,labelY:124,gap:'0.015625'};
  const node=(tag,attributes,text,style={})=>({tag,attributes:Object.fromEntries(Object.entries(attributes).map(([k,v])=>[k,String(v)])),...(text!==undefined?{text}:{}),style:['title','desc'].includes(tag)?{}:{...textStyle,...style}});
  const label=(x,y,text,className='diagram-small',extra={})=>node('text',{x,y,class:className,...extra},text);
  const line=(x1,y1,x2,y2,className='diagram-axis')=>node('line',{x1,y1,x2,y2,class:className},undefined,{stroke:'rgb(115, 144, 175)','stroke-width':'1.5px'});
  const circle=(cx,cy,r,className)=>node('circle',{cx,cy,r,class:className},undefined,{fill:'none',stroke:'rgb(120, 193, 255)','stroke-width':'2.5px'});
  return {attributes:{id:'pin-diagram',viewBox:'0 0 600 350',role:'img','aria-labelledby':'pin-diagram-title pin-diagram-description'},background:'rgb(7, 17, 31)',nodes:[
    node('title',{id:'pin-diagram-title'},'Comparing shrinking space with a fixed exclusion'),
    node('desc',{id:'pin-diagram-description'},`Separation ${r===.5?'0.50':'0.25'}, height gap ${values.gap}. The ring spans radii ${values.ring.replace(' to ',' to ')}. The remote boundary stays at 3.00.`),
    label(202,27,'SPACE','diagram-small',{'text-anchor':'middle'}),label(499,27,'HEIGHT','diagram-small',{'text-anchor':'middle'}),
    ...(region==='remote'?[node('path',{d:'M24 40H380V302H24Z M202 55a117 117 0 1 0 0 234a117 117 0 1 0 0 -234Z','fill-rule':'evenodd',class:'diagram-fill'},undefined,{fill:'rgb(120, 193, 255)','fill-opacity':'0.17'})]:[
      node('circle',{cx:202,cy:172,r:values.mid,fill:'none',stroke:'var(--ul-link)','stroke-opacity':'.2','stroke-width':values.width},undefined,{fill:'none',stroke:'rgb(120, 193, 255)','stroke-opacity':'0.2','stroke-width':`${values.width}px`}),
      circle(202,172,values.inner,'diagram-line'),circle(202,172,values.outer,'diagram-line')]),
    line(26,172,378,172,'diagram-grid'),line(202,42,202,300,'diagram-grid'),circle(202,172,117,'diagram-boundary'),label(206,59,'ρ = 3'),
    line(values.left,179,164,252),circle(values.left,172,5,'diagram-point'),label(154,277,'M','',{'text-anchor':'middle'}),
    line(values.right,179,240,252),circle(values.right,172,5,'diagram-point'),label(246,277,'S','',{'text-anchor':'middle'}),
    label(202,330,region==='remote'?'Outside the fixed boundary':`Ring: ${values.ring}`,'diagram-small',{'text-anchor':'middle'}),line(430,56,430,307),
    node('rect',{x:448,y:78,width:113,height:values.rectHeight,class:'diagram-fill'},undefined,{fill:'rgb(120, 193, 255)','fill-opacity':'0.17'}),line(448,78,561,78,'diagram-line'),line(448,values.bottom,561,values.bottom,'diagram-line'),
    label(503,62,'b = 1','diagram-small',{'text-anchor':'middle'}),label(503,values.labelY,'b − r³','diagram-small',{'text-anchor':'middle'}),label(503,330,'Fixed height scale','diagram-small',{'text-anchor':'middle'})
  ]};
}
function metadata(svg){const raw=svg.match(/<metadata[^>]*>([\s\S]*?)<\/metadata>/)?.[1];assert.ok(raw);return JSON.parse(raw.replaceAll('&quot;','"').replaceAll('&apos;',"'").replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&'));}

test('pin metadata records current bounded controls, teaching choices, scale geometry and full pinned source identities',()=>{
  const actual=api('pinFigureMetadata')({r:.5,region:'annulus'},timestamp);
  assert.equal(actual.schema,'universal-law/pin-teaching-figure/v1'); assert.equal(actual.mode,'teaching_model');
  assert.deepEqual(actual.params,{r:.5,region:'annulus'}); assert.deepEqual(actual.teaching_constants,{A:2,B:4,rho:3,L:24,b:1,k:1});
  assert.deepEqual(actual.pins,[[-.25,0],[.25,0]]); assert.deepEqual(actual.annulus,[1,2]); assert.deepEqual(actual.height_window,[.875,1]); assert.equal(actual.height_gap,.125);
  assert.equal(actual.permalink,canonical); assert.equal(actual.generated_at,timestamp); assert.equal(actual.verification,'not_performed');
  assert.equal(actual.sources.length,2);
  const [remote,annulus]=actual.sources;
  assert.equal(remote.commit,'d6628da09384728992dcbe6e921cc28ba85aebb0'); assert.equal(remote.path,'frontiers/remote_window_20260924/PROOF.md'); assert.equal(remote.blob,'b383bfcc88ec4ad497dff01fb6640e429ba24a84'); assert.equal(remote.bytes,18355); assert.equal(remote.sha256,'a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7');
  assert.equal(annulus.path,'frontiers/rn_annulus_bridge_20260925/PROOF.md'); assert.equal(annulus.blob,'6f317515b3d417661f86e2fed09bc7d950899c2b'); assert.equal(annulus.bytes,16948); assert.equal(annulus.sha256,'d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05');
  assert.match(actual.limits,/non-certifying.*floating-point/i); assert.match(actual.limits,/teaching choices.*theorem constants/i); assert.match(actual.limits,/not mathematical result acceptance/i); assert.match(actual.limits,/verification.*not performed/i);
  const small=api('pinFigureMetadata')({r:.25,region:'remote'},timestamp); assert.deepEqual(small.annulus,[.5,1]); assert.equal(small.height_gap,.015625); assert.deepEqual(small.height_window,[.984375,1]); assert.deepEqual(small.pins,[[-.125,0],[.125,0]]);
  assert.equal(api('pinFigureMetadata')({r:.6,region:'remote'},timestamp).height_gap,.216);
});

test('only displayed hundredth-tick controls and canonical ISO times are accepted',()=>{
  const build=api('pinFigureMetadata');
  for(const params of [undefined,null,[],{r:.5},{r:.5,region:'other'},{r:'0.5',region:'annulus'},{r:NaN,region:'annulus'},{r:Infinity,region:'annulus'},{r:.04,region:'annulus'},{r:.61,region:'annulus'},{r:.255,region:'annulus'},{r:.5,region:['annulus','remote']},{r:.5,region:'annulus',rho:2}]) assert.throws(()=>build(params,timestamp));
  for(const time of ['invalid','2026-10-03','2026-10-03T12:34:56Z','2026-02-31T12:34:56.000Z'])assert.throws(()=>build({r:.5,region:'annulus'},time),/ISO/i);
  for(const [r,want] of [[.05,.05],[.6,.6],[.30000000000000004,.3]]) assert.equal(build({r,region:'annulus'},timestamp).params.r,want);
});

test('citation includes selected highlight, full sources, qualitative limits and a feature-only permalink',()=>{
  const cite=api('pinFigureCitation')({r:.25,region:'remote'});
  assert.match(cite,/r = 0.25/); assert.match(cite,/remote/i); assert.match(cite,/0.015625/); assert.match(cite,/r=0.25&region=remote#distance/); assert.doesNotMatch(cite,/\?(?:s=|R=)|&objects=/);
  assert.match(cite,/fixed-annulus.*all heights/i); assert.match(cite,/fixed-remote.*height window/i); assert.match(cite,/No random field.*count.*probability.*lifetime.*theorem computation/i);
  assert.match(cite,/b383bfcc88ec4ad497dff01fb6640e429ba24a84/); assert.match(cite,/d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05/);
});

test('expected geometry matches the actual independent annulus and remote records at both comparison settings',()=>{
  for(const r of [.5,.25])for(const region of ['annulus','remote']){
    const expected=diagram(r,region).nodes.map(({style,...node})=>node);
    assert.deepEqual(api('pinExpectedNodes')({r,region}),expected);
    const svg=api('serializePinSVG')(diagram(r,region),{r,region},timestamp);
    const meta=metadata(svg); assert.deepEqual(meta.params,{r,region}); assert.equal(meta.sources.length,2);
    assert.match(svg,/Comparing shrinking space with a fixed exclusion/); assert.match(svg,/No field, count, probability, lifetime or theorem computation\./); assert.match(svg,/source identity recorded/i);
    assert.doesNotMatch(svg,/class=|var\(|<script|foreignObject|\shref=|url\(/i);
    if(region==='annulus')assert.match(svg,new RegExp(`stroke-width="${r===.5?'39':'19.5'}"`)); else assert.match(svg,/M24 40H380V302H24Z M202 55a117 117 0 1 0 0 234a117 117 0 1 0 0 -234Z/);
  }
});

test('pin export refuses stale pins, ring width, height window, remote path, labels and extra resources',()=>{
  const serialize=api('serializePinSVG');
  for(const change of [r=>r.nodes.find(n=>n.tag==='circle'&&n.attributes.class==='diagram-point').attributes.cx='197.125',r=>r.nodes[4].attributes['stroke-width']='19.5',r=>r.nodes.find(n=>n.tag==='rect').attributes.height='13.4375',r=>r.nodes[1].text='Old summary',r=>r.nodes[4].style.stroke='url(remote)',r=>r.nodes[4].attributes.onclick='evil()']){const bad=diagram();change(bad);assert.throws(()=>serialize(bad,{r:.5,region:'annulus'},timestamp),/refused/i);}
  assert.throws(()=>serialize(diagram(.5,'annulus'),{r:.25,region:'annulus'},timestamp),/stale|displayed/i);
  const remote=diagram(.5,'remote'); remote.nodes[4].attributes.d=remote.nodes[4].attributes.d.replace('117','116'); assert.throws(()=>serialize(remote,{r:.5,region:'remote'},timestamp),/refused/i);
});

test('the visible pin footer records the same ISO generation timestamp as metadata',()=>{
  const svg=api('serializePinSVG')(diagram(),{r:.5,region:'annulus'},timestamp);
  assert.equal(metadata(svg).generated_at,timestamp);
  assert.match(svg,/<text[^>]*>Generated 2026-10-03T12:34:56\.000Z<\/text>/);
});
