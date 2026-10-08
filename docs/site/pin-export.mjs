// Bounded export of the CURRENT displayed Approach-points teaching diagram.
// Pinned source identities are provenance, not proof acceptance or verification.
import { pinModel } from './explore-models.mjs?site-release=9936ad8dd3c73dd29a736230182b3b8c5e830717f1839dc7941e91c644349b2f';
import { serializeTeachingSVG } from './teaching-export.mjs?site-release=9936ad8dd3c73dd29a736230182b3b8c5e830717f1839dc7941e91c644349b2f';

const PUBLIC_EXPLORE='https://d6g8k5htny-coder.github.io/main/site/explore.html';
const COMMIT='d6628da09384728992dcbe6e921cc28ba85aebb0';
const SOURCES=Object.freeze([
  Object.freeze({repository:'d6g8k5htny-coder/Math-',commit:COMMIT,path:'frontiers/remote_window_20260924/PROOF.md',blob:'b383bfcc88ec4ad497dff01fb6640e429ba24a84',bytes:18355,sha256:'a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7',url:`https://github.com/d6g8k5htny-coder/Math-/blob/${COMMIT}/frontiers/remote_window_20260924/PROOF.md`}),
  Object.freeze({repository:'d6g8k5htny-coder/Math-',commit:COMMIT,path:'frontiers/rn_annulus_bridge_20260925/PROOF.md',blob:'6f317515b3d417661f86e2fed09bc7d950899c2b',bytes:16948,sha256:'d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05',url:`https://github.com/d6g8k5htny-coder/Math-/blob/${COMMIT}/frontiers/rn_annulus_bridge_20260925/PROOF.md`})
]);
const LIMITS='Illustrative, non-certifying floating-point scaling model. A = 2, B = 4, ρ = 3, L = 24, b = 1 and k = 1 are teaching choices, not certified theorem constants or an admissible radius bound. The viewport is a local schematic, not the whole torus. The fixed-annulus source allows all heights; the fixed-remote count uses the shrinking height window. The blue height pane illustrates r³ for either highlight and does not restrict the annulus count. No random field, critical-point count, probability, lifetime or theorem computation. This teaching figure is not mathematical result acceptance. Source identity is recorded provenance; fetched-source and hash verification are not performed by this export.';
function parameters(params) {
  if(!params || typeof params!=='object' || Array.isArray(params) || Object.keys(params).length!==2 || !Object.hasOwn(params,'r') || !Object.hasOwn(params,'region') || !['annulus','remote'].includes(params.region))throw new RangeError('Pin controls must be exactly r and region (annulus or remote)');
  const ticks=Math.round(params.r*100);
  if(!Number.isFinite(params.r) || params.r<.05 || params.r>.6 || Math.abs(params.r*100-ticks)>1e-8)throw new RangeError('r must be within the teaching slider range on a 0.01 step');
  return {r:ticks/100,region:params.region};
}
const permalink=params=>`${PUBLIC_EXPLORE}?r=${params.r}&region=${params.region}#distance`;
export function pinFigureMetadata(params,generatedAt=new Date().toISOString()) {
  const current=parameters(params);
  if(typeof generatedAt!=='string' || !Number.isFinite(Date.parse(generatedAt)) || new Date(generatedAt).toISOString()!==generatedAt)throw new RangeError('Generation time must be a valid ISO timestamp');
  // Integer decimal ticks keep the educational labels canonical. These numbers
  // remain floating-point teaching data, not certified field calculations.
  const ticks=Math.round(current.r*100),gap=ticks**3/1000000;
  return {schema:'universal-law/pin-teaching-figure/v1',mode:'teaching_model',params:current,teaching_constants:{A:2,B:4,rho:3,L:24,b:1,k:1},pins:[[-ticks/200,0],[ticks/200,0]],annulus:[ticks/50,ticks/25],height_gap:gap,height_window:[(1000000-ticks**3)/1000000,1],generated_at:generatedAt,sources:SOURCES.map(source=>({...source})),permalink:permalink(current),verification:'not_performed',limits:LIMITS};
}
export function pinFigureCitation(params) {
  const metadata=pinFigureMetadata(params);
  return `Universal Law, “Approaching points” teaching figure. r = ${metadata.params.r}; highlighted region: ${metadata.params.region}; pins M = (${metadata.pins[0][0]}, 0), S = (${metadata.pins[1][0]}, 0); annulus radii ${metadata.annulus.join(' to ')}; fixed remote radius ρ = 3; height gap r³ = ${metadata.height_gap}; height window ${metadata.height_window[0]} < f(x) < 1. A = 2, B = 4, ρ = 3, L = 24, b = 1, k = 1. ${metadata.permalink}\n${metadata.sources.map((source,index)=>`Teaching source: ${index?'fixed-annulus (all heights)':'fixed-remote (shrinking height window)'}, §1. ${source.url}\nRepository ${source.repository}; commit ${source.commit}; blob ${source.blob}; bytes ${source.bytes}; SHA-256 ${source.sha256}.`).join('\n')}\n${LIMITS}`;
}
const number=(value,places=2)=>value.toFixed(places).replace('-','−');
export function pinExpectedNodes(params) {
  const current=parameters(params),model=pinModel({r:current.r}),remote=current.region==='remote';
  // Use the same model arithmetic and native String conversion as drawPins;
  // require every displayed primitive/label, including its source attributes.
  const node=(tag,attributes,text)=>({tag,attributes:Object.fromEntries(Object.entries(attributes).map(([key,value])=>[key,String(value)])),...(text!==undefined?{text}:{})});
  const text=(x,y,value,className='diagram-small',extra={})=>node('text',{x,y,class:className,...extra},value);
  const line=(x1,y1,x2,y2,className='diagram-axis')=>node('line',{x1,y1,x2,y2,class:className});
  const circle=(cx,cy,r,className)=>node('circle',{cx,cy,r,class:className});
  const description=`Separation ${number(model.r)}, height gap ${number(model.gap,6)}. The ring spans radii ${number(model.annulus[0])} to ${number(model.annulus[1])}. The remote boundary stays at 3.00.`;
  const cx=202,cy=172,scale=39,top=78,bottom=top+model.gap*860;
  const nodes=[node('title',{id:'pin-diagram-title'},'Comparing shrinking space with a fixed exclusion'),node('desc',{id:'pin-diagram-description'},description),text(202,27,'SPACE','diagram-small',{'text-anchor':'middle'}),text(499,27,'HEIGHT','diagram-small',{'text-anchor':'middle'})];
  if(remote) {
    const radius=model.remoteRadius*scale;
    nodes.push(node('path',{d:`M24 40H380V302H24Z M${cx} ${cy-radius}a${radius} ${radius} 0 1 0 0 ${2*radius}a${radius} ${radius} 0 1 0 0 ${-2*radius}Z`,'fill-rule':'evenodd',class:'diagram-fill'}));
  }else {
    const [inner,outer]=model.annulus.map(radius=>radius*scale);
    nodes.push(node('circle',{cx,cy,r:(inner+outer)/2,fill:'none',stroke:'var(--ul-link)','stroke-opacity':'.2','stroke-width':outer-inner}));
    for(const radius of [inner,outer])nodes.push(circle(cx,cy,radius,'diagram-line'));
  }
  nodes.push(line(26,cy,378,cy,'diagram-grid'),line(cx,42,cx,300,'diagram-grid'),circle(cx,cy,model.remoteRadius*scale,'diagram-boundary'),text(206,59,'ρ = 3'));
  model.pins.forEach((pin,index)=>{const px=cx+pin*scale;nodes.push(line(px,cy+7,index?240:164,252),circle(px,cy,5,'diagram-point'),text(index?246:154,277,index?'S':'M','',{'text-anchor':'middle'}));});
  nodes.push(text(202,330,remote?'Outside the fixed boundary':`Ring: ${number(model.annulus[0])} to ${number(model.annulus[1])}`,'diagram-small',{'text-anchor':'middle'}),line(430,56,430,307),node('rect',{x:448,y:top,width:113,height:bottom-top,class:'diagram-fill'}),line(448,top,561,top,'diagram-line'),line(448,bottom,561,bottom,'diagram-line'),text(503,62,'b = 1','diagram-small',{'text-anchor':'middle'}),text(503,Math.max(124,bottom+30),'b − r³','diagram-small',{'text-anchor':'middle'}),text(503,330,'Fixed height scale','diagram-small',{'text-anchor':'middle'}));
  return nodes;
}
export function serializePinSVG(record,params,generatedAt=new Date().toISOString()) {
  const metadata=pinFigureMetadata(params,generatedAt);
  return serializeTeachingSVG(record,{id:'pin-diagram',viewBox:'0 0 600 350',titleId:'pin-diagram-title',descriptionId:'pin-diagram-description',expectedNodes:pinExpectedNodes(metadata.params),metadata,footerLines:[
    'Teaching figure · non-certifying floating-point scaling · local schematic',
    `r = ${metadata.params.r}; highlight: ${metadata.params.region}; gap r³ = ${metadata.height_gap}`,
    'A = 2, B = 4, ρ = 3, L = 24, b = 1, k = 1: teaching choices, not theorem constants.',
    'Annulus source: all heights. Remote count: shrinking height window.',
    'Blue height pane shows r³ for either highlight; it does not restrict the annulus count.',
    'No field, count, probability, lifetime or theorem computation.',
    'Math- sources §1 · remote_window_20260924 / rn_annulus_bridge_20260925',
    `Commit ${COMMIT} · full paths and hashes in metadata`,
    'Source identity recorded; verification not performed. No mathematical result acceptance.',
    metadata.permalink,
    `Generated ${metadata.generated_at}`
  ]});
}
