// Local capture of current teaching primitives. This serializer neither fetches
// sources nor verifies proofs; feature modules supply recorded source identities.
const SVG_NS = 'http://www.w3.org/2000/svg';
const styleKeys = new Set(['fill','stroke','fill-opacity','stroke-opacity','opacity','stroke-width','stroke-dasharray','stroke-linecap','stroke-linejoin','font-family','font-size','font-weight','font-style','text-anchor']);
const classes = new Set(['','diagram-small','diagram-axis','diagram-grid','diagram-boundary','diagram-fill','diagram-line','diagram-point','diagram-object','diagram-muted-object','diagram-box','diagram-on-accent']);
const allowedAttributes = {
  title:['id'],desc:['id'],circle:['cx','cy','r','class','fill','stroke','stroke-opacity','stroke-width'],
  line:['x1','y1','x2','y2','class'],rect:['x','y','width','height','rx','class'],
  path:['d','fill-rule','class'],text:['x','y','class','text-anchor']
};
const requiredAttributes = {title:['id'],desc:['id'],circle:['cx','cy','r'],line:['x1','y1','x2','y2'],rect:['x','y','width','height'],path:['d','fill-rule'],text:['x','y']};
const REMOTE_PATH = 'M24 40H380V302H24Z M202 55a117 117 0 1 0 0 234a117 117 0 1 0 0 -234Z';
// Chrome resolves the two relative arcs to these absolute endpoints in computed
// CSS d. Only this known displayed remote path is admitted, not arbitrary paths.
const REMOTE_COMPUTED_PATH = 'M 24 40 H 380 V 302 H 24 Z M 202 55 A 117 117 0 1 0 202 289 A 117 117 0 1 0 202 55 Z';
function refuse(message) { throw new TypeError(`Teaching SVG refused: ${message}`); }
const plain = value => value && typeof value === 'object' && !Array.isArray(value);
function xml(value) {
  if(typeof value !== 'string' || value.length > 16000 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\ud800-\udfff\ufffe\uffff]/u.test(value)) refuse('unsupported text');
  return value.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&apos;');
}
function identifier(value) { if(typeof value !== 'string' || !/^[a-z][a-z0-9-]{0,79}$/.test(value)) refuse('unsupported id'); return value; }
function boundedNumber(value, min=0, max=600, units=false) {
  if(typeof value !== 'string') refuse('unsupported numeric primitive or style');
  const raw=value.trim(), numberText=units ? raw.replace(/px$/,'') : raw;
  const number=Number(numberText);
  if(!/^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(numberText) || !Number.isFinite(number) || number<min || number>max) refuse('unbounded numeric primitive or style');
  return String(number===0?0:number);
}
function color(value, opaque=false) {
  if(typeof value !== 'string' || value.length>64) refuse('unsupported color');
  if(!opaque && value==='none')return value;
  if(/^#[\da-f]{6}$/i.test(value))return value;
  const match=value.match(/^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$/);
  if(match && match.slice(1).every(part=>Number(part)<=255))return value;
  refuse('unsupported or resource color');
}
function concreteStyles(styles,tag) {
  if(!plain(styles) || !Object.hasOwn(styles,'fill') || !Object.hasOwn(styles,'stroke'))refuse('missing concrete styles');
  if(tag==='text' && ['font-family','font-size'].some(key=>!Object.hasOwn(styles,key)))refuse('missing concrete font');
  const result={};
  for(const [key,value] of Object.entries(styles)) {
    if(!styleKeys.has(key) || typeof value!=='string' || value.length>200)refuse('unsupported style');
    if(['fill','stroke'].includes(key))result[key]=color(value);
    else if(key.endsWith('opacity')) {result[key]=boundedNumber(value,0,1); if(key==='opacity' && Number(result[key])===0)refuse('hidden primitive');}
    else if(key==='stroke-width')result[key]=boundedNumber(value,0,60,true);
    else if(key==='font-size')result[key]=boundedNumber(value,8,40,true);
    else if(key==='stroke-dasharray') {
      if(value==='none')result[key]=value;
      else {const parts=value.trim().split(/[\s,]+/); if(parts.length<1 || parts.length>8)refuse('unsupported dash pattern'); result[key]=parts.map(part=>boundedNumber(part,0,30,true)).join(' ');}
    } else if(key==='font-family') {
      if(!/^-?[a-zA-Z"'][a-zA-Z0-9 .,"'\-]*$/.test(value) || /url|var|https?|import/i.test(value))refuse('unsupported font');
      result[key]=value;
    } else if(key==='font-weight') {
      if(!/^(?:normal|bold|[1-9]\d{0,2}|1000)$/.test(value))refuse('unsupported font weight'); result[key]=value;
    } else {
      const allowed={'font-style':['normal','italic','oblique'],'text-anchor':['start','middle','end'],'stroke-linecap':['butt','round','square'],'stroke-linejoin':['miter','round','bevel']};
      if(!allowed[key]?.includes(value))refuse('unsupported style value'); result[key]=value;
    }
  }
  if(tag==='text' && result.fill==='none' || tag==='line' && (result.stroke==='none' || result['stroke-opacity']==='0' || result['stroke-width']==='0'))refuse('hidden primitive');
  if(result.fill==='none' && (result.stroke==='none' || result['stroke-opacity']==='0' || result['stroke-width']==='0') || result.stroke==='none' && result['fill-opacity']==='0')refuse('hidden primitive');
  return result;
}
function nodeAttributes(node) {
  if(!plain(node) || !Object.hasOwn(allowedAttributes,node.tag) || !plain(node.attributes))refuse('unsupported element');
  if(Object.keys(node).some(key=>!['tag','attributes','text','style'].includes(key)))refuse('unsupported node content');
  const result={};
  for(const [key,value] of Object.entries(node.attributes)) {
    if(!allowedAttributes[node.tag].includes(key) || typeof value!=='string')refuse('unsupported attribute');
    if(key==='class') {if(!classes.has(value))refuse('unsupported class');}
    else if(key==='id')result[key]=identifier(value);
    else if(key==='text-anchor') {if(!['start','middle','end'].includes(value))refuse('unsupported text anchor');result[key]=value;}
    else if(key==='d') {if(value!==REMOTE_PATH)refuse('unsupported path');result[key]=value;}
    else if(key==='fill-rule') {if(value!=='evenodd')refuse('unsupported fill rule');result[key]=value;}
    else if(key==='fill') {if(value!=='none')refuse('unsupported explicit fill');}
    else if(key==='stroke') {if(value!=='var(--ul-link)')refuse('unsupported explicit stroke');}
    else result[key]=boundedNumber(value,0,key==='r'?300:key==='rx'?80:key==='stroke-width'?60:key==='stroke-opacity'?1:600);
  }
  if(requiredAttributes[node.tag].some(key=>!Object.hasOwn(result,key)))refuse('missing geometry or labels');
  return result;
}
function sameAttributes(actual,expected) {
  return plain(expected) && Object.keys(actual).length===Object.keys(expected).length && Object.entries(expected).every(([key,value])=>typeof value==='string' && actual[key]===value);
}
function element(tag,attributes,text='') {
  return `<${tag}${Object.entries(attributes).map(([key,value])=>` ${key}="${xml(String(value))}"`).join('')}>${xml(text)}</${tag}>`;
}
export function serializeTeachingSVG(record,{id,viewBox,titleId,descriptionId,expectedNodes,metadata,footerLines}={}) {
  identifier(id);identifier(titleId);identifier(descriptionId);
  if(typeof viewBox!=='string' || !/^0 0 600 (?:300|350)$/.test(viewBox))refuse('unsupported viewBox');
  if(!plain(record) || !plain(record.attributes) || !Array.isArray(record.nodes) || Object.keys(record).some(key=>!['attributes','background','nodes'].includes(key)))refuse('missing displayed diagram');
  if(!sameAttributes(record.attributes,{id,viewBox,role:'img','aria-labelledby':`${titleId} ${descriptionId}`}))refuse('unsupported displayed root');
  const background=color(record.background,true);
  if(!Array.isArray(expectedNodes) || expectedNodes.length<3 || expectedNodes.length>80 || record.nodes.length!==expectedNodes.length)refuse('missing displayed geometry');
  const nodes=record.nodes.map((node,index)=>{
    const attributes=nodeAttributes(node), expected=expectedNodes[index];
    if(!plain(expected) || Object.keys(expected).some(key=>!['tag','attributes','text'].includes(key)) || node.tag!==expected.tag || !sameAttributes(node.attributes,expected.attributes) || node.text!==expected.text)refuse('stale or unsupported displayed geometry or labels');
    const textual=['title','desc','text'].includes(node.tag);
    if(textual && (typeof node.text!=='string' || !node.text.trim()) || !textual && node.text!==undefined)refuse('missing or unsupported displayed text');
    if(textual)xml(node.text);
    if(['title','desc'].includes(node.tag)) {
      if(index!==(node.tag==='title'?0:1) || attributes.id!==(node.tag==='title'?titleId:descriptionId) || !plain(node.style) || Object.keys(node.style).length)refuse('unsupported accessibility label');
      return {tag:node.tag,attributes,text:node.text};
    }
    return {tag:node.tag,attributes:{...attributes,...concreteStyles(node.style,node.tag)},text:node.text||''};
  });
  if(nodes[0]?.tag!=='title' || nodes[1]?.tag!=='desc' || nodes.filter(node=>node.tag==='title').length!==1 || nodes.filter(node=>node.tag==='desc').length!==1)refuse('missing accessibility labels');
  if(!plain(metadata))refuse('missing metadata');
  const metadataText=JSON.stringify(metadata);xml(metadataText);
  if(!Array.isArray(footerLines) || footerLines.length<1 || footerLines.length>20 || footerLines.some(line=>typeof line!=='string' || !line.trim() || line.length>100))refuse('unsupported caption');
  const label=nodes.find(node=>node.tag==='text');if(!label)refuse('missing displayed font');
  const footerStyle={fill:label.attributes.fill,'font-family':label.attributes['font-family'],'font-size':'11'};
  const diagramHeight=Number(viewBox.split(' ')[3]), height=diagramHeight+34+18*footerLines.length;
  const footer=footerLines.map((line,index)=>element('text',{x:'20',y:String(diagramHeight+24+18*index),...footerStyle},line)).join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="${SVG_NS}" viewBox="0 0 600 ${height}" width="600" height="${height}" role="img" aria-labelledby="${titleId} ${descriptionId}">\n${nodes.slice(0,2).map(node=>element(node.tag,node.attributes,node.text)).join('\n')}\n${element('metadata',{id:`${id}-export-metadata`},metadataText)}\n${element('rect',{x:'0',y:'0',width:'600',height:String(height),fill:background})}\n${nodes.slice(2).map(node=>element(node.tag,node.attributes,node.text)).join('\n')}\n${footer}\n</svg>\n`;
}
export function captureTeachingDiagram(svg,computedStyle=globalThis.getComputedStyle) {
  if(!svg || svg.localName!=='svg' || svg.namespaceURI!==SVG_NS || typeof computedStyle!=='function')refuse('displayed SVG unavailable');
  const attributes=node=>Object.fromEntries([...node.attributes].map(attribute=>[attribute.name,attribute.value]));
  function inspectStyles(node,root=false) {
    const styles=computedStyle(node);
    for(const property of ['filter','clip-path','mask','mask-image','marker-start','marker-mid','marker-end','transform','background-image']) {
      const value=styles.getPropertyValue(property);if(value && value!=='none')refuse('unsupported displayed resource or effect');
    }
    if(!['title','desc'].includes(node.localName) && (styles.getPropertyValue('display')==='none' || styles.getPropertyValue('visibility')!=='visible' || Number(styles.getPropertyValue('opacity'))===0))refuse('hidden displayed diagram');
    // Root opacity affects every primitive but is not represented by the
    // opaque standalone background. Refuse it rather than change the figure.
    if(root && Number(boundedNumber(styles.getPropertyValue('opacity'),0,1))!==1)refuse('unsupported root opacity');
    if(root && typeof node.getClientRects==='function' && !node.getClientRects().length)refuse('hidden displayed diagram');
    return styles;
  }
  if([...svg.childNodes].some(node=>node.nodeType!==1 && (node.nodeType!==3 || node.textContent.trim())))refuse('unsupported SVG root content');
  const rootStyle=inspectStyles(svg,true);
  const nodes=[...svg.children].map(node=>{
    if(node.namespaceURI!==SVG_NS || !Object.hasOwn(allowedAttributes,node.localName) || [...node.childNodes].some(child=>child.nodeType!==3))refuse('unsupported nested diagram content');
    if(!['title','desc','text'].includes(node.localName) && node.textContent.trim())refuse('unsupported primitive text');
    const styles=inspectStyles(node),currentAttributes=attributes(node);
    const geometryKeys={circle:['cx','cy','r'],rect:['x','y','width','height']}[node.localName]||[];
    for(const key of geometryKeys) {
      const displayed=styles.getPropertyValue(key);
      if(!displayed || displayed==='none')continue;
      // SVG2 rect width/height:auto has used value zero, not the native
      // attribute value. auto is unsupported for the other keys here.
      if(displayed==='auto')refuse('auto CSS geometry suppresses or changes current attributes');
      const actual=Number(boundedNumber(displayed,0,key==='r'?300:600,true));
      const expected=Object.hasOwn(currentAttributes,key)?Number(boundedNumber(currentAttributes[key])):0;
      if(Math.abs(actual-expected)>1e-6)refuse('CSS geometry differs from current attributes');
    }
    if(node.localName==='rect') {
      const width=Number(boundedNumber(currentAttributes.width)),height=Number(boundedNumber(currentAttributes.height));
      const native=key=>Object.hasOwn(currentAttributes,key)?Number(boundedNumber(currentAttributes[key],0,80)):null;
      const computed=key=>{
        const value=styles.getPropertyValue(key);
        return !value || value==='none'?native(key):value==='auto'?null:Number(boundedNumber(value,0,80,true));
      };
      // One auto radius copies the other; two auto radii give square corners.
      // Compare effective clamped radii, so Chrome's native rx:12px/ry:auto
      // and the attribute-only export both keep the same rounded rectangle.
      const used=(rx,ry)=>{
        const x=rx??ry??0,y=ry??rx??0;
        return x===0 || y===0?[0,0]:[Math.min(x,width/2),Math.min(y,height/2)];
      };
      const expected=used(native('rx'),native('ry')),actual=used(computed('rx'),computed('ry'));
      if(actual.some((value,index)=>Math.abs(value-expected[index])>1e-6))refuse('CSS corner geometry differs from current attributes');
    }
    if(node.localName==='path') {
      const displayed=styles.getPropertyValue('d');
      if(displayed) {
        const match=displayed.match(/^path\(["']([MmHhVvAaZz\d.,\s-]+)["']\)$/);
        const tokens=value=>value.match(/[MmHhVvAaZz]|-?(?:\d+(?:\.\d*)?|\.\d+)/g)?.map(token=>Number.isFinite(Number(token))?String(Number(token)):token).join(' ');
        if(currentAttributes.d!==REMOTE_PATH || !match || ![tokens(REMOTE_PATH),tokens(REMOTE_COMPUTED_PATH)].includes(tokens(match[1])))refuse('CSS path geometry differs from current attributes');
      }
    }
    return {tag:node.localName,attributes:currentAttributes,style:['title','desc'].includes(node.localName)?{}:Object.fromEntries([...styleKeys].map(key=>[key,styles.getPropertyValue(key)])),...(['title','desc','text'].includes(node.localName)?{text:node.textContent}:{})};
  });
  return {attributes:attributes(svg),background:rootStyle.getPropertyValue('background-color'),nodes};
}
export function createFigureCitationController({buildCitation,writeText,onChange=()=>{}}={}) {
  if(typeof buildCitation!=='function' || typeof onChange!=='function')throw new TypeError('Citation builder and observer must be functions');
  let text='',pending=false,ready=false,revision=0,manualFallback=typeof writeText!=='function';
  let status='Figure export is unavailable until the displayed diagram is drawn.';
  const state=()=>({text,pending,manualFallback,status,canCopy:ready && typeof writeText==='function' && !pending});
  const notify=()=>onChange(state());
  function clear() {text='';ready=false;revision++;status='Figure export is unavailable because the displayed diagram could not be captured.';notify();}
  return {state,clear,setFigure(params) {
    let next;try{next=buildCitation(params);if(typeof next!=='string' || !next.trim())throw new TypeError('Figure citation is unavailable');}catch(error){clear();throw error;}
    if(!ready || next!==text)revision++;text=next;ready=true;
    status=pending?'Copying the previous citation; the current citation is shown below.':manualFallback?'Copy the figure citation manually from the text below.':'Download the displayed teaching figure or copy its citation.';notify();
  },async copy() {
    if(pending || !ready)return false;
    if(typeof writeText!=='function'){manualFallback=true;status='Clipboard unavailable. Copy the figure citation manually below.';notify();return false;}
    const started=revision,currentText=text;pending=true;status='Copying figure citation…';notify();let success=false;
    try {await writeText(currentText);if(ready && started===revision){status='Copied figure citation.';success=true;}else if(ready)status='Settings changed during copying. Copy again for the current figure.';}
    catch {manualFallback=true;if(ready)status='Clipboard access was denied or could not complete. Copy the current citation manually below.';}
    finally {pending=false;notify();}return success;
  }};
}
