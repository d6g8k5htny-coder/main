// Bounded, local export of the Peaks and saddles teaching diagram. No source fetch
// or proof verification occurs here; the identity below is recorded provenance.
import { coneModel } from './explore-models.mjs?site-release=070fbc0fc7e8e79623c93863cb56eca9e739b2337ee3616d7a1488e5b9e876e1';

const SVG_NS = 'http://www.w3.org/2000/svg';
const PUBLIC_EXPLORE = 'https://d6g8k5htny-coder.github.io/main/site/explore.html';
const SOURCE = Object.freeze({
  repository: 'd6g8k5htny-coder/Math-',
  commit: '9d7b6802424fb4715b31999066aafca8ee2f3cca',
  path: 'coefficients/side24_v1/PROOF.md',
  blob: '44b66f04f89fcd87383b3603fa69f1feb64cdddd',
  bytes: 10272,
  sha256: 'c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769',
  url: 'https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/PROOF.md#1-a-nonperiodic-reference-coefficient-with-exact-cone-moments'
});
const LIMITS = 'Illustrative, non-certifying floating-point quadratic model using only the Hessian, eigenvalue and negative-cone identities in source §1. Eigenvalues are s − R and s + R; at λ = 0 the second-order test is inconclusive. No Gaussian sampling, coefficient, persistence or theorem computation. This teaching figure is not mathematical result acceptance. Source identity is recorded provenance; fetched-source and hash verification are not performed by this export.';

function parameters({s, R} = {}) {
  for (const [name, value, min, max] of [['s', s, -3, 3], ['R', R, 0, 2]]) {
    if (!Number.isFinite(value) || value < min || value > max || Math.abs(value * 10 - Math.round(value * 10)) > 1e-8)
      throw new RangeError(`${name} must be within the teaching slider range on a 0.1 step`);
  }
  // Match Explore URL restoration before deriving signs or a permalink.
  return {s: Math.round(s * 10) / 10 || 0, R: Math.round(R * 10) / 10 || 0};
}
export function curvatureFigureModel(params) {
  const {s, R} = parameters(params);
  // Compute on the slider's integer ticks so decimal labels agree everywhere.
  // These returned JavaScript numbers remain non-certifying teaching data.
  const ticks = coneModel(Math.round(s * 10), Math.round(R * 10));
  return {s, radius:R, eigenvalues:ticks.eigenvalues.map(value => value / 10), kind:ticks.kind};
}
function permalink(params) {
  return `${PUBLIC_EXPLORE}?s=${params.s}&R=${params.R}#peaks`;
}
export function buildFigureMetadata(params, generatedAt = new Date().toISOString()) {
  const current = parameters(params);
  if (typeof generatedAt !== 'string' || !Number.isFinite(Date.parse(generatedAt)) || new Date(generatedAt).toISOString() !== generatedAt)
    throw new RangeError('Generation time must be a valid ISO timestamp');
  const model = curvatureFigureModel(current);
  return {
    schema: 'universal-law/curvature-teaching-figure/v1', mode: 'teaching_model',
    params: current, eigenvalues: model.eigenvalues, classification: model.kind,
    generated_at: generatedAt, source: {...SOURCE}, permalink: permalink(current),
    verification: 'not_performed', limits: LIMITS
  };
}
export function buildFigureCitation(params) {
  const current = parameters(params);
  const model = curvatureFigureModel(current);
  return `Universal Law, “Peaks and saddles” teaching figure. s = ${current.s}, R = ${current.R}; eigenvalues s − R = ${model.eigenvalues[0]}, s + R = ${model.eigenvalues[1]}; classification: ${model.kind}. ${permalink(current)}\nTeaching source: SIDE24 derivation, §1 (Hessian, eigenvalue and negative-cone identities), ${SOURCE.url}\n${LIMITS}`;
}

function refuse(message) { throw new TypeError(`Curvature SVG refused: ${message}`); }
function plain(value) { return value && typeof value === 'object' && !Array.isArray(value); }
function xml(value) {
  if (typeof value !== 'string' || value.length > 12000 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\ud800-\udfff\ufffe\uffff]/u.test(value))
    refuse('unsupported text');
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;');
}
const NUMERIC = /^-?(?:\d+(?:\.\d*)?|\.\d+)$/;
function boundedNumber(value, min = 0, max = 600, units = false) {
  const text = String(value).trim();
  const numberText = units ? text.replace(/px$/, '') : text;
  const number = Number(numberText);
  if (!NUMERIC.test(numberText) || !Number.isFinite(number) || number < min || number > max) refuse('unbounded numeric primitive or style');
  return String(number === 0 ? 0 : number);
}
function color(value, opaque = false) {
  if (typeof value !== 'string' || value.length > 64) refuse('unsupported color');
  if (!opaque && value === 'none') return value;
  if (/^#[\da-f]{6}$/i.test(value)) return value;
  const rgb = value.match(/^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$/);
  if (rgb && rgb.slice(1).every(component => Number(component) <= 255)) return value;
  refuse('unsupported or resource color');
}
const styleKeys = new Set(['fill', 'stroke', 'fill-opacity', 'stroke-opacity', 'opacity', 'stroke-width', 'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin', 'font-family', 'font-size', 'font-weight', 'font-style', 'text-anchor']);
function concreteStyles(styles) {
  if (!plain(styles)) refuse('missing concrete styles');
  const result = {};
  for (const [key, value] of Object.entries(styles)) {
    if (!styleKeys.has(key) || typeof value !== 'string' || value.length > 200) refuse('unsupported style');
    if (key === 'fill' || key === 'stroke') result[key] = color(value);
    else if (key.endsWith('opacity')) result[key] = boundedNumber(value, 0, 1);
    else if (key === 'stroke-width') result[key] = boundedNumber(value, 0, 10, true);
    else if (key === 'font-size') result[key] = boundedNumber(value, 8, 40, true);
    else if (key === 'stroke-dasharray') {
      if (value === 'none') result[key] = value;
      else {
        const parts = value.trim().split(/[\s,]+/);
        if (parts.length < 1 || parts.length > 8) refuse('unsupported dash pattern');
        result[key] = parts.map(part => boundedNumber(part, 0, 30, true)).join(' ');
      }
    } else if (key === 'font-family') {
      // Local font names and generic stacks only; no resource syntax or CSS rules.
      if (!/^-?[a-zA-Z][a-zA-Z0-9 .,"'\-]*$/.test(value) || /url|var|https?|import/i.test(value)) refuse('unsupported font');
      result[key] = value;
    } else if (key === 'font-weight') {
      if (!/^(?:normal|bold|[1-9]00)$/.test(value)) refuse('unsupported font weight');
      result[key] = value;
    } else {
      const values = {'font-style': ['normal', 'italic', 'oblique'], 'text-anchor': ['start', 'middle', 'end'], 'stroke-linecap': ['butt', 'round', 'square'], 'stroke-linejoin': ['miter', 'round', 'bevel']};
      if (!values[key]?.includes(value)) refuse('unsupported style value');
      result[key] = value;
    }
  }
  return result;
}
const allowedAttributes = {
  title: ['id'], desc: ['id'], polygon: ['points', 'class'],
  line: ['x1', 'y1', 'x2', 'y2', 'class'],
  text: ['x', 'y', 'class', 'text-anchor'], circle: ['cx', 'cy', 'r', 'class']
};
const classes = new Set(['', 'diagram-fill', 'diagram-warm', 'diagram-grid', 'diagram-small', 'diagram-axis', 'diagram-boundary', 'diagram-point']);
function nodeAttributes(node) {
  if (!plain(node) || !Object.hasOwn(allowedAttributes, node.tag) || !plain(node.attributes)) refuse('unsupported element');
  if (Object.keys(node).some(key => !['tag', 'attributes', 'text', 'style'].includes(key))) refuse('unsupported node content');
  const result = {};
  for (const [key, value] of Object.entries(node.attributes)) {
    if (!allowedAttributes[node.tag].includes(key)) refuse('unsupported attribute');
    if (key === 'class') { if (!classes.has(value)) refuse('unsupported diagram class'); }
    else if (key === 'id') {
      const expected = node.tag === 'title' ? 'curvature-diagram-title' : 'curvature-diagram-description';
      if (value !== expected) refuse('unsupported label id');
      result[key] = value;
    } else if (key === 'points') {
      if (typeof value !== 'string' || value.length > 250) refuse('unsupported polygon');
      const pairs = value.trim().split(/\s+/);
      if (pairs.length !== 4) refuse('unsupported polygon');
      result[key] = pairs.map(pair => {
        const parts = pair.split(',');
        if (parts.length !== 2) refuse('unsupported polygon');
        return parts.map(part => boundedNumber(part)).join(',');
      }).join(' ');
    } else if (key === 'text-anchor') {
      if (!['start', 'middle', 'end'].includes(value)) refuse('unsupported text anchor');
      result[key] = value;
    } else result[key] = boundedNumber(value, 0, key === 'r' ? 30 : 600);
  }
  const required = {title:['id'],desc:['id'],polygon:['points'],line:['x1','y1','x2','y2'],text:['x','y'],circle:['cx','cy','r']}[node.tag];
  if (required.some(key => !Object.hasOwn(result, key))) refuse('missing geometry or labels');
  return result;
}
function element(tag, attributes, text = '') {
  const attrs = Object.entries(attributes).map(([key, value]) => ` ${key}="${xml(String(value))}"`).join('');
  return `<${tag}${attrs}>${xml(text)}</${tag}>`;
}
function validateDiagram(record, params) {
  if (!plain(record) || !plain(record.attributes) || !Array.isArray(record.nodes)) refuse('missing displayed diagram');
  if (Object.keys(record).some(key => !['attributes', 'nodes', 'background'].includes(key))) refuse('unsupported diagram record');
  const expected = {id:'curvature-diagram',viewBox:'0 0 600 330',role:'img','aria-labelledby':'curvature-diagram-title curvature-diagram-description'};
  if (Object.keys(record.attributes).length !== Object.keys(expected).length || Object.entries(expected).some(([key,value]) => record.attributes[key] !== value)) refuse('unsupported diagram root');
  color(record.background, true);
  if (record.nodes.length !== 25 || record.nodes[0]?.tag !== 'title' || record.nodes[1]?.tag !== 'desc') refuse('blank or incomplete displayed diagram');
  const counts = {title:0,desc:0,polygon:0,line:0,text:0,circle:0};
  const clean = record.nodes.map(node => {
    const attributes = nodeAttributes(node);
    counts[node.tag]++;
    const textual = ['title', 'desc', 'text'].includes(node.tag);
    if (textual && (typeof node.text !== 'string' || !node.text.trim())) refuse('missing displayed text');
    if (!textual && node.text !== undefined && node.text !== '') refuse('unsupported primitive text');
    const style = concreteStyles(node.style);
    if (node.tag === 'circle' && (Math.abs(Number(attributes.cx) - (86 + params.R * 155)) > 1e-8 || Math.abs(Number(attributes.cy) - (165 - params.s * 40)) > 1e-8 || Number(attributes.r) !== 9)) refuse('stale displayed marker');
    return {tag:node.tag,attributes:{...attributes,...style},text:textual ? node.text : ''};
  });
  if (Object.entries({title:1,desc:1,polygon:2,line:7,text:13,circle:1}).some(([tag,count]) => counts[tag] !== count)) refuse('blank or incomplete displayed geometry');
  return clean;
}

export function serializeCurvatureSVG(record, params, generatedAt = new Date().toISOString()) {
  const metadata = buildFigureMetadata(params, generatedAt);
  const nodes = validateDiagram(record, metadata.params);
  const label = nodes.find(node => node.tag === 'text');
  const textColor = label.attributes.fill || 'rgb(239, 245, 253)';
  const footerStyle = {fill:textColor,'font-family':label.attributes['font-family'] || 'Arial, Helvetica, sans-serif','font-size':'11'};
  const lines = [
    'Teaching figure · illustrative floating-point quadratic model · non-certifying',
    `s = ${metadata.params.s}, R = ${metadata.params.R}; s − R = ${metadata.eigenvalues[0]}, s + R = ${metadata.eigenvalues[1]}; ${metadata.classification}`,
    'At λ = 0, the second-order test is inconclusive.',
    'No Gaussian sampling, coefficient, persistence or theorem computation.',
    'Source: SIDE24 §1 · Hessian, eigenvalue and negative-cone identities',
    `Math- commit ${SOURCE.commit}`,
    'coefficients/side24_v1/PROOF.md · source identity recorded, verification not performed',
    'This teaching figure is not mathematical result acceptance.',
    `${metadata.permalink}`,
    `Generated ${metadata.generated_at}`
  ];
  const background = element('rect', {x:'0',y:'0',width:'600',height:'535',fill:record.background});
  const footer = lines.map((text,index) => element('text',{x:'20',y:String(354 + index * 18),...footerStyle},text)).join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="${SVG_NS}" viewBox="0 0 600 535" width="600" height="535" role="img" aria-labelledby="curvature-diagram-title curvature-diagram-description">\n${nodes.slice(0,2).map(node => element(node.tag,node.attributes,node.text)).join('\n')}\n${element('metadata',{id:'curvature-export-metadata'},JSON.stringify(metadata))}\n${background}\n${nodes.slice(2).map(node => element(node.tag,node.attributes,node.text)).join('\n')}\n${footer}\n</svg>\n`;
}

export function captureCurvatureDiagram(svg, computedStyle = globalThis.getComputedStyle) {
  if (!svg || svg.localName !== 'svg' || svg.namespaceURI !== SVG_NS || typeof computedStyle !== 'function') refuse('displayed SVG unavailable');
  function attributes(node) { return Object.fromEntries([...node.attributes].map(attribute => [attribute.name,attribute.value])); }
  function inspect(node) {
    if (node.namespaceURI !== SVG_NS || [...node.childNodes].some(child => child.nodeType !== 3)) refuse('unsupported nested diagram content');
    const styles = computedStyle(node);
    for (const property of ['filter','clip-path','mask-image','marker-start','marker-mid','marker-end','transform','background-image']) {
      const value = styles.getPropertyValue(property);
      if (value && value !== 'none') refuse('unsupported displayed resource or effect');
    }
    if (!['title','desc'].includes(node.localName) && (styles.getPropertyValue('display') === 'none' || styles.getPropertyValue('visibility') !== 'visible')) refuse('hidden diagram content');
    const style = ['title','desc'].includes(node.localName) ? {} : Object.fromEntries([...styleKeys].map(property => [property,styles.getPropertyValue(property)]));
    return {tag:node.localName,attributes:attributes(node),style,...(['title','desc','text'].includes(node.localName) ? {text:node.textContent} : {})};
  }
  if ([...svg.childNodes].some(node => node.nodeType !== 1 && (node.nodeType !== 3 || node.textContent.trim()))) refuse('unsupported SVG root content');
  const rootStyle = computedStyle(svg);
  for (const property of ['background-image','filter','clip-path','mask-image','transform']) {
    const value = rootStyle.getPropertyValue(property);
    if (value && value !== 'none') refuse('unsupported SVG resource or effect');
  }
  return {attributes:attributes(svg),background:rootStyle.getPropertyValue('background-color'),nodes:[...svg.children].map(inspect)};
}

export function createCitationController({writeText, onChange = () => {}} = {}) {
  let text = '', pending = false, ready = false, revision = 0;
  let manualFallback = typeof writeText !== 'function';
  let status = 'Figure export becomes available after the diagram is drawn.';
  const state = () => ({text,pending,manualFallback,status,canCopy:ready && typeof writeText === 'function' && !pending});
  const notify = () => onChange(state());
  return {
    state,
    setFigure(params) {
      const next = buildFigureCitation(params);
      if (!ready || next !== text) revision++;
      text = next; ready = true;
      status = pending ? 'Copying the previous citation; the current citation is shown below.'
        : manualFallback ? 'Copy the figure citation manually from the text below.'
        : 'Download the displayed teaching figure or copy its citation.';
      notify();
    },
    clear() {
      text = ''; ready = false; revision++;
      status = 'Figure export is unavailable because the displayed diagram could not be captured.';
      notify();
    },
    async copy() {
      if (pending || !ready) return false;
      if (typeof writeText !== 'function') {
        manualFallback = true; status = 'Clipboard unavailable. Copy the figure citation manually below.'; notify(); return false;
      }
      const started = revision, currentText = text;
      pending = true; status = 'Copying figure citation…'; notify();
      let success = false;
      try {
        await writeText(currentText);
        if (ready && started === revision) { status = 'Copied figure citation.'; success = true; }
        else if (ready) status = 'Settings changed during copying. Copy again for the current figure.';
      } catch {
        manualFallback = true;
        if (ready) status = 'Clipboard access was denied or could not complete. Copy the current citation manually below.';
      } finally { pending = false; notify(); }
      return success;
    }
  };
}
