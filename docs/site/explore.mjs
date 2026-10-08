import { coneModel, pinModel, paletteModel, applyCurvaturePreset } from './explore-models.mjs?site-release=ddce61104c5b934a9ebca5bb9674938c779511f039f99a02bb3c419ec12c64a9';
import { readExploreState, exploreStateURL } from './explore-state.mjs?site-release=ddce61104c5b934a9ebca5bb9674938c779511f039f99a02bb3c419ec12c64a9';

import { captureCurvatureDiagram, serializeCurvatureSVG, createCitationController } from './curvature-export.mjs?site-release=ddce61104c5b934a9ebca5bb9674938c779511f039f99a02bb3c419ec12c64a9';
import { captureTeachingDiagram, createFigureCitationController } from './teaching-export.mjs?site-release=ddce61104c5b934a9ebca5bb9674938c779511f039f99a02bb3c419ec12c64a9';
import { serializePinSVG, pinFigureCitation } from './pin-export.mjs?site-release=ddce61104c5b934a9ebca5bb9674938c779511f039f99a02bb3c419ec12c64a9';
import { serializePaletteSVG, paletteFigureCitation } from './palette-export.mjs?site-release=ddce61104c5b934a9ebca5bb9674938c779511f039f99a02bb3c419ec12c64a9';

const ns = 'http://www.w3.org/2000/svg';
const byId = id => document.getElementById(id);
const number = (value, places = 2) => value.toFixed(places).replace('-', '−');
function shape(tag, attributes, text) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function write(svg, x, y, text, className = '', extra = {}) {
  svg.append(shape('text', { x, y, class: className, ...extra }, text));
}
function line(svg, x1, y1, x2, y2, className = 'diagram-axis') {
  svg.append(shape('line', { x1, y1, x2, y2, class: className }));
}
function clear(svg, title, description) {
  const titleNode = svg.querySelector('title');
  const descriptionNode = svg.querySelector('desc');
  titleNode.textContent = title;
  descriptionNode.textContent = description;
  svg.replaceChildren(titleNode, descriptionNode);
}

const center = byId('curvature-center');
const spread = byId('curvature-spread');
const exportButton = byId('curvature-export-svg');
const copyCitationButton = byId('curvature-copy-citation');
const exportStatus = byId('curvature-export-status');
const citationDetails = byId('curvature-figure-citation');
const citationText = byId('curvature-citation-text');
const downloadSupported = typeof Blob === 'function' && typeof URL.createObjectURL === 'function'
  && typeof URL.revokeObjectURL === 'function' && 'download' in document.createElement('a');
let figureReady = false;
const exportControls = new Set([exportButton, copyCitationButton]);
function teachingFigureTools(prefix, currentParameters, serialize, buildCitation) {
  const download = byId(`${prefix}-export-svg`), copy = byId(`${prefix}-copy-citation`);
  const status = byId(`${prefix}-export-status`), details = byId(`${prefix}-figure-citation`);
  const text = byId(`${prefix}-citation-text`), svg = byId(`${prefix}-diagram`);
  exportControls.add(download); exportControls.add(copy);
  let ready = false;
  const controller = createFigureCitationController({buildCitation,
    writeText: typeof navigator.clipboard?.writeText === 'function' ? value => navigator.clipboard.writeText(value) : undefined,
    onChange(state) {
      text.value = state.text; copy.disabled = !ready || !state.canCopy;
      status.textContent = state.status;
      if (state.manualFallback) details.open = true;
    }
  });
  function capture() { return serialize(captureTeachingDiagram(svg), currentParameters()); }
  function refuse() {
    ready = false; download.disabled = true; controller.clear();
  }
  download.addEventListener('click', () => {
    if (!ready || !downloadSupported) return;
    let objectURL, link;
    try {
      const data = capture();
      objectURL = URL.createObjectURL(new Blob([data], {type:'image/svg+xml;charset=utf-8'}));
      link = document.createElement('a'); link.href = objectURL;
      link.download = `${prefix}-teaching-figure.svg`;
      document.body.append(link); link.click();
      status.textContent = 'SVG download started. This is a teaching figure.';
    } catch { refuse(); details.open = true; }
    finally {
      link?.remove();
      if (objectURL) setTimeout(() => URL.revokeObjectURL(objectURL), 1000);
    }
  });
  copy.addEventListener('click', () => {
    if (!ready) return;
    try { capture(); void controller.copy(); } catch { refuse(); }
  });
  return {refresh() {
    ready = false; download.disabled = true;
    try {
      capture(); ready = true; controller.setFigure(currentParameters());
      download.disabled = !downloadSupported;
      if (!downloadSupported) status.textContent = 'SVG download is unavailable in this browser. The figure citation remains readable below.';
    } catch { refuse(); }
  }};
}
const citation = createCitationController({
  writeText: typeof navigator.clipboard?.writeText === 'function' ? text => navigator.clipboard.writeText(text) : undefined,
  onChange(state) {
    citationText.value = state.text;
    copyCitationButton.disabled = !figureReady || !state.canCopy;
    exportStatus.textContent = state.status;
    if (state.manualFallback) citationDetails.open = true;
  }
});
function currentFigureParameters() { return {s:Number(center.value), R:Number(spread.value)}; }
function updateFigureTools() {
  figureReady = false; exportButton.disabled = true;
  try {
    // Validate actual displayed content before enabling either action. Export
    // recaptures it at the click, including the current light/dark theme.
    serializeCurvatureSVG(captureCurvatureDiagram(byId('curvature-diagram')), currentFigureParameters());
    figureReady = true;
    citation.setFigure(currentFigureParameters());
    exportButton.disabled = !downloadSupported;
    if (!downloadSupported) exportStatus.textContent = 'SVG download is unavailable in this browser. The figure citation can be copied manually below.';
  } catch { citation.clear(); }
}
exportButton.addEventListener('click', () => {
  if (!figureReady || !downloadSupported) return;
  let objectURL, link;
  try {
    const params = currentFigureParameters();
    const svg = serializeCurvatureSVG(captureCurvatureDiagram(byId('curvature-diagram')), params);
    objectURL = URL.createObjectURL(new Blob([svg], {type:'image/svg+xml;charset=utf-8'}));
    link = document.createElement('a');
    link.href = objectURL;
    link.download = `curvature-s${params.s}-R${params.R}.svg`;
    document.body.append(link); link.click();
    exportStatus.textContent = 'Curvature SVG download started. It is an illustrative teaching figure.';
  } catch {
    figureReady = false; exportButton.disabled = true; citation.clear();
    exportStatus.textContent = 'SVG download could not complete: the displayed diagram is unavailable or contains unsupported content.';
    citationDetails.open = true;
  } finally {
    link?.remove();
    // Allow the browser to consume the user-triggered download before release.
    if (objectURL) setTimeout(() => URL.revokeObjectURL(objectURL), 1000);
  }
});
copyCitationButton.addEventListener('click', () => {
  if (!figureReady) return;
  try {
    serializeCurvatureSVG(captureCurvatureDiagram(byId('curvature-diagram')), currentFigureParameters());
    void citation.copy();
  } catch { figureReady = false; exportButton.disabled = true; citation.clear(); }
});
function drawCone() {
  const model = coneModel(Number(center.value), Number(spread.value));
  const labels = {
    peak: ['Downward in both directions', 'Both curvatures are negative: this quadratic model bends downward in every direction. At a stationary point, a negative-definite Hessian identifies a local peak.'],
    saddle: ['Down one way. Up the other.', 'The curvatures have opposite signs. This quadratic model bends downward in one principal direction and upward in the other: a saddle.'],
    bowl: ['Upward in both directions', 'Both curvatures are positive: this quadratic model bends upward in every direction. At a stationary point, a positive-definite Hessian identifies a local minimum.'],
    flat: ['At least one direction is flat', 'At least one curvature is zero. This lies on the boundary: the second-order test alone cannot classify the original surface.']
  };
  byId('curvature-center-value').value = number(model.s, 1);
  byId('curvature-spread-value').value = number(model.radius, 1);
  byId('eigenvalue-first').textContent = number(model.eigenvalues[0], 1);
  byId('eigenvalue-second').textContent = number(model.eigenvalues[1], 1);
  byId('curvature-kind').textContent = labels[model.kind][0];
  byId('curvature-summary').textContent = `${labels[model.kind][1]} The two values are ${number(model.eigenvalues[0], 1)} and ${number(model.eigenvalues[1], 1)}.`;
  const svg = byId('curvature-diagram');
  clear(svg, 'Where the two curvatures change sign', `Center ${number(model.s, 1)}, spread ${number(model.radius, 1)}. ${labels[model.kind][0]}. Eigenvalues ${model.eigenvalues.map(value => number(value, 1)).join(' and ')}.`);
  const x = r => 86 + r * 155;
  const y = s => 165 - s * 40;
  svg.append(shape('polygon', { points: '86,165 474,265 474,294 86,294', class: 'diagram-fill' }));
  svg.append(shape('polygon', { points: '86,36 474,36 474,65 86,165', class: 'diagram-warm' }));
  for (const s of [-3, 0, 3]) {
    line(svg, 86, y(s), 474, y(s), 'diagram-grid');
    write(svg, 66, y(s) + 6, number(s, 0), 'diagram-small', { 'text-anchor': 'end' });
  }
  line(svg, 86, 31, 86, 295);
  line(svg, 86, 295, 498, 295);
  line(svg, x(0), y(0), x(2.5), y(-2.5), 'diagram-boundary');
  line(svg, x(0), y(0), x(2.5), y(2.5), 'diagram-boundary');
  write(svg, 47, 28, 's');
  write(svg, 514, 302, 'R');
  for (const r of [0, 1, 2]) write(svg, x(r), 320, String(r), 'diagram-small', { 'text-anchor': 'middle' });
  write(svg, 225, 65, 'Bowl');
  write(svg, 335, 172, 'Saddle');
  write(svg, 224, 278, 'Peak');
  write(svg, 483, 70, 's = R', 'diagram-small');
  write(svg, 483, 265, 's = −R', 'diagram-small');
  svg.append(shape('circle', { cx: x(model.radius), cy: y(model.s), r: 9, class: 'diagram-point' }));
  updateFigureTools();
}

const distance = byId('pin-distance');
const regions = [...document.querySelectorAll('input[name="region"]')];
const pinTools = teachingFigureTools('pin', () => ({r:Number(distance.value),region:regions.find(input => input.checked).value}), serializePinSVG, pinFigureCitation);
function drawPins() {
  const model = pinModel({ r: Number(distance.value) });
  const remote = regions.find(input => input.checked).value === 'remote';
  byId('pin-distance-value').value = number(model.r);
  byId('pin-separation').textContent = number(model.r);
  byId('pin-gap').textContent = number(model.gap, 6);
  byId('pin-region-title').textContent = remote ? 'The remote boundary stays put' : 'The ring follows the points inward';
  const description = `Separation ${number(model.r)}, height gap ${number(model.gap, 6)}. The ring spans radii ${number(model.annulus[0])} to ${number(model.annulus[1])}. The remote boundary stays at 3.00.`;
  byId('pin-summary').textContent = `${description} ${remote ? 'The remote result also restricts height to the narrow blue window.' : 'The fixed-annulus result allows all heights within that ring.'}`;
  const svg = byId('pin-diagram');
  clear(svg, 'Comparing shrinking space with a fixed exclusion', description);
  const cx = 202, cy = 172, scale = 39;
  write(svg, 202, 27, 'SPACE', 'diagram-small', { 'text-anchor': 'middle' });
  write(svg, 499, 27, 'HEIGHT', 'diagram-small', { 'text-anchor': 'middle' });
  if (remote) {
    const radius = model.remoteRadius * scale;
    svg.append(shape('path', { d: `M24 40H380V302H24Z M${cx} ${cy - radius}a${radius} ${radius} 0 1 0 0 ${2 * radius}a${radius} ${radius} 0 1 0 0 ${-2 * radius}Z`, 'fill-rule': 'evenodd', class: 'diagram-fill' }));
  } else {
    const [inner, outer] = model.annulus.map(radius => radius * scale);
    svg.append(shape('circle', { cx, cy, r: (inner + outer) / 2, fill: 'none', stroke: 'var(--ul-link)', 'stroke-opacity': '.2', 'stroke-width': outer - inner }));
    for (const radius of [inner, outer]) svg.append(shape('circle', { cx, cy, r: radius, class: 'diagram-line' }));
  }
  line(svg, 26, cy, 378, cy, 'diagram-grid');
  line(svg, cx, 42, cx, 300, 'diagram-grid');
  svg.append(shape('circle', { cx, cy, r: model.remoteRadius * scale, class: 'diagram-boundary' }));
  write(svg, 206, 59, 'ρ = 3', 'diagram-small');
  model.pins.forEach((pin, i) => {
    const px = cx + pin * scale;
    line(svg, px, cy + 7, i ? 240 : 164, 252, 'diagram-axis');
    svg.append(shape('circle', { cx: px, cy, r: 5, class: 'diagram-point' }));
    write(svg, i ? 246 : 154, 277, i ? 'S' : 'M', '', { 'text-anchor': 'middle' });
  });
  write(svg, 202, 330, remote ? 'Outside the fixed boundary' : `Ring: ${number(model.annulus[0])} to ${number(model.annulus[1])}`, 'diagram-small', { 'text-anchor': 'middle' });
  const top = 78, bottom = top + model.gap * 860;
  line(svg, 430, 56, 430, 307, 'diagram-axis');
  svg.append(shape('rect', { x: 448, y: top, width: 113, height: bottom - top, class: 'diagram-fill' }));
  for (const y of [top, bottom]) line(svg, 448, y, 561, y, 'diagram-line');
  write(svg, 503, 62, 'b = 1', 'diagram-small', { 'text-anchor': 'middle' });
  write(svg, 503, Math.max(124, bottom + 30), 'b − r³', 'diagram-small', { 'text-anchor': 'middle' });
  write(svg, 503, 330, 'Fixed height scale', 'diagram-small', { 'text-anchor': 'middle' });
  pinTools.refresh();
}

const objects = [...document.querySelectorAll('.object-choices input')];
const paletteTools = teachingFigureTools('palette', () => ({objects:objects.filter(input => input.checked).map(input => Number(input.value) + 1)}), serializePaletteSVG, paletteFigureCitation);
function drawPalette() {
  const model = paletteModel(objects.filter(input => input.checked).map(input => Number(input.value)));
  const names = model.selected.map(value => value + 1);
  const assignments = model.parts?.map((part, i) => `Group ${i + 1}: ${part.length ? `object ${part[0] + 1}` : 'empty'}`).join('. ');
  const summary = model.decomposable
    ? `${names.length} included ${names.length === 1 ? 'object fits' : 'objects fit'} into the two capacity-one groups. ${assignments}.`
    : 'Three included objects need three places, but only two are available. Removing any one object makes a valid assignment possible.';
  byId('palette-kind').textContent = model.decomposable ? (names.length ? 'Everything has a place' : 'Empty groups satisfy the rule too') : 'One object has no place';
  byId('palette-summary').textContent = summary;
  const svg = byId('palette-diagram');
  clear(svg, 'Assigning objects to two groups', summary);
  const centers = [112, 300, 488];
  for (let i = 0; i < 3; i++) {
    const selected = model.selected.includes(i);
    svg.append(shape('circle', { cx: centers[i], cy: 60, r: 27, class: selected ? 'diagram-object' : 'diagram-muted-object' }));
    write(svg, centers[i], 67, String(i + 1), '', { 'text-anchor': 'middle' });
    if (!selected) write(svg, centers[i], 110, 'removed', 'diagram-small', { 'text-anchor': 'middle' });
  }
  // A failed triple displays an attempted placement plus its unavoidable extra.
  // It is not presented as a valid partition.
  const parts = model.parts || [[0], [1]];
  for (let i = 0; i < 2; i++) {
    const x = 36 + i * 208;
    svg.append(shape('rect', { x, y: 157, width: 178, height: 90, rx: 12, class: 'diagram-box' }));
    write(svg, x + 89, 279, `Group ${i + 1}`, '', { 'text-anchor': 'middle' });
    if (parts[i].length) {
      const selected = parts[i][0];
      line(svg, centers[selected], 89, x + 89, 157, 'diagram-line');
      svg.append(shape('circle', { cx: x + 89, cy: 202, r: 25, class: 'diagram-object' }));
      write(svg, x + 89, 209, String(selected + 1), '', { 'text-anchor': 'middle' });
    } else write(svg, x + 89, 209, 'empty', 'diagram-small', { 'text-anchor': 'middle' });
  }
  if (!model.decomposable) {
    line(svg, 488, 88, 510, 169, 'diagram-boundary');
    svg.append(shape('circle', { cx: 510, cy: 203, r: 26, class: 'diagram-point' }));
    write(svg, 510, 210, '3', 'diagram-on-accent', { 'text-anchor': 'middle' });
    write(svg, 510, 279, 'no place', 'diagram-small', { 'text-anchor': 'middle' });
  }
  paletteTools.refresh();
}

const shareLink = byId('explore-state-link');
const stateStatus = byId('explore-state-status');
function currentState() {
  return { s: Number(center.value), R: Number(spread.value), r: Number(distance.value),
    region: regions.find(input => input.checked).value,
    objects: objects.filter(input => input.checked).map(input => Number(input.value) + 1) };
}
function updateLink() {
  shareLink.href = exploreStateURL(location.href, currentState());
}
function commitState() {
  updateLink();
  try {
    if (shareLink.href !== location.href) history.pushState(null, '', shareLink.href);
    stateStatus.textContent = 'These settings are in the page address. Share the link to reopen this teaching example.';
  } catch {
    stateStatus.textContent = 'The page address could not be updated. Use “Link to these settings” to share this teaching example.';
  }
}
function restoreState() {
  const { state, invalid } = readExploreState(location.search);
  center.value = state.s; spread.value = state.R; distance.value = state.r;
  regions.forEach(input => { input.checked = input.value === state.region; });
  objects.forEach(input => { input.checked = state.objects.includes(Number(input.value) + 1); });
  drawCone(); drawPins(); drawPalette(); updateLink();
  stateStatus.textContent = invalid.length
    ? `Some link settings were invalid (${invalid.join(', ')}). Those controls use their starting values; the link below shares the settings shown.`
    : 'Settings you change below are saved in the page address; use “Link to these settings” to share them.';
}
for (const [control, draw] of [[center, drawCone], [spread, drawCone], [distance, drawPins]]) {
  control.addEventListener('input', () => { draw(); updateLink(); });
  control.addEventListener('change', commitState);
}
document.querySelectorAll('[data-curvature-preset]').forEach(button => {
  button.addEventListener('click', () => applyCurvaturePreset(button.dataset.curvaturePreset,
    {center, spread, draw: drawCone, commit: commitState}));
});
byId('curvature-reset').addEventListener('click', () => { center.value = '-2'; spread.value = '1'; drawCone(); commitState(); });
regions.forEach(input => input.addEventListener('change', () => { drawPins(); commitState(); }));
byId('pin-half').addEventListener('click', () => { distance.value = Number(distance.value) === 0.25 ? '0.5' : '0.25'; drawPins(); commitState(); });
byId('pin-reset').addEventListener('click', () => { distance.value = '0.5'; regions[0].checked = true; drawPins(); commitState(); });
objects.forEach(input => input.addEventListener('change', () => { drawPalette(); commitState(); }));
byId('palette-reset').addEventListener('click', () => { objects.forEach(input => { input.checked = true; }); drawPalette(); commitState(); });
window.addEventListener('popstate', restoreState);
window.addEventListener('hashchange', updateLink);

restoreState();
shareLink.hidden = false;
document.querySelectorAll('input[disabled], button[disabled], fieldset[disabled]').forEach(control => {
  if (!exportControls.has(control)) control.disabled = false;
});
