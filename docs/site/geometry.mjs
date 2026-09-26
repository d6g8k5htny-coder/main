// Illustrations only. Source bytes are verified by the detail-page caller before
// mounting; this module additionally binds each visual to its declared source.
export const THREE_CDN = Object.freeze({
  version: '160',
  url: 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.js',
  integrity: 'sha384-ADduDXZ84mYNs9BwgDbfsq8503KaK0OzR6s3AFB6fclZoaXtLa7zmoXdmbFSEptM'
});

const SOURCE_PATHS = Object.freeze({
  ec014: 'imports/hardening_ebedb780/EC-014/proof.md.export.txt',
  remote: 'frontiers/remote_window_20260924/PROOF.md',
  annulus: 'frontiers/rn_annulus_bridge_20260925/PROOF.md',
  p15: 'frontiers/full_price_20260924/PROOF.md'
});
const SVG_NS = 'http://www.w3.org/2000/svg';
const ink = '#162f36', teal = '#07596c', amber = '#986323', pale = '#e0ece8';

function positive(value, name) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and positive`);
  return value;
}

export function ec014Model(r = 1) {
  positive(r, 'r');
  return {
    pins: [{ name: 'M', t: -r / 2, s: 0 }, { name: 'S', t: r / 2, s: 0 }],
    rawRows: ['f(M)', 'f(S)', 'f_t(M)', 'f_s(M)', 'f_t(S)', 'f_s(S)'],
    correctedRows: ['V+', 'V-_corr', 'G_t+', 'G_t-', 'G_s+', 'G_s-']
  };
}

// These membership helpers describe scope; they do not estimate the field,
// evaluate Kac–Rice, supply theorem constants, or replay an analytic proof.
export function remoteModel({ L, rho }) {
  positive(L, 'L'); positive(rho, 'rho');
  if (rho >= L / 4) throw new Error('The fixed exclusion radius must satisfy 0 < rho < L/4');
  return {
    L, rho, heightScope: 'between-pin heights only',
    contains({ distance, height, b, k, r }) {
      positive(k, 'k'); positive(r, 'r');
      if (![distance, height, b].every(Number.isFinite) || distance < 0) return false;
      return distance >= rho && height > b - k * r ** 3 && height < b;
    }
  };
}

export function annulusModel({ A, B, dimension = 2 }) {
  if (dimension !== 2 || !Number.isFinite(A) || !Number.isFinite(B) || A <= 1 || B <= A)
    throw new Error('The fixed annulus requires dimension 2 and finite 1 < A < B');
  return {
    A, B, dimension, heightScope: 'all heights',
    containsPhysical({ x, y, r }) {
      positive(r, 'r');
      if (![x, y].every(Number.isFinite)) return false;
      const scaledRadius = Math.hypot(x / r, y / r);
      return scaledRadius >= A && scaledRadius <= B;
    }
  };
}

export function p15Model(selection = [0, 1, 2]) {
  if (!Array.isArray(selection) || selection.some(v => !Number.isInteger(v) || v < 0 || v > 2) || new Set(selection).size !== selection.length)
    throw new Error('Select a subset of the three original coordinates');
  const selected = [...selection].sort();
  const decomposable = selected.length <= 2;
  return {
    selected, a: 1, d: 2, n: 3, K: 2, macroEdges: [], decomposable,
    parts: decomposable ? [selected.slice(0, 1), selected.slice(1, 2)] : null
  };
}

function element(tag, text, attributes = {}) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}
function shape(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function label(svg, x, y, text, attrs = {}) {
  svg.append(shape('text', { x, y, fill: ink, 'font-size': 17, 'font-family': 'system-ui, sans-serif', ...attrs }, text));
}
function line(svg, x1, y1, x2, y2, attrs = {}) {
  svg.append(shape('line', { x1, y1, x2, y2, stroke: ink, 'stroke-width': 2, ...attrs }));
}
function visual(title, description, viewBox = '0 0 720 360') {
  const figure = element('figure', undefined, { class: 'museum-visual illustration' });
  const svg = shape('svg', { viewBox, role: 'img', 'aria-label': `${title}. ${description}` });
  svg.append(shape('title', {}, title), shape('desc', {}, description));
  figure.append(svg, element('figcaption', description));
  return { figure, svg };
}
function boundary(host, text) { host.append(element('p', text, { class: 'boundary' })); }

function ec014Plan() {
  const result = visual('EC014 two-pin coordinate plane', 'Illustration: pin separation represents r > 0; t is longitudinal and s is transverse. No field values or surface are invented.');
  const { svg } = result;
  svg.append(shape('rect', { x: 24, y: 24, width: 672, height: 296, rx: 12, fill: '#eef3ef', stroke: '#c6d4cd' }));
  line(svg, 65, 176, 662, 176, { stroke: '#829b94' });
  line(svg, 360, 50, 360, 300, { stroke: '#829b94' });
  label(svg, 668, 181, 't'); label(svg, 370, 55, 's');
  for (const [name, x, formula, color] of [['M', 225, '(−r/2, 0)', teal], ['S', 495, '(r/2, 0)', amber]]) {
    svg.append(shape('circle', { cx: x, cy: 176, r: 10, fill: color }));
    label(svg, x, 140, name, { 'text-anchor': 'middle', 'font-size': 23, 'font-weight': 700 });
    label(svg, x, 211, formula, { 'text-anchor': 'middle' });
  }
  line(svg, 225, 255, 495, 255);
  line(svg, 225, 248, 225, 262); line(svg, 495, 248, 495, 262);
  label(svg, 360, 280, 'r > 0', { 'text-anchor': 'middle' });
  return result.figure;
}

function addRows(host) {
  const model = ec014Model();
  const table = element('table', undefined, { 'aria-label': 'Declared ordered six-pin data and corrected frame' });
  const head = element('thead'); const row = element('tr');
  row.append(element('th', 'Order'), element('th', 'Raw vector X_r'), element('th', 'Corrected vector Y_r'));
  head.append(row); table.append(head);
  const body = element('tbody');
  model.rawRows.forEach((raw, i) => {
    const entry = element('tr');
    entry.append(element('td', String(i + 1)), element('td', raw), element('td', model.correctedRows[i]));
    body.append(entry);
  });
  table.append(body); host.append(table);
  host.append(element('p', 'Source identity: det T_r = −r⁻⁵ in these exact row and column orders. The diagram illustrates the six data labels; it does not replay the Jacobian proof or compute a full contact coefficient.', { class: 'muted' }));
}

function drawRemote(host) {
  const { figure, svg } = visual('Fixed remote region and between-pin height window', 'Schematic on the torus: the fixed exterior Dρ is paired only with the shrinking open height window. Diagram distances and heights are not numerical estimates.');
  svg.append(shape('path', { d: 'M24 45H434V315H24Z M229 85A95 95 0 1 0 229 275A95 95 0 1 0 229 85Z', 'fill-rule': 'evenodd', fill: pale }));
  svg.append(shape('rect', { x: 24, y: 45, width: 410, height: 270, fill: 'none', stroke: '#b1c7bd', 'stroke-width': 2 }));
  svg.append(shape('circle', { cx: 229, cy: 180, r: 95, fill: 'none', stroke: teal, 'stroke-width': 2 }));
  label(svg, 42, 75, 'Dρ: dist_X(x, 0) ≥ ρ', { 'font-size': 18, 'font-weight': 700 });
  label(svg, 229, 229, 'excluded interior', { 'text-anchor': 'middle', fill: '#586d6e', 'font-size': 15 });
  line(svg, 229, 180, 308, 129); label(svg, 282, 141, 'ρ fixed', { 'font-size': 15 });
  for (const [x, name, color] of [[203, 'M', teal], [255, 'S', amber]]) {
    svg.append(shape('circle', { cx: x, cy: 180, r: 6, fill: color })); label(svg, x, 164, name, { 'text-anchor': 'middle' });
  }
  label(svg, 229, 341, '0 < ρ < L/4; ρ does not shrink with r', { 'text-anchor': 'middle', 'font-size': 15 });
  svg.append(shape('rect', { x: 495, y: 126, width: 184, height: 94, fill: pale }));
  line(svg, 480, 58, 480, 300, { stroke: '#829b94' });
  line(svg, 495, 126, 679, 126, { 'stroke-dasharray': '5 5', stroke: teal });
  line(svg, 495, 220, 679, 220, { 'stroke-dasharray': '5 5', stroke: teal });
  label(svg, 590, 82, 'HEIGHT', { 'text-anchor': 'middle', 'font-size': 15 });
  label(svg, 590, 111, 'b', { 'text-anchor': 'middle' });
  label(svg, 590, 252, 'b − kr³', { 'text-anchor': 'middle' });
  label(svg, 590, 170, 'open window', { 'text-anchor': 'middle', 'font-size': 16 });
  label(svg, 590, 196, 'length kr³', { 'text-anchor': 'middle', 'font-size': 16 });
  host.append(figure);
  boundary(host, 'FIXED REMOTE SCOPE — between-pin heights only: b − kr³ < f(x) < b, E ⊆ Dρ and fixed 0 < ρ < L/4. No fixed-remote all-height estimate is shown.');
}

function drawAnnulus(host) {
  const { figure, svg } = visual('Fixed scaled annulus and open omitted regions', 'Dimension 2; physical points belong to rE for E ⊆ K_AB and fixed 1 < A < B < ∞. The shaded annulus permits all heights. Dashed regions remain OPEN; radii are symbolic.', '0 0 720 400');
  svg.append(shape('path', { d: 'M260 62A140 140 0 1 0 260 342A140 140 0 1 0 260 62Z M260 114A88 88 0 1 0 260 290A88 88 0 1 0 260 114Z', 'fill-rule': 'evenodd', fill: pale }));
  for (const radius of [88, 140]) svg.append(shape('circle', { cx: 260, cy: 202, r: radius, fill: 'none', stroke: teal, 'stroke-width': 2 }));
  svg.append(shape('circle', { cx: 260, cy: 202, r: 177, fill: 'none', stroke: amber, 'stroke-width': 2, 'stroke-dasharray': '7 6' }));
  line(svg, 260, 202, 323, 141, { stroke: '#68857a' }); label(svg, 289, 181, 'Ar', { 'font-size': 16 });
  line(svg, 260, 202, 369, 289, { stroke: '#68857a' }); label(svg, 334, 250, 'Br', { 'font-size': 16 });
  for (const [x, name, color] of [[231, 'M', teal], [289, 'S', amber]]) {
    svg.append(shape('circle', { cx: x, cy: 202, r: 20, fill: 'none', stroke: amber, 'stroke-width': 2, 'stroke-dasharray': '5 4' }));
    svg.append(shape('circle', { cx: x, cy: 202, r: 5, fill: color }));
    label(svg, x, 234, name, { 'text-anchor': 'middle', 'font-size': 15 });
  }
  label(svg, 260, 104, 'rE: all heights', { 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 700 });
  label(svg, 490, 65, 'OPEN REGIONS', { fill: amber, 'font-weight': 700 });
  label(svg, 490, 99, 'Pin neighborhoods', { 'font-size': 16 });
  label(svg, 490, 129, 'Intermediate scales', { 'font-size': 16 });
  label(svg, 490, 152, 'r ≪ distance ≪ ρ', { 'font-size': 15 });
  svg.append(shape('rect', { x: 480, y: 177, width: 220, height: 147, rx: 8, fill: 'none', stroke: amber, 'stroke-width': 2, 'stroke-dasharray': '7 6' }));
  for (const x of [574, 604]) svg.append(shape('circle', { cx: x, cy: 219, r: 24, fill: 'none', stroke: amber, 'stroke-width': 2, 'stroke-dasharray': '4 4' }));
  label(svg, 590, 269, 'Witness–witness', { 'text-anchor': 'middle', 'font-size': 16 });
  label(svg, 590, 293, 'shrinking separation', { 'text-anchor': 'middle', 'font-size': 15 });
  host.append(figure);
  boundary(host, 'FIXED ANNULUS CANDIDATE — d = 2, fixed 1 < A < B < ∞, physical rE, all heights. Dashed pin neighborhoods, intermediate scales and witness–witness collisions remain OPEN. This is not a solved full plane or all-scales RN statement.');
}

function drawP15(host) {
  const selected = new Set([0, 1, 2]);
  let paletteReversed = false;
  const { figure, svg } = visual('Three-coordinate realized capacity family', 'One original block with a = 1, d = 2, n = 3, K = 2 and no macro edges. Lines show pair capacity conflicts; they are not macro-clutter edges.');
  const controls = element('div', undefined, { class: 'geometry-controls', role: 'group', 'aria-label': 'Choose a subset of the three coordinates' });
  const status = element('p', undefined, { role: 'status', 'aria-live': 'polite' });
  const palette = element('p');
  const vertices = [];
  function redraw() {
    const model = p15Model([...selected]);
    svg.replaceChildren(shape('title', {}, 'Capacity-one conflicts on three coordinates'));
    const positions = [[160, 92], [65, 268], [255, 268]];
    for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) line(svg, ...positions[a], ...positions[b], { stroke: selected.has(a) && selected.has(b) ? '#879c95' : '#cdd7d2', 'stroke-width': 3, 'stroke-dasharray': selected.has(a) && selected.has(b) ? 'none' : '5 5' });
    const colors = paletteReversed ? [amber, teal] : [teal, amber];
    positions.forEach(([x, y], i) => {
      const slot = model.parts?.findIndex(part => part.includes(i)) ?? -1;
      const fill = !selected.has(i) ? '#f6f5f0' : slot >= 0 ? colors[slot] : '#58746d';
      svg.append(shape('circle', { cx: x, cy: y, r: 29, fill, stroke: ink, 'stroke-width': 2 }));
      label(svg, x, y + 7, `x${i + 1}`, { 'text-anchor': 'middle', fill: selected.has(i) ? '#ffffff' : ink, 'font-size': 21, 'font-weight': 700 });
      vertices[i]?.setAttribute('aria-pressed', String(selected.has(i)));
    });
    label(svg, 355, 92, 'Two palette classes', { 'font-size': 21, 'font-weight': 700 });
    label(svg, 355, 126, 'Each can contain at most one', { 'font-size': 17 });
    label(svg, 355, 154, 'selected coordinate.', { 'font-size': 17 });
    for (let i = 0; i < 2; i++) {
      const y = 195 + i * 58;
      svg.append(shape('rect', { x: 355, y, width: 305, height: 43, rx: 6, fill: 'none', stroke: colors[i], 'stroke-width': 3 }));
      const names = model.parts ? model.parts[i].map(v => `x${v + 1}`).join(', ') || 'empty' : 'no valid partition';
      label(svg, 371, y + 28, `Palette ${i + 1}: ${names}`, { 'font-size': 16, fill: colors[i] });
    }
    status.textContent = model.decomposable
      ? `Selected ${model.selected.length} of 3: this proper subset is two-decomposable.`
      : 'Selected 3 of 3: the full triple is not two-decomposable.';
    palette.textContent = 'K = 2; each palette class is a member of D and has size at most a = 1.';
  }
  for (let i = 0; i < 3; i++) {
    const button = element('button', `Toggle x${i + 1}`, { type: 'button', 'data-vertex': i, 'aria-pressed': 'true' });
    button.addEventListener('click', () => { selected.has(i) ? selected.delete(i) : selected.add(i); redraw(); });
    vertices.push(button); controls.append(button);
  }
  const swap = element('button', 'Swap palette colors', { type: 'button' });
  swap.addEventListener('click', () => { paletteReversed = !paletteReversed; redraw(); });
  controls.append(swap);
  host.append(figure, controls, status, palette);
  boundary(host, 'REALIZED FAMILY ONLY — full-price proof §5: one block, a = 1, d = 2, n = 3, no macro edges, K = 2. This finite illustration does not assert an unrestricted downset or prize result.');
  redraw();
}

let threePromise;
function loadThree() {
  if (threePromise) return threePromise;
  threePromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = THREE_CDN.url;
    script.integrity = THREE_CDN.integrity;
    script.crossOrigin = 'anonymous';
    script.referrerPolicy = 'no-referrer';
    script.async = true;
    const timeout = setTimeout(() => reject(new Error('Three.js CDN timed out')), 15000);
    script.onload = () => {
      clearTimeout(timeout);
      if (globalThis.THREE?.REVISION !== THREE_CDN.version) reject(new Error('Unexpected Three.js revision'));
      else resolve(globalThis.THREE);
    };
    script.onerror = () => { clearTimeout(timeout); reject(new Error('Three.js CDN unavailable or integrity failed')); };
    document.head.append(script);
  });
  return threePromise;
}

let activeSceneCleanup = null;
let sceneRequest = 0;
async function mountEC014(host) {
  const request = ++sceneRequest;
  activeSceneCleanup?.();
  const fallback = ec014Plan();
  const status = element('p', '2D coordinate diagram. Checking optional 3D view…', { role: 'status', 'aria-live': 'polite', class: 'muted' });
  const stage = element('div', undefined, { class: 'geometry-3d-stage' });
  stage.hidden = true;
  stage.style.position = 'relative';
  stage.style.width = '100%';
  const controls = element('div', undefined, { class: 'geometry-controls' });
  host.append(stage, fallback, status, controls);
  addRows(host);
  boundary(host, 'ILLUSTRATION — camera orbit changes the view only. The pins and six data labels retain their source definitions. No invented field surface, Jacobian proof replay or full contact coefficient is shown.');
  if (!globalThis.WebGLRenderingContext && !globalThis.WebGL2RenderingContext) {
    status.textContent = '2D fallback: WebGL is unavailable. All pin coordinates and six data labels remain visible.';
    return { destroy() {} };
  }
  let renderer, observer, disposed = false;
  const disposables = [];
  let cleanup = () => {
    if (disposed) return;
    disposed = true;
    observer?.disconnect();
    for (const resource of disposables) resource.dispose();
    renderer?.dispose();
    renderer?.forceContextLoss();
    if (activeSceneCleanup === cleanup) activeSceneCleanup = null;
  };
  try {
    const THREE = await loadThree();
    if (request !== sceneRequest) return { destroy() {} };
    const canvas = element('canvas', undefined, { tabindex: '0', role: 'img', 'aria-label': 'EC014 plane and pins. Drag to orbit, or use arrow keys. Camera movement changes the view only.' });
    canvas.style.width = '100%'; canvas.style.height = '350px'; canvas.style.display = 'block'; canvas.style.touchAction = 'none';
    // One renderer owns one canvas/context. No capability-probe canvas is made.
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
    renderer.setClearColor('#eef3ef');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 40);
    const own = resource => { disposables.push(resource); return resource; };
    const plane = new THREE.Mesh(own(new THREE.PlaneGeometry(3, 1.7)), own(new THREE.MeshBasicMaterial({ color: '#dce7df', side: THREE.DoubleSide })));
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -0.02;
    scene.add(plane);
    function addLine(points, color) {
      const geometry = own(new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p))));
      const material = own(new THREE.LineBasicMaterial({ color }));
      scene.add(new THREE.Line(geometry, material));
    }
    addLine([[-1.4, 0, 0], [1.4, 0, 0]], '#829b94');
    addLine([[0, 0, -0.76], [0, 0, 0.76]], '#829b94');
    addLine([[-0.5, 0.012, 0], [0.5, 0.012, 0]], ink);
    const tags = [];
    for (const [name, x, color] of [['M = (−r/2, 0)', -0.5, teal], ['S = (r/2, 0)', 0.5, amber]]) {
      const pin = new THREE.Mesh(own(new THREE.SphereGeometry(0.065, 16, 12)), own(new THREE.MeshBasicMaterial({ color })));
      pin.position.set(x, 0, 0); scene.add(pin);
      const tag = element('span', name);
      Object.assign(tag.style, { position: 'absolute', transform: 'translate(-50%, -145%)', fontSize: '13px', fontWeight: '700', color, pointerEvents: 'none', whiteSpace: 'nowrap' });
      tags.push({ node: tag, position: new THREE.Vector3(x, 0, 0) });
    }
    for (const [name, position] of [['t', [1.35, 0, 0]], ['s', [0, 0, 0.7]]]) {
      const tag = element('span', name);
      Object.assign(tag.style, { position: 'absolute', transform: 'translate(-50%, -50%)', pointerEvents: 'none', fontWeight: '700' });
      tags.push({ node: tag, position: new THREE.Vector3(...position) });
    }
    stage.append(canvas, ...tags.map(t => t.node));
    let yaw = -0.12, elevation = 0.82, width = 720;
    function render() {
      if (disposed) return;
      camera.position.set(4 * Math.cos(elevation) * Math.sin(yaw), 4 * Math.sin(elevation), 4 * Math.cos(elevation) * Math.cos(yaw));
      camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
      renderer.render(scene, camera);
      for (const tag of tags) {
        const projected = tag.position.clone().project(camera);
        tag.node.style.left = `${(projected.x + 1) * width / 2}px`;
        tag.node.style.top = `${(1 - projected.y) * 350 / 2}px`;
      }
    }
    function resize() {
      width = Math.max(260, stage.clientWidth || host.clientWidth || 720);
      renderer.setSize(width, 350, false);
      camera.aspect = width / 350; camera.updateProjectionMatrix(); render();
    }
    let drag = null;
    canvas.addEventListener('pointerdown', event => {
      drag = { x: event.clientX, y: event.clientY, id: event.pointerId };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!drag || drag.id !== event.pointerId) return;
      yaw -= (event.clientX - drag.x) * 0.008;
      elevation = Math.max(0.2, Math.min(1.45, elevation + (event.clientY - drag.y) * 0.008));
      drag.x = event.clientX; drag.y = event.clientY; render();
    });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(type, () => { drag = null; });
    canvas.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'ArrowLeft') yaw -= 0.12;
      if (event.key === 'ArrowRight') yaw += 0.12;
      if (event.key === 'ArrowUp') elevation = Math.min(1.45, elevation + 0.1);
      if (event.key === 'ArrowDown') elevation = Math.max(0.2, elevation - 0.1);
      render();
    });
    canvas.addEventListener('webglcontextlost', () => {
      stage.hidden = true; fallback.hidden = false; controls.hidden = true;
      status.textContent = '2D fallback: the graphics context became unavailable. The source-defined geometry remains visible.';
      cleanup();
    });
    const reset = element('button', 'Reset camera', { type: 'button' });
    reset.addEventListener('click', () => { yaw = -0.12; elevation = 0.82; render(); });
    const toggle = element('button', 'Show 2D diagram', { type: 'button' });
    toggle.addEventListener('click', () => {
      stage.hidden = !stage.hidden; fallback.hidden = !stage.hidden;
      toggle.textContent = stage.hidden ? 'Show 3D view' : 'Show 2D diagram';
      if (!stage.hidden) resize();
    });
    controls.append(reset, toggle);
    stage.hidden = false; fallback.hidden = true;
    status.textContent = '3D coordinate plane: drag with mouse or touch, or focus the canvas and use arrow keys. Orbit is a camera control only.';
    activeSceneCleanup = cleanup;
    if (globalThis.ResizeObserver) { observer = new ResizeObserver(resize); observer.observe(stage); }
    resize();
    return { destroy: cleanup };
  } catch {
    cleanup(); stage.hidden = true; fallback.hidden = false;
    status.textContent = '2D fallback: the optional 3D library or WebGL could not load. All pin coordinates and six data labels remain visible.';
    return { destroy() {} };
  }
}

export async function mountGeometry(host, kind, source, _quote) {
  if (!Object.hasOwn(SOURCE_PATHS, kind)) throw new Error('Unsupported geometry exhibit');
  if (!source || source.repository !== 'd6g8k5htny-coder/Math-' || source.path !== SOURCE_PATHS[kind] ||
      !/^[0-9a-f]{40}$/.test(source.commit) || !/^[0-9a-f]{64}$/.test(source.sha256) ||
      !Number.isSafeInteger(source.bytes) || source.bytes <= 0) throw new Error('Geometry source identity does not match the exhibit');
  if (!host || typeof host.append !== 'function') throw new Error('A geometry host is required');
  host.setAttribute('data-illustration', kind);
  if (kind === 'ec014') return mountEC014(host);
  if (kind === 'remote') drawRemote(host);
  if (kind === 'annulus') drawAnnulus(host);
  if (kind === 'p15') drawP15(host);
  return { destroy() {} };
}
