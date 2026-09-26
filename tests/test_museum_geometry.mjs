import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const moduleURL = new URL('../docs/site/geometry.mjs', import.meta.url);
assert.ok(fs.existsSync(moduleURL), 'The scope-safe museum geometry module must exist');
const { ec014Model, remoteModel, annulusModel, p15Model, mountGeometry } = await import(moduleURL);

test('EC014 rejects degenerate pins and retains the declared raw and corrected row orders', () => {
  for (const radius of [0, -1, NaN, Infinity]) assert.throws(() => ec014Model(radius));
  const m = ec014Model(2);
  assert.deepEqual(m.pins, [{ name: 'M', t: -1, s: 0 }, { name: 'S', t: 1, s: 0 }]);
  assert.deepEqual(m.rawRows, ['f(M)', 'f(S)', 'f_t(M)', 'f_s(M)', 'f_t(S)', 'f_s(S)']);
  assert.deepEqual(m.correctedRows, ['V+', 'V-_corr', 'G_t+', 'G_t-', 'G_s+', 'G_s-']);
});

test('remote membership keeps rho fixed and excludes all heights outside the open pin window', () => {
  const remote = remoteModel({ L: 12, rho: 2 });
  assert.equal(remote.contains({ distance: 2, height: 9.5, b: 10, k: 1, r: 1 }), true);
  for (const [distance, height] of [[1.999, 9.5], [2, 9], [2, 10], [2, 12]]) {
    assert.equal(remote.contains({ distance, height, b: 10, k: 1, r: 1 }), false);
  }
  assert.equal(remote.contains({ distance: 1.99, height: 9.99, b: 10, k: 1, r: 0.5 }), false);
  assert.throws(() => remoteModel({ L: 12, rho: 3 }));
});

test('annulus membership uses scaled coordinates with fixed A and B in dimension two', () => {
  const annulus = annulusModel({ A: 2, B: 4, dimension: 2 });
  assert.equal(annulus.containsPhysical({ x: 1, y: 0, r: 0.5 }), true);
  assert.equal(annulus.containsPhysical({ x: 2, y: 0, r: 0.5 }), true);
  assert.equal(annulus.containsPhysical({ x: 0.25, y: 0, r: 0.5 }), false);
  assert.equal(annulus.containsPhysical({ x: 2.01, y: 0, r: 0.5 }), false);
  for (const value of [{ A: 1, B: 4 }, { A: 2, B: 2 }, { A: 2, B: Infinity }, { A: 2, B: 4, dimension: 3 }]) {
    assert.throws(() => annulusModel(value));
  }
  assert.equal(annulus.heightScope, 'all heights');
});

test('P15 exact finite example partitions every proper subset and rejects the full triple', () => {
  for (const selection of [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]]) {
    const model = p15Model(selection);
    assert.equal(model.decomposable, true);
    assert.equal(model.parts.length, 2);
    assert.ok(model.parts.every(part => part.length <= 1));
    assert.deepEqual(model.parts.flat().sort(), selection);
  }
  const full = p15Model([0, 1, 2]);
  assert.equal(full.decomposable, false);
  assert.equal(full.parts, null);
  assert.deepEqual(full.macroEdges, []);
  assert.throws(() => p15Model([0, 3]));
});

// Minimal DOM: observes the module's actual rendered scope, interaction and
// network decision. It does not assert browser layout or GPU behavior.
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {}; this.style = {}; this._text = ''; this.hidden = false; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent ?? '').join(''); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this._text = ''; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(key, callback) { this.listeners[key] = callback; }
  removeEventListener() {}
  remove() {}
  querySelectorAll(tag) { return this.children.flatMap(child => [...(child.tagName === tag ? [child] : []), ...(child.querySelectorAll?.(tag) ?? [])]); }
}
let scripts = [];
globalThis.document = {
  createElement: tag => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
  head: { append(script) { scripts.push(script); queueMicrotask(() => script.onerror()); } }
};
const paths = {
  ec014: 'imports/hardening_ebedb780/EC-014/proof.md.export.txt',
  remote: 'frontiers/remote_window_20260924/PROOF.md',
  annulus: 'frontiers/rn_annulus_bridge_20260925/PROOF.md',
  p15: 'frontiers/full_price_20260924/PROOF.md'
};
const source = kind => ({ repository: 'd6g8k5htny-coder/Math-', path: paths[kind], commit: 'd6628da09384728992dcbe6e921cc28ba85aebb0', sha256: 'a'.repeat(64), bytes: 42 });

test('unsupported kinds and mismatched source descriptors do not render geometry or request CDN code', async () => {
  const host = new Element('div');
  await assert.rejects(mountGeometry(host, 'lifetime', source('ec014')));
  await assert.rejects(mountGeometry(host, 'remote', source('annulus')));
  await assert.rejects(mountGeometry(host, 'remote', { ...source('remote'), commit: 'main' }));
  assert.equal(host.children.length, 0);
  assert.equal(scripts.length, 0);
});

test('all static exhibits render with no WebGL or CDN dependency and preserve excluded regions', async () => {
  for (const kind of ['remote', 'annulus', 'p15']) {
    const host = new Element('div');
    await mountGeometry(host, kind, source(kind));
    assert.ok(host.querySelectorAll('svg').length > 0);
    assert.equal(host.querySelectorAll('canvas').length, 0);
    if (kind === 'remote') assert.match(host.textContent, /between-pin heights/);
    if (kind === 'annulus') {
      assert.match(host.textContent, /all heights/);
      assert.ok(host.querySelectorAll('circle').some(circle => circle.attributes['stroke-dasharray']));
      assert.match(host.textContent, /OPEN/);
    }
  }
  assert.equal(scripts.length, 0);
});

test('P15 vertex toggles change the actual decomposition result', async () => {
  const host = new Element('div');
  await mountGeometry(host, 'p15', source('p15'));
  assert.match(host.textContent, /full triple is not two-decomposable/);
  const vertex = host.querySelectorAll('button').find(button => button.attributes['data-vertex'] === '2');
  vertex.listeners.click();
  assert.match(host.textContent, /proper subset is two-decomposable/);
  assert.equal(vertex.attributes['aria-pressed'], 'false');
});

test('EC014 retains complete two-dimensional content when WebGL is missing or CDN fails', async () => {
  delete globalThis.WebGLRenderingContext;
  const first = new Element('div');
  await mountGeometry(first, 'ec014', source('ec014'));
  assert.ok(first.querySelectorAll('svg').length > 0);
  assert.match(first.textContent, /f_t\(M\)/);
  assert.equal(scripts.length, 0);
  globalThis.WebGLRenderingContext = class {};
  const second = new Element('div');
  await mountGeometry(second, 'ec014', source('ec014'));
  assert.equal(scripts.length, 1);
  assert.match(scripts[0].src, /three@0\.160\.0\/build\/three\.js$/);
  assert.match(scripts[0].integrity, /^sha384-/);
  assert.equal(scripts[0].crossOrigin, 'anonymous');
  assert.ok(second.querySelectorAll('svg').length > 0);
  assert.match(second.textContent, /2D/);
  delete globalThis.WebGLRenderingContext;
});
