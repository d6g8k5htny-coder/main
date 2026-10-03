import test from 'node:test';
import assert from 'node:assert/strict';
import * as exploreModels from '../docs/site/explore-models.mjs';
import { buildFigureMetadata, buildFigureCitation } from '../docs/site/curvature-export.mjs';
import { coneModel, pinModel, paletteModel } from '../docs/site/explore-models.mjs';

test('the cone classifies signs, retaining both zero-eigenvalue boundaries', () => {
  for (const [s, radius, eigenvalues, kind] of [
    [-2, 1, [-3, -1], 'peak'], [-1, 1, [-2, 0], 'flat'],
    [0, 1, [-1, 1], 'saddle'], [1, 1, [0, 2], 'flat'],
    [2, 1, [1, 3], 'bowl'], [0, 0, [0, 0], 'flat'],
    [-1, 0, [-1, -1], 'peak'], [1, 0, [1, 1], 'bowl']
  ]) {
    const result = coneModel(s, radius);
    assert.deepEqual(result.eigenvalues, eigenvalues);
    assert.equal(result.kind, kind);
  }
});

test('cone weight is determinant squared only on the negative-definite side', () => {
  assert.equal(coneModel(-2, 1).coneWeight, 9);
  assert.equal(coneModel(-1, 1).coneWeight, 0);
  assert.equal(coneModel(2, 1).coneWeight, 0);
  assert.equal(coneModel(0, 1).coneWeight, 0);
});

test('halving pin distance divides cubic gap by eight and annulus radii by two', () => {
  const large = pinModel({ r: 0.5, k: 2, b: 3, A: 2, B: 4, rho: 3, L: 24 });
  const small = pinModel({ r: 0.25, k: 2, b: 3, A: 2, B: 4, rho: 3, L: 24 });
  assert.deepEqual(large.pins, [-0.25, 0.25]);
  assert.equal(large.gap, 0.25);
  assert.equal(small.gap, 0.03125);
  assert.deepEqual(large.annulus, [1, 2]);
  assert.deepEqual(small.annulus, [0.5, 1]);
  assert.deepEqual(large.heightWindow, [2.75, 3]);
  assert.deepEqual(small.heightWindow, [2.96875, 3]);
  assert.equal(large.remoteRadius, 3);
  assert.equal(small.remoteRadius, 3);
});

test('remote membership uses fixed distance and an open height window', () => {
  const model = pinModel({ r: 0.5, k: 2, b: 3, A: 2, B: 4, rho: 3, L: 24 });
  assert.equal(model.containsRemote(3, 2.875), true);
  assert.equal(model.containsRemote(2.99, 2.875), false);
  assert.equal(model.containsRemote(3, 2.75), false);
  assert.equal(model.containsRemote(3, 3), false);
  assert.equal(model.containsAnnulus(1, 0), true);
  assert.equal(model.containsAnnulus(2, 0), true);
  assert.equal(model.containsAnnulus(0.9, 0), false);
  assert.equal(model.containsAnnulus(2.1, 0), false);
});

test('all eight three-object subsets receive the correct capacity-one partition', () => {
  const cases = [
    [[], [[], []]], [[0], [[0], []]], [[1], [[1], []]], [[2], [[2], []]],
    [[0, 1], [[0], [1]]], [[0, 2], [[0], [2]]], [[1, 2], [[1], [2]]],
    [[0, 1, 2], null]
  ];
  for (const [selected, partition] of cases) {
    const result = paletteModel(selected);
    assert.equal(result.decomposable, partition !== null);
    assert.deepEqual(result.parts, partition);
  }
  const unordered = [2, 0];
  assert.deepEqual(paletteModel(unordered).parts, [[0], [2]]);
  assert.deepEqual(unordered, [2, 0]);
});

test('models reject invalid or unrepresentable teaching parameters', () => {
  for (const args of [[NaN, 1], [0, -1], [Infinity, 0], [1, Infinity], [1e200, 1]])
    assert.throws(() => coneModel(...args));
  for (const args of [{r: 0}, {r: -1}, {r: NaN}, {k: 0}, {b: Infinity}, {A: 1}, {B: 2}, {rho: 6}, {rho: 0}, {L: 0}, {r: 1e200}, {r: 1e-200}])
    assert.throws(() => pinModel(args));
  for (const selected of [[0, 0], [3], [-1], [0.5], '012', null])
    assert.throws(() => paletteModel(selected));
});

let stateModule = {};
try { stateModule = await import('../docs/site/explore-state.mjs'); } catch {}
const readState = query => {
  assert.equal(typeof stateModule.readExploreState, 'function', 'Explore URL reader is available');
  return stateModule.readExploreState(query);
};

test('shared parameters restore all three experiments, including empty selection', () => {
  assert.deepEqual(readState('?s=-1&R=1&r=0.25&region=remote&objects='), {
    state: {s: -1, R: 1, r: 0.25, region: 'remote', objects: []}, invalid: []
  });
  assert.deepEqual(readState('').state, {s: -2, R: 1, r: 0.5, region: 'annulus', objects: [1,2,3]});
  assert.equal(coneModel(readState('?s=-1&R=1').state.s, 1).kind, 'flat');
  assert.equal(pinModel({r: readState('?r=0.25').state.r}).gap, 0.015625);
  for (const selected of [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]) {
    const parsed = readState('?objects='+selected.join(',')).state.objects;
    assert.deepEqual(parsed, selected);
    assert.equal(paletteModel(parsed.map(x => x-1)).decomposable, selected.length <= 2);
  }
});

test('invalid numeric links are reported and defaulted without clamping or partial coercion', () => {
  for (const query of ['s=', 's=Infinity', 's=NaN', 's=0x2', 's=3.1', 's=-3.1', 's=0.15', 's=1junk', 's=1&s=2', 's=%3Cscript%3E']) {
    const result = readState('?'+query+'&r=0.25');
    assert.equal(result.state.s, -2, query);
    assert.equal(result.state.r, 0.25, 'valid fields survive');
    assert.deepEqual(result.invalid, ['s'], query);
  }
  for (const [query, key, fallback] of [['R=-0.1','R',1], ['R=2.1','R',1], ['r=0','r',0.5], ['r=0.61','r',0.5], ['r=0.055','r',0.5]]) {
    const result = readState('?'+query);
    assert.equal(result.state[key], fallback);
    assert.deepEqual(result.invalid, [key]);
  }
  assert.deepEqual(readState('?s=-3&R=0&r=0.05').invalid, []);
  assert.deepEqual(readState('?s=3&R=2&r=0.6').invalid, []);
});

test('invalid choices cannot invent regions or objects', () => {
  for (const q of ['objects=1,1','objects=0','objects=4','objects=1,,2','objects=1&objects=2']) {
    assert.deepEqual(readState('?'+q).state.objects, [1,2,3]);
    assert.deepEqual(readState('?'+q).invalid, ['objects']);
  }
  assert.deepEqual(readState('?objects=3,1').state.objects, [1,3]);
  assert.equal(readState('?region=elsewhere').state.region, 'annulus');
  assert.deepEqual(readState('?region=elsewhere').invalid, ['region']);
});

test('sharing serializes current controls and preserves section and unrelated query fields', () => {
  assert.equal(typeof stateModule.exploreStateURL, 'function');
  const url = new URL(stateModule.exploreStateURL('https://example.test/explore.html?from=reader&s=1&s=2#groups',
    {s: -1, R: 1, r: 0.25, region: 'remote', objects: []}));
  assert.equal(url.origin, 'https://example.test');
  assert.equal(url.pathname, '/explore.html');
  assert.equal(url.hash, '#groups');
  assert.equal(url.searchParams.get('from'), 'reader');
  assert.deepEqual(url.searchParams.getAll('s'), ['-1']);
  assert.deepEqual(readState(url.search), {state: {s:-1,R:1,r:0.25,region:'remote',objects:[]}, invalid:[]});
});

// Catch wrong preset values, omitted control writes/draw/commit, stale exports,
// or accepting an unknown name: expectations are hand-derived sign cases.
test('curvature presets draw and commit the selected controls with matching share and export state', () => {
  assert.equal(typeof exploreModels.applyCurvaturePreset, 'function');
  for (const [name, s, eigenvalues, kind] of [
    ['maximum', -2, [-3, -1], 'peak'], ['saddle', 0, [-1, 1], 'saddle'],
    ['singular-boundary', -1, [-2, 0], 'flat'], ['minimum', 2, [1, 3], 'bowl']
  ]) {
    const center = {value: '0.5'}, spread = {value: '2'};
    let drawn, shared, exported, citation;
    const events = [];
    const current = () => ({s: Number(center.value), R: Number(spread.value)});
    const applied = exploreModels.applyCurvaturePreset(name, {center, spread,
      draw() {
        events.push('draw');
        drawn = coneModel(Number(center.value), Number(spread.value));
        exported = buildFigureMetadata(current(), '2026-10-03T00:00:00.000Z');
        citation = buildFigureCitation(current());
      },
      commit() {
        events.push('commit');
        shared = stateModule.exploreStateURL('https://example.test/explore.html#peaks',
          {...current(), r: 0.25, region: 'remote', objects: []});
      }
    });
    assert.equal(applied, true);
    assert.deepEqual(events, ['draw', 'commit']);
    assert.deepEqual(current(), {s, R: 1});
    assert.deepEqual(drawn.eigenvalues, eigenvalues);
    assert.equal(drawn.kind, kind);
    assert.deepEqual(exported.params, {s, R: 1});
    assert.deepEqual(exported.eigenvalues, eigenvalues);
    assert.equal(exported.classification, kind);
    assert.ok(citation.includes(`s = ${s}, R = 1;`));
    assert.ok(citation.includes(`classification: ${kind}.`));
    assert.deepEqual(readState(new URL(shared).search).state,
      {s, R: 1, r: 0.25, region: 'remote', objects: []});
  }
});

test('unknown curvature presets leave controls and current figure untouched', () => {
  assert.equal(typeof exploreModels.applyCurvaturePreset, 'function');
  for (const name of ['unknown', '__proto__', 'toString', '', null]) {
    const center = {value: '0.5'}, spread = {value: '2'};
    const callbacks = {center, spread,
      draw() { assert.fail('unknown preset must not redraw'); },
      commit() { assert.fail('unknown preset must not commit'); }};
    assert.equal(exploreModels.applyCurvaturePreset(name, callbacks), false);
    assert.equal(center.value, '0.5');
    assert.equal(spread.value, '2');
  }
});
