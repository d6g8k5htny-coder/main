import test from 'node:test';
import assert from 'node:assert/strict';
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
