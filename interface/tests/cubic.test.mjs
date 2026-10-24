// External contract controls; no implementation helper is used for expectations.
// Contract: docs/plans/2026-10-09-canvas-model-contract.md.
// Candidate setup failure is never behavioral RED. Run only in the hosted lane.
import test from 'node:test';
import assert from 'node:assert/strict';

let candidate;
let setup;
try {
  const module = await import('../src/scripts/cubic-model.mjs');
  if (typeof module.cubicModel !== 'function') throw new TypeError('Missing cubicModel export');
  candidate = module.cubicModel;
} catch (error) {
  setup = 'Candidate import/export rejected with an unprintable value';
  try {
    setup = error instanceof Error ? `${error.name}: ${error.message}` : `Non-Error rejection: ${String(error)}`;
  } catch { /* Preserve SETUP_BLOCKED even for a hostile rejection value. */ }
}

test('PREFLIGHT cubic candidate imports and exports cubicModel', () => {
  assert.equal(setup, undefined, `SETUP_BLOCKED; behavior NOT_RUN: ${setup}`);
});

function behavior(name, check) {
  test(`BEHAVIOR cubic ${name}`, (t) => {
    if (!candidate) { t.skip('SETUP_BLOCKED; behavior NOT_RUN'); return; }
    check(candidate);
  });
}

function exactKeys(object, names) {
  assert.ok(object !== null && typeof object === 'object' && !Array.isArray(object));
  assert.deepEqual(Object.keys(object).sort(), [...names].sort());
}

function shape(result) {
  exactKeys(result, ['r', 'k', 'b', 'rCubed', 'pins', 'gap', 'heightWindow', 'normalized']);
  exactKeys(result.normalized, ['xDivisor', 'yOrigin', 'yDivisor', 'pins', 'heightWindow']);
  for (const key of ['r', 'k', 'b', 'rCubed', 'gap']) {
    assert.equal(typeof result[key], 'number');
    assert.ok(Number.isFinite(result[key]));
  }
  for (const tuple of [result.pins, result.heightWindow, result.normalized.pins,
    result.normalized.heightWindow]) {
    assert.ok(Array.isArray(tuple));
    assert.equal(tuple.length, 2);
    for (const value of tuple) assert.ok(typeof value === 'number' && Number.isFinite(value));
  }
}

behavior('literal binary fixture retains all scales and affine teaching coordinates', (model) => {
  const result = model({r: 0.5, k: 2, b: 3});
  shape(result);
  assert.deepEqual(result, {
    r: 0.5, k: 2, b: 3, rCubed: 0.125, pins: [-0.25, 0.25], gap: 0.25,
    heightWindow: [2.75, 3],
    normalized: {xDivisor: 0.5, yOrigin: 3, yDivisor: 0.25,
      pins: [-0.5, 0.5], heightWindow: [-1, 0]},
  });
});

behavior('halving radius divides the height gap by eight without rescaling k', (model) => {
  const large = model({r: 0.5, k: 2, b: 3});
  const small = model({r: 0.25, k: 2, b: 3});
  shape(small);
  assert.equal(small.gap, 0.03125);
  assert.equal(large.gap / small.gap, 8);
  assert.equal(small.rCubed, 0.015625);
  assert.deepEqual(small.pins, [-0.125, 0.125]);
  assert.deepEqual(small.heightWindow, [2.96875, 3]);
  assert.deepEqual(small.normalized, {xDivisor: 0.25, yOrigin: 3, yDivisor: 0.03125,
    pins: [-0.5, 0.5], heightWindow: [-1, 0]});
});

behavior('nonbinary arithmetic and negative origin preserve nominal normalization', (model) => {
  const result = model({r: 0.3, k: 5, b: -2});
  shape(result);
  // Hand-derived decimal quantities, tolerance only for Number arithmetic.
  assert.ok(Math.abs(result.rCubed - 0.027) <= 1e-16);
  assert.ok(Math.abs(result.gap - 0.135) <= 1e-15);
  assert.ok(Math.abs(result.heightWindow[0] - (-2.135)) <= 1e-14);
  assert.deepEqual(result.pins, [-0.15, 0.15]);
  assert.equal(result.normalized.xDivisor, 0.3);
  assert.equal(result.normalized.yOrigin, -2);
  assert.equal(result.normalized.yDivisor, result.gap);
  assert.deepEqual(result.normalized.pins, [-0.5, 0.5]);
  assert.deepEqual(result.normalized.heightWindow, [-1, 0]);
});

behavior('rejects malformed records and coercible or missing numeric fields', (model) => {
  for (const input of [undefined, null, [], true, 4, 'record', {}, {r: 1, k: 1},
    Object.create({r: 1, k: 1, b: 1})]) assert.throws(() => model(input), TypeError);
  for (const key of ['r', 'k', 'b']) {
    for (const value of [undefined, null, true, '1', 1n, {}, [], new Number(1)]) {
      assert.throws(() => model({...{r: 1, k: 1, b: 1}, [key]: value}), TypeError);
    }
  }
});

behavior('rejects invalid ranges overflow underflow and collapsed height windows', (model) => {
  for (const key of ['r', 'k', 'b']) {
    for (const value of [NaN, Infinity, -Infinity]) {
      assert.throws(() => model({...{r: 1, k: 1, b: 1}, [key]: value}), RangeError);
    }
  }
  for (const key of ['r', 'k']) for (const value of [0, -0, -1]) {
    assert.throws(() => model({...{r: 1, k: 1, b: 1}, [key]: value}), RangeError);
  }
  for (const input of [
    {r: 1e200, k: 1, b: 1},       // rCubed overflow
    {r: 1e-200, k: 1e300, b: 0},  // rCubed underflow before multiplication
    {r: 2, k: Number.MAX_VALUE, b: 0}, // gap overflow
    {r: 0.5, k: Number.MIN_VALUE, b: 0}, // gap underflow
    {r: 1, k: 1, b: 1e30},       // subtraction rounds back to b
    {r: 1, k: Number.MAX_VALUE, b: -Number.MAX_VALUE}, // lower overflow
  ]) assert.throws(() => model(input), RangeError);
});

behavior('does not mutate inputs or share mutable results across calls', (model) => {
  const input = Object.freeze({r: 0.5, k: 2, b: 3, unused: 'allowed'});
  const first = model(input);
  const snapshot = structuredClone(first);
  const second = model(input);
  assert.deepEqual(first, second);
  first.pins[0] = 99;
  first.heightWindow[0] = 99;
  first.normalized.pins[0] = 99;
  first.normalized.heightWindow[0] = 99;
  first.normalized.yDivisor = 99;
  assert.deepEqual(second, snapshot);
  assert.deepEqual(model(input), snapshot);
  assert.deepEqual(input, {r: 0.5, k: 2, b: 3, unused: 'allowed'});
});
