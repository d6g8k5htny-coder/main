import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';

const moduleURL = new URL('../docs/site/measurement-model.mjs', import.meta.url);
async function api() {
  assert.ok(existsSync(moduleURL), 'The count-to-density model must be available');
  return import(moduleURL);
}
const fixture = { count: 576, realizations: 2, lower: 0.125, upper: 3.375 };

test('aggregate count uses area, all realizations, and bin width exactly once', async () => {
  const { measurementModel } = await api();
  const result = measurementModel(fixture);
  assert.equal(result.area, 576);
  assert.equal(result.exposure, 1152);
  assert.equal(result.width, 3.25);
  assert.equal(result.mass, 0.5);
  assert.equal(result.density, 2 / 13);
});

test('twice the count gives twice the mass and density', async () => {
  const { measurementModel } = await api();
  const result = measurementModel({ ...fixture, count: 1152 });
  assert.equal(result.mass, 1);
  assert.equal(result.density, 4 / 13);
});

test('adding zero-count realizations still increases the denominator', async () => {
  const { measurementModel } = await api();
  const result = measurementModel({ ...fixture, realizations: 4 });
  assert.equal(result.mass, 0.25);
  assert.equal(result.density, 1 / 13);
});

test('a wider synthetic bin preserves mass and divides density by its width', async () => {
  const { measurementModel } = await api();
  const result = measurementModel({ ...fixture, upper: 6.625 });
  assert.equal(result.width, 6.5);
  assert.equal(result.mass, 0.5);
  assert.equal(result.density, 1 / 13);
});

test('zero counts produce valid zero estimates without a fabricated positive floor', async () => {
  const { measurementModel } = await api();
  for (const count of [0, -0]) {
    const result = measurementModel({ ...fixture, count });
    assert.equal(result.mass, 0);
    assert.equal(result.density, 0);
    assert.equal(Object.is(result.mass, -0), false);
  }
});

test('counts and realization totals reject fractions, unsafe integers, and nonnumbers', async () => {
  const { measurementModel } = await api();
  for (const [field, invalid] of [
    ['count', [-1, 0.5, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity, '576', null]],
    ['realizations', [0, -1, 0.5, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity, '2', null]],
  ]) {
    for (const value of invalid)
      assert.throws(() => measurementModel({ ...fixture, [field]: value }),
        error => error instanceof RangeError && error.field === field);
  }
});

test('only positive ordered finite bin edges are accepted', async () => {
  const { measurementModel } = await api();
  for (const lower of [0, -1, NaN, Infinity, '0.125', null])
    assert.throws(() => measurementModel({ ...fixture, lower }), RangeError);
  for (const upper of [0, 0.125, -1, NaN, Infinity, '3.375', null])
    assert.throws(() => measurementModel({ ...fixture, upper }), RangeError);
});

test('normalization refuses unsafe exposure and overflow or underflow in positive density', async () => {
  const { measurementModel } = await api();
  for (const input of [
    { ...fixture, realizations: Number.MAX_SAFE_INTEGER },
    { ...fixture, lower: Number.MIN_VALUE, upper: 2 * Number.MIN_VALUE },
    { ...fixture, count: 1, realizations: 10000000000000, upper: 1e308 },
  ]) assert.throws(() => measurementModel(input), RangeError);
});

test('input parsing rejects blanks instead of turning them into zero counts', async () => {
  const { parseMeasurementInputs } = await api();
  const text = { count: '576', realizations: '2', lower: '0.125', upper: '3.375' };
  for (const field of Object.keys(text))
    for (const value of ['', '  ', '0x10', '12 items', null, undefined])
      assert.throws(() => parseMeasurementInputs({ ...text, [field]: value }),
        error => error instanceof RangeError && error.field === field);
});

test('decimal input, including scientific notation, reaches the same normalization', async () => {
  const { parseMeasurementInputs, measurementModel } = await api();
  const parsed = parseMeasurementInputs({ count: ' 5.76e2 ', realizations: '2', lower: '.125', upper: '3.375' });
  assert.deepEqual(parsed, fixture);
  assert.equal(measurementModel(parsed).density, 2 / 13);
  assert.equal(measurementModel(parseMeasurementInputs({ count: '0', realizations: '2', lower: '.125', upper: '3.375' })).density, 0);
});

test('decimal conversion cannot round a fractional count to an integer or a nonzero input to zero', async () => {
  const { parseMeasurementInputs } = await api();
  const text = { count: '576', realizations: '2', lower: '0.125', upper: '3.375' };
  for (const [field, value] of [
    ['count', '576.00000000000001'], ['count', '9007199254740990.9'],
    ['realizations', '2.0000000000000001'],
    ['count', '1e-999'], ['lower', '1e-999'],
  ]) assert.throws(() => parseMeasurementInputs({ ...text, [field]: value }),
    error => error instanceof RangeError && error.field === field);
});
