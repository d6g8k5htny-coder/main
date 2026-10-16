// Source-informed external controls with independent graph-search oracles.
// Contract: docs/plans/2026-10-09-canvas-model-contract.md. Hosted execution only.
// Expected barcodes do not call the implementation or its helpers.
import test from 'node:test';
import assert from 'node:assert/strict';

let candidate;
let setup;
try {
  const module = await import('../src/scripts/elder-model.mjs');
  if (typeof module.finiteElderTrace !== 'function') throw new TypeError('Missing finiteElderTrace export');
  candidate = module.finiteElderTrace;
} catch (error) {
  setup = 'Candidate import/export rejected with an unprintable value';
  try {
    setup = error instanceof Error ? `${error.name}: ${error.message}` : `Non-Error rejection: ${String(error)}`;
  } catch { /* Preserve SETUP_BLOCKED even for a hostile rejection value. */ }
}

test('PREFLIGHT elder candidate imports and exports finiteElderTrace', () => {
  assert.equal(setup, undefined, `SETUP_BLOCKED; behavior NOT_RUN: ${setup}`);
});

function behavior(name, check) {
  test(`BEHAVIOR elder ${name}`, (t) => {
    if (!candidate) { t.skip('SETUP_BLOCKED; behavior NOT_RUN'); return; }
    check(candidate);
  });
}

function keys(object, names) {
  assert.ok(object !== null && typeof object === 'object' && !Array.isArray(object));
  assert.deepEqual(Object.keys(object).sort(), [...names].sort());
}

function integer(value, low = 0) {
  assert.ok(typeof value === 'number' && Number.isSafeInteger(value) && value >= low);
}

function dense(array, length) {
  assert.ok(Array.isArray(array));
  assert.equal(array.length, length);
  for (let i = 0; i < length; i++) assert.ok(Object.hasOwn(array, i));
}

function compareBig(a, b) { return a < b ? -1 : a > b ? 1 : 0; }
function older(values, u, v) {
  return values[u] > values[v] || (values[u] === values[v] && u < v);
}

// Independent graph: add positive-direction edges and symmetrize, no DSU.
function graphFor(n) {
  const graph = Array.from({length: n * n}, () => new Set());
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    const u = row * n + col;
    for (const v of [((row + 1) % n) * n + col, row * n + (col + 1) % n]) {
      graph[u].add(v); graph[v].add(u);
    }
  }
  return graph;
}

function connected(graph, start, admitted) {
  assert.ok(admitted(start), 'BFS start must be admitted');
  const seen = new Set([start]);
  const queue = [start];
  for (let i = 0; i < queue.length; i++) for (const v of graph[queue[i]]) {
    if (admitted(v) && !seen.has(v)) { seen.add(v); queue.push(v); }
  }
  return seen;
}

function labelsAt(values, graph, threshold) {
  const labels = Array(values.length).fill(-1);
  let count = 0;
  for (let v = 0; v < values.length; v++) {
    if (values[v] < threshold || labels[v] !== -1) continue;
    for (const u of connected(graph, v, (u) => values[u] >= threshold)) labels[u] = count;
    count++;
  }
  return {labels, count};
}

function recordsFor(barcode) {
  return barcode.intervals.map(([b, d], i) => [b, d, barcode.birth_vertices[i]]);
}

function sortRecords(records) {
  return [...records].sort((a, b) => compareBig(a[0], b[0]) || compareBig(a[1], b[1]) || a[2] - b[2]);
}

function verifyBarcode(values, n, barcode) {
  const size = values.length;
  keys(barcode, ['intervals', 'birth_vertices', 'essential', 'essential_vertices', 'zero_count']);
  assert.ok(Array.isArray(barcode.intervals));
  dense(barcode.intervals, barcode.intervals.length);
  dense(barcode.birth_vertices, barcode.intervals.length);
  dense(barcode.essential, 1); dense(barcode.essential_vertices, 1);
  integer(barcode.zero_count);
  assert.equal(barcode.intervals.length + barcode.zero_count + 1, size);
  for (const [i, pair] of barcode.intervals.entries()) {
    dense(pair, 2);
    assert.equal(typeof pair[0], 'bigint'); assert.equal(typeof pair[1], 'bigint');
    assert.ok(pair[0] > pair[1]);
    integer(barcode.birth_vertices[i]); assert.ok(barcode.birth_vertices[i] < size);
  }
  assert.equal(typeof barcode.essential[0], 'bigint');
  integer(barcode.essential_vertices[0]); assert.ok(barcode.essential_vertices[0] < size);
  assert.equal(new Set(barcode.birth_vertices).size, barcode.birth_vertices.length);
  assert.deepEqual(recordsFor(barcode), sortRecords(recordsFor(barcode)));

  const graph = graphFor(n);
  const visited = new Set();
  const maxima = [];
  // Complete birth coverage via regional plateaus, independent of the sweep.
  for (let v = 0; v < size; v++) {
    if (visited.has(v)) continue;
    const plateau = connected(graph, v, (u) => values[u] === values[v]);
    let hasHigher = false;
    for (const u of plateau) {
      visited.add(u);
      for (const w of graph[u]) if (values[w] > values[v]) hasHigher = true;
    }
    if (!hasHigher) maxima.push(v); // increasing scan gives minimum plateau ID
  }
  const essential = maxima.reduce((best, v) => older(values, v, best) ? v : best);
  assert.deepEqual(barcode.essential, [values[essential]]);
  assert.deepEqual(barcode.essential_vertices, [essential]);
  assert.deepEqual([...barcode.birth_vertices].sort((a, b) => a - b),
    maxima.filter((v) => v !== essential).sort((a, b) => a - b));
  for (const [birth, death, vertex] of recordsFor(barcode)) {
    assert.equal(birth, values[vertex]);
    for (const [threshold, expected] of [[death, true], [death + 1n, false]]) {
      const region = connected(graph, vertex, (u) => values[u] >= threshold);
      assert.equal([...region].some((u) => older(values, u, vertex)), expected,
        `Exact elder death mismatch at vertex ${vertex}, threshold ${threshold}`);
    }
  }
}

function verifyRanks(values, n, barcode) {
  const graph = graphFor(n);
  const actualLevels = [...new Set(values)].sort((a, b) => compareBig(b, a));
  const thresholds = [actualLevels[0] + 1n, ...actualLevels, actualLevels.at(-1) - 1n];
  const labels = new Map(thresholds.map((level) => [level, labelsAt(values, graph, level).labels]));
  for (let i = 0; i < thresholds.length; i++) for (const low of thresholds.slice(i)) {
    const high = thresholds[i];
    const touched = new Set();
    for (let v = 0; v < values.length; v++) if (values[v] >= high) touched.add(labels.get(low)[v]);
    const finite = barcode.intervals.filter(([birth, death]) => birth >= high && death < low).length;
    const essential = barcode.essential.filter((birth) => birth >= high).length;
    assert.equal(finite + essential, touched.size, `Rank mismatch high=${high}, low=${low}`);
  }
}

// Trace verification uses a partial-edge BFS, not union-find or candidate data.
function verifyTrace(values, n, scale, result) {
  keys(result, ['n', 'scale', 'values', 'events', 'levels', 'barcode']);
  assert.equal(result.n, n); assert.equal(result.scale, scale);
  dense(result.values, values.length); assert.deepEqual(result.values, values);
  assert.notEqual(result.values, values);
  verifyBarcode(values, n, result.barcode);
  assert.ok(Array.isArray(result.events) && Array.isArray(result.levels));
  dense(result.events, 3 * values.length);
  dense(result.levels, new Set(values).size);
  const partial = Array.from({length: values.length}, () => new Set());
  const active = new Set();
  const seenEdges = new Set();
  const order = values.map((_, v) => v).sort((a, b) => compareBig(values[b], values[a]) || a - b);
  const records = [];
  let cursor = 0, block = 0, zeros = 0, neutrals = 0, merges = 0;
  const elderOf = (region) => [...region].reduce((best, v) => older(values, v, best) ? v : best);
  for (let index = 0; index < order.length; index++) {
    const vertex = order[index], level = values[vertex];
    assert.deepEqual(result.events[cursor++], {kind: 'activate', vertex, level});
    active.add(vertex);
    const row = Math.floor(vertex / n), col = vertex % n;
    // Branch-based boundaries differ from the source's modular neighbor formula.
    const neighbors = [row ? vertex - n : vertex + n * (n - 1),
      row + 1 < n ? vertex + n : vertex - n * (n - 1),
      col ? vertex - 1 : vertex + n - 1,
      col + 1 < n ? vertex + 1 : vertex - n + 1];
    for (const neighbor of neighbors) {
      if (!active.has(neighbor)) continue;
      const edgeKey = [vertex, neighbor].sort((a, b) => a - b).join(':');
      assert.ok(!seenEdges.has(edgeKey)); seenEdges.add(edgeKey);
      const first = connected(partial, vertex, (u) => active.has(u));
      const firstElder = elderOf(first);
      let expected;
      if (first.has(neighbor)) {
        expected = {kind: 'neutral', edge: [vertex, neighbor], level, elder: firstElder};
        neutrals++;
      } else {
        const secondElder = elderOf(connected(partial, neighbor, (u) => active.has(u)));
        const survivor = older(values, firstElder, secondElder) ? firstElder : secondElder;
        const dying = survivor === firstElder ? secondElder : firstElder;
        const birth = values[dying], death = level, zero = birth === death;
        expected = {kind: 'merge', edge: [vertex, neighbor], level, survivor, dying, birth, death, zero};
        merges++;
        if (zero) zeros++; else records.push([birth, death, dying]);
      }
      assert.deepEqual(result.events[cursor++], expected);
      partial[vertex].add(neighbor); partial[neighbor].add(vertex);
    }
    if (index + 1 === order.length || values[order[index + 1]] !== level) {
      const complete = labelsAt(values, graphFor(n), level);
      assert.deepEqual(result.levels[block++], {level, eventEnd: cursor,
        activeCount: active.size, componentCount: complete.count,
        positiveCount: records.length, zeroCount: zeros});
      assert.equal(active.size - merges, complete.count);
    }
  }
  assert.equal(cursor, result.events.length);
  assert.equal(block, result.levels.length);
  assert.equal(seenEdges.size, 2 * values.length);
  assert.equal(merges, values.length - 1);
  assert.equal(neutrals, values.length + 1);
  assert.equal(zeros, result.barcode.zero_count);
  assert.deepEqual(sortRecords(records), recordsFor(result.barcode));
}

const peaks = () => Array.from({length: 36}, (_, i) => [5n, 1n, 4n, 2n, 3n, 0n][i % 6]);
const golden = () => ({intervals: [[3n, 2n], [4n, 1n]], birth_vertices: [4, 2],
  essential: [5n], essential_vertices: [0], zero_count: 33});

behavior('unequal peaks retain the elder and longest finite bar', (model) => {
  const values = peaks(), result = model({values, n: 6, scale: 1n});
  assert.deepEqual(result.barcode, golden());
  verifyTrace(values, 6, 1n, result); verifyRanks(values, 6, result.barcode);
});

behavior('full tie blocks and neutral edges have distinct meanings', (model) => {
  const values = peaks(), result = model({values, n: 6, scale: 1n});
  verifyTrace(values, 6, 1n, result);
  assert.deepEqual(result.levels.map((x) => [x.level, x.activeCount, x.componentCount]),
    [[5n, 6, 1], [4n, 12, 2], [3n, 18, 3], [2n, 24, 2], [1n, 30, 1], [0n, 36, 1]]);
  assert.equal(result.events.filter((x) => x.kind === 'neutral').length, 37);
  assert.equal(result.events.filter((x) => x.kind === 'merge' && x.zero).length, 33);
  const positive = result.events.filter((x) => x.kind === 'merge' && !x.zero);
  assert.deepEqual(positive.map((x) => [x.birth, x.death, x.dying, x.survivor]),
    [[3n, 2n, 4, 2], [4n, 1n, 2, 0]]);
});

behavior('periodic seams join peaks before a spurious death', (model) => {
  const values = Array.from({length: 16}, (_, i) => [5n, 0n, 1n, 4n][i % 4]);
  const result = model({values, n: 4, scale: 1n});
  assert.deepEqual(result.barcode, {intervals: [], birth_vertices: [], essential: [5n],
    essential_vertices: [0], zero_count: 15});
  verifyTrace(values, 4, 1n, result); verifyRanks(values, 4, result.barcode);
});

behavior('axis graph does not connect diagonal square corners', (model) => {
  const values = Array(16).fill(0n); values[0] = values[5] = 5n;
  const result = model({values, n: 4, scale: 1n});
  assert.deepEqual(result.barcode, {intervals: [[5n, 0n]], birth_vertices: [5],
    essential: [5n], essential_vertices: [0], zero_count: 14});
  verifyTrace(values, 4, 1n, result); verifyRanks(values, 4, result.barcode);
});

behavior('equal peaks prefer the smaller plateau ID', (model) => {
  const values = Array.from({length: 16}, (_, i) => [5n, 1n, 5n, 1n][i % 4]);
  const result = model({values, n: 4, scale: 1n});
  assert.deepEqual(result.barcode, {intervals: [[5n, 1n]], birth_vertices: [2],
    essential: [5n], essential_vertices: [0], zero_count: 14});
  verifyTrace(values, 4, 1n, result); verifyRanks(values, 4, result.barcode);
});

behavior('negative constant plateau has one essential class and eight zero merges', (model) => {
  const values = Array(9).fill(-7n), result = model({values, n: 3, scale: 2n});
  assert.deepEqual(result.barcode, {intervals: [], birth_vertices: [], essential: [-7n],
    essential_vertices: [0], zero_count: 8});
  verifyTrace(values, 3, 2n, result); verifyRanks(values, 3, result.barcode);
});

behavior('large integer translation and positive scaling preserve exact endpoints', (model) => {
  const offset = 1n << 96n, values = peaks().map((v) => v + offset);
  const result = model({values, n: 6, scale: 7n});
  assert.deepEqual(result.barcode.intervals, [[offset + 3n, offset + 2n], [offset + 4n, offset + 1n]]);
  assert.deepEqual(result.barcode.intervals.map(([b, d]) => b - d), [1n, 3n]);
  verifyTrace(values, 6, 7n, result); verifyRanks(values, 6, result.barcode);
  const scaledValues = values.map((v) => 11n * v);
  const scaled = model({values: scaledValues, n: 6, scale: 77n});
  assert.deepEqual(scaled.barcode.intervals.map(([b, d]) => b - d), [11n, 33n]);
  verifyTrace(scaledValues, 6, 77n, scaled); verifyRanks(scaledValues, 6, scaled.barcode);
  // Keep rational endpoints as numerator/scale; no truncating integer division.
  assert.equal(result.scale, 7n); assert.equal(scaled.scale, 77n);
});

behavior('all 512 binary grids satisfy every threshold rank and exact trace', (model) => {
  for (let mask = 0; mask < 512; mask++) {
    const values = Array.from({length: 9}, (_, i) => BigInt((mask >> i) & 1));
    const result = model({values, n: 3, scale: 1n});
    verifyTrace(values, 3, 1n, result); verifyRanks(values, 3, result.barcode);
  }
});

behavior('deterministic tied multilevel fields satisfy complete birth and rank oracles', (model) => {
  // Explicit reproducible synthetic fields; no sampling-law or randomness claim.
  let state = 3934000;
  for (const n of [3, 4, 5]) for (let sample = 0; sample < 25; sample++) {
    const values = Array.from({length: n * n}, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return BigInt((state % 7) - 3);
    });
    const result = model({values, n, scale: 3n});
    verifyTrace(values, n, 3n, result); verifyRanks(values, n, result.barcode);
  }
});

behavior('rejects malformed record dimension scale shape and sparse integer samples', (model) => {
  const valid = {values: Array(9).fill(0n), n: 3, scale: 1n};
  for (const input of [undefined, null, [], true, 3, 'record', {},
    {values: valid.values, n: 3}, Object.create(valid)]) assert.throws(() => model(input), TypeError);
  for (const n of [true, '3', 3n, new Number(3), null]) {
    assert.throws(() => model({...valid, n}), TypeError);
  }
  for (const n of [2, 33, 3.5, NaN, Infinity, -1, 0]) {
    assert.throws(() => model({...valid, n}), RangeError);
  }
  for (const scale of [1, '1', true, null, undefined, Object(1n)]) {
    assert.throws(() => model({...valid, scale}), TypeError);
  }
  for (const scale of [0n, -1n]) assert.throws(() => model({...valid, scale}), RangeError);
  for (const values of [null, '000000000', new BigInt64Array(9), {length: 9}]) {
    assert.throws(() => model({...valid, values}), TypeError);
  }
  for (const values of [[], Array(8).fill(0n), Array(10).fill(0n)]) {
    assert.throws(() => model({...valid, values}), RangeError);
  }
  const sparse = Array(9).fill(0n); delete sparse[4];
  const inherited = Array(9).fill(0n); delete inherited[4];
  const prototype = Object.create(Array.prototype); prototype[4] = 0n;
  Object.setPrototypeOf(inherited, prototype);
  for (const values of [sparse, inherited]) assert.throws(() => model({...valid, values}), TypeError);
  for (const bad of [0, 0.5, NaN, Infinity, false, '0', null, undefined, {}, [], Object(0n)]) {
    const values = Array(9).fill(0n); values[8] = bad;
    assert.throws(() => model({...valid, values}), TypeError);
  }
});

behavior('accepts the teaching cap with linear event storage', (model) => {
  const values = Array(1024).fill(0n), result = model({values, n: 32, scale: 1n});
  assert.equal(result.events.length, 3072);
  assert.deepEqual(result.barcode, {intervals: [], birth_vertices: [], essential: [0n],
    essential_vertices: [0], zero_count: 1023});
  verifyTrace(values, 32, 1n, result);
});

behavior('does not mutate input or share returned state across deterministic calls', (model) => {
  const values = Object.freeze(peaks());
  const input = Object.freeze({values, n: 6, scale: 7n, unused: 'allowed'});
  const first = model(input), second = model(input), snapshot = structuredClone(second);
  assert.deepEqual(first, second); verifyTrace(values, 6, 7n, second);
  first.values[0] = -99n; first.events[0].level = -99n;
  first.events.find((x) => x.kind !== 'activate').edge[0] = -1;
  first.levels[0].componentCount = -1;
  first.barcode.intervals[0][0] = -99n; first.barcode.birth_vertices[0] = -1;
  first.barcode.essential[0] = -99n; first.barcode.essential_vertices[0] = -1;
  assert.deepEqual(second, snapshot); assert.deepEqual(model(input), snapshot);
  assert.deepEqual(values, peaks()); assert.equal(input.scale, 7n);
});

test('ORACLE sensitivity rejects omitted shifted mislabelled and malformed literal barcodes', () => {
  // Checker controls only: not a product substitute and never candidate RED.
  verifyBarcode(peaks(), 6, golden()); verifyRanks(peaks(), 6, golden());
  const mutations = [];
  for (const death of [0n, 2n]) {
    const changed = golden(); changed.intervals[1][1] = death; mutations.push(changed);
  }
  const omitted = golden(); omitted.intervals.pop(); omitted.birth_vertices.pop(); omitted.zero_count++;
  mutations.push(omitted);
  for (const [key, value] of [['birth_vertices', [2, 4]], ['essential', [4n]],
    ['essential_vertices', [2]], ['zero_count', 0], ['zero_count', true]]) {
    mutations.push({...golden(), [key]: value});
  }
  const inexact = golden(); inexact.intervals[0][0] = 3; mutations.push(inexact);
  mutations.push({...golden(), extra: true});
  for (const corrupted of mutations) assert.throws(() => verifyBarcode(peaks(), 6, corrupted));
});
