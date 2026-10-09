// Exact finite periodic H0: faithful first predecessor-computation port.
// Source: d6g8k5htny-coder/main@032267c259162728229e3ddd14d08fe1c3f9a392,
// experiments/periodic_h0/exact_h0.py, compute; source SHA-256:
// 48ee16f2316cb1a233594ed10466340aa321d553c2b888198fe86b4bd9052970.
// The JavaScript wrapper supplies strict BigInt samples, a positive BigInt
// scale and the frozen teaching cap. These admission choices are new here.
// The predecessor exposes a five-key barcode, with no events or levels stream.
// Integer endpoints stay in numerator units; this engine never divides them.
// Finite computation alone has no continuum or scientific acceptance effect.

function admittedInput(input) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('A finite-grid record is required');
  }
  for (const name of ['values', 'n', 'scale']) {
    if (!Object.hasOwn(input, name)) throw new TypeError(`Own ${name} is required`);
  }
  const {values, n, scale} = input;
  if (typeof n !== 'number') throw new TypeError('Grid side must be a Number');
  if (!Number.isSafeInteger(n) || n < 3 || n > 32) {
    throw new RangeError('Grid side must be an integer from three through thirty-two');
  }
  if (typeof scale !== 'bigint') throw new TypeError('Scale must be a BigInt');
  if (scale <= 0n) throw new RangeError('Scale must be positive');
  if (!Array.isArray(values)) throw new TypeError('Flat sample array is required');
  if (values.length !== n * n) throw new RangeError('Square sample array is required');
  const samples = [];
  for (let vertex = 0; vertex < values.length; vertex++) {
    if (!Object.hasOwn(values, vertex)) throw new TypeError('Own dense samples are required');
    const value = values[vertex];
    if (typeof value !== 'bigint') throw new TypeError('Primitive BigInt samples are required');
    samples.push(value);
  }
  return {values: samples, n, scale};
}

function compareInteger(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function computeBarcode(values, n) {
  const count = values.length;
  const parent = new Int32Array(count).fill(-1);
  const sizes = new Uint32Array(count);
  const eldest = new Uint32Array(count);
  // Explicit increasing-ID ties preserve the Python source's stable sort.
  const order = Array.from({length: count}, (_, vertex) => vertex)
    .sort((a, b) => compareInteger(values[b], values[a]) || a - b);
  const records = [];
  let zero = 0;

  function root(vertex) {
    while (parent[vertex] !== vertex) {
      parent[vertex] = parent[parent[vertex]];
      vertex = parent[vertex];
    }
    return vertex;
  }

  for (const vertex of order) {
    parent[vertex] = vertex;
    sizes[vertex] = 1;
    eldest[vertex] = vertex;
    const x = Math.floor(vertex / n), y = vertex % n;
    // Adding n before subtraction preserves Python's nonnegative modulo.
    const neighbors = [((x + n - 1) % n) * n + y, ((x + 1) % n) * n + y,
      x * n + (y + n - 1) % n, x * n + (y + 1) % n];
    for (const neighbor of neighbors) {
      if (parent[neighbor] < 0) continue;
      let u = root(vertex), v = root(neighbor);
      if (u === v) continue;
      let survivor = eldest[u], dying = eldest[v];
      if (values[survivor] < values[dying]
          || (values[survivor] === values[dying] && survivor > dying)) {
        [survivor, dying] = [dying, survivor];
      }
      const birth = values[dying], death = values[vertex];
      if (birth > death) records.push([birth, death, dying]);
      else zero++;
      // Storage balancing is independent of the mathematical elder priority.
      if (sizes[u] < sizes[v]) [u, v] = [v, u];
      parent[v] = u;
      sizes[u] += sizes[v];
      eldest[u] = survivor;
    }
  }
  records.sort((a, b) => compareInteger(a[0], b[0])
    || compareInteger(a[1], b[1]) || a[2] - b[2]);
  const essential = eldest[root(0)];
  return {
    intervals: records.map(([birth, death]) => [birth, death]),
    birth_vertices: records.map(([, , vertex]) => vertex),
    essential: [values[essential]],
    essential_vertices: [essential],
    zero_count: zero,
  };
}

export function finiteElderTrace(input) {
  const {values, n, scale} = admittedInput(input);
  const barcode = computeBarcode(values, n);
  return {n, scale, values, barcode};
}
