// Differential control: base (aff60c50) vs head (e80f370e) current-model.mjs.
// usage: node diff.mjs BASE_DIR HEAD_DIR SEED RANDOM_GRAPHS
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

const [baseDir, headDir, seedArg, countArg] = process.argv.slice(2);
const B = await import(pathToFileURL(baseDir + '/docs/site/current-model.mjs').href);
const H = await import(pathToFileURL(headDir + '/docs/site/current-model.mjs').href);
const D = await import(pathToFileURL(headDir + '/docs/site/dependency-model.mjs').href);

let seed = Number(seedArg) >>> 0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
const pick = a => a[Math.floor(rnd() * a.length)];
const stats = {roots: 0, routes: 0, occurrences: 0, windows: 0, reach: 0, relations: 0, graphs: 0};

function compareRoot(g, id, label) {
  const old = B.requiredDependencyPaths(g, id), neu = H.requiredDependencyPaths(g, id);
  assert.equal(neu.length, old.length, label + ' route count');
  old.forEach((r, i) => {
    assert.deepEqual(neu[i].ids, r.ids, label + ' ids ' + i);
    assert.equal(neu[i].edges.length, r.edges.length, label + ' edge count ' + i);
    r.edges.forEach((e, k) => assert.equal(neu[i].edges[k], e, label + ' edge identity ' + i + '/' + k));
  });
  const a = H.requiredDependencyAnalysis(g, id), total = old.reduce((n, r) => n + r.ids.length, 0);
  assert.equal(a.targetCount, old.length, label + ' targetCount');
  assert.equal(a.totalOccurrences, total, label + ' totalOccurrences');
  const expected = old.flatMap((r, ri) => r.ids.map((n, p) => [ri, r.ids.at(-1), p, n, p ? r.edges[p - 1] : null]));
  for (const limit of [1, 2, 3, 7, 150]) {
    const pages = Math.max(1, Math.ceil(total / limit)), flat = [];
    for (let page = 1; page <= pages; page++) {
      const w = H.requiredDependencyWindow(a, {page, limit});
      stats.windows++;
      assert.equal(w.page, page); assert.equal(w.pages, pages); assert.equal(w.offset, (page - 1) * limit);
      assert.ok(w.occurrences <= limit && w.occurrences >= 0, label + ' bound');
      assert.equal(w.omitted, total - w.occurrences);
      assert.equal(w.segments.reduce((n, s) => n + s.ids.length, 0), w.occurrences);
      for (const s of w.segments) {
        assert.ok(s.ids.length > 0, label + ' empty segment');
        assert.equal(s.complete, s.start === 0 && s.end === s.length);
        assert.equal(s.end - s.start, s.ids.length);
        assert.equal(s.incomingEdges.length, s.ids.length);
        s.ids.forEach((n, i) => flat.push([s.routeIndex, s.target, s.start + i, n, s.incomingEdges[i]]));
      }
    }
    assert.equal(flat.length, expected.length, label + ' flat length limit ' + limit);
    flat.forEach((row, i) => { for (let k = 0; k < 5; k++) assert.equal(row[k], expected[i][k], label + ' row ' + i + ' col ' + k + ' limit ' + limit); });
    // clamp: beyond last page returns the last page; invalid pages return page 1
    assert.equal(H.requiredDependencyWindow(a, {page: pages + 5, limit}).page, pages);
    for (const bad of [0, -1, 1.5, NaN, Infinity, '2', null]) assert.equal(H.requiredDependencyWindow(a, {page: bad, limit}).page, 1);
  }
  stats.roots++; stats.routes += old.length; stats.occurrences += total;
}

function compareReach(g, id, label) {
  const old = B.downstreamReach(g, id), neu = H.downstreamReach(g, id), c = H.downstreamReachCounts(g, id);
  assert.deepEqual(neu, old, label + ' reach');
  assert.deepEqual(c, {known: old.known, allEdge: old.allEdge, requiredOnly: old.requiredOnly, labels: old.labels}, label + ' counts');
  stats.reach++;
}

function compareRelations(g, id, label) {
  for (const [direction, key] of [['outgoing', 'from'], ['incoming', 'to']]) {
    const expected = g.edges.filter(e => e[key] === id && H.hasNode(g, e.from) && H.hasNode(g, e.to)), got = [];
    for (const limit of [1, 4, 150]) {
      got.length = 0;
      const pages = Math.max(1, Math.ceil(expected.length / limit));
      for (let page = 1; page <= pages; page++) {
        const w = H.directRelationWindow(g, id, direction, {page, limit});
        assert.equal(w.total, expected.length); assert.ok(w.rows.length <= limit);
        assert.equal(w.omitted, expected.length - w.rows.length, label + ' relation omitted');
        got.push(...w.rows);
      }
      assert.equal(got.length, expected.length, label + ' relation count');
      got.forEach((e, i) => assert.equal(e, expected[i], label + ' relation order ' + direction));
    }
  }
  stats.relations++;
}

function check(g, label, extraRoots = []) {
  stats.graphs++;
  for (const id of [...Object.keys(g.nodes), ...extraRoots]) {
    compareRoot(g, id, label + ' root ' + id);
    compareReach(g, id, label + ' root ' + id);
    compareRelations(g, id, label + ' root ' + id);
  }
}

// 1. The real admitted corpus at the head.
const snapshot = JSON.parse(readFileSync(headDir + '/docs/site/current.json', 'utf8'));
H.validateSnapshot(snapshot);
check(snapshot.graph.data, 'corpus', ['missing', 'constructor', '__proto__', 'toString']);
const corpus = {...stats};

// 2. Random graphs: cycles, self-loops, parallel edges, relation ties, context
// edges, own "__proto__"/"constructor" node IDs and (unvalidated) unknown endpoints.
const classes = [...D.OPEN_CLASSIFICATIONS, 'HOLD', 'CLOSED', 'ACCEPTED'];
const kinds = [...D.NON_RESEARCH_KINDS, 'candidate', 'lemma', 'theorem'];
const pool = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'B', 'Z', 'ä', 'é', '10', '9', 'x.y', 'constructor', '__proto__', 'toString', 'hasOwnProperty'];
const random = Number(countArg);
for (let t = 0; t < random; t++) {
  const n = 1 + Math.floor(rnd() * 20), ids = [...pool].sort(() => rnd() - 0.5).slice(0, n);
  const nodes = JSON.parse('{' + ids.map(id => JSON.stringify(id) + ':' + JSON.stringify({kind: pick(kinds), classification: pick(classes), layer: pick(['D1', 'D2'])})).join(',') + '}');
  const m = Math.floor(rnd() * n * 3), edges = [], dirty = rnd() < 0.2;
  for (let k = 0; k < m; k++) edges.push({from: dirty && rnd() < 0.1 ? 'missing' : pick(ids), to: dirty && rnd() < 0.1 ? 'missing' : pick(ids), required: rnd() < 0.6, relation: pick(['r', 's', 'depends', 'uses']), ordinal: k});
  check({nodes, edges}, 'random#' + t + (dirty ? '(dirty)' : ''), ['missing']);
}

// 3. Mid-size admitted shapes (old API is quadratic, so 2,000 nodes).
const chain = {nodes: {}, edges: []};
for (let i = 0; i < 2000; i++) { const id = 'c' + String(i).padStart(4, '0'); chain.nodes[id] = {kind: 'candidate', classification: 'OPEN_ACTIVE'}; if (i) chain.edges.push({from: 'c' + String(i - 1).padStart(4, '0'), to: id, required: true, relation: 'chain'}); }
compareRoot(chain, 'c0000', 'chain2000'); compareRoot(chain, 'c1000', 'chain2000 mid');
const fan = {nodes: {root: {kind: 'candidate', classification: 'OPEN_ACTIVE'}}, edges: []};
for (let i = 1; i < 2000; i++) { const id = 'f' + String(i).padStart(4, '0'); fan.nodes[id] = {kind: 'candidate', classification: 'OPEN_ACTIVE'}; fan.edges.push({from: 'root', to: id, required: true, relation: 'fan'}); }
for (let i = 0; fan.edges.length < 10000; i++) fan.edges.push({from: 'f' + String(i % 1998 + 1).padStart(4, '0'), to: 'f1999', required: i % 2 === 0, relation: 'shared.' + i});
compareRoot(fan, 'root', 'fan2000'); compareReach(fan, 'f1999', 'fan2000'); compareRelations(fan, 'f1999', 'fan2000'); compareRelations(fan, 'root', 'fan2000');

console.log(JSON.stringify({corpus, all: stats, seed: seedArg, random}));
console.log('DIFFERENTIAL OK: base and head agree on every compared output');
