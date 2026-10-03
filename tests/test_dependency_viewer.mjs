import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';

import {
  buildGraphIndex,
  classificationClass,
  dependencyPaths,
  metadataEntries,
  normalizeSelection,
  parsePinnedGraph,
  searchNodes,
  unresolvedTargets,
  validateGraph,
} from '../docs/site/dependency-model.mjs';

const root = new URL('../', import.meta.url);
const graphURL = new URL('docs/site/dependency-source/GRAPH.json', root);
const provenanceURL = new URL('docs/site/dependency-source/PROVENANCE.json', root);
const pageURL = new URL('docs/site/dependencies.html', root);
const appURL = new URL('docs/site/dependencies.mjs', root);
const styleURL = new URL('docs/site/dependencies.css', root);

for (const digestDelay of [0, 150]) test(`broad dependency searches expose every matching node after ${digestDelay} ms digest delay`, async () => {
  class Element extends EventTarget {
    constructor(){super();this.children=[];this.value='';this._text='';}
    set textContent(value){this._text=String(value);this.children=[];}
    get textContent(){return this._text+this.children.map(child=>child.textContent||'').join('');}
    append(...children){this.children.push(...children);}
    replaceChildren(...children){this.children=children;this._text='';}
  }
  const html=await readFile(pageURL,'utf8');
  const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const saved=new Map(['window','document','fetch','crypto'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  if (digestDelay) Object.defineProperty(globalThis,'crypto',{configurable:true,value:{subtle:{async digest(...args){
    await new Promise(resolve=>setTimeout(resolve,digestDelay));
    return webcrypto.subtle.digest(...args);
  }}}});
  const window=new EventTarget();window.location={hash:'',search:'',href:'https://example.test/dependencies.html'};
  globalThis.window=window;
  globalThis.document={getElementById:id=>nodes.get(id),createElement:()=>new Element(),createTextNode:text=>({textContent:text})};
  globalThis.fetch=async url=>{
    assert.ok(['./dependency-source/GRAPH.json','./dependency-source/PROVENANCE.json'].includes(url));
    return new Response(await readFile(new URL('docs/site/'+url,root)));
  };
  let initialization;
  try {
    ({initialization}=await import(`../docs/site/dependencies.mjs?visitor-search-${digestDelay}`));
    // Await the actual startup lifecycle, including byte validation and rendering.
    await initialization;
    assert.match(nodes.get('load-status').textContent,/^Pinned Math/);
    const search=nodes.get('dependency-search');search.value='math';search.dispatchEvent(new Event('input'));
    assert.equal(nodes.get('search-results').children.length,39);
    assert.match(nodes.get('search-results').textContent,/math.side24-coefficient/);
    assert.match(nodes.get('search-results').textContent,/math.uniform-matrix-cap-lifetime/);
  } finally {
    // Keep the DOM/network globals alive until all startup work has settled.
    await initialization;
    for(const [name,descriptor] of saved)if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];
  }
});

async function fixture() {
  return JSON.parse(await readFile(graphURL, 'utf8'));
}

test('the pinned graph is complete and structurally valid', async () => {
  const graph = await fixture();
  assert.deepEqual(validateGraph(graph), []);
  assert.equal(Object.keys(graph.nodes).length, 49);
  assert.equal(graph.edges.length, 55);
});

test('unresolved targets include open leaves with no recorded dependents', async () => {
  const graph = await fixture();
  const index = buildGraphIndex(graph);
  const targets = unresolvedTargets(index);
  const ids = targets.map(target => target.id);
  assert.ok(ids.includes('hist.LOGQ-TAIL'));
  assert.ok(ids.includes('math.rn-region.witness-collision'));
  assert.equal(index.nodes.get('hist.LOGQ-TAIL').dependents.length, 0);
  assert.equal(index.nodes.get('math.rn-region.witness-collision').dependents.length, 0);
  assert.ok(!ids.includes('hist.lemma_closed'));
  assert.ok(!ids.includes('eng.hard-gate'));
});

test('unresolved targets rank by transitive impact without hiding zero-impact leaves', async () => {
  const targets = unresolvedTargets(buildGraphIndex(await fixture()));
  assert.equal(targets.length, 15);
  assert.ok(targets[0].impact >= targets.at(-1).impact);
  assert.ok(targets.some(target => target.impact === 0));
  for (let i = 1; i < targets.length; i += 1) {
    const previous = targets[i - 1];
    const current = targets[i];
    assert.ok(previous.impact > current.impact ||
      (previous.impact === current.impact && previous.id.localeCompare(current.id) <= 0));
  }
});

test('search covers ids, notes, source paths, classifications and edge relations', async () => {
  const index = buildGraphIndex(await fixture());
  assert.ok(searchNodes(index, 'shrinking charts').some(node => node.id === 'hist.CH-LIFT'));
  assert.ok(searchNodes(index, 'UNIFORM_MATRIX_CAP').some(node => node.id === 'math.uniform-matrix-cap-lifetime'));
  assert.ok(searchNodes(index, 'OPEN_ACTIVE').some(node => node.id === 'hist.Piece-2-annulus'));
  assert.ok(searchNodes(index, 'reads_with_congruence_erratum').some(node => node.id === 'math.uniform-matrix-cap-lifetime'));
  assert.ok(searchNodes(index, 'reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md')
    .some(node => node.id === 'math.rn-fixed-annulus-window'));
  assert.ok(searchNodes(index, 'xAI/Grok via Cursor')
    .some(node => node.id === 'math.rn-fixed-annulus-window'));
  assert.ok(searchNodes(index, '74')
    .some(node => node.id === 'math.p15-full-price'));
  assert.ok(searchNodes(index, 'ACCEPT_AT_STATED_SCOPE')
    .some(node => node.id === 'math.rn-region.mesoscopic-scaled-annulus'));
  assert.equal(searchNodes(index, 'definitely-no-such-node').length, 0);
});

test('the app verifies graph bytes against a compiled source identity before parsing', async () => {
  const [bytes, provenance] = await Promise.all([
    readFile(graphURL),
    readFile(provenanceURL, 'utf8').then(JSON.parse),
  ]);
  const graph = await parsePinnedGraph(bytes, provenance);
  assert.equal(Object.keys(graph.nodes).length, 49);

  const changed = Buffer.from(bytes);
  changed[100] ^= 1;
  await assert.rejects(parsePinnedGraph(changed, provenance), /digest mismatch/i);
  await assert.rejects(parsePinnedGraph(bytes, { ...provenance, commit: '0'.repeat(40) }), /provenance mismatch/i);
  await assert.rejects(parsePinnedGraph(bytes, { ...provenance, captured_at: '2099-01-01T00:00:00Z' }), /provenance mismatch/i);
});

test('classification styling and metadata preserve source review lineage', async () => {
  const index = buildGraphIndex(await fixture());
  const reviewed = index.nodes.get('math.rn-fixed-annulus-window');
  const entries = new Map(metadataEntries(reviewed));
  assert.equal(classificationClass('PROVED_REVIEWED'), 'classification-proved-reviewed');
  assert.equal(classificationClass('OPEN_ACTIVE'), 'classification-open-active');
  assert.equal(entries.get('Review source'), 'reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md');
  assert.equal(entries.get('Review provider'), 'xAI/Grok via Cursor');
  const issueEntries = new Map(metadataEntries(index.nodes.get('math.p15-full-price')));
  assert.equal(issueEntries.get('Review issue'), '74');
  assert.equal(issueEntries.get('Review disposition'), undefined);
  const dispositionEntries = new Map(metadataEntries(index.nodes.get('math.rn-region.mesoscopic-scaled-annulus')));
  assert.equal(dispositionEntries.get('Review disposition'), 'ACCEPT_AT_STATED_SCOPE');
  const nonDischarge = index.nodes.get('math.d1-component.reconciliation-record');
  const nonDischargeEntries = new Map(metadataEntries(nonDischarge));
  assert.equal(
    nonDischargeEntries.get('Technical checks (non-discharge)'),
    JSON.stringify(nonDischarge.technical_checks_non_discharge[0]),
  );
  assert.ok(searchNodes(index, 'never discharge')
    .some(node => node.id === 'math.d1-component.reconciliation-record'));
  const component = index.nodes.get('math.d5-component.punctured-pin-continuum-check');
  const componentEntries = new Map(metadataEntries(component));
  assert.equal(componentEntries.get('Component role'), 'review_checker');
  assert.ok(searchNodes(index, 'review_checker')
    .some(node => node.id === component.id));

  const remainderEntries = metadataEntries(index.nodes.get('math.lifetime-remainder'));
  const source = remainderEntries.find(([label]) => label === 'Source');
  const reviewSource = remainderEntries.find(([label]) => label === 'Review source');
  assert.equal(source[2],
    'https://github.com/d6g8k5htny-coder/Math-/blob/7858329974e28be79f29b22644370084ff43da4f/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md');
  assert.equal(reviewSource[2],
    'https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206');
  const externalSource = metadataEntries(index.nodes.get('math.rn-mesoscopic-reduction'))
    .find(([label]) => label === 'Source');
  assert.equal(externalSource[2], undefined);
  assert.equal(metadataEntries({
    id: 'synthetic', classification: 'OPEN_ACTIVE', layer: 'test', kind: 'candidate',
    controlling: false, impact: 0, source: '../private.txt',
    review_source: 'https://example.com/review',
  }).filter(([, , href]) => href !== undefined).length, 0);
});

test('classification CSS identifiers use locale-neutral ASCII case folding', () => {
  const original = String.prototype.toLocaleLowerCase;
  String.prototype.toLocaleLowerCase = () => 'classification-broken-by-locale';
  try {
    assert.equal(classificationClass('ENGINEERING_CONTROL'), 'classification-engineering-control');
    assert.equal(classificationClass('AUTHOR_SIDE_CANDIDATE'), 'classification-author-side-candidate');
  } finally {
    String.prototype.toLocaleLowerCase = original;
  }
});

test('dependency paths use required source edges and retain relation metadata', async () => {
  const index = buildGraphIndex(await fixture());
  const paths = dependencyPaths(index, 'math.rn-mesoscopic-reduction');
  const target = paths.find(path => path.ids.at(-1) === 'hist.Piece-2-annulus');
  assert.equal(target, undefined);
  const nested = paths.find(path => path.ids.at(-1) === 'math.rn-count-interface');
  assert.ok(nested);
  assert.ok(nested.ids.includes('math.rn-fixed-remote-window'));
  assert.equal(nested.ids[0], 'math.rn-mesoscopic-reduction');
  assert.equal(nested.edges.length, nested.ids.length - 1);
  assert.ok(nested.edges.every(edge => edge.required && edge.relation.length));

  const annulusPaths = dependencyPaths(index, 'math.rn-fixed-annulus-window');
  assert.deepEqual(annulusPaths.map(path => path.ids.at(-1)), ['math.rn-count-interface']);
  assert.ok(!annulusPaths.some(path => path.ids.includes('regional.fixed-annulus.high-jet-route')));
  assert.ok(!annulusPaths.some(path => path.ids.includes('hist.CH-LIFT')));
});

test('saved node links fail visibly instead of selecting an unrelated node', async () => {
  const index = buildGraphIndex(await fixture());
  assert.deepEqual(normalizeSelection(index, 'hist.CH-LIFT'), {
    id: 'hist.CH-LIFT', error: null,
  });
  assert.deepEqual(normalizeSelection(index, 'missing.node'), {
    id: null, error: 'Unknown node “missing.node”. The saved link does not match this source snapshot.',
  });
  assert.deepEqual(normalizeSelection(index, ''), { id: null, error: null });
});

test('provenance pins the exact Math source bytes and labels the snapshot boundary', async () => {
  const provenance = JSON.parse(await readFile(provenanceURL, 'utf8'));
  assert.equal(provenance.repository, 'd6g8k5htny-coder/Math-');
  assert.equal(provenance.commit, '7858329974e28be79f29b22644370084ff43da4f');
  assert.equal(provenance.files['GRAPH.json'].bytes, 38753);
  assert.equal(provenance.files['GRAPH.json'].sha256, '8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09');
  assert.equal(provenance.files['hard_gate.py'].bytes, 26735);
  assert.equal(provenance.files['hard_gate.py'].sha256, 'a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8');
  assert.match(provenance.scope, /dated read-only snapshot/i);
  assert.equal(provenance.scientific_effect, 'NONE');
});

test('the viewer page exposes its source boundary and accessible interaction contract', async () => {
  const [page, app, style] = await Promise.all([
    readFile(pageURL, 'utf8'),
    readFile(appURL, 'utf8'),
    readFile(styleURL, 'utf8'),
  ]);
  assert.match(page, /Content-Security-Policy/);
  assert.match(page, /script type="module" src="dependencies\.mjs"/);
  assert.match(page, /id="dependency-search"/);
  assert.match(page, /id="unresolved-targets"/);
  assert.match(page, /id="node-detail"/);
  assert.match(page, /id="selection-error"[^>]*role="alert"/);
  assert.match(page, /dated read-only snapshot/i);
  assert.match(page, /scientific effect:\s*NONE/i);
  assert.match(page, /dependency-source\/GRAPH\.json/);
  assert.match(page, /dependency-source\/PROVENANCE\.json/);
  assert.match(page, />Pinned gate implementation</);
  assert.doesNotMatch(page, />Pinned generator</);
  assert.doesNotMatch(page, /<script(?![^>]*src=)/i);
  assert.doesNotMatch(page, /\son(?:click|change|input|submit)=/i);
  assert.match(app, /URLSearchParams/);
  assert.match(app, /replaceState/);
  assert.match(app, /popstate[\s\S]*writeURL:\s*false,\s*focus:\s*false/);
  assert.match(app, /prepareInitialSelectionRestore/);
  assert.match(app, /\['wheel',[\s\S]*'pagehide'\]/);
  assert.match(app, /event\.type === 'focusin'[\s\S]*node-detail/);
  assert.match(app, /requestAnimationFrame/);
  assert.match(app, /scrollIntoView\(\{\s*block:\s*'start',\s*behavior:\s*'instant'\s*\}\)/);
  assert.match(app, /dependencyPaths/);
  assert.match(app, /prefers-reduced-motion:\s*reduce/);
  assert.match(style, /@media\s*\(max-width:\s*760px\)/);
  assert.match(style, /:focus-visible/);
  assert.match(style, /\.path-arrow[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(style, /classification-proved-reviewed/);
});
