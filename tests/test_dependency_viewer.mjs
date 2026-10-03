import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import * as evidenceModel from '../docs/site/dependency-model.mjs';

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

test('evidence shows recorded links without promoting missing review or formal evidence', () => {
  assert.equal(typeof evidenceModel.evidenceRows, 'function');
  const rows = evidenceModel.evidenceRows({id:'x', classification:'PROVED_REVIEWED', impact:0,
    source:'proof/PROOF.md',review_provider:'OpenAI',review_disposition:'ACCEPT'});
  const byLayer = new Map(rows.map(row=>[row.label,row]));
  assert.equal(byLayer.get('Source').state,'Record linked');
  assert.equal(byLayer.get('Review').state,'Metadata only');
  assert.equal(byLayer.get('Formal proof').state,'Not recorded');
  assert.equal(byLayer.get('Reproduction').state,'Not recorded');
  assert.match(byLayer.get('Review').detail,/OpenAI/);
  assert.ok(!JSON.stringify(rows).includes('independent'));
  assert.ok(!JSON.stringify(rows).includes('Verified'));
});

test('evidence retains unsafe paths as metadata and never turns a reference into verification', () => {
  assert.equal(typeof evidenceModel.evidenceRows,'function');
  const rows=evidenceModel.evidenceRows({source:'../private.txt',review_source:'javascript:alert(1)'});
  assert.equal(rows[0].state,'Metadata only');
  assert.equal(rows[0].href,undefined);
  assert.equal(rows[1].href,undefined);
  const linked=evidenceModel.evidenceRows({review_source:'reviews/r/REVIEW.md',review_disposition:'AMEND'});
  assert.equal(linked[1].state,'Record linked');
  assert.match(linked[1].detail,/AMEND/);
  assert.ok(linked[1].href.endsWith('/reviews/r/REVIEW.md'));
});

test('all-records and exact classification filters intersect text without relabeling nodes', async () => {
  assert.equal(typeof evidenceModel.filterNodes,'function');
  const index=buildGraphIndex(await fixture());
  assert.equal(evidenceModel.filterNodes(index,'','').length,49);
  assert.equal(evidenceModel.filterNodes(index,'math','').length,39);
  const results=evidenceModel.filterNodes(index,'math','PROVED_REVIEWED');
  assert.ok(results.length>0 && results.length<39);
  assert.ok(results.every(node=>node.classification==='PROVED_REVIEWED'));
  assert.equal(evidenceModel.filterNodes(index,'','MISSING_CLASS').length,0);
  assert.deepEqual(evidenceModel.classificationOptions(index),[
    'AUTHOR_SIDE_CANDIDATE','AUTHOR_SIDE_REDUCTION','BLOCKED_ABSENT','COVERED_BY_CANDIDATE','ENGINEERING_CONTROL','FALSE','HOLD',
    'OPEN_ACTIVE','OPEN_HISTORICAL','PROVED_REVIEWED','REFUTED','SUPERSEDED_NONBLOCKING']);
});

test('saved filters restore exact state and visibly reject unknown classifications', async () => {
  assert.equal(typeof evidenceModel.readFilters,'function');
  const index=buildGraphIndex(await fixture());
  assert.deepEqual(evidenceModel.readFilters(index,'?q=cone+moments&classification=PROVED_REVIEWED&node=x'),
    {query:'cone moments',classification:'PROVED_REVIEWED',error:null});
  const bad=evidenceModel.readFilters(index,'?classification=ACCEPT_ALL');
  assert.equal(bad.classification,'ACCEPT_ALL');
  assert.match(bad.error,/Unknown classification/);
});

test('recorded context keeps reading-rule order and resolves only recorded same-graph references', async () => {
  assert.equal(typeof evidenceModel.recordedContextRows, 'function');
  const index = buildGraphIndex(await fixture());
  const rows = evidenceModel.recordedContextRows(index, index.nodes.get('math.uniform-matrix-cap-lifetime'));
  assert.deepEqual(rows[0], {
    field: 'reading_rule', label: 'Reading rule', state: 'Recorded', ordered: true,
    values: [
      { text: 'math.d1-component.congruence-erratum', id: 'math.d1-component.congruence-erratum' },
      { text: 'math.d1-component.section9-replacement-v1_1', id: 'math.d1-component.section9-replacement-v1_1' },
      { text: 'math.d1-component.marked-cylinder-cap', id: 'math.d1-component.marked-cylinder-cap' },
      { text: 'math.d1-component.reconciliation-record', id: 'math.d1-component.reconciliation-record' },
    ],
  });
  const component = evidenceModel.recordedContextRows(index, index.nodes.get('math.d5-component.punctured-pin-continuum-check'));
  assert.deepEqual(component[1].values, [{ text: 'math.d5-pin-neighborhood-first-moment', id: 'math.d5-pin-neighborhood-first-moment' }]);
  assert.deepEqual(component[2].values, [{ text: 'review_checker' }]);
  const coverage = evidenceModel.recordedContextRows(index, index.nodes.get('math.rn-region.fixed-remote'));
  assert.deepEqual(coverage[3].values, [{ text: 'math.rn-fixed-remote-window', id: 'math.rn-fixed-remote-window' }]);
});

test('absent context is visibly not recorded and is never inferred from dependencies', async () => {
  assert.equal(typeof evidenceModel.recordedContextRows, 'function');
  const index = buildGraphIndex(await fixture());
  const parent = index.nodes.get('math.d5-pin-neighborhood-first-moment');
  assert.ok(parent.dependencies.length > 0);
  assert.deepEqual(evidenceModel.recordedContextRows(index, parent).map(row => [row.field, row.state, row.values]), [
    ['reading_rule', 'Not recorded', []], ['component_of', 'Not recorded', []],
    ['component_role', 'Not recorded', []], ['coverage_source', 'Not recorded', []],
  ]);
  const empty = evidenceModel.recordedContextRows(index, { reading_rule: [] })[0];
  assert.equal(empty.state, 'Recorded');
  assert.deepEqual(empty.values, []);
});

test('unresolved and unsafe context references stay literal without executable destinations', async () => {
  assert.equal(typeof evidenceModel.recordedContextRows, 'function');
  const graph = await fixture();
  graph.nodes['javascript:alert(1)'] = {classification:'OPEN_ACTIVE',layer:'test',kind:'candidate'};
  const index = buildGraphIndex(graph);
  const rows = evidenceModel.recordedContextRows(index, {
    reading_rule: ['missing.node', 'javascript:alert(1)', '<img src=x onerror=alert(1)>'],
    component_of: '../private', coverage_source: 'https://example.test/proof',
    component_role: '<img src=x onerror=alert(1)>',
  });
  assert.deepEqual(rows[0].values, [
    { text:'missing.node', note:'not found in this snapshot' },
    { text:'javascript:alert(1)', note:'not found in this snapshot' },
    { text:'<img src=x onerror=alert(1)>', note:'not found in this snapshot' },
  ]);
  assert.deepEqual(rows[1].values, [{ text:'../private', note:'not found in this snapshot' }]);
  assert.deepEqual(rows[3].values, [{ text:'https://example.test/proof', note:'not found in this snapshot' }]);
  assert.deepEqual(rows[2].values, [{ text:'<img src=x onerror=alert(1)>' }]);
  assert.ok(rows.every(row => row.values.every(value => !('href' in value) && !('id' in value))));
});

test('malformed context preserves complete recorded values instead of inventing references', async () => {
  assert.equal(typeof evidenceModel.recordedContextRows, 'function');
  const index = buildGraphIndex(await fixture());
  const rows = evidenceModel.recordedContextRows(index, {
    reading_rule: ['hist.CH-LIFT', { id:'hist.LOGQ-TAIL', extra:['keep', null] }, null, 42, ['hist.CH-LIFT']],
    component_of: { id:'hist.CH-LIFT' }, component_role: ['proof', { detail:'keep' }], coverage_source: null,
  });
  assert.deepEqual(rows[0].values, [
    { text:'hist.CH-LIFT', id:'hist.CH-LIFT' },
    { text:'{"id":"hist.LOGQ-TAIL","extra":["keep",null]}', note:'Invalid recorded value' },
    { text:'null', note:'Invalid recorded value' },
    { text:'42', note:'Invalid recorded value' },
    { text:'["hist.CH-LIFT"]', note:'Invalid recorded value' },
  ]);
  assert.equal(rows[1].state, 'Invalid recorded value');
  assert.deepEqual(rows[1].values, [{ text:'{"id":"hist.CH-LIFT"}' }]);
  assert.equal(rows[2].state, 'Invalid recorded value');
  assert.deepEqual(rows[2].values, [{ text:'["proof",{"detail":"keep"}]' }]);
  assert.equal(rows[3].state, 'Invalid recorded value');
  assert.deepEqual(rows[3].values, [{ text:'null' }]);
  for (const value of ['hist.CH-LIFT', {id:'hist.CH-LIFT'}, null, 0]) {
    const row = evidenceModel.recordedContextRows(index, {reading_rule:value})[0];
    assert.equal(row.state, 'Invalid recorded value');
    assert.deepEqual(row.values, [{text:typeof value === 'string' ? value : JSON.stringify(value)}]);
  }
});

test('context presentation does not change graph edges, impact ranking, classifications or existing search', async () => {
  assert.equal(typeof evidenceModel.recordedContextRows, 'function');
  const graph = await fixture();
  const index = buildGraphIndex(graph);
  const before = JSON.stringify(graph);
  const ranked = unresolvedTargets(index).map(node => [node.id, node.impact, node.classification]);
  const search = searchNodes(index, 'reading_rule_record').map(node => node.id);
  for (const node of index.nodes.values()) evidenceModel.recordedContextRows(index, node);
  assert.equal(JSON.stringify(graph), before);
  assert.deepEqual(index.counts, {nodes:49, edges:55});
  assert.deepEqual(unresolvedTargets(index).map(node => [node.id, node.impact, node.classification]), ranked);
  assert.deepEqual(searchNodes(index, 'reading_rule_record').map(node => node.id), search);
  assert.deepEqual(index.nodes.get('math.rn-region.fixed-remote').dependencies, []);
});

for (const fragment of ['', '#object-evidence', '#object-audit', '#object-reading-context']) for (const digestDelay of [0, 150]) test(`dependency startup preserves complete source metadata and ${fragment || 'search'} after ${digestDelay} ms digest delay`, async () => {
  class Element extends EventTarget {
    constructor(tagName=''){super();this.tagName=tagName;this.children=[];this.value='';this._text='';}
    set textContent(value){this._text=String(value);this.children=[];}
    get textContent(){return this._text+this.children.map(child=>child.textContent||'').join('');}
    append(...children){this.children.push(...children);}
    focus(){this.focused=true;}
    scrollIntoView(){this.scrolled=true;}
    replaceChildren(...children){this.children=children;this._text='';}
  }
  const html=await readFile(pageURL,'utf8');
  const nodes=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const saved=new Map(['window','document','fetch','crypto'].map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  if (digestDelay) Object.defineProperty(globalThis,'crypto',{configurable:true,value:{subtle:{async digest(...args){
    await new Promise(resolve=>setTimeout(resolve,digestDelay));
    return webcrypto.subtle.digest(...args);
  }}}});
  const window=new EventTarget();window.location=new URL('https://example.test/dependencies.html'+(fragment?'?node=math.rn-fixed-annulus-window'+fragment:''));
  window.requestAnimationFrame=callback=>callback();
  window.history={replaceState(state,title,url){window.location=new URL(url);}};
  globalThis.window=window;
  globalThis.document={getElementById:id=>nodes.get(id),createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element(),createTextNode:text=>({textContent:text})};
  globalThis.fetch=async url=>{
    assert.ok(['./dependency-source/GRAPH.json','./dependency-source/PROVENANCE.json'].includes(url));
    return new Response(await readFile(new URL('docs/site/'+url,root)));
  };
  let initialization;
  try {
    ({initialization}=await import(`../docs/site/dependencies.mjs?visitor-search-${digestDelay}-${encodeURIComponent(fragment)}`));
    // Await the actual startup lifecycle, including byte validation and rendering.
    await initialization;
    assert.match(nodes.get('load-status').textContent,/^Pinned Math/);
    if(fragment){
      assert.ok(nodes.has(fragment.slice(1)), 'saved selected-object section exists');
      assert.equal(nodes.get(fragment.slice(1)).focused,true);assert.equal(nodes.get(fragment.slice(1)).scrolled,true);
    }
    const search=nodes.get('dependency-search');search.value='math';search.dispatchEvent(new Event('input'));
    assert.equal(nodes.get('search-results').children.length,39);
    assert.match(nodes.get('search-results').textContent,/math.side24-coefficient/);
    assert.match(nodes.get('search-results').textContent,/math.uniform-matrix-cap-lifetime/);
    assert.equal(new URLSearchParams(window.location.search).get('q'),'math');
    const filter=nodes.get('classification-filter');
    filter.value='PROVED_REVIEWED';filter.dispatchEvent(new Event('change'));
    assert.ok(nodes.get('search-results').children.length<39);
    assert.match(window.location.search,/classification=PROVED_REVIEWED/);
    filter.value='';filter.dispatchEvent(new Event('change'));
    search.value='reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md';search.dispatchEvent(new Event('input'));
    nodes.get('search-results').children[0].children[0].dispatchEvent(new Event('click'));
    assert.match(nodes.get('object-scope').textContent,/compact positive gaps/);
    assert.match(nodes.get('evidence-body').textContent,/Record linked/);
    assert.match(nodes.get('evidence-body').textContent,/Not recorded/);
    assert.match(nodes.get('node-metadata').textContent,/xAI\/Grok/);
    assert.ok(nodes.has('node-record'),'complete source node is available');
    assert.deepEqual(JSON.parse(nodes.get('node-record').textContent),(await fixture()).nodes['math.rn-fixed-annulus-window']);
    const context=nodes.get('recorded-context');
    assert.ok(context,'recorded context is available beside scope and evidence');
    assert.equal(context.textContent.match(/Not recorded/g)?.length,4);
    const descendants=node=>[node,...(node.children||[]).flatMap(descendants)];
    const buttons=()=>descendants(context).filter(node=>node.tagName==='button');
    function select(id){
      search.value=id;search.dispatchEvent(new Event('input'));
      const result=nodes.get('search-results').children.find(item=>item.children[0].children[0].textContent===id);
      assert.ok(result,`existing search selects ${id}`);
      result.children[0].dispatchEvent(new Event('click'));
    }
    const rankBefore=nodes.get('unresolved-targets').textContent;
    select('math.uniform-matrix-cap-lifetime');
    assert.deepEqual(buttons().map(button=>button.textContent),[
      'math.d1-component.congruence-erratum','math.d1-component.section9-replacement-v1_1',
      'math.d1-component.marked-cylinder-cap','math.d1-component.reconciliation-record']);
    assert.equal(descendants(context).filter(node=>node.tagName==='ol').length,1);
    assert.ok(buttons().every(button=>button.type==='button' && !button.href));
    buttons()[2].dispatchEvent(new Event('click'));
    assert.equal(nodes.get('detail-heading').textContent,'math.d1-component.marked-cylinder-cap');
    assert.equal(new URLSearchParams(window.location.search).get('node'),'math.d1-component.marked-cylinder-cap');
    assert.equal(new URLSearchParams(window.location.search).get('q'),'math.uniform-matrix-cap-lifetime');
    assert.equal(window.location.hash,'#node-detail');
    assert.equal(nodes.get('node-detail').focused,true);
    assert.equal(nodes.get('node-detail').scrolled,true);
    assert.match(nodes.get('object-scope').textContent,/sections 1-5.*sections 6-8 are not consumed/);
    assert.match(nodes.get('evidence-body').textContent,/Record linked/);
    assert.match(context.textContent,/deterministic_support/);
    assert.deepEqual(buttons().map(button=>button.textContent),['math.uniform-matrix-cap-lifetime']);
    buttons()[0].dispatchEvent(new Event('click'));
    assert.equal(nodes.get('detail-heading').textContent,'math.uniform-matrix-cap-lifetime');
    select('math.rn-region.fixed-remote');
    assert.deepEqual(buttons().map(button=>button.textContent),['math.rn-fixed-remote-window']);
    assert.match(nodes.get('object-classification').textContent,/COVERED_BY_CANDIDATE/);
    assert.match(nodes.get('dependency-list').textContent,/No direct dependencies/);
    buttons()[0].dispatchEvent(new Event('click'));
    assert.equal(new URLSearchParams(window.location.search).get('node'),'math.rn-fixed-remote-window');
    assert.match(nodes.get('object-classification').textContent,/AUTHOR_SIDE_CANDIDATE/);
    assert.equal(context.textContent.match(/Not recorded/g)?.length,4);
    assert.equal(buttons().length,0,'previous context links are cleared for the next object');
    assert.equal(nodes.get('node-count').textContent,'49');
    assert.equal(nodes.get('edge-count').textContent,'55');
    assert.equal(nodes.get('unresolved-targets').textContent,rankBefore);
    window.location=new URL('https://example.test/dependencies.html?q=bad&classification=NO_SUCH_CLASS');
    window.dispatchEvent(new Event('popstate'));
    assert.equal(context.textContent,'','no selection clears stale context');
    assert.equal(nodes.get('search-results').children.length,0);
    assert.match(nodes.get('filter-error').textContent,/Unknown classification/);
    nodes.get('clear-filters').dispatchEvent(new Event('click'));
    assert.equal(nodes.get('search-results').children.length,49);
    assert.equal(nodes.get('filter-error').hidden,true);
    assert.equal(window.location.search,'');
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
  assert.match(page, /script type="module" src="dependencies\.mjs\?site-release=[0-9a-f]{64}"/);
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
  assert.match(app, /requestAnimationFrame/);
  assert.match(app, /scrollIntoView\(\{\s*block:\s*'start',\s*behavior:\s*'instant'\s*\}\)/);
  assert.match(app, /dependencyPaths/);
  assert.match(app, /prefers-reduced-motion:\s*reduce/);
  assert.match(style, /@media\s*\(max-width:\s*760px\)/);
  assert.match(style, /:focus-visible/);
  assert.match(style, /\.path-arrow[^}]*overflow-wrap:\s*anywhere/s);
  assert.match(style, /classification-proved-reviewed/);
});
