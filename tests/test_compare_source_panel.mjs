import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';

const moduleURL = new URL('../docs/site/compare-source-panel.mjs', import.meta.url);
test('Compare source panel demand loader exists', () => assert.ok(existsSync(moduleURL)));
const load = async () => existsSync(moduleURL) ? import(moduleURL) : null;
function section() {
  const inside = {id: 'current-date-explanation'};
  const outside = {id: 'results-heading'};
  const root = {id:'current-research', contains:target=>target===inside||target===root};
  root.ownerDocument={getElementById:id=>({[inside.id]:inside,[outside.id]:outside,[root.id]:root})[id]||null};
  return root;
}
test('An explicit current-source query requests even the empty source search', async () => {
  const m=await load();assert.ok(m,'demand loader missing');
  for(const query of ['?source-q=SIDE24','?source-q=','?source-q'])
    assert.equal(m.sourcePanelRequested(section(),'https://example.test/compare.html'+query),true,query);
});
test('Only a target in the actual source panel requests its verifier', async () => {
  const m=await load();assert.ok(m,'demand loader missing');
  for(const hash of ['#current-research','#current-date-explanation','#current%2Ddate-explanation'])
    assert.equal(m.sourcePanelRequested(section(),'https://example.test/compare.html'+hash),true,hash);
  for(const suffix of ['', '?from=archive:abc', '#results-heading', '#missing', '#%EF%ZZ'])
    assert.equal(m.sourcePanelRequested(section(),'https://example.test/compare.html'+suffix),false,suffix);
});
test('Both pagers use distinct identities and ordinary, non-live count spans', async () => {
  const html=await readFile(new URL('../docs/site/compare.html',import.meta.url),'utf8');
  for(const id of ['pagination-top','previous-top','next-top','page-status-top','pagination','previous','next','page-status'])
    assert.equal([...html.matchAll(new RegExp(`id="${id}"`,'g'))].length,1,id);
  assert.match(html,/aria-label="Result pages, above differences"/);
  assert.match(html,/aria-label="Result pages, below differences"/);
  for(const id of ['page-status','page-status-top']) assert.match(html,new RegExp(`<span id="${id}"></span>`));
});
