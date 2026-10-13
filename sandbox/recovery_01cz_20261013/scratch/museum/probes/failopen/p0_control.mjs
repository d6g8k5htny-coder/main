// Control: the committed cd32303 museum.json validates and renders fully verified against the fixture.
import {museum, realManifest, render, card, links, sourceBytes} from './lib.mjs';
const m = realManifest();
const index = sourceBytes(m.index_source.url).toString('utf8'), status = sourceBytes(m.status_source.url).toString('utf8');
museum.validateBoundManifest(m, index, status); console.log('validateBoundManifest(committed museum.json): PASS');
const d = await render(m);
console.log('museum-state:', d.getElementById('museum-state').textContent);
for (const id of ['d5-pin-neighborhoods-open', 'sard-g-a1-a6-open', 'd2-lifetime-remainder', 'd5-all-height-annulus']) {
  const c = card(d, id); console.log(id, '| links:', JSON.stringify(links(c).filter(l => /proof|source contract|review/i.test(l))));
  console.log('   contains "NO COMPLETE PROOF YET":', c.textContent.includes('NO COMPLETE PROOF YET'), '| contains "OPEN / NOT LANDED":', c.textContent.includes('OPEN / NOT LANDED'), '| contains "not byte-frozen":', c.textContent.includes('not byte-frozen'));
}
