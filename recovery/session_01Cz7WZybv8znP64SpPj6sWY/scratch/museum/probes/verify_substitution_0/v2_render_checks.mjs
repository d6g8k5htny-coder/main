// Verifier probe (read-only, no network): (1) which views fail in the baseline offline render, so the
// "3 unavailable" in p4 is attributable; (2) end state of the path-swap case (exhibits.remote := D5
// height-window proof), with and without a matching exhibit_quotes.remote; (3) the reported annulus
// substitution, recording what the exhibit card displays next to the D5 claim card.
import fs from 'node:fs';
import {shipped, render, findCard, texts, makeDescriptor, HERE, walk} from './common.mjs';
const states = d => ['museum-state'].map(id => d.getElementById(id).textContent)[0];
const active = d => texts(d.getElementById('active-exhibit')).slice(0, 4);
{ const {document, log} = await render(shipped(), {search: '?view=annulus', withGeometry: true});
  console.log('[1] baseline state:', states(document)); console.log('[1] fetch misses:', JSON.stringify(log));
  const refused = []; walk(document.getElementById('claim-cards'), n => { if (n.className === 'museum-card refused') refused.push(texts(n)[0]); });
  console.log('[1] refused claim cards:', JSON.stringify(refused), '| active exhibit:', JSON.stringify(active(document))); }
for (const changeQuote of [false, true]) {
  const m = shipped(); m.exhibits.remote = {...m.claims[4].proof, id: 'remote', class: 'illustration', notice: m.canvas_notice};
  if (changeQuote) { const t = fs.readFileSync(`${HERE}/fixture/Math-/${m.claims[4].proof.path}`, 'utf8'); m.exhibit_quotes.remote = t.slice(0, 200); }
  const {document} = await render(m, {search: '?view=remote', withGeometry: true});
  console.log(`[2] path swap, quote ${changeQuote ? 'matched' : 'unchanged'}: final active-exhibit =`, JSON.stringify(active(document)));
}
{ const p = 'frontiers/rn_annulus_bridge_20260925/PROOF.md';
  const real = fs.readFileSync(`${HERE}/fixture/Math-/${p}`, 'utf8');
  const altText = real.replace('and 1<A<B<infinity.', 'and 0<A<B<=infinity (pin neighborhoods and all distances included).');
  console.log('[3] real annulus source contains "1<A<B<infinity.":', real.includes('and 1<A<B<infinity.'), '| replaced:', altText !== real);
  const alt = Buffer.from(altText), start = '## 1. Statement and exact scope';
  const m = shipped(); const commit = '4'.repeat(40);
  m.exhibits.annulus = {...makeDescriptor('d6g8k5htny-coder/Math-', commit, p, alt), id: 'annulus', class: 'illustration', notice: m.canvas_notice};
  m.exhibit_quotes.annulus = start + altText.split(start, 2)[1].split('## 2.', 1)[0];
  const {document} = await render(m, {search: '?view=annulus', overrides: new Map([[m.exhibits.annulus.url, alt]]), withGeometry: true});
  const card = findCard(document.getElementById('active-exhibit'), 'D5 all-height fixed annulus'); const t = texts(card);
  console.log('[3] exhibit card texts (first 12):', JSON.stringify(t.slice(0, 12).map(s => s.slice(0, 70))));
  console.log('[3] excerpt shows widened scope:', t.some(s => s.includes('0<A<B<=infinity')));
  const claim = findCard(document.getElementById('claim-cards'), 'd5-all-height-annulus'); const ct = texts(claim);
  console.log('[3] D5 claim card still shows its own commit:', ct[ct.indexOf('Commit') + 1], '| label:', ct.find(s => s.startsWith('Source label')));
}
