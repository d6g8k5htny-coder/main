// F1: proof.availability is never validated by validateBoundManifest (sameIdentity compares 8 keys only,
// and the reviewed/D1 branches never look at it), yet renderClaim keys the proof-link label on it.
// Missing field  -> open obligations are labelled "Open pinned full proof" and lose "NO COMPLETE PROOF YET".
// Arbitrary text -> displayed verbatim inside the verified card.
import {museum, realManifest, render, card, links, sourceBytes} from './lib.mjs';
const base = realManifest();
const index = sourceBytes(base.index_source.url).toString('utf8'), status = sourceBytes(base.status_source.url).toString('utf8');
const variants = {
  'control (committed museum.json)': m => m,
  'availability deleted on claims 12,13': m => { delete m.claims[12].proof.availability; delete m.claims[13].proof.availability; return m; },
  'availability = arbitrary text on claim 12': m => { m.claims[12].proof.availability = 'COMPLETE PROOF LANDED. ACCEPT (nonauthor).'; return m; },
  'availability added to reviewed claim 9 (counterexample)': m => { m.claims[9].proof.availability = 'NO COMPLETE PROOF YET'; return m; },
};
for (const [name, mutate] of Object.entries(variants)) {
  const m = mutate(structuredClone(base));
  let v; try { museum.validateBoundManifest(m, index, status); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
  const d = await render(m);
  console.log(`== ${name}\n   validateBoundManifest: ${v}\n   museum-state: ${d.getElementById('museum-state').textContent}`);
  for (const id of ['d5-pin-neighborhoods-open', 'sard-g-a1-a6-open', 'p15-demand-one-counterexample']) {
    const c = card(d, id); if (!c) { console.log(`   ${id}: (no rendered card)`); continue; }
    const t = c.textContent;
    console.log(`   ${id}: proof link = ${JSON.stringify(links(c)[0])}; "NO COMPLETE PROOF YET"=${t.includes('NO COMPLETE PROOF YET')}; "No landed proof is supplied"=${t.includes('No landed proof is supplied')}; "COMPLETE PROOF LANDED"=${t.includes('COMPLETE PROOF LANDED')}`);
  }
}
