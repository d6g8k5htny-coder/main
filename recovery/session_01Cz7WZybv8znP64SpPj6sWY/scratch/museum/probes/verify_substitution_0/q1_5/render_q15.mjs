// Verifier probe Q1-5: render through the cd32303 museum.mjs startMuseum (no network; fetch stubbed in
// common.mjs). Manifest = the museum.json the patched COPY's builder generated (README re-pinned to the
// 10e1f191 stand-in bytes, which are served for the README URL). Prints what each card shows.
import fs from 'node:fs';
import {render, findCard, texts, HERE} from './common.mjs';
const readme = fs.readFileSync(`${HERE}/readme_10e1f191.md`);
const load = v => JSON.parse(fs.readFileSync(`${HERE}/runs/${v}/docs/site/museum.json`, 'utf8'));
async function show(label, manifest, ids) {
  const url = manifest.claims[0].replay.source.url;
  const {document, log} = await render(manifest, {overrides: new Map([[url, readme]])});
  console.log(`[${label}] state: ${document.getElementById('museum-state').textContent}`);
  for (const id of ids) {
    const card = findCard(document.getElementById('claim-cards'), id);
    if (!card) { console.log(`  ${id}: NO CARD (refused)`); continue; }
    const t = texts(card);
    const cmd = t[t.indexOf('Replay') + 2];
    console.log(`  ${id}: class=${card.attributes['data-object-class']} | replay shown: ${JSON.stringify(cmd)} | notice: ${JSON.stringify(t[t.indexOf('Replay') + 1])}`);
  }
  if (log.length) console.log('  misses:', log.join('; '));
}
await show('baseline', load('baseline'), ['d2-lifetime-remainder', 'd4-fixed-remote-rn', 'd6-p15-full-price']);
await show('SWAP', load('SWAP_D2_D6__D4_gets_D0_hard_gate'), ['d2-lifetime-remainder', 'd4-fixed-remote-rn', 'd6-p15-full-price']);
const m = load('baseline'); const x = m.claims.find(c => c.id === 'p15-demand-one-counterexample');
console.log(`p15 counterexample replay in built manifest: command=${JSON.stringify(x.replay.command)}`);
x.replay = structuredClone(m.claims[10].replay);
await show('P15X (frontend-only edit of museum.json)', m, ['p15-demand-one-counterexample']);
