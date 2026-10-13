// Q1/P6 (frontend half): for 'd5-pin-neighborhoods-open' nothing in validateBoundManifest depends on
// the row content except title/quote/status_quote equality with whatever row sits at position 2:
// its proof must equal index_source (79) and its review pointer is checked against the PROOF_INDEX
// "D5 pin neighborhoods" line (84-85), not against the row. The D5 PR55 AMEND review is displayed on
// the card of a different obligation.
import fs from 'node:fs';
import {tryValidate, render, findCard, anchors, texts, indexText, HERE, pinnedBytes, MAIN_PIN} from './common.mjs';
const m = JSON.parse(fs.readFileSync(`${HERE}/p6_manifest.json`, 'utf8'));
const status = fs.readFileSync(`${HERE}/p6_STATUS_successor.md`);
console.log('validateBoundManifest(successor):', tryValidate(m, indexText(), status.toString('utf8')));
const overrides = new Map([[m.status_source.url, status]]);
for (const d of [m.claims[13].review, ...m.packets.flatMap(p => [p.result]), m.claims[1].replay.source]) {
  if (d.repository === 'd6g8k5htny-coder/main' && d.path !== 'STATUS.md') overrides.set(d.url, pinnedBytes(d.repository, MAIN_PIN, d.path));
}
const {document} = await render(m, {overrides});
const card = findCard(document.getElementById('claim-cards'), 'd5-pin-neighborhoods-open');
const t = texts(card);
console.log('card id d5-pin-neighborhoods-open rendered:', !!card);
console.log('  title:', t[2]);
console.log('  shows:', JSON.stringify(t.filter(x => /NO COMPLETE PROOF|Review pointer|Unmerged review/.test(x))));
console.log('  review link:', JSON.stringify(anchors(card).find(([l]) => /review/i.test(l))));
