// Baseline: the committed cd32303 museum.json renders cleanly against the byte-verified fixture.
import {realManifest, validate, render, reviewBlock} from './lib.mjs';
const m = realManifest();
console.log('validateBoundManifest(committed museum.json):', validate(m));
const r = await render(m);
console.log('museum-state:', r.state);
for (const id of ['d2-lifetime-remainder','d3-side24-coefficient','d4-fixed-remote-rn','p15-demand-one-counterexample','d1-parent-selection-open','d5-pin-neighborhoods-open','d5-all-height-annulus']) {
  console.log(id, JSON.stringify(reviewBlock(r.cards[id])));
}
const nonFixture = r.requested.filter(u => u !== 'museum.json' && !u.startsWith('https://raw.githubusercontent.com/'));
console.log('requests to anything other than pinned raw URLs:', JSON.stringify(nonFixture));
console.log('requests to github.com issue/comment URLs:', r.requested.filter(u => /github\.com\/.*\/issues\//.test(u)).length);
