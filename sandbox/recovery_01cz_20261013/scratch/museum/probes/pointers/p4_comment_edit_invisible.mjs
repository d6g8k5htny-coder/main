// F5: on the COMMITTED manifest, every comment-pointer card still prints "Review wording is quoted at its declared
// scope" and the page reports "displayed source bytes verified", while no review wording is quoted and nothing binds
// the comment's content. Simulate an edited/retracted comment: the rendered page is byte-identical.
import {realManifest, render} from './lib.mjs';
const m = realManifest();
const pointerCards = m.claims.filter(c => c.review && c.review.pointer_only);
const variants = {
  original: 'R5/R6 delta review: ACCEPT for Theorem R at its stated scope.',
  edited: 'EDITED 2026-10-12: ACCEPT WITHDRAWN. Theorem R is AMEND pending a new gap.',
};
const outs = {};
for (const [k, body] of Object.entries(variants)) {
  const extra = Object.fromEntries(pointerCards.map(c => [c.review.review_url, body]));   // served if ever requested
  const r = await render(m, extra);
  outs[k] = r;
  console.log(`[${k}] github.com review URLs requested by the viewer: ${r.requested.filter(u => u.startsWith('https://github.com/')).length}`);
}
const same = JSON.stringify(outs.original.cards) === JSON.stringify(outs.edited.cards) && outs.original.state === outs.edited.state;
console.log('rendered claim cards identical after the simulated comment edit:', same);
console.log('museum-state:', outs.edited.state);
for (const c of pointerCards) {
  const card = outs.edited.cards[c.id];
  const strip = card.text.match(/Source byte count[^.]*\.[^.]*\./)?.[0];
  console.log(`- ${c.id} [${c.class}] review_url=${c.review.review_url}`);
  console.log(`    strip: "${strip}"`);
  console.log(`    quotes on card come from: scope_quote <- ${c.id.endsWith('-open') ? 'STATUS AMEND row' : 'PROOF_INDEX bullet'}; status_quote <- ${c.status_quote === null ? '(none)' : 'STATUS cell'}; review text quoted: none`);
  console.log(`    review descriptor fields: ${Object.keys(c.review).sort().join(',')}`);
  console.log(`    identity shown for the review = ${c.review.repository}:${c.review.path}@${c.review.commit.slice(0, 8)} (the pointer source, not the comment)`);
}
