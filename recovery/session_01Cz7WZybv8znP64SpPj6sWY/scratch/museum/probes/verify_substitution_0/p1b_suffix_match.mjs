// Q1/P1b: sourceContainsPath (museum.mjs:51-53) also accepts `/${path})` anywhere in the quote, i.e. a
// path SUFFIX of any link, including a mutable branch URL. The D5 AMEND row links
// .../Math-/blob/main/PROOF_INDEX.md, so the pinned index itself is accepted as the D5 card's
// "Byte-verified review"; the P15 bullet's LANDING_CLAIMS.json link is accepted likewise.
import {shipped, tryValidate, render, findCard, anchors, texts} from './common.mjs';
for (const [id, pick] of [['d5-pin-neighborhoods-open', m => ({...m.index_source})], ['p15-demand-one-counterexample', m => null]]) {
  const m = shipped(); const c = m.claims.find(x => x.id === id);
  if (id === 'p15-demand-one-counterexample') {
    const {makeDescriptor, pinnedBytes, MATH_PIN} = await import('./common.mjs');
    c.review = makeDescriptor('d6g8k5htny-coder/Math-', MATH_PIN, 'claims/LANDING_CLAIMS.json', pinnedBytes('d6g8k5htny-coder/Math-', MATH_PIN, 'claims/LANDING_CLAIMS.json'));
  } else c.review = pick(m);
  console.log(`\n[${id}] review := ${c.review.path} (no pointer_only)\n  ${tryValidate(m)}`);
  const {document} = await render(m);
  const card = findCard(document.getElementById('claim-cards'), id);
  console.log('  rendered:', JSON.stringify(texts(card).filter(x => /verified review|Review pointer|Unmerged/.test(x))), '\n  link:', JSON.stringify(anchors(card).find(([l]) => /review/i.test(l))));
}
