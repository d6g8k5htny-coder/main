// F1: review_url is validated only when pointer_only===true, yet renderClaim links "Open review" to
// review_url || html_url. A non-pointer ("Byte-verified review") descriptor can carry any review_url.
import {realManifest, validate, render, reviewBlock, museum} from './lib.mjs';
const cases = {
  'mutable issue comment (not in the D5 quote)': 'https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841570965',
  'foreign host': 'https://example.invalid/forged-review',
  'branch-mutable blob/main path': 'https://github.com/d6g8k5htny-coder/Math-/blob/main/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md',
};
for (const [name, url] of Object.entries(cases)) {
  const m = realManifest();
  m.claims[3].review.review_url = url;          // d5-all-height-annulus; pointer_only stays absent
  console.log(`--- ${name}`);
  console.log('validateDescriptor(review):', (() => { try { museum.validateDescriptor(m.claims[3].review); return 'PASS'; } catch (e) { return 'REFUSED: ' + e.message; } })());
  console.log('validateBoundManifest:', validate(m));
  const r = await render(m);
  const c = r.cards['d5-all-height-annulus'];
  console.log('card class:', c.text.includes('ACCEPT-scoped') ? 'ACCEPT-scoped' : '?', '| review block:', JSON.stringify(reviewBlock(c)));
  console.log('identity shown under "Review source identity" still names:', m.claims[3].review.path, '@', m.claims[3].review.commit.slice(0, 12));
  console.log('museum-state:', r.state);
}
// Control: the same URL with pointer_only:true IS refused, i.e. the check exists but is gated on the flag.
const ctl = realManifest(); Object.assign(ctl.claims[3].review, {pointer_only: true, review_url: 'https://example.invalid/forged-review', review_notice: 'x'});
console.log('--- control (pointer_only:true, same foreign URL):', validate(ctl));
