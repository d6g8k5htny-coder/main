// Q1/P3: a review pointer is accepted when pointerText.includes(review_url) (museum.mjs:85), an
// undelimited substring test. Any URL that is a prefix of a URL in the claim's quote passes the
// URL-shape regex (museum.mjs:40) and the binding, so a pointer to a DIFFERENT issue or comment is
// displayed as "Review pointer in the verified index". The builder's own check is delimited
// ("](" + url + ")"), so this is a frontend-guard gap.
import {shipped, tryValidate, render, findCard, anchors, texts, indexText} from './common.mjs';
const cases = [
  ['d3-side24-coefficient', 'https://github.com/d6g8k5htny-coder/main/issues/6', 'issue #6 (bullet links #65)'],
  ['d3-side24-coefficient', 'https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-58412', 'truncated comment id'],
  ['d4-fixed-remote-rn', 'https://github.com/d6g8k5htny-coder/main/issues/7', 'issue #7 (bullet links #76)'],
  ['d1-parent-selection-open', 'https://github.com/d6g8k5htny-coder/main/issues/6', 'issue #6 on the AMEND D1 card'],
];
for (const [id, url, label] of cases) {
  const m = shipped(); const c = m.claims.find(x => x.id === id);
  const original = c.review.review_url; c.review.review_url = url;
  const quote = c.scope_quote;
  console.log(`\n[${id}: ${label}]\n  quote contains "](${url})"? ${quote.includes('](' + url + ')')} | quote.includes(url)? ${quote.includes(url)}\n  ${tryValidate(m)}`);
  if (id !== 'd3-side24-coefficient' || !url.endsWith('/6')) continue;
  const {document} = await render(m);
  const card = findCard(document.getElementById('claim-cards'), id);
  console.log('  rendered:', JSON.stringify(texts(card).filter(x => /Review pointer|not byte-frozen/.test(x))), '\n  link:', JSON.stringify(anchors(card).find(([l]) => /review/i.test(l))));
}
// Variant: pointer_only is checked with === true in validateDescriptor (museum.mjs:40) but by
// truthiness in validateBoundManifest (82) and renderClaim (151-154). pointer_only:1 skips the URL
// regex and the review_notice requirement; review_url '' is a substring of every quote.
const m = shipped(); const c = m.claims.find(x => x.id === 'd5-two-scale');
c.review = {...m.index_source, pointer_only: 1, review_url: ''};
console.log('\n[d5-two-scale: review := {...index_source, pointer_only: 1, review_url: ""}]\n ', tryValidate(m));
const {document} = await render(m);
const card = findCard(document.getElementById('claim-cards'), 'd5-two-scale');
console.log('  rendered:', JSON.stringify(texts(card).filter(x => /Review pointer|not byte-frozen|Identity of the pointer/.test(x))), '\n  link:', JSON.stringify(anchors(card).find(([l]) => /review/i.test(l))));
