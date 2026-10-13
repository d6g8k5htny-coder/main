// F2: validateBoundManifest binds a review pointer with pointerText.includes(review_url): a substring test.
// Any regex-valid URL that is a PREFIX of a URL in the quote passes: other issue numbers, truncated comment ids.
import {realManifest, validate, render, reviewBlock} from './lib.mjs';
const B = 'https://github.com/d6g8k5htny-coder/main/issues/';
const cases = [
  [0, 'd2-lifetime-remainder', B + '6', 'issue #6: prefix of .../issues/67#issuecomment-...'],
  [0, 'd2-lifetime-remainder', B + '67#issuecomment-584', 'truncated comment id 584 (a different comment anchor)'],
  [2, 'd4-fixed-remote-rn', B + '7', 'issue #7: prefix of .../issues/76#issuecomment-...'],
  [1, 'd3-side24-coefficient', B + '65#issuecomment-58412', 'truncated comment id'],
  [11, 'd1-parent-selection-open', B + '6', 'issue #6 on the AMEND/open D1 card'],
  [0, 'd2-lifetime-remainder', B + '99#issuecomment-1', 'CONTROL: not a substring of the quote'],
];
for (const [i, id, url, why] of cases) {
  const m = realManifest(); m.claims[i].review.review_url = url;
  const v = validate(m);
  let shown = '';
  if (v === 'PASS') { const r = await render(m); shown = ' | displayed: ' + JSON.stringify(reviewBlock(r.cards[id])); }
  console.log(`${id} review_url=${url} (${why}) -> ${v}${shown}`);
}
