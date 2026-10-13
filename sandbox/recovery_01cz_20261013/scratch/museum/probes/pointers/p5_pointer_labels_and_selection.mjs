// F6: on the committed manifest, the pointer label and the single "Open review" link misdescribe the pointer source.
import {realManifest, render, reviewBlock} from './lib.mjs';
const m = realManifest(); const r = await render(m);
for (const c of m.claims.filter(c => c.review && c.review.pointer_only)) {
  const card = r.cards[c.id]; const rb = reviewBlock(card);
  const reviewLinks = [...c.scope_quote.matchAll(/\[([^\]]+)\]\((https:\/\/github\.com\/[^)]+\/issues\/[^)]+)\)/g)].map(x => `${x[1]} -> ${x[2].replace('https://github.com/d6g8k5htny-coder/', '')}`);
  const reviewAnchors = card.anchors.filter(a => a.text === 'Open review ↗').length;
  console.log(`${c.id}: label="${rb.label}" | pointer source actually = ${c.review.repository.split('/')[1]}:${c.review.path}`);
  console.log(`    issue links in the quoted source line: ${JSON.stringify(reviewLinks)}`);
  console.log(`    "Open review" anchors on card: ${reviewAnchors} -> ${rb.open_review_href.replace('https://github.com/d6g8k5htny-coder/', '')}`);
}
