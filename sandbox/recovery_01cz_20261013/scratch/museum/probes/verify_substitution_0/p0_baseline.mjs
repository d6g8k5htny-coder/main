// Baseline: the shipped cd32303 museum.json validates against the pinned index/status bytes
// and renders; list each claim's proof and review binding as shipped.
import {shipped, tryValidate, render, findCard} from './common.mjs';
const m = shipped();
console.log('shipped manifest:', tryValidate(m));
for (const c of m.claims) console.log(c.id.padEnd(32), '| proof', c.proof.path, '| review', c.review.pointer_only ? `POINTER ${c.review.review_url} (identity ${c.review.path})` : `${c.review.path}@${c.review.commit.slice(0,8)}`);
const {document, log} = await render(m);
console.log('state:', document.getElementById('museum-state').textContent);
const cards = document.getElementById('claim-cards').children.map(h => h.children[0]);
console.log('rendered cards:', cards.filter(c => c.attributes['data-object-id']).length, '| refused:', cards.filter(c => c.className === 'museum-card refused').map(c => c.textContent).join(' || '));
console.log('fetch misses:', [...new Set(log)].join('\n  '));
