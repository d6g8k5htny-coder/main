// F2: validateDescriptor enforces the review-pointer URL shape and the non-empty "not byte-frozen" notice only
// when pointer_only === true, but validateBoundManifest and renderClaim branch on truthiness. pointer_only:1
// (or "yes") skips the shape/notice guard while still taking the pointer branch, where
// pointerText.includes(review_url) is vacuously true for review_url "" (String.prototype.includes("") === true).
import {museum, realManifest, render, card, links, sourceBytes} from './lib.mjs';
const base = realManifest();
const index = sourceBytes(base.index_source.url).toString('utf8'), status = sourceBytes(base.status_source.url).toString('utf8');
const ptr = (m, i, extra) => { const r = {...m.index_source, ...extra}; m.claims[i].review = r; return m; };
const variants = {
  'control: claim 0 committed pointer': m => m,
  'pointer_only:true, review_url:"", notice removed (guard applies)': m => ptr(m, 0, {pointer_only: true, review_url: ''}),
  'pointer_only:1, review_url:"", notice removed': m => ptr(m, 0, {pointer_only: 1, review_url: ''}),
  'pointer_only:"yes", review_url:"D2", notice removed': m => ptr(m, 0, {pointer_only: 'yes', review_url: 'D2'}),
  'claim 5 (byte-frozen review file) replaced by pointer_only:1 to the index itself': m => ptr(m, 5, {pointer_only: 1, review_url: ''}),
  'claim 13 sard-g (byte-frozen review file) replaced by pointer_only:1': m => ptr(m, 13, {pointer_only: 1, review_url: ''}),
};
for (const [name, mutate] of Object.entries(variants)) {
  const m = mutate(structuredClone(base));
  let v; try { museum.validateBoundManifest(m, index, status); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
  const d = await render(m);
  console.log(`== ${name}\n   validateBoundManifest: ${v}\n   museum-state: ${d.getElementById('museum-state').textContent}`);
  for (const id of ['d2-lifetime-remainder', 'd5-two-scale', 'sard-g-a1-a6-open']) {
    const c = card(d, id); if (!c) { console.log(`   ${id}: refused/no card`); continue; }
    const t = c.textContent;
    console.log(`   ${id}: review label=${t.includes('Review pointer in the verified index') ? '"Review pointer in the verified index"' : t.includes('Byte-verified review') ? '"Byte-verified review"' : '(none)'}; ${JSON.stringify(links(c).find(l => l.startsWith('Open review')))}; "not byte-frozen" notice shown=${t.includes('not byte-frozen')}`);
  }
}
