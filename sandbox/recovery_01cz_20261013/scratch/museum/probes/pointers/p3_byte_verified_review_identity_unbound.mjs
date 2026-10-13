// F4: a non-pointer review is bound only by sourceContainsPath(quote, review): quote contains `](path)`, the html_url,
// or ANY `/<path>)` suffix. Repository and commit of the review descriptor are never compared with the pointer
// source (the pinned index / STATUS snapshot), unlike claim.proof (which must equal index_source.repository/commit).
import {realManifest, validate, render, reviewBlock} from './lib.mjs';
const rebind = (d, repository, commit, path = d.path) => {
  const p = path.split('/').map(encodeURIComponent).join('/');
  return {...d, repository, commit, path, url: `https://raw.githubusercontent.com/${repository}/${commit}/${p}`, html_url: `https://github.com/${repository}/blob/${commit}/${p}`};
};
const show = async (label, m, id, extra = {}) => {
  const v = validate(m); let out = `${label} -> validateBoundManifest: ${v}`;
  if (v === 'PASS') { const r = await render(m, extra); const c = r.cards[id];
    out += c ? ` | card ${id}: ${JSON.stringify(reviewBlock(c))}` : ' | card refused at fetch';
    out += ` | state: ${r.state.slice(0, 60)}`; }
  console.log(out);
};
// (a) The AMEND d5-pin review is a pointer whose identity is PROOF_INDEX@d6628da. Drop pointer_only: the STATUS row's
//     MUTABLE link https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md) satisfies the `/${path})` clause,
//     so the guard accepts PROOF_INDEX@d6628da as a "Byte-verified review" bound by a branch-relative URL.
{ const m = realManifest(); const r = m.claims[12].review; delete r.pointer_only; delete r.review_url; delete r.review_notice;
  await show('(a) d5-pin: pointer_only removed', m, 'd5-pin-neighborhoods-open'); }
// (b) Same review path, different commit or repository: guard and fetch-verification both pass if the bytes at that
//     identity hash to the descriptor's own sha256. Simulated with the genuine bytes served at the foreign identity.
for (const [repo, commit] of [['d6g8k5htny-coder/Math-', 'bebb77ae9fe3b22c425a979b780fc99a72c8e761'], ['d6g8k5htny-coder/query-', 'f'.repeat(40)]]) {
  const m = realManifest(); const orig = m.claims[3].review; const moved = rebind(orig, repo, commit);
  m.claims[3].review = moved;
  const bytes = Buffer.from((await import('./lib.mjs')).fixtures[orig.url], 'base64');
  await show(`(b) d5-all-height review moved to ${repo}@${commit.slice(0, 8)} (index pins Math-@d6628da)`, m, 'd5-all-height-annulus', {[moved.url]: bytes});
}
// (c) Suffix clause: a review file named just "REVIEW.md" (any repo root, any commit) is "contained" in every quote that
//     links some .../REVIEW.md.
{ const m = realManifest(); const moved = rebind(m.claims[3].review, 'd6g8k5htny-coder/query-', 'e'.repeat(40), 'REVIEW.md');
  const forged = 'Forged review text: R1-R7 ACCEPT globally.\n';
  m.claims[3].review = {...moved, bytes: Buffer.byteLength(forged), sha256: (await import('node:crypto')).createHash('sha256').update(forged).digest('hex')};
  await show('(c) d5-all-height review = query-@eeee../REVIEW.md (forged bytes, self-consistent digest)', m, 'd5-all-height-annulus', {[moved.url]: forged}); }
// (d) The claim's own proof file is in its quote, so it can fill the review slot of a claim whose real review is a
//     mutable comment pointer: D2's comment pointer becomes a "Byte-verified review" of the proof itself.
{ const m = realManifest(); m.claims[0].review = structuredClone(m.claims[0].proof);
  await show('(d) d2 review := d2 proof descriptor', m, 'd2-lifetime-remainder'); }
// control: a path not present in the quote is refused
{ const m = realManifest(); m.claims[3].review = structuredClone(m.claims[4].review);
  await show('(control) d5-all-height review := d5-height-window review (path not in quote)', m, 'd5-all-height-annulus'); }
