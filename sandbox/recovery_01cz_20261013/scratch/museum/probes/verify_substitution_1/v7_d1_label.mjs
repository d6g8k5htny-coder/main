// Verifier 2 of 2 for Q1-7. Read-only against the cd32303 bytes; network stubbed in common.mjs.
// (1) shipped museum.json: the D1 AMEND card's source column, in order.
// (2) what each pinned source calls the target link; which source the validator binds it to.
// (3) counterfactual: remove the D1 target URL from PROOF_INDEX (and re-pin the index consistently);
//     does the card still say "Review pointer in the verified index"?
import {shipped, render, findCard, anchors, texts, sha256, gitBlob, TREE, indexText, statusText, tryValidate, walk} from './common.mjs';
import fs from 'node:fs';
const m = shipped();
const D1 = 'd1-parent-selection-open', URL63 = 'https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743';
const idx = indexText(), st = statusText();
console.log('pins: index sha', sha256(Buffer.from(idx)) === m.index_source.sha256, '| STATUS sha', sha256(Buffer.from(st)) === m.status_source.sha256);

const {document, log} = await render(m);
const state = document.getElementById('museum-state').textContent;
console.log('museum-state:', JSON.stringify(state), '| 404s:', log.length);
const card = findCard(document.getElementById('claim-cards'), D1);
console.log('D1 card rendered (not refused):', !!card);
const t = texts(card), s = t.indexOf('Source and review');
console.log('D1 source column texts:', JSON.stringify(t.slice(s, t.indexOf('Replay', s))));
console.log('D1 anchors:', JSON.stringify(anchors(card).filter(([l]) => /review/i.test(l))));
const c = m.claims.find(x => x.id === D1);
console.log('review descriptor path/commit:', c.review.path, c.review.commit.slice(0, 8), '| = status_source:', c.review.sha256 === m.status_source.sha256, '| = index_source:', c.review.sha256 === m.index_source.sha256);

const label = (text) => (text.match(new RegExp('\\[([^\\]]+)\\]\\(' + URL63.replace(/[.?#/]/g, '\\$&') + '\\)')) || [])[1];
console.log('STATUS row link label for target :', JSON.stringify(label(st)));
console.log('PROOF_INDEX link label for target:', JSON.stringify(label(idx)), '| PROOF_INDEX link label for the D1 review it reopens:', JSON.stringify((idx.match(/\[([^\]]+)\]\(https:\/\/github\.com\/d6g8k5htny-coder\/main\/issues\/63#issuecomment-5841570965\)/) || [])[1]));

// (3) counterfactual index without the D1 target URL, consistently re-pinned (same commit/path/url, new bytes).
const idx2 = idx.replace(`[main #63 was reopened](${URL63})`, 'main #63 was reopened');
console.log('counterfactual index still contains target URL:', idx2.includes(URL63), '| changed:', idx2 !== idx);
const b2 = Buffer.from(idx2), m2 = structuredClone(m), oldSha = m.index_source.sha256;
const repin = d => { if (d && d.sha256 === oldSha && d.url === m.index_source.url) Object.assign(d, {sha256: sha256(b2), blob: gitBlob(b2), bytes: b2.length}); };
repin(m2.index_source); for (const cl of m2.claims) { repin(cl.proof); repin(cl.review); }
console.log('validateBoundManifest(counterfactual):', tryValidate(m2, idx2, st));
const r2 = await render(m2, {overrides: new Map([[m.index_source.url, b2]])});
const card2 = findCard(r2.document.getElementById('claim-cards'), D1);
console.log('counterfactual D1 card rendered:', !!card2, '| label:', JSON.stringify(texts(card2).filter(x => /Review pointer/.test(x))), '| state:', JSON.stringify(r2.document.getElementById('museum-state').textContent));
