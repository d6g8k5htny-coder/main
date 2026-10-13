// Q1/P7 (shipped bytes, no modification): attribution/label checks on the actual cd32303 museum.json.
import fs from 'node:fs';
import {shipped, render, findCard, anchors, texts, TREE, walk} from './common.mjs';
const m = shipped();
const {document} = await render(m);
// (a) D1 AMEND card: pointer descriptor identity is STATUS.md, but the card says "verified index".
const d1 = findCard(document.getElementById('claim-cards'), 'd1-parent-selection-open'); const t = texts(d1);
const i = t.indexOf('Identity of the pointer source');
console.log('(a) D1 card text:', JSON.stringify(t.filter(x => /Review pointer/.test(x))), '| pointer identity Path:', t[t.indexOf('Path', i) + 1], '| STATUS row calls the target:', JSON.stringify((m.claims[11].scope_quote.match(/\[([^\]]+)\]\(https:\/\/github\.com\/d6g8k5htny-coder\/main\/issues\/63#issuecomment-5841830743\)/) || [])[1]));
// (b) d5-height-window card links to the annulus exhibit, whose pinned source is the D5 all-height proof.
const hw = findCard(document.getElementById('claim-cards'), 'd5-height-window-annulus');
const link = anchors(hw).find(([l]) => /illustration/.test(l));
console.log('(b) d5-height-window proof:', m.claims[4].proof.path, m.claims[4].proof.blob.slice(0, 8), '| illustration link:', link[1], '-> exhibit source', m.exhibits.annulus.path, m.exhibits.annulus.blob.slice(0, 8), '(= d5-all-height proof:', (m.exhibits.annulus.sha256 === m.claims[3].proof.sha256) + ')');
const r = await render(m, {search: '?view=annulus', withGeometry: true});
const ex = r.document.getElementById('active-exhibit');
console.log('    exhibit card title/id:', JSON.stringify(texts(ex).slice(0, 3)));
// (c) the builder's only scope-separation note for the annulus exhibit is never rendered.
console.log('(c) exhibits.annulus.scope_boundary =', JSON.stringify(m.exhibits.annulus.scope_boundary));
console.log('    referenced in museum.mjs:', fs.readFileSync(`${TREE}/docs/site/museum.mjs`, 'utf8').includes('scope_boundary'), '| present in rendered exhibit text:', ex.textContent.includes('distinct scopes'));
