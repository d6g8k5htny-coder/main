// Verifier 1 for Q1-8, against cd32303 bytes only. Network stubbed in common.mjs.
import fs from 'node:fs';
import {shipped, render, findCard, anchors, texts, walk, TREE, HERE, sha256} from './common.mjs';
const m = shipped();
const all = m.claims.find(c => c.id === 'd5-all-height-annulus'), hw = m.claims.find(c => c.id === 'd5-height-window-annulus');
console.log('[1] claims: all-height', all.class, all.proof.path, '| height-window', hw.class, hw.proof.path);
console.log('    exhibit.annulus sha256 == all-height proof sha256:', m.exhibits.annulus.sha256 === all.proof.sha256, '| == height-window proof sha256:', m.exhibits.annulus.sha256 === hw.proof.sha256);
const {document, log} = await render(m);
const card = findCard(document.getElementById('claim-cards'), 'd5-height-window-annulus');
console.log('[2] height-window card links:', JSON.stringify(anchors(card).map(([t,h]) => t + ' -> ' + h)));
const r = await render(m, {search: '?view=annulus', withGeometry: true});
const ex = r.document.getElementById('active-exhibit');
let label = null, h3 = null; walk(ex, n => { if (n.className === 'object-label' && !label) label = n._text; if (n.tagName === 'H3' && !h3) h3 = n.textContent; });
console.log('[3] exhibit object-label:', JSON.stringify(label), '| h3:', JSON.stringify(h3), '| refusal?', /Unavailable/.test(ex.textContent), '| render log:', JSON.stringify(r.log));
const t = ex.textContent;
for (const s of ['all heights', 'height-window', 'height window', 'FIXED_ANNULUS_CANDIDATE', 'rn_annulus_bridge_20260925/PROOF.md', 'pin neighborhoods', 'Dashed pin neighborhoods', 'distinct scopes', 'between-pin-height count is a subset'])
  console.log('    rendered exhibit contains', JSON.stringify(s) + ':', t.includes(s));
console.log('[4] scope_boundary in museum.mjs:', fs.readFileSync(`${TREE}/docs/site/museum.mjs`,'utf8').includes('scope_boundary'), '| in geometry.mjs:', fs.readFileSync(`${TREE}/docs/site/geometry.mjs`,'utf8').includes('scope_boundary'), '| in museum_check.py:', fs.readFileSync(`${TREE}/tools/museum_check.py`,'utf8').includes('scope_boundary'), '| in tests:', fs.readdirSync(`${TREE}/tests`).filter(f=>/museum/.test(f)).some(f=>fs.readFileSync(`${TREE}/tests/${f}`,'utf8').includes('scope_boundary')));
const hwSrc = fs.readFileSync(`${HERE}/fixture/Math-/${hw.proof.path}`, 'utf8'), allSrc = fs.readFileSync(`${HERE}/fixture/Math-/${all.proof.path}`, 'utf8');
console.log('[5] fixture bytes match pins: hw', sha256(Buffer.from(hwSrc)) === hw.proof.sha256, '| all', sha256(Buffer.from(allSrc)) === all.proof.sha256);
console.log('    height-window source says:', JSON.stringify(hwSrc.match(/\(A2\) is NOT an all-height assertion[^.]*\./)[0]));
console.log('    all-height source says:', JSON.stringify(allSrc.match(/The between-pin-height count is a subset[^.]*\./)[0]));
