// Verifier 2 probe for Q1-8, against the shipped cd32303 museum.json/museum.mjs/geometry.mjs (unmodified).
// No network: common.mjs replaces global fetch with a throwing stub; all bytes are served locally and hash-checked by museum.mjs.
import fs from 'node:fs';
import {shipped, render, findCard, anchors, texts, TREE, HERE} from './common.mjs';
const m = shipped();
const hwIdx = m.claims.findIndex(c => c.id === 'd5-height-window-annulus');
const ahIdx = m.claims.findIndex(c => c.id === 'd5-all-height-annulus');
const hw = m.claims[hwIdx], ah = m.claims[ahIdx], ex = m.exhibits.annulus;
console.log('1. claims: height-window idx', hwIdx, hw.proof.path, hw.proof.sha256.slice(0,12), '| review', hw.review.path);
console.log('   all-height idx', ahIdx, ah.proof.path, ah.proof.sha256.slice(0,12), '| review', ah.review.path);
console.log('   exhibit.annulus', ex.path, ex.sha256.slice(0,12), '| == all-height proof:', ex.sha256 === ah.proof.sha256, '| == height-window proof:', ex.sha256 === hw.proof.sha256);
// 2. Render the full museum (no view) and read the height-window card.
const base = await render(m);
console.log('2. museum-state:', base.document.getElementById('museum-state').textContent);
const card = findCard(base.document.getElementById('claim-cards'), 'd5-height-window-annulus');
console.log('   height-window card rendered:', !!card, '| links:', JSON.stringify(anchors(card).map(([l,u]) => l + ' -> ' + u.replace(/https:\/\/[^ ]+\/blob\/[0-9a-f]{40}\//, '<blob>/'))));
// 3. Follow the illustration link and render the exhibit with geometry.
const r = await render(m, {search: '?view=annulus', withGeometry: true});
const exNode = r.document.getElementById('active-exhibit');
const t = exNode.textContent;
console.log('3. exhibit state:', r.document.getElementById('museum-state').textContent, '| fetch misses:', r.log.length);
console.log('   exhibit header texts:', JSON.stringify(texts(exNode).slice(0, 3)));
const ids = []; (function w(n){ if(!n||typeof n!=='object')return; if(n.tagName==='DD')ids.push(n._text); (n.children||[]).forEach(w);})(exNode);
console.log('   exhibit identity dd:', JSON.stringify(ids));
for (const needle of ['all heights', 'between-pin', 'height window', 'height-window', 'FIXED_ANNULUS_CANDIDATE', 'Pin neighborhoods', 'remain OPEN', 'distinct scopes', 'AMEND'])
  console.log(`   exhibit text contains ${JSON.stringify(needle)}:`, t.includes(needle));
console.log('   boundary line:', JSON.stringify(texts(exNode).find(s => s.startsWith('FIXED ANNULUS CANDIDATE'))));
// 4. What the two pinned proof sources say about height scope.
const hwText = fs.readFileSync(`${HERE}/fixture/Math-/${hw.proof.path}`, 'utf8');
const ahText = fs.readFileSync(`${HERE}/fixture/Math-/${ah.proof.path}`, 'utf8');
console.log('4. height-window source says:', JSON.stringify(hwText.match(/Unlike the earlier thin-tube result, \(A2\) is NOT an all-height assertion\./)?.[0]), '| bound:', JSON.stringify(hwText.match(/E_Qr\^W N_j\(r E0\) <= C\*k\*r\^3\*area\(E0\)/)?.[0]));
console.log('   all-height source says:', JSON.stringify(ahText.match(/The between-pin-height count is a subset and satisfies the same bound\./)?.[0]));
// 5. Does any guard bind the annulus exhibit to the claim that links to it?
const mj = fs.readFileSync(`${TREE}/docs/site/museum.mjs`, 'utf8'), ck = fs.readFileSync(`${TREE}/tools/museum_check.py`, 'utf8');
console.log('5. museum.mjs compares exhibits.annulus to a claim proof:', /exhibits\.annulus|exhibits\[['"]annulus/.test(mj), '| lifetime binding present:', mj.includes("sameIdentity(manifest.exhibits.lifetime,manifest.claims[0].proof)"));
console.log('   museum_check.py mentions scope_boundary:', ck.includes('scope_boundary'), '| mentions VIEW_LINKS/height-window:', /VIEW_LINKS|height-window/.test(ck));
console.log('   museum.mjs references scope_boundary:', mj.includes('scope_boundary'));
