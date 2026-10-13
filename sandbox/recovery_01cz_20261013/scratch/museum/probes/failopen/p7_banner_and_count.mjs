// (a) A refusal in the exhibit area for an undeclared ?view= is not pushed to jobs, so the banner still reports
//     "Source projections and displayed source bytes verified".
// (b) The pinned integration test's `claim-cards.children.length === 14` counts placeholder hosts, which exist
//     whether each card rendered or was refused; it cannot detect a refusal.
import {realManifest, render} from './lib.mjs';
const m = realManifest();
const d = await render(m, {search: '?view=lifetime'});
console.log('(a) ?view=lifetime  museum-state:', d.getElementById('museum-state').textContent);
console.log('    active-exhibit:', d.getElementById('active-exhibit').textContent);
const bad = structuredClone(m); for (const i of [0, 3, 4, 11]) bad.claims[i].proof.sha256 = '0'.repeat(64);
bad.claims[0].proof.sha256 = m.claims[0].proof.sha256; // keep D2 (shared with lifetime) intact
const e = await render(bad);
const cards = e.getElementById('claim-cards');
console.log('(b) 3 proofs with wrong digests  museum-state:', e.getElementById('museum-state').textContent);
console.log('    claim-cards.children.length =', cards.children.length, '(test asserts 14) | refusal cards =', cards.walk().filter(n => n.className === 'museum-card refused').length);
