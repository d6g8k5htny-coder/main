// Q1/P4: only exhibits.lifetime is bound to a claim proof (museum.mjs:101). exhibits.remote/annulus/p15
// are linked from the D4/D5/D6 ACCEPT-scoped cards (VIEW_LINKS) and titled after those claims
// (EXHIBITS), but validateBoundManifest only shape-checks them (museum.mjs:100), the quote check
// (199) compares the excerpt with the exhibit's OWN bytes, and geometry.mjs:388 binds repository+path,
// not commit/digest. A different version of the file is displayed as the claim's exhibit.
import fs from 'node:fs';
import {shipped, tryValidate, render, findCard, anchors, texts, makeDescriptor, HERE, walk} from './common.mjs';
const p = 'frontiers/rn_annulus_bridge_20260925/PROOF.md';
const real = fs.readFileSync(`${HERE}/fixture/Math-/${p}`, 'utf8');
// SYNTHETIC stand-in for "the same path at another commit": scope widened to what the review excludes.
const altText = real.replace('and 1<A<B<infinity.', 'and 0<A<B<=infinity (pin neighborhoods and all distances included).').replace('**Candidate theorem.**', '**Theorem (global, all scales).**');
const alt = Buffer.from(altText);
const start = '## 1. Statement and exact scope', end = '## 2.';
const altQuote = start + altText.split(start, 2)[1].split(end, 1)[0];
const m = shipped(); const commit = '4'.repeat(40);
m.exhibits.annulus = {...makeDescriptor('d6g8k5htny-coder/Math-', commit, p, alt), id: 'annulus', class: 'illustration', notice: m.canvas_notice};
m.exhibit_quotes.annulus = altQuote;
console.log('validateBoundManifest:', tryValidate(m));
console.log('D5 all-height card links here:', 'museum.html?view=annulus#active-exhibit', '| D5 proof commit', m.claims[3].proof.commit.slice(0, 8), '| exhibit commit', commit.slice(0, 8));
const {document} = await render(m, {search: '?view=annulus', overrides: new Map([[m.exhibits.annulus.url, alt]]), withGeometry: true});
const host = document.getElementById('active-exhibit');
const card = findCard(host, 'D5 all-height fixed annulus');
console.log('exhibit card rendered:', !!card, '| title/id:', JSON.stringify(texts(card).slice(0, 3)));
const t = texts(card); console.log('identity shown: Commit', t[t.indexOf('Commit') + 1]);
let ill = null; walk(card, n => { if (n.attributes?.['data-illustration']) ill = n.attributes['data-illustration']; });
console.log('geometry mounted for kind:', ill);
const excerpt = t.find(x => x.startsWith(start)) || '';
console.log('"Exact source excerpt" shown contains widened scope:', excerpt.includes('0<A<B<=infinity'), '| contains "global, all scales":', excerpt.includes('global, all scales'));
console.log('state:', document.getElementById('museum-state').textContent);
// Control: the identical change on exhibits.lifetime is refused.
const c = shipped(); c.exhibits.lifetime = {...c.exhibits.lifetime, ...makeDescriptor('d6g8k5htny-coder/Math-', commit, c.exhibits.lifetime.path, alt)};
console.log('control (same change on exhibits.lifetime):', tryValidate(c));
// Path swap is also accepted by validateBoundManifest; only geometry.mjs refuses it later, after the card was appended.
const s = shipped(); s.exhibits.remote = {...s.claims[4].proof, id: 'remote', class: 'illustration', notice: s.canvas_notice};
console.log('exhibits.remote := D5 height-window proof:', tryValidate(s));
