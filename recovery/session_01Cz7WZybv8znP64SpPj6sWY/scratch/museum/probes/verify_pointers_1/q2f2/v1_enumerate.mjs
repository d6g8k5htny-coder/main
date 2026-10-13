// Q2-F2 verifier 2, browser side. Imports the cd32303 museum.mjs read-only; network stubbed.
// For every pointer_only review (except D5-pin, which has an exact-equality pin), enumerate EVERY string that
// (a) passes validateDescriptor's line-40 pointer regex and (b) is a substring of the exact pinned pointer text,
// then run the real validateBoundManifest on the real museum.json with only review_url replaced.
// Expected of a link-equality binding: PASS iff the candidate is a markdown link target in the pointer text.
import fs from 'node:fs'; import path from 'node:path'; import {fileURLToPath, pathToFileURL} from 'node:url';
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const HERE = path.dirname(fileURLToPath(import.meta.url)); const CD = path.resolve(HERE, '../../../cd32303');
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const fx = JSON.parse(fs.readFileSync(path.join(HERE, 'fixture/verified-inputs.json'), 'utf8'));
const real = () => JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));
const m0 = real(); const txt = s => Buffer.from(fx[s.url], 'base64').toString('utf8');
const index = txt(m0.index_source), status = txt(m0.status_source);
console.log('baseline real museum.json:', (() => { try { museum.validateBoundManifest(m0, index, status); return 'PASS'; } catch (e) { return 'REFUSED ' + e.message; } })());
const RE = /^https:\/\/github\.com\/d6g8k5htny-coder\/(main|Math-)\/issues\/\d+(?:#issuecomment-\d+)?$/; // museum.mjs:40 verbatim
const P = 'https://github.com/d6g8k5htny-coder/';
let totalNonLink = 0, passNonLink = 0, failLink = 0;
for (const [i, claim] of m0.claims.entries()) {
  if (!claim.review?.pointer_only || claim.id === 'd5-pin-neighborhoods-open') continue;
  const text = claim.scope_quote;                     // museum.mjs:84 pointerText for these claims
  const links = new Set([...text.matchAll(/\]\(([^)]+)\)/g)].map(x => x[1]));
  const cands = new Set();
  for (let s = text.indexOf(P); s >= 0; s = text.indexOf(P, s + 1))
    for (let e = s + P.length + 1; e <= text.length; e++) { const c = text.slice(s, e); if (RE.test(c)) cands.add(c); }
  const rows = [];
  for (const c of cands) {
    const m = real(); m.claims[i].review.review_url = c;
    let v; try { museum.validateBoundManifest(m, index, status); v = 'PASS'; } catch (e) { v = 'REFUSED'; }
    const isLink = links.has(c);
    if (!isLink) { totalNonLink++; if (v === 'PASS') passNonLink++; } else if (v !== 'PASS') failLink++;
    rows.push(`${v.padEnd(7)} ${isLink ? 'link-target ' : 'NOT-A-LINK  '} ${c}${c === claim.review.review_url ? '   <= generated value' : ''}`);
  }
  console.log(`\n[${i}] ${claim.id} (${claim.class}); regex-valid substrings of the pinned quote: ${cands.size}`);
  for (const r of rows) console.log('  ' + r);
}
console.log(`\nSUMMARY: ${passNonLink}/${totalNonLink} regex-valid NON-link substrings PASS validateBoundManifest; link targets refused: ${failLink}`);
// Control: a URL that is not a substring is refused (the guard is not simply absent).
const c = real(); c.claims[0].review.review_url = 'https://github.com/d6g8k5htny-coder/main/issues/99';
try { museum.validateBoundManifest(c, index, status); console.log('CONTROL issues/99 on D2: PASS (unexpected)'); } catch (e) { console.log('CONTROL issues/99 on D2: REFUSED', e.message); }
