// Verifier probe for Q1-4, sub-claim "exhibits.remote := another claim's proof passes
// validateBoundManifest; only geometry.mjs refuses it later, after the card has been appended".
// Checks the END STATE of the page and whether the excerpt is ever shown. No network.
import {shipped, tryValidate, render, findCard, texts} from './common.mjs';
const s = shipped();
s.exhibits.remote = {...s.claims[4].proof, id: 'remote', class: 'illustration', notice: s.canvas_notice};
s.exhibit_quotes.remote = s.exhibit_quotes.annulus; // any excerpt that occurs in the substituted bytes? (annulus excerpt is from a different file)
console.log('validateBoundManifest:', tryValidate(s));
// Use an excerpt that does occur in the substituted bytes so the quote check passes.
import fs from 'node:fs';
import {HERE} from './common.mjs';
const hw = fs.readFileSync(`${HERE}/fixture/Math-/${s.claims[4].proof.path}`, 'utf8');
s.exhibit_quotes.remote = hw.split('\n').find(l => l.trim().length > 40);
let appended = false;
const {document} = await render(s, {search: '?view=remote', withGeometry: true});
const host = document.getElementById('active-exhibit');
console.log('final active-exhibit card present:', !!findCard(host, 'D4 fixed-remote RN'), '| final text:', JSON.stringify(texts(host)));
// Commit-only substitution on remote and p15 (same as the annulus case in the finding).
for (const [kind, title, start] of [['remote', 'D4 fixed-remote RN', '## 1. The count and the new scope'], ['p15', 'D6 P15 full price', '## 5. Sharpness and the exact demand boundary']]) {
  const {makeDescriptor} = await import('./common.mjs');
  const m = shipped(); const p = m.exhibits[kind].path;
  const real = fs.readFileSync(`${HERE}/fixture/Math-/${p}`, 'utf8');
  const alt = Buffer.from(real.replace(start, start + '\n\nSYNTHETIC WIDENED SCOPE LINE.'));
  const commit = '5'.repeat(40);
  m.exhibits[kind] = {...makeDescriptor('d6g8k5htny-coder/Math-', commit, p, alt), id: kind, class: 'illustration', notice: m.canvas_notice};
  m.exhibit_quotes[kind] = start + '\n\nSYNTHETIC WIDENED SCOPE LINE.';
  const v = tryValidate(m);
  const r = await render(m, {search: `?view=${kind}`, overrides: new Map([[m.exhibits[kind].url, alt]]), withGeometry: true});
  const c = findCard(r.document.getElementById('active-exhibit'), title);
  console.log(`${kind}: ${v} | card "${title}" rendered: ${!!c} | widened excerpt shown: ${!!c && texts(c).some(x => x.includes('SYNTHETIC WIDENED'))}`);
}
