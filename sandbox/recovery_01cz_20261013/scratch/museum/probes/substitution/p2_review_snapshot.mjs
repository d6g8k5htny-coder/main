// Q1/P2 (frontend half): for a byte-verified (non-pointer) review, validateBoundManifest checks only
// that the review PATH occurs in the claim quote (museum.mjs:86, sourceContainsPath). Repository and
// commit are not bound to index_source, unlike the proof (museum.mjs:75). A different version of the
// review file, at another commit or in the query- repository, is displayed as "Byte-verified review".
import fs from 'node:fs';
import {shipped, tryValidate, render, findCard, anchors, texts, makeDescriptor, HERE} from './common.mjs';
const real = fs.readFileSync(`${HERE}/fixture/Math-/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md`);
// SYNTHETIC stand-in for "the same path at another commit": verdict row R7 flipped to AMEND.
const alt = Buffer.from(real.toString('utf8').replace('| R7 crossover, absorption, and full cover (17)–(20), (2) | **ACCEPT** |', '| R7 crossover, absorption, and full cover (17)–(20), (2) | **AMEND** |'));
console.log('synthetic alternate differs from pinned review:', !alt.equals(real), '| contains AMEND row:', alt.includes('| **AMEND** |'));
for (const [label, repo, commit] of [['other Math- commit', 'd6g8k5htny-coder/Math-', '1'.repeat(40)], ['query- repository', 'd6g8k5htny-coder/query-', '2'.repeat(40)]]) {
  const m = shipped();
  const review = makeDescriptor(repo, commit, 'reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md', alt);
  m.claims[3].review = review;
  console.log(`\n[${label}] review ${repo}@${commit.slice(0, 8)}: ${tryValidate(m)}`);
  const {document} = await render(m, {overrides: new Map([[review.url, alt]])});
  const card = findCard(document.getElementById('claim-cards'), 'd5-all-height-annulus');
  const t = texts(card);
  console.log('  card renders:', JSON.stringify(t.filter(x => /verified review|ACCEPT-scoped|Source label/.test(x))));
  console.log('  review link:', JSON.stringify(anchors(card).find(([l]) => /review/i.test(l))));
  console.log('  review identity shown: Repository', t[t.indexOf('Repository', t.indexOf('Review source identity')) + 1], '| Commit', t[t.indexOf('Commit', t.indexOf('Review source identity')) + 1]);
}
const control = shipped(); const p = control.claims[3].proof;
control.claims[3].proof = makeDescriptor(p.repository, '1'.repeat(40), p.path, fs.readFileSync(`${HERE}/fixture/Math-/${p.path}`));
console.log('\n[control: same change applied to the PROOF]', tryValidate(control));
