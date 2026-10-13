// Q1/P1 (frontend half): feed the builder's accepted review:=proof claims (p1_builder_claims.json)
// into the shipped manifest; validateBoundManifest accepts, and the card renders the author's
// proof file under "Byte-verified review" / "Open review".
import fs from 'node:fs';
import {shipped, tryValidate, render, findCard, anchors, texts, HERE} from './common.mjs';
const built = JSON.parse(fs.readFileSync(`${HERE}/p1_builder_claims.json`, 'utf8'));
for (const [name, claim] of Object.entries(built)) {
  const m = shipped(); const i = m.claims.findIndex(c => c.id === claim.id);
  m.claims[i] = claim;
  console.log(`\n[${name}]\n  ${tryValidate(m)}`);
  const {document} = await render(m);
  const card = findCard(document.getElementById('claim-cards'), claim.id);
  if (!card) { const host = document.getElementById('claim-cards').children[i]; console.log('  card refused at render:', host.textContent); continue; }
  const t = texts(card);
  console.log('  card says:', JSON.stringify(t.filter(x => /review|Review/.test(x)).slice(0, 3)));
  console.log('  links:', JSON.stringify(anchors(card).filter(([label]) => /review|proof/i.test(label))));
}
// D2 (claim 0) review:=proof: render is blocked only by the missing README replay bytes; show the
// review half of verifyClaim succeeds on the real proof bytes.
const m = shipped(); const d2 = built['B: d2 review := its own proof'];
const {museum, makeFetcher} = await import('./common.mjs');
const noReplay = structuredClone(d2); noReplay.replay = {command: null, url: null, notice: 'x'};
await museum.verifyClaim(noReplay, makeFetcher(m)).then(() => console.log('\n[B] verifyClaim (review bytes = D2 proof bytes) passed; renderClaim would print "Byte-verified review" ->', d2.review.html_url));
