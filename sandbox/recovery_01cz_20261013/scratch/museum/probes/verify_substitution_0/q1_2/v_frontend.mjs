// Verifier probe for Q1-2 (frontend half, fed with manifests the cd32303 builder itself produced
// in v_builder_path.py). No network: common.mjs stubs fetch; alt bytes served by URL override.
import fs from 'node:fs';
import {tryValidate, render, findCard, texts, HERE} from './common.mjs';
for (const i of [10, 5, 3]) {
  const f = `${HERE}/out/manifest_${i}.json`; if (!fs.existsSync(f)) { console.log(`[${i}] no manifest`); continue; }
  const m = JSON.parse(fs.readFileSync(f, 'utf8')); const c = m.claims[i]; const alt = fs.readFileSync(`${HERE}/out/alt_${i}.bin`);
  console.log(`\n[${c.id}] review ${c.review.repository}@${c.review.commit.slice(0, 8)} vs index ${m.index_source.commit.slice(0, 8)}: ${tryValidate(m)}`);
  if (!tryValidate(m).startsWith('ACCEPTED')) continue;
  const {document, log} = await render(m, {overrides: new Map([[c.review.url, alt]])});
  const card = findCard(document.getElementById('claim-cards'), c.id);
  if (!card) { console.log('  card refused/not rendered'); continue; }
  const t = texts(card);
  console.log('  card renders:', JSON.stringify(t.filter(x => /verified review|ACCEPT-scoped|engineering-only|Source label|Unavailable/.test(x))));
  const k = t.indexOf('Review source identity');
  console.log('  review identity shown: Commit', t[t.indexOf('Commit', k) + 1], '| SHA-256', t[t.indexOf('SHA-256', k) + 1].slice(0, 16));
  console.log('  fetch misses:', log.length);
}
