// Verifier 2 of Q1-6, frontend half. Read-only against cd32303 docs/site/museum.mjs; fetch is stubbed
// (common.mjs replaces globalThis.fetch with a thrower; museum.mjs verifies every served byte against its pin).
import fs from 'node:fs';
import {tryValidate, render, findCard, anchors, texts, indexText, pinnedBytes, MAIN_PIN} from '../substitution/common.mjs';
const H = '<SCRATCH>/museum/probes/verify_substitution_1';
for (const name of ['A_finding_row', 'B_row_mentions_inner_microdisk', 'C_swap_D5_SARD']) {
  const m = JSON.parse(fs.readFileSync(`${H}/${name}_manifest.json`, 'utf8'));
  const status = fs.readFileSync(`${H}/${name}_STATUS.md`);
  const v = tryValidate(m, indexText(), status.toString('utf8'));
  console.log(`[${name}] validateBoundManifest: ${v}`);
  if (!v.startsWith('ACCEPTED')) continue;
  const overrides = new Map([[m.status_source.url, status]]);
  for (const d of [m.claims[13].review, ...m.packets.map(p => p.result), m.claims[1].replay.source])
    if (d.repository === 'd6g8k5htny-coder/main' && d.path !== 'STATUS.md') overrides.set(d.url, pinnedBytes(d.repository, MAIN_PIN, d.path));
  const {document, log} = await render(m, {overrides});
  const state = document.getElementById('museum-state').textContent;
  const card = findCard(document.getElementById('claim-cards'), 'd5-pin-neighborhoods-open');
  const t = texts(card ?? {});
  console.log(`    museum-state: ${state}`);
  console.log(`    d5-pin-neighborhoods-open card rendered: ${!!card}; refused cards: ${texts(document.getElementById('claim-cards')).filter(x => /^Unavailable/.test(x)).length}; fetch misses: ${log.length}`);
  console.log(`    h3 title: ${t[2]}`);
  console.log(`    shows: ${JSON.stringify(t.filter(x => /NO COMPLETE PROOF|OPEN \/ NOT LANDED|Review pointer in|Unmerged review/.test(x)))}`);
  console.log(`    review link: ${JSON.stringify(anchors(card).find(([l]) => /review/i.test(l)))}`);
}
