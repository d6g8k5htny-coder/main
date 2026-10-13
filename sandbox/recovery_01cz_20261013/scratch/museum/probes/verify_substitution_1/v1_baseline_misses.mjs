// Which views are unavailable in this offline harness, for the shipped cd32303 museum.json and for variant B?
import fs from 'node:fs';
import {render, texts, shipped, pinnedBytes, MAIN_PIN} from '../substitution/common.mjs';
const H = '<SCRATCH>/museum/probes/verify_substitution_1';
for (const [label, m, ov] of [['shipped', shipped(), new Map()], ['B', JSON.parse(fs.readFileSync(`${H}/B_row_mentions_inner_microdisk_manifest.json`,'utf8')), null]]) {
  let overrides = ov;
  if (!overrides) { overrides = new Map([[m.status_source.url, fs.readFileSync(`${H}/B_row_mentions_inner_microdisk_STATUS.md`)]]);
    for (const d of [m.claims[13].review, ...m.packets.map(p => p.result), m.claims[1].replay.source]) if (d.repository==='d6g8k5htny-coder/main'&&d.path!=='STATUS.md') overrides.set(d.url, pinnedBytes(d.repository, MAIN_PIN, d.path)); }
  const {document, log} = await render(m, {overrides});
  console.log(label, '| misses:', log.join('; '));
  console.log(label, '| refused:', JSON.stringify(texts(document.getElementById('claim-cards')).filter((x,i,a)=>/^Unavailable/.test(a[i+1]??''))));
}
