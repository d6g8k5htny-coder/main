// Why the d6 card is not rendered offline: README.md@d6628da (its replay source) is absent from the fixture.
import fs from 'node:fs';
import {render, texts, walk, HERE} from './common.mjs';
const m = JSON.parse(fs.readFileSync(`${HERE}/out/manifest_10.json`, 'utf8')); const c = m.claims[10];
const {document, log} = await render(m, {overrides: new Map([[c.review.url, fs.readFileSync(`${HERE}/out/alt_10.bin`)]])});
const refused = []; walk(document.getElementById('claim-cards'), n => { if (n.className === 'museum-card refused') refused.push(texts(n).join(' / ')); });
console.log('refused cards:', JSON.stringify(refused)); console.log('fetch misses:', JSON.stringify(log));
