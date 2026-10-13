// Render a CI-green sim/<name>/docs/site/museum.json through the cd32303 display code and print the review block.
import fs from 'node:fs';
import {render, reviewBlock, validate} from './lib.mjs';
const [name, id] = process.argv.slice(2);
const m = JSON.parse(fs.readFileSync(new URL(`./sim/${name}/docs/site/museum.json`, import.meta.url)));
console.log(`[${name}] validateBoundManifest:`, validate(m));
const r = await render(m);
const c = r.cards[id];
console.log(`[${name}] ${id}:`, JSON.stringify(reviewBlock(c)));
const notice = c.paras.find(p => /byte-frozen|Byte-frozen|sealed/.test(p)); if (notice) console.log(`[${name}] notice shown:`, notice);
console.log(`[${name}] museum-state:`, r.state);
