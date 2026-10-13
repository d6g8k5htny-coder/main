// Informational: does current main 513261bb's validateBoundManifest bind proof.availability / replay.notice /
// pointer_only type? Validation only (no rendering). Index text from the verified fixture; STATUS text from the
// read-only blob 3aeb0412 (sha256 checked against main's museum.json before use).
import fs from 'node:fs'; import {createHash} from 'node:crypto'; import {pathToFileURL} from 'node:url';
const MAIN = new URL('../../main_513261bb/', import.meta.url);
const museum = await import(new URL('docs/site/museum.mjs', MAIN));
const base = JSON.parse(fs.readFileSync(new URL('docs/site/museum.json', MAIN), 'utf8'));
const fx = JSON.parse(fs.readFileSync(new URL('fixture/verified-inputs.json', import.meta.url), 'utf8'));
const index = Buffer.from(fx[base.index_source.url], 'base64');
const status = fs.readFileSync(new URL('work/main_status_f2e432ea.md', import.meta.url));
for (const [b, s] of [[index, base.index_source], [status, base.status_source]]) if (createHash('sha256').update(b).digest('hex') !== s.sha256) throw Error('pin mismatch ' + s.path);
const run = (name, f) => { const m = structuredClone(base); f(m); let v; try { museum.validateBoundManifest(m, index.toString('utf8'), status.toString('utf8')); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; } console.log(`${name}: ${v}`); };
run('control (main museum.json)', () => {});
run('delete availability on d5-pin-neighborhoods-open and sard-g', m => { for (const c of m.claims) if (['d5-pin-neighborhoods-open', 'sard-g-a1-a6-open'].includes(c.id)) delete c.proof.availability; });
run('AMEND claims replay.notice = "Replay reproduces the closed result."', m => { for (const c of m.claims) if (c.class === 'AMEND/open') c.replay.notice = 'Replay reproduces the closed result.'; });
run('claim 0 review -> pointer_only:1, review_url:"", no notice', m => { m.claims[0].review = {...m.index_source, pointer_only: 1, review_url: ''}; });
