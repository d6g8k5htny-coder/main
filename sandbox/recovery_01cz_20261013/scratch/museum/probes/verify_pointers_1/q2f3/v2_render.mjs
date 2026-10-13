// Q2-F3 verifier 2: render a sim/<variant>/docs/site/museum.json through the UNMODIFIED cd32303 museum.mjs
// (imported read-only from the extracted tree) with a minimal DOM stub; network disabled; the fetcher
// serves only the fixture bytes re-verified by v1_ci.py. Prints, per pointer card, the source-column
// text in display order and whether the SHA-256 shown in the details block is the pointer SOURCE's digest.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const CD = new URL('../../../cd32303/', import.meta.url);
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(new URL('docs/site/museum.mjs', CD));
const fx = JSON.parse(fs.readFileSync(new URL('../fixture/verified-inputs.json', import.meta.url), 'utf8'));
const variant = process.argv[2] || 'control', only = process.argv[3];
const raw = fs.readFileSync(new URL(`./sim/${variant}/docs/site/museum.json`, import.meta.url), 'utf8');
const manifest = JSON.parse(raw);
class El { constructor(t){this.tag=t;this.kids=[];this.text='';this.attrs={};this.className='';}
  set textContent(v){this.text=String(v);this.kids=[];} get textContent(){return this.text+this.kids.map(k=>k.textContent).join('');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this.text='';this.kids=k;} setAttribute(k,v){this.attrs[k]=v;} }
const nodes = Object.fromEntries(['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(id => [id, new El('div')]));
const doc = {createElement: t => new El(t), getElementById: id => nodes[id] ?? null};
const fetcher = async url => { const k = String(url);
  if (k === 'museum.json') return new Response(raw);
  if (Object.hasOwn(fx, k)) return new Response(Buffer.from(fx[k], 'base64'));
  throw Error('unexpected request ' + k); };
const idx = fs.readFileSync(new URL('../fixture/Math-/PROOF_INDEX.md', import.meta.url), 'utf8');
const st = fs.readFileSync(new URL('../fixture/main/STATUS.md', import.meta.url), 'utf8');
let v; try { museum.validateBoundManifest(structuredClone(manifest), idx, st); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
console.log(`[${variant}] validateBoundManifest: ${v}`);
await museum.startMuseum({document: doc, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
const walk = (n, o=[]) => { o.push(n); n.kids.forEach(k => walk(k, o)); return o; };
for (const host of nodes['claim-cards'].kids) {
  const art = host.kids[0], id = art?.attrs?.['data-object-id'];
  const claim = manifest.claims.find(c => c.id === id);
  if (!claim?.review?.pointer_only || (only && id !== only)) continue;
  const col = art.kids[1].kids[1];                       // claim-columns > [scope, source, replay]
  const seq = col.kids.map(n => n.tag === 'a' ? `<a ${n.text} -> ${n.href}>` : n.tag === 'details' ? `<details ${n.kids[0].text}: sha256=${walk(n).filter(x=>x.tag==='dd').map(x=>x.text).find(t=>/^[0-9a-f]{64}$/.test(t))}>` : n.text).filter(Boolean);
  const shown = walk(col).filter(x=>x.tag==='dd').map(x=>x.text).find(t=>/^[0-9a-f]{64}$/.test(t));
  const pointerSrc = id === 'd1-parent-selection-open' ? manifest.status_source : manifest.index_source;
  console.log(`[${variant}] ${id} (${art.attrs['data-object-class']}):`);
  for (const s of seq) console.log('    ' + s);
  console.log(`    => digest shown below the notice is the pointer SOURCE (${pointerSrc.path}) digest: ${shown === pointerSrc.sha256}; review target is a GitHub issue/comment: ${/\/issues\/\d+/.test(claim.review.review_url)}`);
}
console.log(`[${variant}] museum-state: ${nodes['museum-state'].textContent}`);
