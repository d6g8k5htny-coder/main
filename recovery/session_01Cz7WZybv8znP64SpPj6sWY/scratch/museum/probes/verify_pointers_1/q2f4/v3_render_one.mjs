// Render <root>/docs/site/museum.json with <root>/docs/site/museum.mjs (a byte copy of cd32303's), serving only
// <inputs.json>. Network disabled. Prints validateBoundManifest and the review block of four cards.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [ROOT, FIX] = process.argv.slice(2).map(p => path.resolve(p));
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(ROOT, 'docs/site/museum.mjs')));
const fx = JSON.parse(fs.readFileSync(FIX, 'utf8'));
const m = JSON.parse(fs.readFileSync(path.join(ROOT, 'docs/site/museum.json'), 'utf8'));
const t = u => Buffer.from(fx[u], 'base64').toString('utf8');
let v; try { museum.validateBoundManifest(m, t(m.index_source.url), t(m.status_source.url)); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
console.log('validateBoundManifest(regenerated museum.json): ' + v);
class El { constructor(x){this.tag=x.toUpperCase();this.kids=[];this.attrs={};this._t='';this.className='';}
  set textContent(s){this._t=String(s);this.kids=[];} get textContent(){return [this._t,...this.kids.map(k=>k.textContent)].filter(Boolean).join(' ');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this._t='';this.kids=k;} setAttribute(k,s){this.attrs[k]=String(s);} addEventListener(){} }
const all = (n, o = []) => { o.push(n); n.kids.forEach(k => all(k, o)); return o; };
const html = fs.readFileSync(path.join(ROOT, 'docs/site/museum.html'), 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(x => [x[2], new El(x[1])]));
await museum.startMuseum({document: {createElement: x => new El(x), getElementById: id => nodes.get(id) ?? null}, search: '',
  geometryLoader: () => { throw Error('no geometry'); },
  fetcher: async u => { u = String(u); if (u === 'museum.json') return new Response(JSON.stringify(m)); return Object.hasOwn(fx, u) ? new Response(Buffer.from(fx[u], 'base64')) : new Response('absent', {status: 404}); }});
for (const host of nodes.get('claim-cards').kids) {
  const a = host.kids[0], id = a?.attrs?.['data-object-id'];
  if (!['d2-lifetime-remainder', 'd5-all-height-annulus', 'd6-p15-full-price', 'd5-pin-neighborhoods-open'].includes(id)) { if (!id) console.log('card REFUSED: ' + host.textContent.slice(0, 120)); continue; }
  const ns = all(a), ps = ns.filter(n => n.tag === 'P').map(n => n._t);
  const label = ps.find(s => /^(Review pointer in the verified index|Byte-verified review|No separate byte-frozen review)/.test(s));
  const link = ns.find(n => n.tag === 'A' && n._t === 'Open review ↗')?.href ?? null;
  console.log(`card ${id}: ${label} -> ${link}`);
}
console.log('museum-state: ' + nodes.get('museum-state').textContent);
