// Verifier 2 for Q2-F5. Independent harness: renders the cd32303 museum.mjs against a local fixture
// (each served byte string is re-checked here against museum.json's pinned sha256 before use). No network.
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const CD = path.resolve(HERE, '../../cd32303');
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const fx = JSON.parse(fs.readFileSync(path.resolve(HERE, '../pointers/fixture/verified-inputs.json'), 'utf8'));
const manifest = JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));
// independent pin check of every fixture entry the manifest names
const pins = [manifest.index_source, manifest.status_source, ...manifest.claims.flatMap(c => [c.proof, c.review, c.replay?.source]).filter(Boolean)];
let bad = 0; for (const p of pins) { const b = Buffer.from(fx[p.url] ?? '', 'base64'); if (b.length !== p.bytes || crypto.createHash('sha256').update(b).digest('hex') !== p.sha256) bad++; }
console.log(`fixture pins re-checked against museum.json: ${pins.length} checked, ${bad} mismatched`);
class El { constructor(t){this.tag=t.toUpperCase();this.kids=[];this.attrs={};this._t='';this.className='';}
  set textContent(v){this._t=String(v);this.kids=[];} get textContent(){return [this._t,...this.kids.map(k=>k.textContent)].filter(Boolean).join(' ');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this._t='';this.kids=k;} setAttribute(k,v){this.attrs[k]=String(v);} addEventListener(){} }
const ids = ['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'];
async function render(serveExtra) {
  const nodes = new Map(ids.map(i => [i, new El('div')])); const doc = {createElement: t => new El(t), getElementById: i => nodes.get(i) ?? null};
  const req = [];
  const fetcher = async u => { u = String(u); req.push(u); if (u === 'museum.json') return new Response(JSON.stringify(manifest));
    if (serveExtra && Object.hasOwn(serveExtra, u)) return new Response(serveExtra[u]);
    if (Object.hasOwn(fx, u)) return new Response(Buffer.from(fx[u], 'base64')); return new Response('nf', {status: 404}); };
  await museum.startMuseum({document: doc, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const cards = {}; for (const h of nodes.get('claim-cards').kids) { const a = h.kids[0]; cards[a.attrs['data-object-id'] ?? 'REFUSED'] = a; }
  return {state: nodes.get('museum-state').textContent, cards, req};
}
const walk = (n, o = []) => { o.push(n); for (const k of n.kids) walk(k, o); return o; };
const A = await render();
const B = await render(Object.fromEntries(manifest.claims.filter(c => c.review?.review_url).map(c => [c.review.review_url, 'EDITED: ACCEPT WITHDRAWN'])));
console.log('state:', A.state);
console.log('cards rendered:', Object.keys(A.cards).length, '| any refused:', Object.hasOwn(A.cards, 'REFUSED'));
console.log('requests to github.com (orig/edited):', A.req.filter(u => u.startsWith('https://github.com/')).length, B.req.filter(u => u.startsWith('https://github.com/')).length);
const ser = r => JSON.stringify(Object.fromEntries(Object.entries(r.cards).map(([k, v]) => [k, v.textContent])));
console.log('page identical after simulated edit:', ser(A) === ser(B) && A.state === B.state);
const STRIP = 'Review wording is quoted at its declared scope.';
const withStrip = Object.entries(A.cards).filter(([, a]) => a.textContent.includes(STRIP)).map(([k]) => k);
console.log(`strip sentence on ${withStrip.length}/14 cards (pointer and byte-verified alike)`);
console.log('"checked on 2026-09-26" qualifier in rendered page:', Object.values(A.cards).some(a => a.textContent.includes('checked on')) || A.state.includes('checked on'));
const idx = Buffer.from(fx[manifest.index_source.url], 'base64').toString('utf8');
console.log('qualifier present in pinned PROOF_INDEX:', /Review descriptions below report the linked records, checked on 2026-09-26/.test(idx));
for (const c of manifest.claims) {
  if (!c.review) continue; const r = c.review; const a = A.cards[c.id]; const all = walk(a);
  const paras = all.filter(n => n.tag === 'P').map(n => n._t);
  const href = all.find(n => n.tag === 'A' && n._t === 'Open review ↗')?.href;
  const src = c.id === 'd1-parent-selection-open' ? manifest.status_source : manifest.index_source;
  const same = r.pointer_only ? ['url', 'sha256', 'bytes', 'commit', 'path'].every(k => r[k] === src[k]) : null;
  const verdictInQuote = (c.scope_quote.match(/(record ACCEPT[^.]*|verifies the coefficient calculation|disposition EXACT_COUNTEREXAMPLE|fail-closed reopening|has an AMEND review)/) || [])[0] ?? '-';
  const immut = r.pointer_only ? (/\/blob\/[0-9a-f]{40}\//.test(r.review_url) ? 'commit-pinned blob' : /#issuecomment-/.test(r.review_url) ? 'mutable comment' : 'mutable issue thread') : 'byte-verified file';
  console.log(`- ${c.id} [${c.class}] review=${immut}; review descriptor == pointer source identity: ${same}`);
  console.log(`    label: ${paras.find(p => /Review pointer|Byte-verified review/.test(p))} | notice shown: ${paras.includes(r.review_notice ?? '\u0000')} | Open review -> ${href}`);
  console.log(`    index-authored review wording quoted on card: "${verdictInQuote}"`);
}
