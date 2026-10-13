// Verifier 1 for Q2-F5. Independent harness: imports cd32303 museum.mjs read-only, stubs ALL network,
// serves only the sha256-pinned fixture bytes (verifiedBytes rejects any byte that does not match the manifest digest).
import fs from 'node:fs'; import path from 'node:path'; import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const CD = path.resolve(HERE, '../../cd32303');
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const fx = JSON.parse(fs.readFileSync(path.resolve(HERE, '../pointers/fixture/verified-inputs.json'), 'utf8'));
const manifestText = fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8');
const manifest = JSON.parse(manifestText);
const html = fs.readFileSync(path.join(CD, 'docs/site/museum.html'), 'utf8');
const mjs = fs.readFileSync(path.join(CD, 'docs/site/museum.mjs'), 'utf8');
class El { constructor(t){this.tag=t;this.kids=[];this.t='';this.attrs={};this.className='';}
  set textContent(v){this.t=String(v);this.kids=[];} get textContent(){return [this.t,...this.kids.map(k=>k.textContent)].filter(Boolean).join(' ');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this.t='';this.kids=k;} setAttribute(k,v){this.attrs[k]=v;} addEventListener(){} }
async function run(extra={}) {
  const req = [];
  const nodes = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new El('x')]));
  const document = {createElement: t => new El(t), getElementById: id => nodes.get(id) ?? null};
  const fetcher = async url => { const k = String(url); req.push(k);
    if (k === 'museum.json') return new Response(manifestText);
    if (Object.hasOwn(extra, k)) return new Response(extra[k]);
    if (Object.hasOwn(fx, k)) return new Response(Buffer.from(fx[k], 'base64'));
    return new Response('absent', {status: 404}); };
  await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const cards = {}; for (const host of nodes.get('claim-cards').kids) { const a = host.kids[0]; cards[a.attrs['data-object-id'] ?? 'REFUSED'] = a.textContent; }
  return {req, cards, state: nodes.get('museum-state').textContent, page: [...nodes.values()].map(n => n.textContent).join('\n')};
}
const base = await run();
console.log('A. cards rendered:', Object.keys(base.cards).length, '| refused:', Object.keys(base.cards).filter(k => k === 'REFUSED').length);
console.log('   museum-state:', base.state);
const STRIP = 'Review wording is quoted at its declared scope.';
console.log('B. strip sentence present on cards:');
for (const c of manifest.claims) {
  const kind = !c.review ? 'no-review' : c.review.pointer_only ? 'POINTER' : 'byte-verified-file';
  console.log(`   ${c.id.padEnd(32)} ${kind.padEnd(19)} strip=${base.cards[c.id].includes(STRIP)} review_url=${c.review?.review_url ?? '-'}`);
}
// C. what does the "review" fetch in verifyClaim fetch for pointer cards? Use an uncached recording fetcher.
console.log('C. verifyClaim review-step URL for pointer cards:');
for (const c of manifest.claims.filter(c => c.review?.pointer_only)) {
  const urls = []; const f = async u => { urls.push(String(u)); return new Response(Buffer.from(fx[String(u)], 'base64')); };
  await museum.verifyClaim(c, f);
  console.log(`   ${c.id}: review-step fetched ${urls[1]} ; equals index/status pointer source url: ${urls[1] === manifest.index_source.url || urls[1] === manifest.status_source.url}`);
}
// D. Date qualifier from pinned PROOF_INDEX
const idx = Buffer.from(fx[manifest.index_source.url], 'base64').toString('utf8');
const q = 'checked on 2026-09-26';
console.log('D. pinned PROOF_INDEX contains qualifier:', idx.includes('Review descriptions below report the linked records, ' + q));
console.log('   qualifier in museum.json:', manifestText.includes(q), '| museum.mjs:', mjs.includes(q), '| museum.html:', html.includes(q), '| rendered page:', base.page.includes(q));
// E. Simulated withdrawal served at every review_url
const edit = 'EDITED: ACCEPT WITHDRAWN';
const extra = Object.fromEntries(manifest.claims.filter(c => c.review?.pointer_only).map(c => [c.review.review_url, edit]));
const edited = await run(extra);
console.log('E. github.com requests:', base.req.filter(u => u.startsWith('https://github.com/')).length, '/', edited.req.filter(u => u.startsWith('https://github.com/')).length,
  '| page identical after edit:', base.page === edited.page && JSON.stringify(base.cards) === JSON.stringify(edited.cards));
// F. Is each pointer review_url actually mutable?
console.log('F. pointer mutability:');
for (const c of manifest.claims.filter(c => c.review?.pointer_only)) {
  const u = c.review.review_url;
  const kind = /\/blob\/[0-9a-f]{40}\//.test(u) ? 'commit-pinned blob (content-addressed; not a mutable comment)' : /#issuecomment-\d+$/.test(u) ? 'mutable issue comment' : 'mutable whole issue thread';
  console.log(`   ${c.id}: ${kind}`);
}
// G. Per-card mitigation shown on pointer cards
for (const c of manifest.claims.filter(c => c.review?.pointer_only)) {
  const t = base.cards[c.id];
  console.log(`G. ${c.id}: label 'Review pointer in the verified index'=${t.includes('Review pointer in the verified index')} notice 'not byte-frozen'=${t.includes('not byte-frozen')} summary 'Identity of the pointer source'=${t.includes('Identity of the pointer source')}`);
}
// H. Does any byte-verified-file card quote review wording? (e.g. scope_quote contains only a link)
for (const id of ['d5-height-window-annulus','d5-two-scale','d5-fixed-transverse']) {
  const c = manifest.claims.find(x => x.id === id); console.log(`H. ${id} scope_quote: ${c.scope_quote}`);
}
