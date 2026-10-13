// Verifier-2 independent probe for Q2-F1 (no reuse of the finder's lib.mjs).
// Imports the cd32303 museum.mjs read-only, network stubbed; serves only fixture bytes already
// checked against cd32303 museum_data.SOURCES pins (size, git blob, sha256).
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const CD = '<SCRATCH>/museum/cd32303';
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(CD + '/docs/site/museum.mjs'));
const fx = JSON.parse(fs.readFileSync(new URL('./fixture/verified-inputs.json', import.meta.url), 'utf8'));
const realBytes = fs.readFileSync(CD + '/docs/site/museum.json');
console.log('museum.json sha256 (cd32303):', createHash('sha256').update(realBytes).digest('hex'));
const base = () => JSON.parse(realBytes);
const txt = u => Buffer.from(fx[u], 'base64').toString('utf8');
const m0 = base();
const index = txt(m0.index_source.url), status = txt(m0.status_source.url);
const nonPointer = m0.claims.map((c, i) => [i, c]).filter(([, c]) => c.review && !c.review.pointer_only);
console.log('non-pointer review claims at cd32303:', nonPointer.map(([i, c]) => `${i}:${c.id}:${c.class}`).join(', '));
console.log('any committed non-pointer review with review_url:', nonPointer.some(([, c]) => 'review_url' in c.review));

// minimal DOM stub
class El { constructor(t){this.tag=t;this.kids=[];this.text='';this.attrs={};this.className='';}
  set textContent(v){this.text=String(v);this.kids=[];} get textContent(){return this.text+this.kids.map(k=>k.textContent).join(' ');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this.text='';this.kids=k;} setAttribute(k,v){this.attrs[k]=v;} }
const walk = (n, o=[]) => { o.push(n); n.kids.forEach(k => walk(k, o)); return o; };
async function render(manifest) {
  const nodes = Object.fromEntries(['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(id => [id, new El('div')]));
  const doc = {createElement: t => new El(t), getElementById: id => nodes[id] ?? null};
  const requested = [];
  const fetcher = async url => { const k = String(url); requested.push(k);
    if (k === 'museum.json') return new Response(JSON.stringify(manifest));
    if (Object.hasOwn(fx, k)) return new Response(Buffer.from(fx[k], 'base64'));
    return new Response('no', {status: 404}); };
  await museum.startMuseum({document: doc, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  return {nodes, requested};
}
function reviewOf(nodes, id) {
  const art = nodes['claim-cards'].kids.map(h => h.kids[0]).find(a => a?.attrs?.['data-object-id'] === id);
  if (!art) return 'CARD NOT RENDERED';
  const all = walk(art);
  const label = all.find(n => n.tag === 'p' && /Byte-verified review|Review pointer in the verified index/.test(n.text))?.text;
  const link = all.find(n => n.tag === 'a' && n.text === 'Open review ↗')?.href;
  const shownSha = all.filter(n => n.tag === 'dd').map(n => n.text).filter(t => /^[0-9a-f]{64}$/.test(t));
  return {class: art.attrs['data-object-class'], label, open_review_href: link, review_identity_sha_shown: shownSha.includes(m0.claims.find(c=>c.id===id).review.sha256)};
}
const tryV = f => { try { f(); return 'PASS'; } catch (e) { return 'REFUSED: ' + e.message; } };

// baseline
{ const {nodes} = await render(base()); console.log('baseline state:', nodes['museum-state'].textContent);
  for (const [, c] of nonPointer) console.log('  baseline', c.id, JSON.stringify(reviewOf(nodes, c.id))); }

// cross-claim review substitution: d5-all-height-annulus "Open review" -> d6-p15 review's own pinned html_url
const other = m0.claims.find(c => c.id === 'd6-p15-full-price').review.html_url;
const urls = {
  'cross-claim pinned review (d6-p15)': other,
  'mutable issue comment': 'https://github.com/d6g8k5htny-coder/main/issues/151#issuecomment-1',
  'branch-relative blob/main': 'https://github.com/d6g8k5htny-coder/Math-/blob/main/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md',
  'foreign host': 'https://example.invalid/forged-review',
};
for (const [i, c] of nonPointer) {
  for (const [name, url] of Object.entries(urls)) {
    if (url === c.review.html_url) continue;
    const m = base(); m.claims[i].review.review_url = url;
    const v = tryV(() => museum.validateBoundManifest(m, index, status));
    const {nodes, requested} = await render(m);
    console.log(`[${c.id}] ${name}: validateBoundManifest=${v} | ${JSON.stringify(reviewOf(nodes, c.id))} | fetched_review_url=${requested.includes(url)} | state="${nodes['museum-state'].textContent.slice(0, 60)}"`);
  }
}
// controls: (a) pointer_only:true with foreign url refused; (b) tampering the html_url (the validated field) refused
{ const m = base(); Object.assign(m.claims[3].review, {pointer_only: true, review_url: urls['foreign host'], review_notice: 'x'});
  console.log('control a (pointer_only:true, foreign):', tryV(() => museum.validateBoundManifest(m, index, status))); }
{ const m = base(); m.claims[3].review.html_url = urls['foreign host'];
  console.log('control b (html_url foreign, no review_url):', tryV(() => museum.validateBoundManifest(m, index, status))); }
{ const m = base(); m.claims[3].review.path = 'reviews/p15_full_price_nonauthor_20260926/REVIEW.md';
  console.log('control c (non-pointer review path swapped to another claim):', tryV(() => museum.validateBoundManifest(m, index, status))); }
