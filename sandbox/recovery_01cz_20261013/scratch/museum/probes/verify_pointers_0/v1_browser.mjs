// Verifier 1 for Q2-F1. Independent harness (does not reuse probes/pointers/lib.mjs).
// Imports the cd32303 docs/site/museum.mjs read-only, serves only the independently byte-verified
// fixture, and disables the network. Prints, per case: validateDescriptor, validateBoundManifest,
// the rendered card's review label and "Open review" href, the identity rows shown, and museum-state.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const CD = path.resolve(HERE, '../../cd32303');
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const FIX = JSON.parse(fs.readFileSync(path.join(HERE, 'fixture/verified-inputs.json'), 'utf8'));
const bytes = u => Buffer.from(FIX[u], 'base64');
const committed = () => JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));

class El { constructor(t){this.tag=t.toUpperCase();this.kids=[];this.attrs={};this.className='';this.t='';this.href=undefined;}
  set textContent(v){this.t=String(v);this.kids=[];} get textContent(){return this.t+this.kids.map(k=>k.textContent).join(' ');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this.t='';this.kids=k;} setAttribute(k,v){this.attrs[k]=String(v);} addEventListener(){} }
function doc(){ const html=fs.readFileSync(path.join(CD,'docs/site/museum.html'),'utf8');
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m=>[m[2],new El(m[1])]));
  return {createElement:t=>new El(t),getElementById:id=>nodes.get(id)??null}; }
const all = (n, o=[]) => { if (n?.tag) { o.push(n); n.kids.forEach(k => all(k, o)); } return o; };
const tryv = f => { try { f(); return 'PASS'; } catch (e) { return 'REFUSED: ' + e.message; } };

async function run(label, mutate, cardId) {
  const m = committed(); mutate(m);
  const idx = bytes(m.index_source.url).toString('utf8'), st = bytes(m.status_source.url).toString('utf8');
  const dv = tryv(() => museum.validateDescriptor(m.claims.find(c => c.id === cardId).review));
  const bv = tryv(() => museum.validateBoundManifest(m, idx, st));
  const requested = [];
  const fetcher = async u => { const k = String(u); requested.push(k);
    if (k === 'museum.json') return new Response(JSON.stringify(m));
    if (Object.hasOwn(FIX, k)) return new Response(bytes(k));
    return new Response('absent', {status: 404}); };
  const d = doc();
  await museum.startMuseum({document: d, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const art = d.getElementById('claim-cards').kids.map(h => h.kids[0]).find(a => a?.attrs?.['data-object-id'] === cardId);
  const nodes = art ? all(art) : [];
  const ps = nodes.filter(n => n.tag === 'P').map(n => n.textContent);
  const lbl = ps.find(p => /Byte-verified review|Review pointer in the verified index/.test(p)) ?? null;
  const a = nodes.find(n => n.tag === 'A' && n.textContent === 'Open review ↗');
  const dd = nodes.filter(n => n.tag === 'DD').map(n => n.textContent);
  console.log(`--- ${label} [${cardId}]`);
  console.log(`  validateDescriptor(review)=${dv} | validateBoundManifest=${bv}`);
  console.log(`  rendered: class=${art?.attrs?.['data-object-class'] ?? 'REFUSED/absent'} | label=${JSON.stringify(lbl)} | Open review href=${a ? a.href : null}`);
  console.log(`  review identity rows shown: ${dd.filter(v => /REVIEW|reviews\//.test(v)).join(' ; ')}`);
  console.log(`  fetched the link target? ${requested.includes(a?.href)} | museum-state=${JSON.stringify(d.getElementById('museum-state').textContent)}`);
}

const D6_REVIEW_PINNED = () => committed().claims[10].review.html_url;   // a different claim's pinned, immutable review
await run('baseline (committed museum.json, no mutation)', () => {}, 'd5-all-height-annulus');
await run('mutable issue comment on a non-pointer review', m => { m.claims[3].review.review_url = 'https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172'; }, 'd5-all-height-annulus');
await run('branch-relative blob/main URL', m => { m.claims[3].review.review_url = 'https://github.com/d6g8k5htny-coder/Math-/blob/main/reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md'; }, 'd5-all-height-annulus');
await run('foreign host', m => { m.claims[3].review.review_url = 'https://example.invalid/forged-review'; }, 'd5-all-height-annulus');
await run('cross-claim: D5 card links to the pinned D6 P15 review', m => { m.claims[3].review.review_url = D6_REVIEW_PINNED(); }, 'd5-all-height-annulus');
await run('AMEND/open card (sard-g) links to an ACCEPT review', m => { m.claims[13].review.review_url = D6_REVIEW_PINNED(); }, 'sard-g-a1-a6-open');
await run('explicit pointer_only:false, foreign host', m => { Object.assign(m.claims[3].review, {pointer_only: false, review_url: 'https://example.invalid/forged-review'}); }, 'd5-all-height-annulus');
// Controls: the regex and quote binding do exist, but only when pointer_only === true.
await run('CONTROL pointer_only:true + foreign host', m => { Object.assign(m.claims[3].review, {pointer_only: true, review_url: 'https://example.invalid/forged-review', review_notice: 'x'}); }, 'd5-all-height-annulus');
await run('CONTROL non-pointer review path swapped to another claim\'s review', m => { m.claims[3].review = structuredClone(committed().claims[10].review); }, 'd5-all-height-annulus');
