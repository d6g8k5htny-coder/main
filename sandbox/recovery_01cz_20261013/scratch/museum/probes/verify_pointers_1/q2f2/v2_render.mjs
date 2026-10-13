// Q2-F2 verifier 2: full startMuseum() render (cd32303 museum.mjs, read-only) with a minimal DOM stub and a
// fetcher that serves ONLY verified fixture bytes. Prints, per substituted card, the review label, the Open review
// href, the notice, and the SHA-256 shown under "Identity of the pointer source", plus the page state line.
import fs from 'node:fs'; import path from 'node:path'; import {fileURLToPath, pathToFileURL} from 'node:url';
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const HERE = path.dirname(fileURLToPath(import.meta.url)); const CD = path.resolve(HERE, '../../../cd32303');
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const fx = JSON.parse(fs.readFileSync(path.join(HERE, 'fixture/verified-inputs.json'), 'utf8'));
class El { constructor(t){this.tag=t;this.kids=[];this.text='';this.attrs={};this.className='';}
  set textContent(v){this.text=String(v);this.kids=[];} get textContent(){return this.text+this.kids.map(k=>k.textContent).join('');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this.text='';this.kids=k;} setAttribute(a,v){this.attrs[a]=v;} }
const all = (n, o=[]) => { o.push(n); n.kids.forEach(k => all(k, o)); return o; };
async function render(manifest) {
  const ids = ['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit']; const nodes = Object.fromEntries(ids.map(i => [i, new El('div')]));
  const document = {createElement: t => new El(t), getElementById: i => nodes[i] ?? null};
  const fetcher = async u => { u = String(u); if (u === 'museum.json') return new Response(JSON.stringify(manifest));
    if (Object.hasOwn(fx, u)) return new Response(Buffer.from(fx[u], 'base64')); return new Response('absent', {status: 404}); };
  await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('none'); }});
  const cards = {}; for (const host of nodes['claim-cards'].kids) { const a = host.kids[0]; if (a?.attrs['data-object-id']) cards[a.attrs['data-object-id']] = a; }
  return {state: nodes['museum-state'].textContent, cards};
}
function reviewView(card) {
  const n = all(card); const i = n.findIndex(x => x.tag === 'p' && /Review pointer in the verified index|Byte-verified review/.test(x.text));
  const href = n.find(x => x.tag === 'a' && x.text === 'Open review ↗')?.href;
  const det = n.find(x => x.tag === 'details' && x.kids[0]?.text === 'Identity of the pointer source');
  const dl = det ? all(det).filter(x => x.tag === 'dt' || x.tag === 'dd').map(x => x.text) : [];
  const sha = dl[dl.indexOf('SHA-256') + 1], p = dl[dl.indexOf('Path') + 1];
  const quoteHasExactLink = card.kids.length && all(card).find(x => x.tag === 'blockquote')?.text.includes('](' + href + ')');
  return {label: n[i]?.text, open_review_href: href, pointer_source: p + ' sha256=' + sha, quote_contains_link_to_href: !!quoteHasExactLink, class: card.attrs['data-object-class']};
}
const real = () => JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));
const B = 'https://github.com/d6g8k5htny-coder/main/issues/';
for (const [i, url] of [[null, null], [0, B + '6'], [0, B + '67#issuecomment-584178220'], [11, B + '6'], [11, B + '63#issuecomment-5841830']]) {
  const m = real(); if (i !== null) m.claims[i].review.review_url = url; const id = m.claims[i ?? 11].id;
  const r = await render(m);
  console.log(`${i === null ? 'BASELINE' : 'SUBSTITUTED'} ${id} review_url=${m.claims[i ?? 11].review.review_url}`);
  console.log('   ', JSON.stringify(reviewView(r.cards[id])));
  console.log('    state:', r.state);
}
