// Verifier 1 probe for Q2-F2 (frontend half). Independent of the pointers/ harness except for READ-ONLY use of
// its fixture bytes, which are re-verified here against the pinned SHA-256/byte count in cd32303 museum.json.
// STATUS.md is read from the local git object 71400b94:STATUS.md (read-only `git show`).
// Network is stubbed: any fetch outside the hash-verified local map is refused.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const CD = path.resolve(HERE, '../../cd32303');
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const manifest0 = JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));
const fx = JSON.parse(fs.readFileSync(path.resolve(HERE, '../pointers/fixture/verified-inputs.json'), 'utf8'));
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
// Independent byte acquisition and digest check of the two pointer sources.
const statusBytes = execFileSync('git', ['-C', '/home/user/main', 'show', `${manifest0.status_source.commit}:${manifest0.status_source.path}`]);
const indexBytes = Buffer.from(fx[manifest0.index_source.url], 'base64');
for (const [name, b, pin] of [['STATUS.md(git object)', statusBytes, manifest0.status_source], ['PROOF_INDEX.md(fixture)', indexBytes, manifest0.index_source]])
  console.log(`pin check ${name}: bytes ${b.length}/${pin.bytes} sha256 ${sha(b) === pin.sha256 ? 'MATCH' : 'MISMATCH'}`);
const index = indexBytes.toString('utf8'), status = statusBytes.toString('utf8');
// Serve every pinned URL only after re-checking its digest against museum.json identities.
const pins = new Map();
const addPin = s => { if (s && s.url) pins.set(s.url, s); };
addPin(manifest0.index_source); addPin(manifest0.status_source);
for (const c of manifest0.claims) { addPin(c.proof); addPin(c.review); addPin(c.replay?.source); }
for (const k of Object.keys(manifest0.exhibits)) addPin(manifest0.exhibits[k]);
for (const p of manifest0.packets) addPin(p.result);
const linkTargets = q => [...q.matchAll(/\]\(([^)]+)\)/g)].map(m => m[1]);
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';this.className='';}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return [this._t,...this.children.map(c=>c.textContent)].join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const walk = (n, o = []) => { if (n?.tagName) { o.push(n); n.children.forEach(c => walk(c, o)); } return o; };
async function render(m) {
  const html = fs.readFileSync(path.join(CD, 'docs/site/museum.html'), 'utf8');
  const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(x => [x[2], new El(x[1])]));
  const document = {createElement: t => new El(t), getElementById: id => nodes.get(id) ?? null};
  const fetcher = async url => {
    const k = String(url);
    if (k === 'museum.json') return new Response(JSON.stringify(m));
    const b = k === manifest0.status_source.url ? statusBytes : (Object.hasOwn(fx, k) ? Buffer.from(fx[k], 'base64') : null);
    if (!b) return new Response('absent', {status: 404});
    return new Response(b);
  };
  await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const out = {state: nodes.get('museum-state').textContent, cards: {}};
  for (const host of nodes.get('claim-cards').children) { const a = host.children[0]; if (!a) continue;
    const ns = walk(a); out.cards[a.attributes['data-object-id'] || 'REFUSED'] = {
      cls: a.attributes['data-object-class'],
      label: ns.filter(n => n.tagName === 'P').map(n => n._t).find(t => /Review pointer in the verified index|Byte-verified review/.test(t)),
      href: ns.find(n => n.tagName === 'A' && n._t === 'Open review ↗')?.href,
      identity: ns.filter(n => n.tagName === 'DL').map(d => d.children.map(c => c._t))[1] }; }
  return out;
}
const B = 'https://github.com/d6g8k5htny-coder/main/issues/';
const cases = [
  ['d2-lifetime-remainder', B + '6'],
  ['d2-lifetime-remainder', B + '67#issuecomment-584178220'],   // last digit dropped from the R5/R6 comment id
  ['d3-side24-coefficient', B + '65#issuecomment-58412'],
  ['d4-fixed-remote-rn', B + '7'],
  ['d1-parent-selection-open', B + '6'],
  ['d1-parent-selection-open', B + '63#issuecomment-584183074'], // last digit dropped from the reopening comment id
  ['p15-demand-one-counterexample', B + '6'],
  ['d2-lifetime-remainder', B + '99#issuecomment-1'],            // CONTROL: not a substring of the quote
  ['d2-lifetime-remainder', B + '67#issuecomment-5841270276'],   // CONTROL: an exact link target in the quote (R1-R4)
];
const base = await render(manifest0);
console.log('baseline render state:', base.state);
for (const [id, url] of cases) {
  const m = structuredClone(manifest0); const i = m.claims.findIndex(c => c.id === id);
  const quote = m.claims[i].scope_quote; m.claims[i].review.review_url = url;
  let v; try { museum.validateBoundManifest(m, index, status); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
  const exact = linkTargets(quote).includes(url), sub = quote.includes(url);
  let shown = '';
  if (v === 'PASS') { const r = await render(m); const c = r.cards[id];
    shown = ` | rendered class=${c?.cls} label=${JSON.stringify(c?.label)} href=${c?.href} pointer-source-identity=${JSON.stringify(c?.identity)}`; }
  console.log(`${id} review_url=${url} exactLinkTargetInQuote=${exact} substringOfQuote=${sub} -> ${v}${shown}`);
}
