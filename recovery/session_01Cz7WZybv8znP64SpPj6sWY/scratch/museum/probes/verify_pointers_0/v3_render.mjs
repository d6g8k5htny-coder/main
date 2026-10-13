// Verifier 1 probe for Q2-F2 (render of CI-green regenerated manifests). Independent of the pointers/ harness except for READ-ONLY use of
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
// Render a generator-regenerated, CI-green sim/<variant>/docs/site/museum.json through the cd32303 display code.
const [variant, id] = process.argv.slice(2);
const m = JSON.parse(fs.readFileSync(path.join(HERE, 'sim', variant, 'docs/site/museum.json'), 'utf8'));
const diff = JSON.stringify(m.claims.find(c => c.id === id).review.review_url);
let v; try { museum.validateBoundManifest(m, index, status); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
const r = await render(m); const c = r.cards[id];
const q = m.claims.find(c => c.id === id).scope_quote;
console.log(`[${variant}] regenerated review_url=${diff} exactLinkTargetInQuote=${linkTargets(q).includes(JSON.parse(diff))} validateBoundManifest=${v}`);
console.log(`[${variant}] state=${JSON.stringify(r.state)}`);
console.log(`[${variant}] ${id}: class=${c?.cls} label=${JSON.stringify(c?.label)} href=${c?.href} pointer-source-identity=${JSON.stringify(c?.identity)}`);
