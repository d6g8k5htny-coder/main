// Verifier 2 of Q2-F6. Independent of the finder's lib.mjs: own fixture integrity check, own DOM stub.
// Imports the cd32303 museum.mjs read-only. All network is stubbed; any fetch outside the fixture map is refused.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
const ROOT = '<SCRATCH>/museum';
const CD = path.join(ROOT, 'cd32303');
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const manifestBytes = fs.readFileSync(path.join(CD, 'docs/site/museum.json'));
const manifest = JSON.parse(manifestBytes.toString('utf8'));
const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, 'probes/pointers/fixture/verified-inputs.json'), 'utf8'));

// 1. Re-verify every fixture entry against the descriptors that cd32303 museum.json itself pins (bytes, git blob, sha256).
const pins = new Map();
const addPin = d => { if (d && d.url) pins.set(d.url, d); };
addPin(manifest.index_source); addPin(manifest.status_source);
for (const c of manifest.claims) { addPin(c.proof); addPin(c.review); addPin(c.replay?.source); }
for (const k of Object.keys(manifest.exhibits)) addPin(manifest.exhibits[k]);
for (const p of manifest.packets) addPin(p.result);
let ok = 0;
for (const [url, d] of pins) {
  const raw = Buffer.from(fixture[url] ?? '', 'base64');
  const blob = crypto.createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${raw.length}\0`), raw])).digest('hex');
  const sha = crypto.createHash('sha256').update(raw).digest('hex');
  if (raw.length !== d.bytes || blob !== d.blob || sha !== d.sha256) { console.log('FIXTURE MISMATCH', url); process.exit(2); }
  ok++;
}
console.log(`fixture: ${ok}/${pins.size} pinned sources match museum.json@cd32303 (bytes+blob+sha256)`);

// 2. Minimal DOM and a full startMuseum render.
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this.className='';this._t='';this.href=undefined;}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return this._t+this.children.map(c=>c.textContent).join('\n');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const ids = ['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'];
const nodes = Object.fromEntries(ids.map(i => [i, new El('div')]));
const document = {createElement: t => new El(t), getElementById: i => nodes[i] ?? null};
const outside = [];
const fetcher = async url => { const k = String(url);
  if (k === 'museum.json') return new Response(manifestBytes);
  if (Object.hasOwn(fixture, k)) return new Response(Buffer.from(fixture[k], 'base64'));
  outside.push(k); return new Response('refused', {status: 404}); };
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
console.log('state:', nodes['museum-state'].textContent);
console.log('requests outside fixture:', JSON.stringify(outside));
const walk = (n, o = []) => { o.push(n); for (const c of n.children) walk(c, o); return o; };
const cards = {};
for (const host of nodes['claim-cards'].children) { const a = host.children[0]; if (a?.attributes['data-object-id']) cards[a.attributes['data-object-id']] = walk(a); }

const index = Buffer.from(fixture[manifest.index_source.url], 'base64').toString('utf8');
const status = Buffer.from(fixture[manifest.status_source.url], 'base64').toString('utf8');
const links = s => [...s.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)].map(m => [m[1], m[2]]);
const short = u => u.replace('https://github.com/d6g8k5htny-coder/', '');
const statusRowFor = id => status.split('\n').find(l => l.startsWith(`| **${id} —`));
const d5IndexLine = index.split('\n').find(l => l.startsWith('- D5 pin neighborhoods:'));

for (const c of manifest.claims.filter(c => c.review?.pointer_only)) {
  const n = cards[c.id]; const paras = n.filter(x => x.tagName === 'P').map(x => x._t);
  const anchors = n.filter(x => x.tagName === 'A');
  const openReview = anchors.filter(a => a._t === 'Open review ↗');
  const label = paras.find(p => /Review pointer in the verified index|Byte-verified review/.test(p));
  const allText = n.map(x => x._t).join('\n');
  const href = openReview[0].href;
  const labelled = links(c.scope_quote).find(([, u]) => u === href);
  console.log(`\n[${c.id}] label="${label}" | pointer-source identity = ${c.review.repository}:${c.review.path}@${c.review.commit.slice(0,8)}`);
  console.log(`  "Open review" anchors: ${openReview.length} -> ${short(href)} | label of that URL in displayed scope_quote: ${labelled ? JSON.stringify(labelled[0]) : 'NOT IN DISPLAYED QUOTE'}`);
  console.log(`  review/comment links in displayed scope_quote: ${JSON.stringify(links(c.scope_quote).filter(([, u]) => /issues\/\d+|REVIEW\.md/.test(u)).map(([t, u]) => `${t} -> ${short(u)}`))}`);
  const sid = {'d2-lifetime-remainder':'D2','d3-side24-coefficient':'D3','d4-fixed-remote-rn':'D4'}[c.id];
  if (sid) { const row = statusRowFor(sid); const cell = row.slice(1,-1).split('|').map(s=>s.trim())[2];
    console.log(`  STATUS ${sid} 'Source and review' cell links (cell displayed on card: ${allText.includes(cell)}): ${JSON.stringify(links(cell).map(([t,u])=>`${t} -> ${short(u)}`))}`); }
  if (c.id === 'd5-pin-neighborhoods-open') {
    console.log(`  D5_OPEN_REVIEW in displayed scope_quote: ${c.scope_quote.includes(href)}; in status_quote: ${String(c.status_quote).includes(href)}; anywhere in rendered card text except the anchor href: ${allText.includes(href)}`);
    console.log(`  binding PROOF_INDEX line contains it: ${d5IndexLine.includes(href)}; that line rendered on card: ${allText.includes(d5IndexLine)}; card links pinned PROOF_INDEX blob: ${anchors.some(a => a.href === manifest.index_source.html_url)}`);
  }
}
// D1: what the PROOF_INDEX (not displayed) calls the same comment, and which review it names.
const d1line = index.split('\n').find(l => l.startsWith('- D1 parent lifetime theorem:'));
console.log(`\nPROOF_INDEX D1 line (not the D1 pointer source) links: ${JSON.stringify(links(d1line).filter(([,u])=>/issues/.test(u)).map(([t,u])=>`${t} -> ${short(u)}`))}`);
