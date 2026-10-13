// Q2-F3 verifier-1: render a sim/<name> tree's museum.json through THAT tree's museum.mjs (identical to cd32303
// except where the mutation edits it), serving only byte-verified fixture bytes; network is stubbed.
// Prints the 'Source and review' column of the chosen card in document order.
import fs from 'node:fs'; import path from 'node:path'; import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const [name, id] = process.argv.slice(2);
const SIM = path.join(HERE, 'sim', name);
const museum = await import(pathToFileURL(path.join(SIM, 'docs/site/museum.mjs')));
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, '../../pointers/fixture/verified-inputs.json'), 'utf8'));
const manifest = JSON.parse(fs.readFileSync(path.join(SIM, 'docs/site/museum.json'), 'utf8'));
class El { constructor(t){this.tag=t.toUpperCase();this.children=[];this.attrs={};this._t='';this.className='';}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return this._t+this.children.map(c=>c.textContent).join('');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attrs[k]=String(v);} addEventListener(){} }
const html = fs.readFileSync(path.join(SIM, 'docs/site/museum.html'), 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new El(m[1])]));
const document = {createElement: t => new El(t), getElementById: i => nodes.get(i) ?? null};
const requested = [];
const fetcher = async url => { const k = String(url); requested.push(k);
  if (k === 'museum.json') return new Response(JSON.stringify(manifest));
  if (Object.hasOwn(fixtures, k)) return new Response(Buffer.from(fixtures[k], 'base64'));
  return new Response('absent', {status: 404}); };
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
const flat = (n, out = []) => { if (n && n.tag) { out.push(n); n.children.forEach(c => flat(c, out)); } return out; };
const art = [...document.getElementById('claim-cards').children].map(h => h.children[0]).find(a => a?.attrs?.['data-object-id'] === id);
console.log(`[${name}] museum-state: ${document.getElementById('museum-state').textContent}`);
if (!art) { console.log(`[${name}] card ${id}: NOT RENDERED`); process.exit(0); }
const col = flat(art).filter(n => n.tag === 'DIV' && n.className === 'claim-column')[1];
console.log(`[${name}] card ${id} class=${art.attrs['data-object-class']} — Source and review column:`);
for (const n of col.children) {
  if (n.tag === 'A') console.log(`   <a> ${n.textContent} -> ${n.href}`);
  else if (n.tag === 'DETAILS') console.log(`   <details> ${n.children[0].textContent} :: ${flat(n.children[1]).filter(x=>x.tag==='DD').map(x=>x.textContent).join(' | ')}`);
  else if (n.textContent) console.log(`   <${n.tag.toLowerCase()}> ${n.textContent}`);
}
const reviewFetched = requested.some(u => u.includes('issuecomment'));
console.log(`[${name}] linked comment fetched by viewer: ${reviewFetched}`);
