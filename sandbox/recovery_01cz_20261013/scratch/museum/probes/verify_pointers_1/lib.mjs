// Shared harness: imports the cd32303 museum.mjs read-only, serves ONLY the byte-verified local fixture,
// and stubs all network. Any request not in the fixture map is recorded and refused.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
export const CD = path.resolve(HERE, '../../cd32303');
globalThis.fetch = async (u) => { throw Error('NETWORK DISABLED IN PROBE: ' + u); };
export const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
export const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'fixture/verified-inputs.json'), 'utf8'));
export const realManifest = () => JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));
export const text = (url) => Buffer.from(fixtures[url], 'base64').toString('utf8');
export function sources(manifest) { return {index: text(manifest.index_source.url), status: text(manifest.status_source.url)}; }
export function validate(manifest) {
  const {index, status} = sources(manifest);
  try { museum.validateBoundManifest(manifest, index, status); return 'PASS'; } catch (e) { return 'REFUSED: ' + e.message; }
}
class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.className='';this._text='';}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;}
  setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){}
}
function documentFromHTML(){
  const html=fs.readFileSync(path.join(CD,'docs/site/museum.html'),'utf8');
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m=>[m[2],new Element(m[1])]));
  return {nodes,createElement:t=>new Element(t),getElementById:id=>nodes.get(id)??null};
}
function walk(node, out=[]) { if (node && node.tagName) { out.push(node); for (const c of node.children) walk(c, out); } return out; }
// Render the full museum with the given manifest; extra maps URL -> bytes for hypothetical files.
export async function render(manifest, extra = {}) {
  const requested = [];
  const fetcher = async (url) => {
    const key = String(url); requested.push(key);
    if (key === 'museum.json') return new Response(JSON.stringify(manifest));
    if (Object.hasOwn(extra, key)) return new Response(extra[key]);
    if (Object.hasOwn(fixtures, key)) return new Response(Buffer.from(fixtures[key], 'base64'));
    return new Response('not in fixture', {status: 404});
  };
  const document = documentFromHTML();
  await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const cards = {};
  for (const host of document.getElementById('claim-cards').children) {
    const art = host.children[0]; if (!art) continue;
    const id = art.attributes['data-object-id'] || ('refused:' + (art.children[0]?.textContent || ''));
    const nodes = walk(art);
    cards[id] = {text: art.textContent, anchors: nodes.filter(n => n.tagName === 'A').map(n => ({text: n.textContent, href: n.href})),
                 paras: nodes.filter(n => n.tagName === 'P').map(n => n.textContent)};
  }
  return {state: document.getElementById('museum-state').textContent, cards, requested};
}
export function reviewBlock(card) {
  const i = card.paras.findIndex(p => /Review pointer in the verified index|Byte-verified review|No separate byte-frozen review/.test(p));
  const a = card.anchors.find(x => x.text === 'Open review ↗');
  return {label: card.paras[i], open_review_href: a ? a.href : null};
}
