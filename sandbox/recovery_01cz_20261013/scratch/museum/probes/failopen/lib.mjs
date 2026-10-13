// Shared harness for the fail-open probes. Network is stubbed: the fetcher serves museum.json from the
// supplied object and raw.githubusercontent URLs only from the byte-verified local fixture; anything else throws.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
export const TREE = process.env.MUSEUM_TREE || path.resolve(HERE, '../../cd32303');
export const museum = await import(pathToFileURL(path.join(TREE, 'docs/site/museum.mjs')));
export const realManifest = () => JSON.parse(fs.readFileSync(path.join(TREE, 'docs/site/museum.json'), 'utf8'));
const fixtures = JSON.parse(fs.readFileSync(path.join(HERE, 'fixture/verified-inputs.json'), 'utf8'));
export const sourceBytes = url => { if (!Object.hasOwn(fixtures, url)) throw Error(`no fixture for ${url}`); return Buffer.from(fixtures[url], 'base64'); };
export const requests = [];
export function makeFetcher(manifest) {
  return async url => {
    url = String(url); requests.push(url);
    if (url === 'museum.json') return new Response(JSON.stringify(manifest));
    if (Object.hasOwn(fixtures, url)) return new Response(Buffer.from(fixtures[url], 'base64'));
    throw Error(`network stubbed: refused ${url}`);
  };
}
export class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.listeners={};this.hidden=false;this._text='';this.style={};}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join('');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;}
  setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(k,f){this.listeners[k]=f;}
  walk(){return [this,...this.children.flatMap(c=>c instanceof Element?c.walk():[])];}
}
export function documentFromHTML(){
  const html=fs.readFileSync(path.join(TREE,'docs/site/museum.html'),'utf8');
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m=>[m[2],new Element(m[1])]));
  return {nodes,createElement:t=>new Element(t),getElementById:id=>nodes.get(id)??null};
}
export async function render(manifest, {search='', geometryLoader=()=>{throw Error('geometry not loaded in this probe');}}={}) {
  const document = documentFromHTML();
  await museum.startMuseum({document, fetcher: makeFetcher(manifest), search, geometryLoader});
  return document;
}
export function card(document, id) { return document.getElementById('claim-cards').walk().find(n => n.attributes?.['data-object-id'] === id) || null; }
export function links(node) { return node.walk().filter(n => n.tagName === 'A').map(a => `${a.textContent} -> ${a.href}`); }
export const squash = s => s.replace(/\s+/g, ' ').trim();
