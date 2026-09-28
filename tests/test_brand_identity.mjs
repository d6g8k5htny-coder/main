import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const root=new URL('../',import.meta.url);
const read=p=>fs.existsSync(new URL(p,root))?fs.readFileSync(new URL(p,root),'utf8'):'';
const css=()=>read('docs/site/brand.css');
const colors=()=>JSON.parse(read('docs/site/brand/tokens.json'));
function luminance(hex){const v=hex.match(/^#([0-9a-f]{6})$/i);assert.ok(v,'six-digit hex');return v[1].match(/../g).map(x=>parseInt(x,16)/255).map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4).reduce((a,x,i)=>a+x*[.2126,.7152,.0722][i],0);}
export function contrast(a,b){const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);}
test('brand assets exist and declare no scientific authority',()=>{
 for(const p of ['docs/site/brand.css','docs/site/brand/tokens.json','docs/site/brand/mark.svg','docs/site/brand/banner.svg','docs/BRAND_STYLE.md'])assert.ok(read(p),`missing ${p}`);
 assert.equal(colors().scientific_effect,'NONE');
});
test('dark and light text, links and state labels meet 4.5:1 contrast',()=>{
 assert.ok(read('docs/site/brand/tokens.json'),'palette missing');
 for(const [mode,t] of Object.entries(colors().themes))for(const bg of ['background','surface','raised'])for(const fg of ['text','muted','link','gold','accept','amend','error','engineering'])assert.ok(contrast(t[fg],t[bg])>=4.5,`${mode}/${fg}/${bg} ${contrast(t[fg],t[bg])}`);
});
test('buttons and focus ring have explicit contrast',()=>{
 assert.ok(read('docs/site/brand/tokens.json'),'palette missing');
 for(const [mode,t] of Object.entries(colors().themes)){
 assert.ok(contrast(t.buttonText,t.button)>=4.5,mode+' button');
 assert.ok(contrast(t.focus,t.background)>=3,mode+' focus');
 assert.ok(contrast(t.edge,t.surface)>=3,mode+' control border');
 }
});
test('palette tokens are actually applied, not decorative metadata',()=>{
 assert.ok(read('docs/site/brand/tokens.json'),'palette missing');
 for(const t of Object.values(colors().themes))for(const [key,value] of Object.entries(t))assert.ok(css().includes(`--ul-${key}: ${value}`),key);
 assert.ok(css().includes('background: var(--ul-background)'));
});
test('all public pages load local branding after legacy styles with no CSP relaxation',()=>{
 for(const name of ['index','museum','formal']){
 const h=read(`docs/site/${name}.html`);
 assert.match(h,/<link rel="stylesheet" href="brand\.css">/);
 assert.ok(h.indexOf('brand.css')>h.indexOf('style.css'));
 assert.match(h,/href="brand\/mark\.svg"/);
 assert.match(h,/class="skip-link" href="#main-content"/);
 assert.match(h,/<main id="main-content"/);
 assert.ok(!h.includes("'unsafe-inline'")&&!h.includes("'unsafe-eval'"));
 }
});
test('light, reduced-motion, forced-color and print modes are supplied',()=>{
 for(const v of ['prefers-color-scheme: light','prefers-reduced-motion: reduce','forced-colors: active','@media print'])assert.ok(css().includes(v),v);
});
test('status classes retain redundant patterns and non-brand acceptance color',()=>{
 assert.ok(css().includes('.museum-card.amend-open'));
 assert.ok(css().includes('border-style: dashed'));
 assert.ok(css().includes('border-style: double'));
 assert.ok(css().includes('border-style: dotted'));
 assert.ok(css().includes('.cards article.accept'));
 assert.ok(read('docs/site/brand/tokens.json'),'palette missing');
 for(const t of Object.values(colors().themes))assert.notEqual(t.accept,t.gold);
});
test('decorative SVGs are self-contained, named and contain no scripts or external embeds',()=>{
 for(const name of ['mark','banner']){
 const s=read(`docs/site/brand/${name}.svg`);assert.ok(s,'missing '+name);
 assert.match(s,/<title id="/);assert.match(s,/<desc id="/);assert.match(s,/role="img"/);assert.match(s,/viewBox="/);
 assert.ok(!/<script|<foreignObject|<image|\bon[a-z]+=|https?:\/\/(?!www\.w3\.org\/2000\/svg)/i.test(s));
 assert.ok(!/ACCEPT|PROVEN|kernel.checked|\btheorems?\b/i.test(s),'No status or result claims in artwork');
 assert.ok(Buffer.byteLength(s)<15000,'small vector asset');
 }
});
test('body links remain underlined and keyboard focus is not removed',()=>{
 assert.match(css(),/a\s*\{[^}]*text-decoration-thickness:/s);
 assert.match(css(),/:focus-visible/);
 assert.match(css(),/\.skip-link:focus/);
 assert.ok(!/outline:\s*(0|none)/.test(css()));
});
test('navigation and grids can reflow on small screens',()=>{
 assert.match(css(),/overflow-wrap: anywhere/);
 assert.match(css(),/minmax\(0, ?1fr\)/);
 assert.match(css(),/flex-wrap: wrap/);
 assert.match(css(),/max-width: 640px/);
});
test('styles do not hide evidence, scope disclaimers or refusal states',()=>{
 assert.ok(css(),'brand stylesheet missing');
 assert.ok(!/(?:\.boundary|\.source-header|\.error|\.canvas-disclaimer|\.status-object)[^{]*\{[^}]*display:\s*none/s.test(css()));
 assert.ok(!/url\(["']?https?:|@import/i.test(css()),'No external theme dependencies');
 for(const p of ['config.json','museum.json','status.json'])assert.ok(!css().includes(p),'No theme-owned evidence loading');
});
test('low-contrast adversarial palette fails the same numerical threshold',()=>assert.ok(contrast('#777777','#888888')<4.5));
