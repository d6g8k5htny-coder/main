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
 for(const name of ['index','explore','research','workspace','museum','formal']){
 const h=read(`docs/site/${name}.html`);
 assert.match(h,/<link rel="stylesheet" href="brand\.css\?site-release=[0-9a-f]{64}">/);
 assert.ok(h.indexOf('brand.css')>h.indexOf('style.css'));
 assert.match(h,/href="brand\/mark\.svg\?site-release=[0-9a-f]{64}"/);
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
// Rules of brand.css and museum.css (the museum page loads both; comments removed), each with its normalised selector list and enclosing at-rules, so rules inside @media count too.
// These checks read literal px lengths only: a custom property (var(--x)) or other unit in a ring width, ring offset or padding is not resolved and fails, so keep those declarations literal px.
const sheets=['docs/site/brand.css','docs/site/museum.css'];
const normSel=s=>s.trim().replace(/\s*([>+~])\s*/g,' $1 ').replace(/\s+/g,' ');
function cssRules(){const out=[];for(const file of sheets){const stack=[];let buf='';for(const ch of read(file).replace(/\/\*[\s\S]*?\*\//g,'')){if(ch==='{'){stack.push(buf.trim());buf='';}else if(ch==='}'){const pre=stack.pop();if(pre!==undefined&&!pre.startsWith('@'))out.push({file,selectors:pre.split(',').map(normSel),body:buf,media:stack.filter(x=>x.startsWith('@'))});buf='';}else buf+=ch;}}return out;}
// A rule names `sel` when a selector in its list is `sel` or `sel` in a narrower context (`.museum-card sel`, `main > sel`).
const rulesFor=sel=>cssRules().filter(r=>r.selectors.some(s=>s===sel||s.endsWith(' '+sel)));
// The last declaration of `prop` in a rule body: a later declaration overrides an earlier one in the same rule.
const decl=(body,prop)=>{let v;for(const d of body.split(';')){const i=d.indexOf(':');if(i>0&&d.slice(0,i).trim()===prop)v=d.slice(i+1).replace(/!important/,'').trim();}return v;};
const px=v=>{const m=String(v).match(/^(-?\d*\.?\d+)(px)?$/);assert.ok(m&&(m[2]||Number(m[1])===0),`${v} is not a literal px length`);return Number(m[1]);};
const outlineWidths=body=>{const out=[];const w=decl(body,'outline-width');if(w!==undefined)out.push(px(w));const sh=decl(body,'outline');if(sh!==undefined){const t=sh.split(/\s+/).find(x=>/^-?\d*\.?\d+(px)?$/.test(x));assert.ok(t,`outline: ${sh} has no literal px width`);out.push(px(t));}return out;};
const outlineOffset=body=>{const o=decl(body,'outline-offset');return o===undefined?[]:[px(o)];};
// Padding per side from every padding property of one rule (each property's last declaration; the pages are LTR, so inline-start is left).
function paddingSides(body){
 const s={top:[],right:[],bottom:[],left:[]};const put=(side,v)=>s[side].push(px(v));
 const sh=decl(body,'padding');if(sh!==undefined){const [t,r=t,b=t,l=r]=sh.split(/\s+/);put('top',t);put('right',r);put('bottom',b);put('left',l);}
 for(const [prop,a,b] of [['padding-inline','left','right'],['padding-block','top','bottom']]){const v=decl(body,prop);if(v!==undefined){const [x,y=x]=v.split(/\s+/);put(a,x);put(b,y);}}
 for(const [prop,side] of [['padding-top','top'],['padding-right','right'],['padding-bottom','bottom'],['padding-left','left'],['padding-inline-start','left'],['padding-inline-end','right'],['padding-block-start','top'],['padding-block-end','bottom']]){const v=decl(body,prop);if(v!==undefined)put(side,v);}
 return s;
}
const where=r=>`${r.file}${r.media.length?' '+r.media.join(' '):''}`;
test('the museum figure focus ring keeps its 3px width, 4px offset and the clip room drawn for it',()=>{
 const sel='.geometry-host > .museum-visual:focus-visible';const rules=rulesFor(sel);
 assert.ok(rules.some(r=>!r.media.length&&outlineWidths(r.body).length&&outlineOffset(r.body).length),`no unconditional outline rule for ${sel}`);
 for(const r of rules){
  for(const w of outlineWidths(r.body))assert.equal(w,3,`figure ring width (${where(r)})`);
  for(const o of outlineOffset(r.body)){assert.ok(o>0,`figure ring offset ${o}px is not outside the figure (${where(r)})`);assert.equal(o,4,`figure ring offset (${where(r)})`);}
 }
 const need=4+3;const host=rulesFor('.geometry-host').map(r=>({r,s:paddingSides(r.body)}));
 for(const side of ['top','right','bottom','left'])assert.ok(host.some(x=>!x.r.media.length&&x.s[side].length),`no unconditional .geometry-host padding on the ${side}`);
 for(const {r,s} of host)for(const [side,v] of Object.entries(s))for(const x of v)assert.ok(x>=need,`.geometry-host ${side} padding ${x}px < ${need}px (${where(r)})`);
});
test('the conditional-route card leaves room inside its clip for the focus ring of its summaries and links',()=>{
 // Room the ring needs: the widest width plus the largest offset set by any rule for a link's or summary's focus in either sheet.
 const ringRules=cssRules().filter(r=>r.selectors.some(s=>/^(?:a|summary)(?![\w-])[^ ]*:focus(?:-visible)?$/.test(s.split(' ').pop())));
 const widths=ringRules.flatMap(r=>outlineWidths(r.body)),offsets=ringRules.flatMap(r=>outlineOffset(r.body));
 assert.ok(widths.length&&offsets.length,'link and summary focus ring width and offset');
 const need=Math.max(...widths)+Math.max(...offsets);
 const sel='#conditional-route > .museum-card';const card=rulesFor(sel).map(r=>({r,s:paddingSides(r.body)}));
 assert.ok(card.some(x=>!x.r.media.length),`no rule for ${sel} outside @media`);
 for(const side of ['left','right'])assert.ok(card.some(x=>!x.r.media.length&&x.s[side].length),`no ${side} padding for ${sel} outside @media`);
 for(const {r,s} of card){const v=[...s.left,...s.right];if(v.length)assert.ok(Math.min(...v)>=need,`${sel} inline padding ${Math.min(...v)}px < ${need}px (${where(r)})`);}
});
