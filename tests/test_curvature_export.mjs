import test from 'node:test';
import assert from 'node:assert/strict';

let feature = {};
try { feature = await import('../docs/site/curvature-export.mjs'); } catch {}
function api(name) {
  assert.equal(typeof feature[name], 'function', `Curvature export ${name} is available`);
  return feature[name];
}
const timestamp = '2026-10-03T12:34:56.000Z';
const canonical = 'https://d6g8k5htny-coder.github.io/main/site/explore.html?s=-2&R=1#peaks';
const sourceURL = 'https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/PROOF.md#1-a-nonperiodic-reference-coefficient-with-exact-cone-moments';
// A hand-checked record of the current diagram, with concrete dark-mode styles.
function diagram(s = -2, R = 1) {
  const textStyle = {fill: 'rgb(239, 245, 253)', 'font-family': 'system-ui, sans-serif', 'font-size': '21px', 'font-weight': '400', 'font-style': 'normal', 'text-anchor': 'start'};
  const edge = {stroke: 'rgb(115, 144, 175)', 'stroke-width': '1.5px', fill: 'none'};
  const node = (tag, attributes = {}, text, style = {}) => ({tag, attributes, ...(text !== undefined ? {text} : {}), style});
  return {
    attributes: {id: 'curvature-diagram', viewBox: '0 0 600 330', role: 'img', 'aria-labelledby': 'curvature-diagram-title curvature-diagram-description'},
    background: 'rgb(7, 17, 31)',
    nodes: [
      node('title', {id: 'curvature-diagram-title'}, 'Where the two curvatures change sign'),
      node('desc', {id: 'curvature-diagram-description'}, 'Center −2.0, spread 1.0. Downward in both directions. Eigenvalues -3 and -1.'),
      node('polygon', {points: '86,165 474,265 474,294 86,294', class: 'diagram-fill'}, undefined, {fill: 'rgb(120, 193, 255)', 'fill-opacity': '0.17'}),
      node('polygon', {points: '86,36 474,36 474,65 86,165', class: 'diagram-warm'}, undefined, {fill: 'rgb(233, 198, 117)', 'fill-opacity': '0.11'}),
      ...[-3, 0, 3].flatMap(value => [node('line', {x1: '86', y1: String(165 - value * 40), x2: '474', y2: String(165 - value * 40), class: 'diagram-grid'}, undefined, edge), node('text', {x: '66', y: String(171 - value * 40), 'text-anchor': 'end', class: 'diagram-small'}, String(value), textStyle)]),
      node('line', {x1: '86', y1: '31', x2: '86', y2: '295', class: 'diagram-axis'}, undefined, edge),
      node('line', {x1: '86', y1: '295', x2: '498', y2: '295', class: 'diagram-axis'}, undefined, edge),
      node('line', {x1: '86', y1: '165', x2: '473.5', y2: '265', class: 'diagram-boundary'}, undefined, {...edge, 'stroke-dasharray': '7px, 6px'}),
      node('line', {x1: '86', y1: '165', x2: '473.5', y2: '65', class: 'diagram-boundary'}, undefined, {...edge, 'stroke-dasharray': '7px, 6px'}),
      node('text', {x: '47', y: '28'}, 's', textStyle), node('text', {x: '514', y: '302'}, 'R', textStyle),
      ...[0, 1, 2].map(value => node('text', {x: String(86 + value * 155), y: '320', 'text-anchor': 'middle', class: 'diagram-small'}, String(value), textStyle)),
      node('text', {x: '225', y: '65'}, 'Bowl', textStyle), node('text', {x: '335', y: '172'}, 'Saddle', textStyle), node('text', {x: '224', y: '278'}, 'Peak', textStyle),
      node('text', {x: '483', y: '70', class: 'diagram-small'}, 's = R', textStyle), node('text', {x: '483', y: '265', class: 'diagram-small'}, 's = −R', textStyle),
      node('circle', {cx: String(86 + R * 155), cy: String(165 - s * 40), r: '9', class: 'diagram-point'}, undefined, {fill: 'rgb(233, 198, 117)', stroke: 'rgb(7, 17, 31)', 'stroke-width': '3px'})
    ]
  };
}
function metadata(svg) {
  const raw = svg.match(/<metadata[^>]*>([\s\S]*?)<\/metadata>/)?.[1];
  assert.ok(raw, 'Export has machine-readable provenance');
  return JSON.parse(raw.replaceAll('&quot;', '"').replaceAll('&apos;', "'").replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&'));
}

test('export records bounded current parameters, eigenvalues and zero boundary meaning', () => {
  const build = api('buildFigureMetadata');
  const record = build({s: -2, R: 1}, timestamp);
  assert.deepEqual(record.params, {s: -2, R: 1});
  assert.deepEqual(record.eigenvalues, [-3, -1]);
  assert.equal(record.classification, 'peak');
  assert.equal(record.generated_at, timestamp);
  for (const [s, R, classification, eigenvalues] of [[0,1,'saddle',[-1,1]], [2,1,'bowl',[1,3]], [-1,1,'flat',[-2,0]], [0,0,'flat',[0,0]], [3,2,'bowl',[1,5]]]) {
    const actual = build({s,R}, timestamp);
    assert.equal(actual.classification, classification);
    assert.deepEqual(actual.eigenvalues, eigenvalues);
  }
  assert.match(record.limits, /second-order.*inconclusive/i);
});

test('invalid, out-of-range or off-step parameters and false timestamps are refused', () => {
  const build = api('buildFigureMetadata');
  for (const params of [{s: NaN,R:1},{s:Infinity,R:1},{s:-3.1,R:1},{s:3.1,R:1},{s:0,R:-0.1},{s:0,R:2.1},{s:0.15,R:1},{s:0,R:0.15},{s:'-2',R:1}]) assert.throws(() => build(params, timestamp));
  for (const invalid of ['not-a-date', '', '2026-02-30T00:00:00.000Z']) assert.throws(() => build({s:-2,R:1}, invalid));
});

test('citation keeps only the canonical figure permalink and pinned teaching source', () => {
  const citation = api('buildFigureCitation')({s: -2, R: 1, r: 0.25, region: 'remote', url: 'https://evil.test/?private=secret#groups'});
  assert.ok(citation.includes(canonical));
  assert.ok(citation.includes(sourceURL));
  assert.match(citation, /teaching figure/i);
  assert.match(citation, /s = -2.*R = 1/);
  assert.doesNotMatch(citation, /evil|private|secret|region|groups|r=0.25/);
  assert.match(citation, /not.*acceptance/i);
});

test('SVG retains actual geometry, concrete theme, aria names and visible teaching footer', () => {
  const svg = api('serializeCurvatureSVG')(diagram(), {s: -2, R: 1}, timestamp);
  assert.match(svg, /^<\?xml version="1.0" encoding="UTF-8"\?>\n<svg/);
  assert.match(svg, /viewBox="0 0 600 [4-9]\d\d"/);
  assert.match(svg, /aria-labelledby="curvature-diagram-title curvature-diagram-description"/);
  assert.match(svg, /<circle[^>]*cx="241"[^>]*cy="245"[^>]*r="9"/);
  assert.match(svg, /points="86,165 474,265 474,294 86,294"/);
  assert.match(svg, /fill="rgb\(7, 17, 31\)"/);
  assert.match(svg, /font-family="system-ui, sans-serif"/);
  assert.match(svg, /Teaching figure.*floating-point quadratic model/);
  assert.match(svg, /s − R.*s \+ R/);
  assert.match(svg, /Source: SIDE24 §1/);
  assert.match(svg, /non-certifying/);
  assert.doesNotMatch(svg, /<script|foreignObject|\bon[a-z]+=|\bhref=|\sclass=|\sstyle=|var\(|url\(|<style|<a\b/i);
  const record = metadata(svg);
  assert.equal(record.schema, 'universal-law/curvature-teaching-figure/v1');
  assert.equal(record.mode, 'teaching_model');
  assert.equal(record.verification, 'not_performed');
  assert.equal(record.permalink, canonical);
  assert.deepEqual(record.source, {repository:'d6g8k5htny-coder/Math-',commit:'9d7b6802424fb4715b31999066aafca8ee2f3cca',path:'coefficients/side24_v1/PROOF.md',blob:'44b66f04f89fcd87383b3603fa69f1feb64cdddd',bytes:10272,sha256:'c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769',url:sourceURL});
});

test('light theme colors are exported from the captured record', () => {
  const record = diagram(); record.background = 'rgb(246, 248, 252)';
  record.nodes.at(-1).style.fill = 'rgb(120, 87, 11)';
  const svg = api('serializeCurvatureSVG')(record, {s:-2,R:1}, timestamp);
  assert.match(svg, /fill="rgb\(246, 248, 252\)"/);
  assert.match(svg, /<circle[^>]*fill="rgb\(120, 87, 11\)"/);
});

test('inherited native system font stacks remain self-contained without remote font syntax', () => {
  const record = diagram();
  record.nodes[2].style['font-family'] = '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';
  const svg = api('serializeCurvatureSVG')(record, {s:-2,R:1}, timestamp);
  assert.match(svg, /font-family="-apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, system-ui, sans-serif"/);
  assert.doesNotMatch(svg, /url\(/);
});

test('all diagram text and attribute values are XML escaped', () => {
  const record = diagram();
  record.nodes[0].text = 'Curvature <test> & "quoted"';
  record.nodes[14].text = '<script>alert("x")</script> & apostrophe\'';
  const svg = api('serializeCurvatureSVG')(record, {s:-2,R:1}, timestamp);
  assert.match(svg, /Curvature &lt;test&gt; &amp; &quot;quoted&quot;/);
  assert.match(svg, /&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt; &amp; apostrophe&apos;/);
  assert.doesNotMatch(svg, /<script>/);
});

test('blank, stale or unsupported diagram content is refused instead of exported', () => {
  const serialize = api('serializeCurvatureSVG');
  const bad = [];
  const blank = diagram(); blank.nodes = blank.nodes.slice(0,2); bad.push(blank);
  const stale = diagram(); stale.nodes.at(-1).attributes.cy = '244'; bad.push(stale);
  const script = diagram(); script.nodes.push({tag:'script',attributes:{},text:'alert(1)',style:{}}); bad.push(script);
  for (const [key,value] of [['onload','alert(1)'],['href','https://evil.test'],['style','fill:red'],['transform','translate(1)']]) { const record=diagram(); record.nodes.at(-1).attributes[key]=value; bad.push(record); }
  for (const [key,value] of [['fill','url(https://evil.test/p.svg)'],['stroke','var(--ul-link)'],['font-family','url(https://evil.test/font)'],['stroke-width','10000px']]) { const record=diagram(); record.nodes.at(-1).style[key]=value; bad.push(record); }
  const points=diagram(); points.nodes[2].attributes.points='86,165 Infinity,265'; bad.push(points);
  const badId=diagram(); badId.nodes[0].attributes.id='name" onload="evil'; bad.push(badId);
  const unknownStyle=diagram(); unknownStyle.nodes[14].style.filter='blur(1px)'; bad.push(unknownStyle);
  for (const record of bad) assert.throws(() => serialize(record,{s:-2,R:1},timestamp));
  assert.throws(() => serialize(diagram(),{s:0,R:1},timestamp), /displayed|marker|stale/i);
});

test('clipboard absence exposes manual citation and never permits copy', async () => {
  const controller = api('createCitationController')({});
  controller.setFigure({s:-2,R:1});
  const state = controller.state();
  assert.equal(state.canCopy, false);
  assert.equal(state.manualFallback, true);
  assert.ok(state.text.includes(canonical));
  assert.equal(await controller.copy(), false);
  assert.match(controller.state().status, /manual/i);
});

test('clipboard denial preserves the current manual text and reports failure', async () => {
  const controller = api('createCitationController')({writeText: async () => { throw new Error('permission denied'); }});
  controller.setFigure({s:0,R:1});
  assert.equal(await controller.copy(), false);
  assert.equal(controller.state().manualFallback, true);
  assert.match(controller.state().status, /denied|could not/i);
  assert.ok(controller.state().text.includes('?s=0&R=1#peaks'));
});

test('pending clipboard writes are serialized and cannot report stale settings as copied', async () => {
  let finish; const writes=[];
  const controller = api('createCitationController')({writeText: value => { writes.push(value); return new Promise(resolve => {finish=resolve;}); }});
  controller.setFigure({s:-2,R:1});
  const first = controller.copy();
  assert.equal(controller.state().pending, true);
  assert.equal(controller.state().canCopy, false);
  controller.setFigure({s:2,R:1});
  assert.equal(await controller.copy(), false);
  assert.equal(writes.length, 1);
  assert.ok(writes[0].includes('?s=-2&R=1#peaks'));
  finish(); assert.equal(await first, false);
  assert.doesNotMatch(controller.state().status, /copied figure citation/i);
  assert.match(controller.state().status, /changed.*copy again/i);
  assert.equal(controller.state().canCopy, true);
  assert.ok(controller.state().text.includes('?s=2&R=1#peaks'));
});

test('a successful current write reports success, while unavailable figures block later writes', async () => {
  const writes=[];
  const controller = api('createCitationController')({writeText: async value => {writes.push(value);}});
  assert.equal(await controller.copy(), false);
  controller.setFigure({s:-1,R:1});
  assert.equal(await controller.copy(), true);
  assert.match(controller.state().status, /copied figure citation/i);
  controller.clear();
  assert.equal(controller.state().canCopy, false);
  assert.equal(await controller.copy(), false);
  assert.equal(writes.length, 1);
});
