// Native reader presentation only. No model import, wire adapter or data fetch.
// Pins are declarations, not authentication of this consumer or a future build.
const SOURCE_BASE = 'https://github.com/d6g8k5htny-coder/main/blob/'
  + '77d35feff69c4df1b55a7449f3deaa68ed75de7d/';
const SOURCE_LINKS = Object.freeze([
  ['Cubic model',SOURCE_BASE + 'interface/src/scripts/cubic-model.mjs'],
  ['Finite elder model',SOURCE_BASE + 'interface/src/scripts/elder-model.mjs'],
  ['Frozen model contract',SOURCE_BASE + 'docs/plans/2026-10-09-canvas-model-contract.md'],
]);
const mounted = new WeakMap();
const COLORS = ['#174a7e','#8b2f52','#526d18','#67429a','#97600b','#006b65'];

function text(value) {
  if (!['string','number','bigint','boolean'].includes(typeof value))
    throw new TypeError('A native text value is required');
  return String(value);
}
function escape(value) {
  return text(value).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function numberText(value) {
  return Object.is(value,-0) ? '-0' : Object.is(value,+0) ? '+0' : String(value);
}
function snapshot(value) {
  if (!value || value.kind !== 'Snapshot'
      || !['cubic','elder'].includes(value.model))
    throw new TypeError('A committed or prepared native reader snapshot is required');
  return value;
}
function prefixLink(E, label, extra = '') {
  const end=escape(E);
  return '<a href="#event-prefix-' + end + '" data-reader-event-end="' + end
    + '" data-reader-focus="event-link-' + end + '"' + extra + '>'
    + escape(label) + '</a>';
}
function table(name, caption, headers, rows, options = {}) {
  return '<div class="reader-table-scroll"><table data-reader-table="' + name + '">'
    + '<caption>' + escape(caption) + '</caption><thead><tr>'
    + headers.map(h => '<th scope="col">' + escape(h) + '</th>').join('')
    + '</tr></thead><tbody>' + rows.map((row,i) => {
      const id=options.rowId ? ' id="' + escape(options.rowId(row,i)) + '"' : '';
      return '<tr' + id + '>' + row.map((value,column) => {
        const tag=column === 0 ? 'th' : 'td';
        const rendered=options.cell ? options.cell(value,column,i) : escape(value);
        return '<' + tag + (column === 0 ? ' scope="row"' : '') + '>'
          + rendered + '</' + tag + '>';
      }).join('') + '</tr>';
    }).join('') + '</tbody></table></div>';
}
function provenance(source) {
  const b=source.binding;
  const rows=[
    ['repository',b.repository],['modelCommit',b.modelCommit],
    ['cubicSha256',b.cubicSha256],['elderSha256',b.elderSha256],
    ['contractSha256',b.contractSha256],['label',b.label],
    ['composition.commit',b.composition ? b.composition.commit : 'UNKNOWN'],
    ['composition.tree',b.composition ? b.composition.tree : 'UNKNOWN'],
    ['qualification',source.qualification],['inputProducer',source.inputProducer],
    ['consumer',source.consumer],['build',source.build],['assets',source.assets],
    ['graph',source.graph],['review',source.review],['kernel',source.kernel],
    ['scientificEffect',source.scientificEffect],
    ['scientificStatusAuthority',source.scientificStatusAuthority],
  ];
  return '<h3>Source declarations and limits</h3>'
    + '<p>These declared pins do not authenticate the consumer or build. '
    + 'Graph, review and kernel joins remain as stated below.</p>'
    + table('provenance','Complete source and status declarations',['Field','Value'],rows)
    + '<ul>' + SOURCE_LINKS.map(([label,url]) => '<li><a href="' + url + '">'
      + escape(label) + '</a></li>').join('') + '</ul>';
}
function cubicTables(r) {
  const fields=[
    ['r',r.r],['k',r.k],['b',r.b],['rCubed',r.rCubed],
    ['pins[0]',r.pins[0]],['pins[1]',r.pins[1]],['gap',r.gap],
    ['heightWindow[0]',r.heightWindow[0]],['heightWindow[1]',r.heightWindow[1]],
    ['normalized.xDivisor',r.normalized.xDivisor],
    ['normalized.yOrigin',r.normalized.yOrigin],
    ['normalized.yDivisor',r.normalized.yDivisor],
    ['normalized.pins[0]',r.normalized.pins[0]],
    ['normalized.pins[1]',r.normalized.pins[1]],
    ['normalized.heightWindow[0]',r.normalized.heightWindow[0]],
    ['normalized.heightWindow[1]',r.normalized.heightWindow[1]],
  ];
  return '<p>Static cubic document. Binary64 teaching arithmetic; '
    + 'the nominal affine coordinates do not assert exact recovery from rounded endpoints. '
    + 'This height window is not a persistence interval.</p>'
    + table('cubic-fields','All physical and nominal cubic fields',['Field','Value'],
      fields.map(([name,value]) => [name,numberText(value)]));
}
function controls(s) {
  if (s.model !== 'elder') return '';
  const r=s.result, E=s.selection.eventEnd, nav=s.navigation;
  const button=(action,label,target) => '<button type="button" data-reader-action="'
    + action + '" data-reader-focus="' + action + '"'
    + (target === null ? ' disabled' : '') + '>' + label + '</button>';
  return '<nav aria-label="Manual reader navigation"><p>Paused; manual steps only. '
    + 'On the drawing, use Left/Right for events, Page Up/Down for complete levels, '
    + 'Home for empty and End for the final event.</p><div class="reader-buttons">'
    + button('previous-event','Previous event',nav.previousEvent)
    + button('next-event','Next event',nav.nextEvent)
    + button('previous-level','Previous complete level',nav.previousLevel)
    + button('next-level','Next complete level',nav.nextLevel)
    + button('reset','Reset to empty',nav.reset) + '</div>'
    + '<label>Insertion prefix <input type="range" min="0" max="' + escape(r.events.length)
    + '" step="1" value="' + escape(E) + '" data-reader-action="event-range"'
    + ' data-reader-focus="event-range" aria-valuetext="'
    + (E === 0 ? 'Empty insertion prefix' : 'After event ' + (E-1)) + '"></label>'
    + '<label>Complete level <select data-reader-action="level-select"'
    + ' data-reader-focus="level-select"><option value="" disabled'
    + (s.selection.mode === 'complete_level' ? '' : ' selected')
    + '>Choose a complete equal-value block</option>'
    + r.levels.map((l,i) => '<option value="' + i + '"'
      + (s.selection.mode === 'complete_level' && s.selection.levelIndex === i
        ? ' selected' : '') + '>Complete equal-value block: '
      + escape(l.level) + '/' + escape(r.scale) + ' (E' + l.eventEnd + ')</option>').join('')
    + '</select></label></nav>';
}
function elderTables(s) {
  const r=s.result, p=s.selected, E=s.selection.eventEnd, active=new Set(p.activeVertices);
  const grid=r.values.map((value,i) =>
    [i,Math.floor(i/r.n),i%r.n,value,active.has(i)]);
  const events=r.events.map((e,i) => [
    i,e.kind,e.vertex === undefined ? '' : e.vertex,
    e.edge ? e.edge[0] : '',e.edge ? e.edge[1] : '',e.level,
    e.survivor === undefined ? '' : e.survivor,e.dying === undefined ? '' : e.dying,
    e.birth === undefined ? '' : e.birth,e.death === undefined ? '' : e.death,
    e.zero === undefined ? '' : e.zero,e.elder === undefined ? '' : e.elder,i<E]);
  const levels=r.levels.map((l,i) => [i,l.level,l.eventEnd,l.activeCount,
    l.componentCount,l.positiveCount,l.zeroCount,'Complete equal-value block']);
  const intervals=r.barcode.intervals.map(([b,d],i) => [i,b,d,
    r.barcode.birth_vertices[i],'(' + d + '/' + r.scale + ', ' + b + '/' + r.scale + ']',
    (b-d) + '/' + r.scale]);
  const essential=[[r.barcode.essential[0],r.barcode.essential_vertices[0],
    '(-infinity, ' + r.barcode.essential[0] + '/' + r.scale + ']']];
  const summary=[
    ['n',r.n],['scale',r.scale],['zero_count',r.barcode.zero_count],
    ['selection.mode',s.selection.mode],['selection.eventEnd',E],
    ['selection.levelIndex',s.selection.mode === 'complete_level'
      ? s.selection.levelIndex : 'NONE'],
    ['selected.activeCount',p.activeVertices.length],
    ['selected.componentCount',p.components.length],
    ['selected.positiveCount',p.positiveCount],['selected.zeroCount',p.zeroCount],
    ['selected.neutralCount',p.neutralCount],
    ['selected.lastEventIndex',E === 0 ? 'NONE' : E-1],
    ['interpretation',s.selection.mode === 'complete_level'
      ? 'Complete equal-value block' : 'Insertion-order prefix'],
  ];
  return controls(s)
    + '<p>' + prefixLink(0,'Empty insertion prefix',' id="event-prefix-0"') + '. '
    + 'Prefix E means state after event E-1; only complete equal-value blocks '
    + 'are superlevel thresholds. Zero merges and redundant neutral edges are distinct.</p>'
    + table('summary','Selected state and whole-sweep metadata',['Field','Value'],summary)
    + table('grid','Every periodic grid input and its selected activation',
      ['Vertex','First coordinate','Second coordinate','Value numerator','Active'],grid,
      {rowId:row => 'reader-vertex-' + row[0]})
    + table('components','Selected components and mathematical elders',
      ['Elder vertex','Vertices'],p.components.map(c => [c.elder,c.vertices.join(',')]))
    + table('events','Complete explanatory insertion stream',
      ['Index','Kind','Vertex','Edge start','Edge end','Level numerator','Survivor',
        'Dying','Birth numerator','Death numerator','Zero','Elder','Applied'],events,
      {rowId:(_,i) => 'event-prefix-' + (i+1),
        cell:(value,column,i) => column === 0 ? prefixLink(i+1,value) : escape(value)})
    + table('levels','Every complete equal-value block',
      ['Index','Level numerator','Exclusive event end','Active','Components',
        'Positive merges','Zero merges','Interpretation'],levels,
      {cell:(value,column,i) => column === 2
        ? '<a href="#event-prefix-' + escape(value) + '" data-reader-level-index="' + i
          + '" data-reader-focus="level-link-' + i + '">' + escape(value) + '</a>'
        : escape(value)})
    + '<p>Whole-sweep intervals stay fixed during playback. Finite endpoints are '
    + 'open at death and closed at birth. Numerators and the positive scale below '
    + 'are exact; fractions are not truncated.</p>'
    + table('intervals','All finite intervals, including the longest',
      ['Index','Birth numerator','Death numerator','Birth vertex','Physical interval',
        'Lifetime'],intervals)
    + table('essential','Separate essential class',
      ['Birth numerator','Birth vertex','Physical interval'],essential);
}
export function renderStaticReader(value) {
  const s=snapshot(value);
  return '<section id="canvas-reader" aria-labelledby="reader-title"'
    + ' data-reader-model="' + s.model + '" data-reader-revision="' + escape(s.revision)
    + '" data-document-revision="' + escape(s.documentRevision)
    + '" data-selection-mode="' + escape(s.selection.mode) + '"'
    + (s.model === 'elder' ? ' data-event-end="' + escape(s.selection.eventEnd) + '"' : '') + '>'
    + '<style>#canvas-reader{font:1rem/1.5 system-ui,sans-serif;overflow-wrap:anywhere}'
    + '#canvas-reader .reader-table-scroll{overflow-x:auto;margin-block:1rem}'
    + '#canvas-reader table{border-collapse:collapse;width:100%}'
    + '#canvas-reader th,#canvas-reader td{padding:.35rem;border:1px solid #999;text-align:left}'
    + '#canvas-reader caption{font-weight:700;text-align:left}'
    + '#canvas-reader button,#canvas-reader select,#canvas-reader input{font:inherit;max-width:100%}'
    + '#canvas-reader label{display:block;margin-block:.6rem}'
    + '#canvas-reader .reader-buttons{display:flex;flex-wrap:wrap;gap:.5rem}'
    + '#canvas-reader :focus-visible{outline:3px solid #174a7e;outline-offset:3px}'
    + '.native-canvas-reader :focus-visible{outline:3px solid #174a7e;outline-offset:3px}'
    + '</style><h2 id="reader-title">Finite mathematical reader</h2>'
    + '<p>NON_CERTIFYING. Scientific effect NONE; scientific-status authority false. '
    + 'Exact native tables accompany an approximate drawing.</p>'
    + '<p id="reader-canvas-note">No animation or timer. Drawing uses labels and shapes '
    + 'as well as color. Complete tables remain available without Canvas or JavaScript.</p>'
    + (s.model === 'cubic' ? cubicTables(s.result) : elderTables(s))
    + provenance(s.source) + '</section>';
}

// Geometry uses bounded ratios AFTER exact BigInt subtraction, never Number endpoints.
function ratio(numerator,denominator) {
  if (denominator <= 0n) return 0.5;
  const clamped=numerator < 0n ? 0n : numerator > denominator ? denominator : numerator;
  return Number((clamped*4096n)/denominator)/4096;
}
function shortInteger(value) {
  const s=value.toString();
  return s.length <= 22 ? s : s.slice(0,12) + '…' + s.slice(-8);
}
function line(ctx,x1,y1,x2,y2,color='#243342',width=1) {
  ctx.strokeStyle=color; ctx.lineWidth=width;
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
}
function circle(ctx,x,y,r,fill,stroke='#243342') {
  ctx.beginPath(); ctx.arc(x,y,r,0,2*Math.PI);
  ctx.fillStyle=fill; ctx.fill(); ctx.strokeStyle=stroke; ctx.stroke();
}
function triangle(ctx,x,y,r,color) {
  ctx.beginPath(); ctx.moveTo(x,y-r); ctx.lineTo(x-r,y+r);
  ctx.lineTo(x+r,y+r); ctx.closePath(); ctx.fillStyle=color; ctx.fill();
}
function drawCubic(ctx,s) {
  const r=s.result;
  ctx.fillStyle='#17212b'; ctx.font='bold 18px system-ui';
  ctx.fillText('Physical and nominal teaching coordinates',24,30);
  const panel=(left,title,pins,window,notes) => {
    const x=left+45, y=115, width=325, bottom=355;
    ctx.fillStyle='#17212b'; ctx.font='bold 16px system-ui';
    ctx.fillText(title,left+15,70);
    line(ctx,x,y,x+width,y); line(ctx,x,bottom,x+width,bottom);
    ctx.fillStyle='#174a7e'; ctx.fillRect(x+width*0.25-6,y-6,12,12);
    circle(ctx,x+width*0.75,y,6,'#8b2f52');
    ctx.fillStyle='#17212b'; ctx.font='13px system-ui';
    ctx.fillText('square pin: ' + numberText(pins[0]),x,150);
    ctx.fillText('circle pin: ' + numberText(pins[1]),x,175);
    line(ctx,x+width/2,210,x+width/2,bottom,'#526d18',3);
    ctx.fillText('upper: ' + numberText(window[1]),x,205);
    ctx.fillText('lower: ' + numberText(window[0]),x,385);
    notes.forEach((note,i) => ctx.fillText(note,x,420+i*24));
  };
  panel(15,'Physical coordinates',r.pins,r.heightWindow,[
    'r=' + numberText(r.r) + '; k=' + numberText(r.k),
    'b=' + numberText(r.b) + '; gap=' + numberText(r.gap)]);
  panel(480,'Nominal affine coordinates',r.normalized.pins,r.normalized.heightWindow,[
    'X=x/r; Y=(y-b)/gap',
    'Retained normalizers appear in the exact table.']);
  ctx.fillStyle='#17212b'; ctx.font='13px system-ui';
  ctx.fillText('Panel positions are illustrative; binary64 endpoints and all normalizers are in the tables.',24,520);
}
function drawElder(ctx,s) {
  const r=s.result, p=s.selected, n=r.n, left=35, top=80, size=440, cell=size/n;
  const point=vertex => [left+(vertex%n+0.5)*cell,top+(Math.floor(vertex/n)+0.5)*cell];
  const active=new Set(p.activeVertices), elders=new Map();
  for (const c of p.components) for (const vertex of c.vertices) elders.set(vertex,c.elder);
  let minimum=r.values[0], maximum=r.values[0];
  for (const value of r.values) {
    if (value<minimum) minimum=value;
    if (value>maximum) maximum=value;
  }
  const edge=(vertices,color,width) => {
    const [a,b]=vertices, [ax,ay]=point(a), [bx,by]=point(b);
    const seamX=Math.abs(a%n-b%n) === n-1;
    const seamY=Math.abs(Math.floor(a/n)-Math.floor(b/n)) === n-1;
    ctx.setLineDash(seamX || seamY ? [4,3] : []);
    if (seamX) {
      line(ctx,ax,ay,ax<bx ? left : left+size,ay,color,width);
      line(ctx,bx,by,bx<ax ? left : left+size,by,color,width);
    } else if (seamY) {
      line(ctx,ax,ay,ax,ay<by ? top : top+size,color,width);
      line(ctx,bx,by,bx,by<ay ? top : top+size,color,width);
    } else line(ctx,ax,ay,bx,by,color,width);
    ctx.setLineDash([]);
  };
  ctx.fillStyle='#17212b'; ctx.font='bold 17px system-ui';
  ctx.fillText('Periodic grid; E' + s.selection.eventEnd + ' / ' + r.events.length,24,28);
  ctx.font='12px system-ui';
  ctx.fillText('Squares active; circles inactive; triangles mark mathematical elders.',24,52);
  ctx.fillText(p.lastEvent ? 'Dashed stubs wrap seams. Last event: ' + p.lastEvent.kind
    + (p.lastEvent.kind === 'merge' ? '; zero=' + p.lastEvent.zero : '')
    : 'Empty insertion prefix; no last event.',24,68);
  for (const e of p.edges) edge(e,'#566675',1.5);
  for (let v=0;v<r.values.length;v++) {
    const [x,y]=point(v), isActive=active.has(v);
    const shade=Math.round(245-55*ratio(r.values[v]-minimum,maximum-minimum));
    if (isActive) {
      ctx.fillStyle='rgb(' + shade + ',' + shade + ',255)';
      ctx.fillRect(x-cell*0.38,y-cell*0.38,cell*0.76,cell*0.76);
      ctx.strokeStyle=COLORS[elders.get(v)%COLORS.length];
      ctx.strokeRect(x-cell*0.38,y-cell*0.38,cell*0.76,cell*0.76);
    } else circle(ctx,x,y,cell*0.28,'white','#9b9b9b');
    if (cell>=20) {
      ctx.fillStyle='#17212b'; ctx.font='10px system-ui';
      ctx.fillText(String(v),x-cell*0.28,y+3);
    }
  }
  for (const c of p.components) {
    const [x,y]=point(c.elder), color=COLORS[c.elder%COLORS.length];
    triangle(ctx,x,y-Math.min(cell*0.30,12),Math.min(cell*0.20,6),color);
    if (cell>=20) {
      ctx.fillStyle='#17212b'; ctx.font='10px system-ui';
      ctx.fillText('e'+c.elder,x-cell*0.3,y+cell*0.3);
    }
  }
  if (p.lastEvent) {
    if (p.lastEvent.edge) edge(p.lastEvent.edge,'#111',3);
    else {
      const [x,y]=point(p.lastEvent.vertex);
      circle(ctx,x,y,Math.min(cell*0.45,15),'rgba(255,255,255,0)','#111');
    }
  }
  const bars=r.barcode.intervals, essential=r.barcode.essential[0];
  let lower=essential, upper=essential;
  for (const [birth,death] of bars) {
    if (death<lower) lower=death;
    if (birth>upper) upper=birth;
  }
  const bx=550, bw=365, by=105, bh=340;
  const position=value => bx+bw*ratio(value-lower,upper-lower);
  ctx.fillStyle='#17212b'; ctx.font='bold 16px system-ui';
  ctx.fillText('Whole-sweep barcode',540,72);
  ctx.font='12px system-ui';
  ctx.fillText('Open death; closed birth. Essential class is separate.',540,92);
  const spacing=bh/Math.max(1,bars.length);
  bars.forEach(([birth,death],i) => {
    const y=by+(i+0.5)*spacing;
    const color=COLORS[r.barcode.birth_vertices[i]%COLORS.length];
    line(ctx,position(death),y,position(birth),y,color,Math.min(3,spacing*0.6));
    if (spacing>=7) {
      circle(ctx,position(death),y,3,'white');
      circle(ctx,position(birth),y,3,color);
    }
    if (spacing>=12) {
      ctx.fillStyle='#17212b'; ctx.font='11px system-ui';
      ctx.fillText('v'+r.barcode.birth_vertices[i],bx-40,y+4);
    }
  });
  const ey=480, ex=position(essential);
  line(ctx,bx,ey,ex,ey,'#111',2);
  line(ctx,bx,ey,bx+8,ey-5,'#111',2); line(ctx,bx,ey,bx+8,ey+5,'#111',2);
  circle(ctx,ex,ey,4,'#111');
  ctx.fillStyle='#17212b'; ctx.font='12px system-ui';
  ctx.fillText('Essential: (-infinity, ' + shortInteger(essential) + '/'+shortInteger(r.scale)+']',540,505);
  ctx.fillText('Zero pairs: ' + r.barcode.zero_count + '; finite bars: ' + bars.length,540,530);
  ctx.fillText('All bars drawn; dense bars overlap. Exact full fractions remain in the tables.',24,560);
  ctx.fillText('Geometry uses bounded ratios of exact differences; long drawing labels are abbreviated.',24,580);
}
function stageCanvas(doc,s) {
  try {
    const canvas=doc.createElement('canvas');
    canvas.width=960; canvas.height=600;
    canvas.style.width='100%'; canvas.style.maxWidth='960px'; canvas.style.height='auto';
    canvas.tabIndex=0; canvas.setAttribute('role','img');
    canvas.setAttribute('aria-describedby','reader-canvas-note');
    canvas.setAttribute('aria-label',s.model === 'elder'
      ? 'Approximate periodic grid and complete barcode at prefix ' + s.selection.eventEnd
      : 'Physical and nominal cubic teaching-coordinate panels');
    canvas.dataset.readerFocus='drawing';
    canvas.textContent='The complete exact native tables follow this drawing.';
    const ctx=canvas.getContext('2d');
    if (!ctx) return {canvas:null,error:'Canvas2D unavailable; exact tables retained.'};
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='white'; ctx.fillRect(0,0,canvas.width,canvas.height);
    if (s.model === 'cubic') drawCubic(ctx,s); else drawElder(ctx,s);
    if (ctx.isContextLost?.())
      return {canvas:null,error:'Drawing context lost; exact current tables retained.'};
    canvas.dataset.readerRevision=s.revision.toString();
    canvas.dataset.documentRevision=s.documentRevision.toString();
    return {canvas,error:null};
  } catch {
    return {canvas:null,error:'Drawing failed; exact current tables retained.'};
  }
}

export function mountReader(container,controller) {
  const doc=container?.ownerDocument;
  if (!doc || typeof container.appendChild !== 'function'
      || typeof container.replaceChildren !== 'function'
      || typeof controller?.current !== 'function'
      || typeof controller?.prepareSelection !== 'function'
      || typeof controller?.commit !== 'function')
    throw new TypeError('A real DOM container and native reader controller are required');
  if (mounted.has(container)) throw new TypeError('This container already owns a reader');
  const root=doc.createElement('div'), canvasHost=doc.createElement('div');
  const textHost=doc.createElement('div'), live=doc.createElement('p');
  const redraw=doc.createElement('button');
  root.className='native-canvas-reader'; root.setAttribute('role','region');
  root.tabIndex=-1; root.dataset.readerFocus='reader-region';
  root.style.minWidth='0'; root.style.maxWidth='100%';
  root.style.font='1rem/1.5 system-ui,sans-serif';
  root.setAttribute('aria-label','Finite reader drawing and exact tables');
  live.setAttribute('role','status'); live.setAttribute('aria-live','polite');
  live.setAttribute('aria-atomic','true');
  redraw.type='button'; redraw.textContent='Refresh current drawing';
  redraw.style.font='inherit';
  redraw.dataset.readerAction='refresh-drawing'; redraw.dataset.readerFocus='refresh-drawing';
  root.append(redraw,canvasHost,textHost,live);
  let disposed=false, attached=false, rendered=null, detachCanvas=() => {};
  const media=doc.defaultView?.matchMedia?.('(prefers-reduced-motion: reduce)') || null;
  const reducedMotion=() => {
    root.dataset.reducedMotion=media ? String(media.matches) : 'unknown';
  };
  const announce=message => {live.textContent=message;};
  const clearCanvas=() => {
    detachCanvas(); detachCanvas=() => {};
    try {
      for (const canvas of canvasHost.querySelectorAll('canvas')) canvas.width=canvas.width;
    } finally {canvasHost.replaceChildren();}
  };
  const focusKey=() => {
    const active=doc.activeElement;
    return root.contains(active) ? active?.dataset?.readerFocus || null : null;
  };
  const restoreFocus=key => {
    if (!key) return;
    const target=[...root.querySelectorAll('[data-reader-focus]')]
      .find(el => el.dataset.readerFocus === key && !el.disabled)
      || root.querySelector('input[data-reader-action="event-range"]')
      || canvasHost.querySelector('canvas') || root;
    if (target) {
      try {target.focus({preventScroll:true});} catch {target.focus();}
    }
  };
  const stage=s => {
    if (s === null) {
      const empty=doc.createElement('p');
      empty.textContent='Awaiting a trusted native document. The view does not run models.';
      return {snapshot:null,fragment:empty,canvas:null,error:null};
    }
    const html=renderStaticReader(s), template=doc.createElement('template');
    template.innerHTML=html;
    const drawing=stageCanvas(doc,s);
    return {snapshot:s,fragment:template.content,canvas:drawing.canvas,error:drawing.error};
  };
  const publish=(prepared,key) => {
    textHost.replaceChildren(prepared.fragment);
    // Removing the old canvas also removes stale paint when drawing is unavailable.
    clearCanvas();
    if (prepared.canvas) {
      const canvas=prepared.canvas;
      canvasHost.appendChild(canvas);
      const lost=() => {
        if (disposed) return;
        const key=focusKey();
        clearCanvas();
        const note=doc.createElement('p');
        note.textContent='Drawing context lost; exact current tables retained. Refresh to draw again.';
        canvasHost.appendChild(note);
        restoreFocus(key);
        announce(note.textContent);
      };
      canvas.addEventListener('contextlost',lost);
      detachCanvas=() => canvas.removeEventListener('contextlost',lost);
    }
    else if (prepared.error) {
      const note=doc.createElement('p'); note.textContent=prepared.error;
      canvasHost.appendChild(note);
    }
    rendered=prepared.snapshot;
    root.dataset.readerRevision=rendered ? rendered.revision.toString() : 'empty';
    root.dataset.documentRevision=rendered ? rendered.documentRevision.toString() : 'empty';
    redraw.disabled=rendered === null;
    if (!attached) {
      // Trusted native state replaces built fallback HTML; display text is never input.
      if (rendered === null && container.childNodes.length > 0) {
        if (root.parentNode !== container) container.appendChild(root);
      } else {container.replaceChildren(root); attached=true;}
    }
    reducedMotion();
    restoreFocus(key);
    announce(prepared.error || (rendered
      ? 'Paused ' + rendered.model + ' reader, revision ' + rendered.revision
        + (rendered.model === 'elder' ? ', prefix ' + rendered.selection.eventEnd : '') + '.'
      : 'Reader has no document.'));
  };
  const refresh=() => {
    if (disposed) return false;
    const current=controller.current(), key=focusKey();
    let prepared;
    try {prepared=stage(current);} catch {
      clearCanvas(); announce('Text rendering failed; reader update not published.');
      return false;
    }
    if (controller.current() !== current) {
      clearCanvas(); announce('Reader changed during staging; refresh the current document.');
      return false;
    }
    publish(prepared,key);
    return true;
  };
  const actionIsCurrent=() => {
    if (disposed) return false;
    if (controller.current() !== rendered) {
      refresh(); announce('Document changed; choose the action again.'); return false;
    }
    return true;
  };
  const choose=selection => {
    if (!actionIsCurrent()) return false;
    const key=focusKey(), ready=controller.prepareSelection(selection);
    if (ready.kind !== 'Ready') {
      refresh(); announce('Selection refused; committed reader retained.'); return false;
    }
    let prepared;
    try {prepared=stage(ready.snapshot);} catch {
      refresh(); announce('Text rendering failed; committed reader retained.'); return false;
    }
    const committed=controller.commit(ready.ticket);
    if (committed !== ready.snapshot) {
      refresh(); announce('Selection became stale; current committed reader retained.'); return false;
    }
    publish(prepared,key);
    return true;
  };
  const navigate=action => {
    if (!actionIsCurrent()) return;
    const s=controller.current();
    if (!s || s.model !== 'elder') return;
    const fields={'previous-event':'previousEvent','next-event':'nextEvent',
      'previous-level':'previousLevel','next-level':'nextLevel',reset:'reset'};
    const target=s.navigation[fields[action]];
    if (target === null || target === undefined) return;
    if (action === 'reset') choose({mode:'reset'});
    else if (action.endsWith('level')) choose({mode:'complete_level',levelIndex:target});
    else choose({mode:'event_prefix',eventEnd:target});
  };
  const click=event => {
    const target=event.target?.closest?.('[data-reader-action],a[data-reader-event-end],a[data-reader-level-index]');
    if (!target || !root.contains(target) || target.disabled) return;
    if (target.tagName === 'A') {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
      const accepted=target.dataset.readerLevelIndex !== undefined
        ? choose({mode:'complete_level',levelIndex:Number(target.dataset.readerLevelIndex)})
        : choose({mode:'event_prefix',eventEnd:Number(target.dataset.readerEventEnd)});
      if (!accepted) event.preventDefault();
      // Keep native href behavior: each exact event/block target really exists.
    } else if (target.tagName === 'BUTTON') {
      if (target.dataset.readerAction === 'refresh-drawing') refresh();
      else navigate(target.dataset.readerAction);
    }
  };
  const change=event => {
    const target=event.target;
    if (!root.contains(target)) return;
    if (target.dataset?.readerAction === 'event-range')
      choose({mode:'event_prefix',eventEnd:Number(target.value)});
    else if (target.dataset?.readerAction === 'level-select' && target.value !== '')
      choose({mode:'complete_level',levelIndex:Number(target.value)});
  };
  const keydown=event => {
    if (event.target?.dataset?.readerFocus !== 'drawing'
        || event.altKey || event.ctrlKey || event.metaKey) return;
    const actions={ArrowLeft:'previous-event',ArrowRight:'next-event',
      PageUp:'previous-level',PageDown:'next-level',Home:'reset'};
    if (!Object.hasOwn(actions,event.key) && event.key !== 'End') return;
    if (!actionIsCurrent()) {event.preventDefault(); return;}
    const s=controller.current();
    if (!s || s.model !== 'elder') return;
    event.preventDefault();
    if (event.key === 'End')
      choose({mode:'event_prefix',eventEnd:s.result.events.length});
    else navigate(actions[event.key]);
  };
  const dispose=() => {
    if (disposed) return;
    disposed=true;
    root.removeEventListener('click',click);
    root.removeEventListener('change',change);
    root.removeEventListener('keydown',keydown);
    if (typeof media?.removeEventListener === 'function')
      media.removeEventListener('change',reducedMotion);
    else media?.removeListener?.(reducedMotion);
    try {clearCanvas();} finally {
      root.remove(); mounted.delete(container); rendered=null;
    }
  };
  root.addEventListener('click',click);
  root.addEventListener('change',change);
  root.addEventListener('keydown',keydown);
  if (typeof media?.addEventListener === 'function')
    media.addEventListener('change',reducedMotion);
  else media?.addListener?.(reducedMotion);
  mounted.set(container,true);
  reducedMotion();
  try {
    if (!refresh() && !attached) container.appendChild(root);
  } catch (error) {dispose(); throw error;}
  return Object.freeze({refresh,dispose});
}
