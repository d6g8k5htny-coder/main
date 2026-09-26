import {verifiedJSON, verifiedBytes, validateStatus, validateCoefficient, safeSourceURL, hex40, hex64} from './core.mjs';
const el=id=>document.getElementById(id);
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const error=(id,e)=>{el(id).textContent=`Unavailable: ${e.message}. No result inferred.`;el(id).className='error';};
const link=(text,url)=>{const a=node('a',text);a.href=url;return a;};
// Small safe renderer: links to this public GitHub owner; no HTML execution.
function cell(target,text,base) {
  const re=/\[([^\]]+)\]\(([^)]+)\)/g;let start=0;
  for(const match of text.matchAll(re)) {
    target.append(document.createTextNode(text.slice(start,match.index).replace(/\*\*|`/g,'')));
    let url;try{url=new URL(match[2],base);}catch{url=null;}
    if(url?.origin==='https://github.com' && url.pathname.startsWith('/d6g8k5htny-coder/'))target.append(link(match[1],url.href));
    else target.append(document.createTextNode(match[1]));
    start=match.index+match[0].length;
  }
  target.append(document.createTextNode(text.slice(start).replace(/\*\*|`/g,'')));
}
function identity(target,pin) {
  for(const [label,value] of [['Commit',pin.commit],['Path',pin.path],['SHA-256',pin.sha256],['Bytes',pin.bytes]]) {
    const p=node('p');p.append(node('strong',label+': '),node('code',String(value)));target.append(p);
  }
  const url=pin.url||pin.raw_url;if(url)target.append(link('Read the pinned public bytes ↗',url));
}
async function board(config) {
  const s=validateStatus(await verifiedJSON(config.status_json));
  await verifiedBytes(s.source);
  const base=`https://github.com/d6g8k5htny-coder/main/blob/${s.source.commit}/`;
  el('status-note').textContent=`Snapshot ${s.snapshot_date}. Counts cover the selected rows below, not every theorem or artifact. ${s.meaning}`;
  const labels={accept:'ACCEPT — scoped',amend:'AMEND / open',engineering:'Engineering only'};
  for(const section of s.sections) {
    const card=node('article',undefined,section.key);card.append(node('span',String(section.count),'count'),node('h3',labels[section.key]),node('p','Selected source rows; read the scope below.'));el('counts').append(card);
    const details=node('details',undefined,'status-group');details.append(node('summary',`${labels[section.key]} · ${section.count}`));
    for(const row of section.rows) {
      const article=node('article',undefined,'status-object');const title=node('h4');cell(title,row[0],base);article.append(title);
      row.slice(1).forEach((c,i)=>{const p=node('p');p.append(node('strong',section.headers[i+1]+': '));cell(p,c,base);article.append(p);});details.append(article);
    }el('status-rows').append(details);
  }
  const details=node('details');details.append(node('summary','Status source identity'));identity(details,s.source);el('status-rows').append(details);
}
function svgElement(tag,attributes,text) {const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attributes).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text)n.textContent=text;return n;}
function drawCoefficient(data,dimension) {
  const selected=data.dimensions[dimension];el('lower').textContent=selected.lower;el('upper').textContent=selected.upper;el('moment').textContent=data.cone_moments[dimension];
  const svg=el('coefficient-chart');svg.replaceChildren();
  svg.append(svgElement('line',{x1:80,y1:175,x2:475,y2:175,stroke:'#829793'}));
  ['2','3'].forEach((d,i)=>{const value=Number(data.dimensions[d].lower),y=40+i*65;
    svg.append(svgElement('text',{x:8,y:y+23,fill:'#304e55','font-size':15},`d = ${d}`));
    svg.append(svgElement('rect',{x:80,y,width:380*value/0.08,height:32,rx:3,fill:d===dimension?'#2b7565':'#adc6ba'}));
    svg.append(svgElement('text',{x:80,y:y+49,fill:'#526b70','font-size':12},`≈ ${value.toFixed(6)} (display)`));
  });
  for(const [x,text] of [[80,'0'],[270,'0.04'],[460,'0.08']])svg.append(svgElement('text',{x,y:201,fill:'#526b70','font-size':12},text));
}
async function coefficients(config) {
  const data=validateCoefficient(await verifiedJSON(config.coefficient));
  el('coefficient-state').textContent='SHA-256 and byte count verified. Exact endpoints shown verbatim.';
  el('proof-link').href=`https://github.com/d6g8k5htny-coder/Math-/blob/${config.proof.commit}/${config.proof.path}`;
  identity(el('coefficient-identity'),config.coefficient);el('coefficient-identity').append(node('p',data.method),node('p',`Original source scope: ${data.scope}`));
  drawCoefficient(data,el('dimension').value);el('dimension').addEventListener('change',()=>drawCoefficient(data,el('dimension').value));
}
async function inventory(config) {
  const index=await verifiedJSON(config.inventory);
  if(index.source_count!==2138||index.pages.length!==config.inventory.pages.length)throw new Error('Catalog index mismatch');
  const chunks=await Promise.all(config.inventory.pages.map(async (pin,i)=>{
    if(index.pages[i].path!==pin.path.split('/').at(-1)||index.pages[i].count!==pin.count)throw new Error('Shard selector mismatch');
    const data=await verifiedJSON(pin);
    if(!Array.isArray(data.sources)||data.sources.length!==pin.count)throw new Error('Shard count mismatch');
    data.sources.forEach(row=>{safeSourceURL(row);if(!Number.isSafeInteger(row.bytes)||row.bytes<0||!hex40.test(row.blob))throw new Error('Catalog row identity missing');});return data.sources;
  }));
  const rows=chunks.flat();if(rows.length!==index.source_count)throw new Error('Catalog total mismatch');
  let limit=50;
  const render=()=>{const query=el('search').value.trim().toLowerCase();const matches=rows.filter(row=>[row.repository,row.path,row.commit,row.sha256].some(s=>s.toLowerCase().includes(query)));el('catalog').replaceChildren();
    matches.slice(0,limit).forEach(row=>{const tr=node('tr'),name=node('td'),commit=node('td'),hash=node('td');name.append(node('small',row.repository),link(row.path,safeSourceURL(row)));commit.append(node('code',row.commit));hash.append(node('code',row.sha256));tr.append(name,commit,hash);el('catalog').append(tr);});
    el('inventory-state').textContent=`${matches.length.toLocaleString()} matches · ${Math.min(limit,matches.length)} shown · all 2,138 source records loaded from the original hash-verified shards.`;el('more').hidden=matches.length<=limit;};
  el('search').disabled=false;el('search').addEventListener('input',()=>{limit=50;render();});el('more').addEventListener('click',()=>{limit+=50;render();});render();
}
async function custody(config) {
  const manifest=await verifiedJSON(config.imports);
  if(manifest.scientific_effect!=='NONE'||manifest.byte_copies.length!==config.imports.count||manifest.byte_copies.some(row=>row.kind!=='BYTE_COPY'||row.source_label_adopted!==false))throw new Error('Import custody mismatch');
  const observed=config.observations;
  el('custody-note').textContent=`${config.imports.count} byte-copy imports landed at Math ${config.imports.commit}; source labels were not adopted. ${observed.open_math_prs} Math PRs were open when observed ${observed.observed_at}. An open PR is not landed math.`;
  el('custody-note').append(' ',link('Import identities ↗',config.imports.manifest_url));
  const q=config.query;const bundle=await verifiedJSON(q);if(bundle.math_tip!==q.math_pin||bundle.scientific_status_authority!==false)throw new Error('Query source mismatch');el('query-note').textContent=`Query checkout ${q.commit}; recorded Math commit ${q.math_pin}. --check-math-tip checks seven file identities, not commit equality. The button below checks commit currency only.`;
  if(!hex40.test(q.math_pin)||!hex40.test(q.commit))throw new Error('Query pin unavailable');
  el('check-tip').disabled=false;el('check-tip').addEventListener('click',async()=>{el('check-tip').disabled=true;el('tip-result').textContent='Reading current public Math ref…';
    try{const r=await fetch('https://api.github.com/repos/d6g8k5htny-coder/Math-/git/ref/heads/main',{credentials:'omit',redirect:'error'});if(!r.ok)throw new Error(`GitHub API ${r.status}`);const ref=await r.json(),tip=ref.object?.sha;if(!hex40.test(tip)||ref.ref!=='refs/heads/main')throw new Error('Unexpected ref response');const same=tip===q.math_pin;el('tip-result').className=same?'good':'error';el('tip-result').textContent=`${same?'Commit pin current':'Commit pin differs'} at ${new Date().toISOString()}: Math ${tip}. Gate-byte equality was not tested by this browser check.`;
    }catch(e){error('tip-result',e);}finally{el('check-tip').disabled=false;}});
}
try {
  const response=await fetch('config.json',{credentials:'omit'});if(!response.ok)throw new Error('Shop config unavailable');const config=await response.json();
  const jobs=[['status-note',()=>board(config)],['coefficient-state',()=>coefficients(config)],['inventory-state',()=>inventory(config)],['custody-note',()=>custody(config)]];
  await Promise.all(jobs.map(async([id,job])=>{try{await job();}catch(e){error(id,e);}}));
} catch(e) {['status-note','coefficient-state','inventory-state','custody-note'].forEach(id=>error(id,e));}
