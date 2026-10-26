// Screen-space presentation only: never changes source IDs, graph edges or nodes.
export function placeSpatialLabels(entries, width, height) {
  const margin=4,gap=4;
  if(!Number.isFinite(width)||!Number.isFinite(height)||width<=margin*2||height<=margin*2)return [];
  const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
  const overlaps=(a,b,padding=0)=>a.left<b.left+b.width+padding&&a.left+a.width+padding>b.left&&a.top<b.top+b.height+padding&&a.top+a.height+padding>b.top;
  const candidates=entries.filter(e=>typeof e.id==='string'&&[e.x,e.y,e.z,e.width,e.height].every(Number.isFinite)&&e.z>-1&&e.z<1&&e.x>=0&&e.x<=width&&e.y>=0&&e.y<=height&&e.width>0&&e.height>0&&e.width<=width-margin*2&&e.height<=height-margin*2)
    .sort((a,b)=>Number(Boolean(b.selected))-Number(Boolean(a.selected))||a.z-b.z||(a.id<b.id?-1:a.id>b.id?1:0));
  const placed=[];
  for(const e of candidates){
    const positions=[[e.x-e.width/2,e.y-e.height-12],[e.x-e.width/2,e.y+12],[e.x+12,e.y-e.height/2],[e.x-e.width-12,e.y-e.height/2]];
    for(const [x,y] of positions){
      const box={id:e.id,left:clamp(x,margin,width-margin-e.width),top:clamp(y,margin,height-margin-e.height),width:e.width,height:e.height};
      if(overlaps(box,{left:e.x-5,top:e.y-5,width:10,height:10})||placed.some(other=>overlaps(box,other,gap)))continue;
      placed.push(box);break;
    }
  }
  return placed;
}
