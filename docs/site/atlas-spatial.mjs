import * as THREE from './vendor/three-0.160.0/three.module.js?site-release=f2988301e38ea3d44631f3220044accc019d63ae79b737e79b7b8ab7a2d5f778';

// An on-demand renderer: no animation loop, damping, physics or auto-rotation.
export function createSpatialRenderer({host,onPick,onFailure}) {
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(50,1,.1,3000);
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));host.append(renderer.domElement);
  const canvas=renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('role','img');canvas.setAttribute('aria-label','3D dependency graph. Arrow keys orbit, Shift and arrows pan, plus and minus zoom, Home resets. Use the node list to select with a keyboard.');
  const labels=document.createElement('div');labels.className='atlas-space-labels';host.append(labels);
  const group=new THREE.Group();scene.add(group);const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
  let meshes=[],labelNodes=[],radius=32,fittedRadius=32,theta=.12,phi=1.1,target=new THREE.Vector3(),paused=false,disposed=false,frame=0,drag=null,draws=0;
  const cleanup=[];
  function listen(node,type,fn,options){node.addEventListener(type,fn,options);cleanup.push(()=>node.removeEventListener(type,fn,options));}
  function clear(){for(const child of [...group.children]){child.geometry?.dispose();if(Array.isArray(child.material))child.material.forEach(m=>m.dispose());else child.material?.dispose();group.remove(child);}meshes=[];labelNodes=[];labels.replaceChildren();}
  function schedule(){if(!disposed&&!paused&&!document.hidden&&!frame)frame=requestAnimationFrame(draw);}
  function draw(){frame=0;if(disposed||paused||document.hidden||!host.getClientRects().length)return;
    const started=performance.now();
    const {width,height}=host.getBoundingClientRect();if(width<1||height<1)return;
    renderer.setSize(width,height,false);camera.aspect=width/height;camera.position.set(target.x+radius*Math.sin(phi)*Math.sin(theta),target.y+radius*Math.cos(phi),target.z+radius*Math.sin(phi)*Math.cos(theta));camera.lookAt(target);camera.updateProjectionMatrix();camera.updateMatrixWorld();
    renderer.render(scene,camera);draws++;host.dataset.frames=String(draws);
    // Keep selected + nearest readable labels; the complete list remains below.
    const placed=[];
    for(const {node,position,selected}of labelNodes){const p=position.clone().project(camera);const anchorX=(p.x+1)*width/2,y=(1-p.y)*height/2;const w=Math.min(node.offsetWidth||230,width-8),h=38;const x=Math.max(w/2+4,Math.min(width-w/2-4,anchorX));const box={x:x-w/2,y:y-36,w,h};const visible=p.z<1&&p.z>-1&&x>0&&x<width&&y>35&&y<height-10&&(selected||!placed.some(b=>Math.abs(box.x-b.x)<(box.w+b.w)/2&&Math.abs(box.y-b.y)<h));node.hidden=!visible;if(visible){node.style.left=`${x}px`;node.style.top=`${y-18}px`;placed.push(box);}}
    host.dataset.ready='true';host.dataset.renderMs=String(performance.now()-started);
  }
  function reset(){target.set(0,0,0);theta=.12;phi=1.1;radius=fittedRadius;schedule();}
  function zoom(factor){radius=THREE.MathUtils.clamp(radius*factor,5,500);schedule();}
  function pan(x,y){const right=new THREE.Vector3().setFromMatrixColumn(camera.matrix,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);target.addScaledVector(right,-x*radius*.0015).addScaledVector(up,y*radius*.0015);}
  function update(view,selected){clear();const bounds=new THREE.Box3();
    for(const id of view.ids){const position=new THREE.Vector3(...view.positions[id]);bounds.expandByPoint(position);const mesh=new THREE.Mesh(new THREE.SphereGeometry(id===selected?.5:.33,12,8),new THREE.MeshBasicMaterial({color:id===selected?0xf4c76b:0x72bded}));mesh.position.copy(position);mesh.userData.id=id;group.add(mesh);meshes.push(mesh);
      const label=document.createElement('button');label.type='button';label.textContent=id;label.tabIndex=-1;label.setAttribute('aria-hidden','true');label.className=id===selected?'selected':'';label.addEventListener('click',()=>onPick(id));labels.append(label);labelNodes.push({node:label,position,selected:id===selected});}
    labelNodes.sort((a,b)=>Number(b.selected)-Number(a.selected));
    for(const edge of view.edges){const a=new THREE.Vector3(...view.positions[edge.from]),b=new THREE.Vector3(...view.positions[edge.to]),direction=b.clone().sub(a),length=direction.length();if(length<.01)continue;direction.normalize();const end=b.clone().addScaledVector(direction,-.6),start=a.clone().addScaledVector(direction,.6);
      const geometry=new THREE.BufferGeometry().setFromPoints([start,end]);const material=edge.required?new THREE.LineBasicMaterial({color:0x91b9db}):new THREE.LineDashedMaterial({color:0x91b9db,dashSize:.25,gapSize:.2});const line=new THREE.Line(geometry,material);line.computeLineDistances();group.add(line);
      const cone=new THREE.Mesh(new THREE.ConeGeometry(.14,.45,7),new THREE.MeshBasicMaterial({color:0x91b9db}));cone.position.copy(end).addScaledVector(direction,-.22);cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction);group.add(cone);}
    fittedRadius=bounds.isEmpty()?32:Math.max(22,bounds.getSize(new THREE.Vector3()).length()*1.1);
    if(!host.dataset.ready)radius=fittedRadius;
    host.dataset.nodes=String(view.ids.length);host.dataset.edges=String(view.edges.length);schedule();
  }
  listen(canvas,'pointerdown',e=>{if(e.button!==0&&e.button!==2)return;drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};canvas.setPointerCapture(e.pointerId);});
  listen(canvas,'pointermove',e=>{if(!drag)return;const x=e.clientX-drag.x,y=e.clientY-drag.y;if(drag.pan)pan(x,y);else{theta-=x*.007;phi=THREE.MathUtils.clamp(phi+y*.007,.1,Math.PI-.1);}drag.x=e.clientX;drag.y=e.clientY;schedule();});
  listen(canvas,'pointerup',e=>{if(!drag)return;const moved=Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY);drag=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);if(moved>5)return;const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(meshes)[0];if(hit)onPick(hit.object.userData.id);});
  listen(canvas,'pointercancel',()=>{drag=null;});listen(canvas,'contextmenu',e=>e.preventDefault());
  listen(canvas,'wheel',e=>{e.preventDefault();zoom(e.deltaY>0?1.12:1/1.12);},{passive:false});
  listen(canvas,'keydown',e=>{const keys=['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'];if(keys.includes(e.key)){e.preventDefault();const x=e.key==='ArrowLeft'?-20:e.key==='ArrowRight'?20:0,y=e.key==='ArrowUp'?-20:e.key==='ArrowDown'?20:0;if(e.shiftKey)pan(x,y);else{theta-=x*.007;phi=THREE.MathUtils.clamp(phi+y*.007,.1,Math.PI-.1);}schedule();}else if(['+','=','-','Home'].includes(e.key)){e.preventDefault();if(e.key==='Home')reset();else zoom(e.key==='-'?1.2:1/1.2);}});
  listen(canvas,'webglcontextlost',e=>{e.preventDefault();onFailure('WebGL context lost. The graph and list remain available; activate 3D again to retry.');});
  listen(document,'visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;}else schedule();});
  const resize=new ResizeObserver(schedule);resize.observe(host);
  return {update,reset,zoom,pause(value){paused=value;if(value){cancelAnimationFrame(frame);frame=0;}else schedule();},dispose(){if(disposed)return;disposed=true;cancelAnimationFrame(frame);resize.disconnect();cleanup.forEach(fn=>fn());clear();renderer.dispose();renderer.forceContextLoss();canvas.remove();labels.remove();}};
}
