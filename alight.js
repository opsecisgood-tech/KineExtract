// vid2anim → Alight Motion exporter (.xml project). Default look: white stickman, thin black edge, square head.
// Pure browser, no dependencies. Wired up from app.js with initAlight(ctx).
//
// Export pipeline (v6):
//   1. Layout model   every body part becomes a thin bar (centre, length, angle) on the output canvas.
//   2. CHECK 1        scans the analysed motion for bars that overlap where they should not
//                     (crossing bars, a bar lying on top of its neighbour, duplicate "helper" bars).
//   3. Build the .xml (optionally after the auto-fix: duplicates left out, parts nudged apart).
//   4. CHECK 2        reads the finished .xml back, validates it, checks that it moves like the analysis
//                     and runs the overlap scan again on what Alight Motion will really draw.
//   5. CHECK 3        you look at the preview and confirm (manual check) before the file can be saved.
//   If something is wrong you can let the auto-fix move parts apart or move/hide parts yourself.
//
// v7: every shape now carries an explicit <size> (half-extent) tag alongside <scale>. Without it, Alight Motion fell
// back to its own default half-extent instead of the one our Scale math assumes, which is what made every bar come
// out a consistent fraction of its intended length ("bunched" limbs) once a real export was opened in the app,
// even though our own preview — built from the same geometry model — looked right the whole time. Default size is
// 50 (so a rect's drawn size is 2×50×scale = 100×scale, matching the existing "/100" convention used everywhere
// below); "Shape size" in Compatibility exposes it in case a future AM build changes the real default.

/* ───────── helpers ───────── */
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const f6=v=>(Number.isFinite(v)?v:0).toFixed(6);
const num=v=>{const s=(+v).toFixed(8);return s.replace(/(\.\d*?)0+$/,'$1').replace(/\.$/,'.0')};
const AM_FPS=[12,15,18,20,24,25,30,48,50,60];
const even=n=>Math.max(2,Math.round(n/2)*2);
const LINEAR=[0,0,1,1];
const isLin=e=>Array.isArray(e)&&e.every((v,i)=>Math.abs(v-LINEAR[i])<1e-6);
const easeStr=e=>e&&!isLin(e)?`cubicBezier ${e.map(num).join(' ')}`:'';
const sideColor=d=>d.side==='L'?'#FF3DE8FF':d.side==='R'?'#FFFF8A4C':'#FFCFD3D9';
const thickBase=id=>id==='head'?16:/tor|pelvis|neck/.test(id)?14:9; // same proportions as the Rig preview (480 px wide)
const clamp01=v=>v<0?0:v>1?1:v;
const HELPER=/^[LR](knee|ank|wr|sh)$/;            // analysis-only bars (chords) that duplicate other bars
const HEAVY=/^(head|neck|torso|utor3?|ltor3?|mtor|pelvis)$/; // parts that move less when bars are pushed apart

/* ───────── key planning ─────────
   A transform channel (location = x+y, scale = len, rotation = rot) can only have ONE set of keyframes in Alight Motion.
   x and y have their own keyframes in vid2anim, so we take the union of their times. Where the original easing can't be carried over
   exactly (different curves on x and y, elastic/bounce, or a union key that cuts an eased segment in two) the segment is baked
   with one key per frame, which reproduces the curve exactly. */
function planKeys(arrs,n,dense){
  if(dense)return Array.from({length:n},(_,i)=>({i,e:null}));
  const set=new Set();for(const k of arrs)if(k)for(const q of k)set.add(q.i);
  if(!set.size)set.add(0);
  const idx=[...set].sort((a,b)=>a-b),out=[];
  for(let j=0;j<idx.length;j++){
    const a=idx[j],b=idx[j+1];
    if(b===undefined){out.push({i:a,e:null});break}
    const classes=new Set();let single=null;
    for(const k of arrs){
      if(!k||k.length<2)continue;
      let qi=-1;for(let q=0;q<k.length-1;q++)if(k[q].i<=a&&k[q+1].i>=b){qi=q;break}
      if(qi<0)continue;
      const s=k[qi],t=k[qi+1];
      if(Math.abs(t.v-s.v)<1e-9)continue;
      const exact=s.i===a&&t.i===b,e=s.e;
      if(typeof e==='string'){classes.add('S:'+e);single=e;continue}
      if(isLin(e)){classes.add('lin');continue}
      if(!exact){classes.add('partial');continue}
      classes.add('B:'+e.join(','));single=e;
    }
    if(classes.size===0||(classes.size===1&&classes.has('lin'))){out.push({i:a,e:null});continue}
    if(classes.size===1&&[...classes][0].startsWith('B:')){out.push({i:a,e:single});continue}
    if(classes.size===1&&[...classes][0]==='S:hold'){out.push({i:a,e:null});out.push({i:b-.05,e:null,holdFrom:a});continue}
    out.push({i:a,e:null});for(let f=a+1;f<b;f++)out.push({i:f,e:null}); // bake
  }
  return out;
}

/* ───────── geometry ───────── */
// closest points between two segments {x0,y0,x1,y1}; distance 0 when they cross
function segSeg(P,Q){
  const d1x=P.x1-P.x0,d1y=P.y1-P.y0,d2x=Q.x1-Q.x0,d2y=Q.y1-Q.y0,rx=P.x0-Q.x0,ry=P.y0-Q.y0,
    a=d1x*d1x+d1y*d1y,e=d2x*d2x+d2y*d2y,f=d2x*rx+d2y*ry,EPS=1e-9;let s,t;
  if(a<=EPS&&e<=EPS){s=t=0}
  else if(a<=EPS){s=0;t=clamp01(f/e)}
  else{const c=d1x*rx+d1y*ry;
    if(e<=EPS){t=0;s=clamp01(-c/a)}
    else{const b=d1x*d2x+d1y*d2y,den=a*e-b*b;
      s=den>EPS?clamp01((b*f-c*e)/den):0;t=(b*s+f)/e;
      if(t<0){t=0;s=clamp01(-c/a)}else if(t>1){t=1;s=clamp01((b-c)/a)}}}
  const ax=P.x0+d1x*s,ay=P.y0+d1y*s,bx=Q.x0+d2x*t,by=Q.y0+d2y*t;
  return{d:Math.hypot(ax-bx,ay-by),ax,ay,bx,by};
}
function ptSeg(px,py,Q){
  const dx=Q.x1-Q.x0,dy=Q.y1-Q.y0,l=dx*dx+dy*dy,t=l>1e-9?clamp01(((px-Q.x0)*dx+(py-Q.y0)*dy)/l):0;
  return Math.hypot(px-(Q.x0+dx*t),py-(Q.y0+dy*t));
}
// a bar is {id,cx,cy,hl,ux,uy,r,x0,y0,x1,y1}: centre line of half length hl (direction ux,uy) and half width r
function mkSeg(id,cx,cy,hl,ux,uy,r){return{id,cx,cy,hl,ux,uy,r,x0:cx-ux*hl,y0:cy-uy*hl,x1:cx+ux*hl,y1:cy+uy*hl}}
const ends=A=>[[A.x0,A.y0],[A.x1,A.y1]];
// the joint of two bars = where an end of one bar touches the other. For directly connected bars (tee=true) an end may also touch the
// middle of the other bar (a T, like the torso meeting the pelvis); bars two joints apart only get an end-to-end joint.
function joint(A,B,tee){let best=1e18,J=[A.cx,A.cy];
  if(tee){
    const probe=(X,Y)=>{for(const p of ends(X)){const d=ptSeg(p[0],p[1],Y);if(d<best){best=d;
      const dx=Y.x1-Y.x0,dy=Y.y1-Y.y0,l=dx*dx+dy*dy,t=l>1e-9?clamp01(((p[0]-Y.x0)*dx+(p[1]-Y.y0)*dy)/l):0;J=[(p[0]+Y.x0+dx*t)/2,(p[1]+Y.y0+dy*t)/2]}}};
    probe(A,B);probe(B,A);
  }else for(const p of ends(A))for(const q of ends(B)){const d=Math.hypot(p[0]-q[0],p[1]-q[1]);if(d<best){best=d;J=[(p[0]+q[0])/2,(p[1]+q[1])/2]}}
  return{J,d:best}}
// length of A's centre line that lies inside B but farther than `ex` from the joint J
function excess(A,B,J,ex,thr,step){let n=0;
  for(let t=-A.hl;t<=A.hl+1e-6;t+=step){const px=A.cx+A.ux*t,py=A.cy+A.uy*t;
    if(ptSeg(px,py,B)<thr&&Math.hypot(px-J[0],py-J[1])>ex)n++}
  return n*step}
/* Overlap test for two bars. rel: 0 = not connected, 1 = share a joint, 2 = two joints apart.
   Unconnected bars may not touch at all. Connected bars may overlap only inside the joint (a small disc around the shared joint);
   a bar lying along its neighbour beyond that is an overlap, and nearly full cover is a duplicate. */
function testPair(A,B,rel,P){
  const T=A.r+B.r,sd=segSeg(A,B);
  if(sd.d>=T*.98)return null;
  if(!rel){
    let nx,ny;
    if(sd.d>.5){nx=(sd.ax-sd.bx)/sd.d;ny=(sd.ay-sd.by)/sd.d}
    else{nx=A.cx-B.cx;ny=A.cy-B.cy;const l=Math.hypot(nx,ny);if(l<1){nx=-A.uy;ny=A.ux}else{nx/=l;ny/=l}}
    return{kind:'cross',depth:T-sd.d,need:T-sd.d,nx,ny};
  }
  const{J}=joint(A,B,rel===1),ex=(rel===1?2.2:3.2)*P.lineW+Math.max(0,Math.max(A.r,B.r)-P.lineW/2),step=Math.max(P.lineW/2,2),
    e=Math.max(excess(A,B,J,ex,T*.9,step),excess(B,A,J,ex,T*.9,step));
  if(e<P.lineW)return null;
  const mn=Math.min(2*A.hl,2*B.hl)||e,kind=e>=.6*mn?'dup':'overlap',L=A.hl>=B.hl?A:B;
  let dx=A.cx-B.cx,dy=A.cy-B.cy;const dot=dx*L.ux+dy*L.uy;dx-=dot*L.ux;dy-=dot*L.uy;let l=Math.hypot(dx,dy);
  if(l<.5){dx=-L.uy;dy=L.ux;l=1}
  return{kind,depth:e,need:Math.min(e,P.lineW*1.2),nx:dx/l,ny:dy/l};
}
const KIND_RANK={cross:1,overlap:2,dup:3};
const KIND_TXT={cross:'cross or touch',overlap:'lie on top of each other',dup:'are a duplicate of each other'};

/* ───────── layout model ───────── */
function prepare(c,o,adj,extraHide){
  const fps=c.fps(),n=c.n(),K=c.K(),S=c.S(),D=c.D(),allIds=c.act();
  if(!n||!allIds.length)throw new Error('Analyze a video first.');
  const vw=c.W(),vh=c.H(),sc=o.long/Math.max(vw,vh),cw=even(vw*sc),ch=even(vh*sc),k=o.long/1080,stick=o.style!=='classic',
    off=c.off(),totalMs=Math.max(200,Math.round(n/fps*1000)),dur=totalMs/1000,
    hidden=id=>(adj.hide&&adj.hide.has(id))||(extraHide&&extraHide.has(id)),ids=allIds.filter(id=>!hidden(id));
  const series=(id,p)=>{const arr=S[id]?.[p],kk=K[id]?.[p];return{k:o.dense||!kk||!kk.length?null:kk,s:arr}};
  const valAt=(id,p,i,hold)=>{const{k:kk,s}=series(id,p);if(hold!==undefined)i=hold;if(kk)return c.curve(kk,i);const a=s[Math.max(0,Math.min(s.length-1,Math.round(i)))];return Number.isFinite(a)?a:0};
  const lineW=Math.max(1,.0075*o.long*(o.thick/100)),edge=Math.max(0,(Number.isFinite(+o.edge)?+o.edge:2)*o.long/1080);
  const thick=id=>thickBase(id)/480*cw*(o.thick/100);
  // square head: steady size (the ear-to-nose line changes length every time the head turns, so it is not used for the size)
  let hd=0;
  if(stick&&ids.includes('head')){
    const medv=a=>{const q=(a||[]).filter(Number.isFinite).sort((x,y)=>x-y);return q.length?q[q.length>>1]:0},tl=medv(S.torso?.len),hl=medv(S.head?.len);
    hd=tl>0?.42*tl*ch:hl>0?2.4*hl*ch:lineW*8;hd=Math.max(lineW*4,hd*((+o.headPct||100)/100));
  }
  // base values per frame (as the graphs show them) – the layout is always built from these
  const base={};
  for(const id of ids){const b={};for(const p of['x','y','len','rot']){const a=new Float64Array(n);for(let i=0;i<n;i++)a[i]=valAt(id,p,i);b[p]=a}base[id]=b}
  const mv={};for(const id of ids){const m=adj.move&&adj.move[id];if(m&&(m[0]||m[1]))mv[id]=[m[0]*k,m[1]*k]}
  const P={lineW:stick?lineW:thick('x')};
  const rOf=id=>stick?(id==='head'?hd/2*1.1:lineW/2):thick(id)/2;
  // bars at frame i; dyn = {id:Float32Array(2n)} auto-fix offsets, or tmp = {id:[dx,dy]} for the solver
  const segAt=(id,i,dyn,tmp)=>{
    const b=base[id],m=mv[id]||[0,0];let dx=m[0],dy=m[1];
    if(tmp&&tmp[id]){dx+=tmp[id][0];dy+=tmp[id][1]}else if(dyn&&dyn[id]){dx+=dyn[id][2*i];dy+=dyn[id][2*i+1]}
    const cx=b.x[i]*cw+dx,cy=b.y[i]*ch+dy;
    if(stick&&id==='head')return mkSeg(id,cx,cy,0,1,0,rOf(id));
    const a=(b.rot[i]+off)*Math.PI/180,L=b.len[i]*ch+(stick?lineW:0);
    return mkSeg(id,cx,cy,Math.max(.5,L/2),Math.cos(a),Math.sin(a),rOf(id));
  };
  const segsAt=(i,dyn,tmp)=>ids.map(id=>segAt(id,i,dyn,tmp));
  const M={c,o,D,fps,n,cw,ch,k,stick,off,totalMs,dur,ids,allIds,series,valAt,lineW,edge,hd,thick,base,mv,P,segAt,segsAt,rOf};
  M.relMap=relations(M,allIds);
  M.rel=(a,b)=>M.relMap.get(a<b?a+'|'+b:b+'|'+a)||0;
  return M;
}
// which bars are connected: share a landmark, or sit at the same joint in the analysed motion (level 1); one bar in between (level 2)
function relations(M,allIds){
  const{D,n}=M,lm={};for(const id of allIds)lm[id]=new Set([...(D[id].a||[]),...(D[id].b||[])]);
  const key=(a,b)=>a<b?a+'|'+b:b+'|'+a;
  // every bar takes part (even hidden ones), so connections do not change when a bar is left out
  const tmp=prepareLite(M,allIds),frames=[];const st=Math.max(1,Math.floor(n/40));for(let i=0;i<n;i+=st)frames.push(i);
  const dist={};
  for(const i of frames){const sg=allIds.map(id=>tmp(id,i));
    for(let a=0;a<sg.length;a++)for(let b=a+1;b<sg.length;b++){
      const A=sg[a],B=sg[b];let best=1e18;for(const p of ends(A))for(const q of ends(B))best=Math.min(best,Math.hypot(p[0]-q[0],p[1]-q[1]));
      best-=Math.max(0,A.r+B.r-M.P.lineW);const kk=key(A.id,B.id);(dist[kk]||(dist[kk]=[])).push(best)}}
  const L1=new Map(),nb={};for(const id of allIds)nb[id]=new Set();
  for(let a=0;a<allIds.length;a++)for(let b=a+1;b<allIds.length;b++){
    const x=allIds[a],y=allIds[b];let share=false;for(const v of lm[x])if(lm[y].has(v)){share=true;break}
    const d=dist[key(x,y)]||[],med=d.length?d.slice().sort((p,q)=>p-q)[d.length>>1]:1e9;
    if(share||med<3.5*M.P.lineW){L1.set(key(x,y),1);nb[x].add(y);nb[y].add(x)}}
  const out=new Map(L1);
  for(let a=0;a<allIds.length;a++)for(let b=a+1;b<allIds.length;b++){
    const x=allIds[a],y=allIds[b];if(L1.has(key(x,y)))continue;
    for(const m of nb[x])if(nb[y].has(m)){out.set(key(x,y),2);break}}
  return out;
}
// segments for every active bar regardless of hiding (used only to work out the connections)
function prepareLite(M,allIds){
  const{c,o,n,cw,ch,off,stick,lineW,thick}=M,S=c.S(),K=c.K();
  const get=(id,p,i)=>{const kk=K[id]?.[p];if(!o.dense&&kk&&kk.length)return c.curve(kk,i);const s=S[id]?.[p];const a=s&&s[Math.max(0,Math.min(s.length-1,Math.round(i)))];return Number.isFinite(a)?a:0};
  return(id,i)=>{const cx=get(id,'x',i)*cw,cy=get(id,'y',i)*ch;
    if(stick&&id==='head')return mkSeg(id,cx,cy,0,1,0,M.hd?M.hd/2*1.1:lineW*2);
    const a=(get(id,'rot',i)+off)*Math.PI/180,L=get(id,'len',i)*ch+(stick?lineW:0);
    return mkSeg(id,cx,cy,Math.max(.5,L/2),Math.cos(a),Math.sin(a),stick?lineW/2:thick(id)/2)};
}
const dynAt=(dyn,i)=>{if(!dyn)return null;const o={};for(const id in dyn)o[id]=[dyn[id][2*i],dyn[id][2*i+1]];return o};
const frameList=(n,max)=>{const st=Math.max(1,Math.ceil(n/max)),a=[];for(let i=0;i<n;i+=st)a.push(i);if(a[a.length-1]!==n-1)a.push(n-1);return a};

/* scan many frames; segsFn(i) returns the bars of frame i */
function scan(M,segsFn,frames){
  const pairs=new Map(),score=new Map(),nm=id=>M.D[id]?.n||id;
  for(const i of frames){
    const sg=segsFn(i);let s=0;
    for(let a=0;a<sg.length;a++)for(let b=a+1;b<sg.length;b++){
      const A=sg[a],B=sg[b],r=testPair(A,B,M.rel(A.id,B.id),M.P);if(!r)continue;
      const kk=A.id+'|'+B.id;let q=pairs.get(kk);
      if(!q){q={a:A.id,b:B.id,names:nm(A.id)+' ↔ '+nm(B.id),kind:r.kind,count:0,worst:0,frame:i};pairs.set(kk,q)}
      q.count++;if(KIND_RANK[r.kind]>KIND_RANK[q.kind])q.kind=r.kind;
      if(r.depth>q.worst){q.worst=r.depth;q.frame=i}
      s+=r.depth+10;
    }
    if(s)score.set(i,s);
  }
  const list=[...pairs.values()].sort((x,y)=>y.count-x.count);
  let wf=-1,ws=0;for(const[i,s]of score)if(s>ws){ws=s;wf=i}
  return{pairs:list,worstFrame:wf,badFrames:score.size,scanned:frames.length,
    issues:list.map(q=>({text:`${q.names} ${KIND_TXT[q.kind]} in ${q.count} of ${frames.length} checked frames (worst ${q.worst.toFixed(0)} px, frame ${q.frame})`,frame:q.frame,ids:[q.a,q.b]}))};
}
function hitsAt(M,sg){const s=new Set();
  for(let a=0;a<sg.length;a++)for(let b=a+1;b<sg.length;b++)if(testPair(sg[a],sg[b],M.rel(sg[a].id,sg[b].id),M.P)){s.add(sg[a].id);s.add(sg[b].id)}
  return s}

/* ───────── auto-fix ───────── */
// step 1: analysis helper bars (knee chord, heel-toe, wrist, shoulder) that just repeat another bar are left out
function findDups(M){
  const frames=frameList(M.n,60),cnt=new Map(),hide=new Set();
  for(const i of frames){const sg=M.segsAt(i);
    for(let a=0;a<sg.length;a++)for(let b=a+1;b<sg.length;b++){const r=testPair(sg[a],sg[b],M.rel(sg[a].id,sg[b].id),M.P);
      if(r&&(r.kind==='dup'||(r.kind==='overlap'&&(HELPER.test(sg[a].id)||HELPER.test(sg[b].id))))){const kk=sg[a].id+'|'+sg[b].id;cnt.set(kk,(cnt.get(kk)||0)+1)}}}
  // only analysis helper bars are ever dropped automatically; real limbs are never removed, they are moved apart instead
  for(const[kk,v]of[...cnt].sort((x,y)=>y[1]-x[1])){
    if(v<.4*frames.length)continue;const[a,b]=kk.split('|'),ha=HELPER.test(a),hb=HELPER.test(b);
    if(ha===hb)continue;hide.add(ha?a:b)}
  return hide;
}
// step 2: where bars still overlap, push them apart frame by frame (limbs move more than the trunk); capped at maxPx
function solveDyn(M,maxPx){
  const{n,ids,P}=M,cur={},dyn={};
  for(const id of ids){cur[id]=[0,0];dyn[id]=new Float32Array(2*n)}
  const mass=id=>HEAVY.test(id)?5:1;
  const solveFrame=i=>{
    for(let it=0;it<18;it++){
      const sg=M.segsAt(i,null,cur),push={};let any=false,ax=null;
      {// spine axis (trunk bars only, pelvis and head excluded): arms are pushed sideways away from it, never along it
        let ux=0,uy=0,cx=0,cy=0,k=0;
        for(const q of sg)if(HEAVY.test(q.id)&&q.id!=='head'&&q.id!=='pelvis'&&q.hl>0){const sgn=(ux*q.ux+uy*q.uy)<0?-1:1;ux+=sgn*q.ux;uy+=sgn*q.uy;cx+=q.cx;cy+=q.cy;k++}
        const l=Math.hypot(ux,uy);if(k&&l>1e-6)ax={cx:cx/k,cy:cy/k,px:-uy/l,py:ux/l}}
      for(let a=0;a<sg.length;a++)for(let b=a+1;b<sg.length;b++){
        const A=sg[a],B=sg[b],r=testPair(A,B,M.rel(A.id,B.id),P);if(!r)continue;any=true;
        const tA=HEAVY.test(A.id),tB=HEAVY.test(B.id);
        if(ax&&tA!==tB){const Lm=tA?B:A;let d=(Lm.cx-ax.cx)*ax.px+(Lm.cy-ax.cy)*ax.py;
          if(Math.abs(d)<1){const e0=(Lm.x0-ax.cx)*ax.px+(Lm.y0-ax.cy)*ax.py,e1=(Lm.x1-ax.cx)*ax.px+(Lm.y1-ax.cy)*ax.py;d=Math.abs(e0)>Math.abs(e1)?e0:e1}
          const sd=d>=0?1:-1,f=tA?-1:1;r.nx=f*sd*ax.px;r.ny=f*sd*ax.py}
        const mg=Math.min(r.need,P.lineW)*.7+.5,ma=mass(A.id),mb=mass(B.id),wa=mb/(ma+mb),wb=ma/(ma+mb);
        (push[A.id]||(push[A.id]=[0,0]));(push[B.id]||(push[B.id]=[0,0]));
        push[A.id][0]+=r.nx*mg*wa*2;push[A.id][1]+=r.ny*mg*wa*2;push[B.id][0]-=r.nx*mg*wb*2;push[B.id][1]-=r.ny*mg*wb*2}
      if(!any)return true;
      // head and trunk are one rigid block: whatever pushes on any of them is shared, so the spine never breaks apart
      const trunk=ids.filter(id=>HEAVY.test(id));
      if(trunk.length>1){let gx=0,gy=0;for(const id of trunk)if(push[id]){gx+=push[id][0];gy+=push[id][1]}gx/=trunk.length;gy/=trunk.length;for(const id of trunk)push[id]=[gx,gy]}
      for(const id in push){const c=cur[id];c[0]+=push[id][0];c[1]+=push[id][1];const l=Math.hypot(c[0],c[1]);if(l>maxPx){c[0]*=maxPx/l;c[1]*=maxPx/l}}
    }
    return false;
  };
  const store=i=>{for(const id of ids){dyn[id][2*i]=cur[id][0];dyn[id][2*i+1]=cur[id][1]}};
  // pass 1: frame by frame, each frame starts from a faded copy of the previous solution (keeps motion smooth)
  for(let i=0;i<n;i++){for(const id of ids){cur[id][0]*=.6;cur[id][1]*=.6}solveFrame(i);store(i)}
  // pass 2: 3-tap smoothing of the offsets, then re-solve only the frames that smoothing broke
  const sm={};for(const id of ids){const a=dyn[id],b=new Float32Array(2*n);
    for(let i=0;i<n;i++)for(let q=0;q<2;q++){const p=a[2*Math.max(0,i-1)+q],c=a[2*i+q],d=a[2*Math.min(n-1,i+1)+q];b[2*i+q]=.25*p+.5*c+.25*d}sm[id]=b}
  for(let i=0;i<n;i++){for(const id of ids){cur[id][0]=sm[id][2*i];cur[id][1]=sm[id][2*i+1]}
    if(M.segsAt(i,null,cur).some((A,a,sg)=>sg.some((B,b)=>b>a&&testPair(A,B,M.rel(A.id,B.id),P))))solveFrame(i);
    store(i)}
  // parts that never moved by a visible amount stay exactly where they were
  const out={};for(const id of ids){let mx=0;for(const v of dyn[id])mx=Math.max(mx,Math.abs(v));if(mx>.3)out[id]=dyn[id]}
  return Object.keys(out).length?out:null;
}

/* ───────── scene builder ───────── */
function emit(M,dyn){
  const{c,o,n,fps,cw,ch,stick,ids,D,totalMs,dur,lineW,edge,hd,mv}=M,sgn=o.inv?-1:1,k=M.k;
  const projFps=o.fps==='match'?(AM_FPS.includes(fps)?fps:30):+o.fps;
  const T=i=>{const s=i/fps;return(o.tmode==='frac'?s/dur:s).toFixed(9)};
  const channel=(tag,id,props,fmt,opt={})=>{
    const arrs=props.map(p=>M.series(id,p).k),keys=planKeys(arrs,n,o.dense||opt.force);
    let vals=keys.map(q=>props.map(p=>M.valAt(id,p,q.i,q.holdFrom)));
    if(opt.add)vals=vals.map((v,j)=>opt.add(v,keys[j].i));
    const cols=vals[0].map((_,ci)=>vals.map(v=>v[ci])),still=cols.every(col=>Math.max(...col)-Math.min(...col)<1e-7);
    if(still||keys.length<2)return `<${tag} value="${fmt(vals[0])}"/>`;
    const kf=keys.map((q,j)=>{
      const e=o.easeOn==='arrive'?(j>0?keys[j-1].e:null):(j<keys.length-1?q.e:null),es=easeStr(e);
      return `<kf t="${T(q.i)}" v="${fmt(vals[j])}"${es?` e="${es}"`:''}/>`});
    return `<${tag}>\n        ${kf.join('\n        ')}\n      </${tag}>`;
  };
  const locFmt=v=>`${f6(v[0])},${f6(v[1])},0.000000`;
  // location in canvas pixels; manual moves are a constant shift, auto-fix offsets change per frame (the channel is then keyed on every frame)
  const locCh=id=>{
    const m=mv[id]||[0,0],d=dyn&&dyn[id];
    return channel('location',id,['x','y'],locFmt,{force:!!d,add:(v,i)=>{const fi=Math.max(0,Math.min(n-1,Math.round(i)));
      return[v[0]*cw+m[0]+(d?d[2*fi]:0),v[1]*ch+m[1]+(d?d[2*fi+1]:0)]}});
  };
  const rotCh=id=>channel('rotation',id,['rot'],v=>f6(sgn*(v[0]+M.off)));
  // Alight Motion's .rect draws at 2 × size × scale (size = half-extent; verified against real 5.0.27x/ffver106 exports).
  // Every shape below must carry an explicit <size>, or AM falls back to its own default half-extent instead of ours —
  // that mismatch is what was making every bar come out a consistent fraction of its intended length ("bunched" limbs).
  const szVal=(Number.isFinite(+o.size)&&+o.size>0)?+o.size:50,sdiv=2*szVal,SZ=`<size value="${f6(szVal)},${f6(szVal)}"/>`;
  const WHITE='#FFFFFFFF',BLACK='#FF000000',items=[];
  if(!stick){
    for(const id of ids){const d=D[id],thick=M.thick(id);
      items.push({pid:id,role:'main',label:d.n,sid:'.rect',color:sideColor(d),tf:[locCh(id),SZ,channel('scale',id,['len'],v=>`${f6(Math.max(.001,v[0]*ch/sdiv))},${f6(Math.max(.001,thick/sdiv))}`),rotCh(id)]})}
  }else{
    // Stickman: every part is a thin white bar. The thin black edge is a slightly bigger black bar underneath (so it only uses the same
    // rectangle + colour tags as the classic look). All edges go under all white bars, which makes the joints merge into one clean outline.
    const lines=ids.filter(id=>id!=='head'),pre={};
    for(const id of lines)pre[id]={loc:locCh(id),rot:rotCh(id)};
    // each bar is lengthened by one line width so neighbouring bars overlap at the joints instead of leaving notches
    const sclOf=(id,pad)=>channel('scale',id,['len'],v=>`${f6(Math.max(.001,(v[0]*ch+lineW+pad)/sdiv))},${f6(Math.max(.001,(lineW+pad)/sdiv))}`);
    if(edge>0)for(const id of lines)items.push({pid:id,role:'edge',label:D[id].n+' edge',sid:'.rect',color:BLACK,tf:[pre[id].loc,SZ,sclOf(id,2*edge),pre[id].rot]});
    for(const id of lines)items.push({pid:id,role:'main',label:D[id].n,sid:'.rect',color:WHITE,tf:[pre[id].loc,SZ,sclOf(id,0),pre[id].rot]});
    if(ids.includes('head')){
      // square head: a plain white square (same .rect shape as the bars) with a slightly bigger black square under it
      const loc=locCh('head'),sq=d=>`<scale value="${f6(d/sdiv)},${f6(d/sdiv)}"/>`;
      if(edge>0)items.push({pid:'head',role:'edge',label:'Head edge',sid:'.rect',color:BLACK,tf:[loc,SZ,sq(hd+2*edge)]});
      items.push({pid:'head',role:'main',label:'Head',sid:'.rect',color:WHITE,tf:[loc,SZ,sq(hd)]});
    }
  }
  if(o.order==='top')items.reverse();
  const layers=items.map((it,li)=>`  <shape id="${li+1}" label="${esc(it.label)}" startTime="0" endTime="${totalMs}" fillType="color" s="${it.sid}">
    <transform>
      ${it.tf.join('\n      ')}
    </transform>
    <fillColor value="${it.color}"/>
  </shape>`).join('\n');
  const title=(o.title||'vid2anim').slice(0,60);
  const xml=`<?xml version="1.0" encoding="UTF-8"?>
<scene title="${esc(title)}" width="${cw}" height="${ch}" exportWidth="${cw}" exportHeight="${ch}" bgcolor="${o.bg}" totalTime="${totalMs}" fps="${projFps}" modifiedTime="${Date.now()}" amver="1028425" ffver="106" am="com.alightcreative.motion/5.0.273.1028425" amplatform="android">
${layers}
</scene>
`;
  return{xml,items:items.map(i=>({pid:i.pid,role:i.role})),info:{layers:items.length,canvas:`${cw}×${ch}`,ms:totalMs,fps:projFps,bytes:new Blob([xml]).size}};
}

/* ───────── CHECK 1: the analysed layout ───────── */
function check1(M,dyn){
  const fr=frameList(M.n,500),r=scan(M,i=>M.segsAt(i,dyn),fr);
  const problems=[];
  for(const id of M.ids){const b=M.base[id];let bad=0;for(let i=0;i<M.n;i++)if(![b.x[i],b.y[i],b.len[i],b.rot[i]].every(Number.isFinite))bad++;if(bad)problems.push({text:`${M.D[id].n}: ${bad} frames have no valid position`,frame:0,ids:[id]})}
  return{ok:!r.pairs.length&&!problems.length,pairs:r.pairs,issues:[...problems,...r.issues],worstFrame:r.worstFrame,badFrames:r.badFrames,scanned:r.scanned};
}

/* ───────── CHECK 2: read the finished .xml back ───────── */
const attr=(s,k)=>{const m=s.match(new RegExp('(?:^|\\s)'+k+'="([^"]*)"'));return m?m[1]:undefined};
function parseScene(xml){
  const head=(xml.match(/<scene\b[^>]*>/)||[''])[0],layers=[];
  for(const m of xml.matchAll(/<shape\b([^>]*)>([\s\S]*?)<\/shape>/g)){
    const L={id:attr(m[1],'id'),label:attr(m[1],'label'),start:+attr(m[1],'startTime'),end:+attr(m[1],'endTime'),shape:attr(m[1],'s'),ch:{}};
    L.color=(m[2].match(/<fillColor value="([^"]*)"/)||[])[1];
    for(const tag of['location','scale','rotation','size']){
      const st=m[2].match(new RegExp(`<${tag}\\s+value="([^"]*)"\\s*/>`));
      if(st){L.ch[tag]={v:st[1].split(',').map(Number)};continue}
      const kb=m[2].match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
      if(kb){const kfs=[];for(const q of kb[1].matchAll(/<kf\s+t="([^"]+)"\s+v="([^"]+)"(?:\s+e="([^"]+)")?\s*\/>/g))kfs.push({t:+q[1],v:q[2].split(',').map(Number),e:q[3]});L.ch[tag]={kfs}}
    }
    layers.push(L);
  }
  return{head,layers,w:+attr(head,'width'),h:+attr(head,'height'),total:+attr(head,'totalTime'),fps:+attr(head,'fps')};
}
function bez(x1,y1,x2,y2,p){ // CSS-style cubic bezier easing
  if(p<=0)return 0;if(p>=1)return 1;
  const X=t=>3*(1-t)*(1-t)*t*x1+3*(1-t)*t*t*x2+t*t*t,Y=t=>3*(1-t)*(1-t)*t*y1+3*(1-t)*t*t*y2+t*t*t;
  let lo=0,hi=1,t=p;for(let q=0;q<40;q++){t=(lo+hi)/2;if(X(t)<p)lo=t;else hi=t}return Y(t);
}
function evalCh(ch,f,o,toF){
  if(!ch)return[0];if(ch.v)return ch.v;
  const k=ch.kfs,F=k.map(q=>toF(q.t));
  if(f<=F[0])return k[0].v;if(f>=F[F.length-1])return k[k.length-1].v;
  let lo=0,hi=k.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(F[m]<=f)lo=m;else hi=m}
  const p=(f-F[lo])/(F[hi]-F[lo]),e=o.easeOn==='arrive'?k[hi].e:k[lo].e;let q=p;
  if(e){const m=e.match(/cubicBezier\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)/);if(m)q=bez(+m[1],+m[2],+m[3],+m[4],p)}
  return k[lo].v.map((v,i)=>v+(k[hi].v[i]-v)*q);
}
function check2(M,em,dyn){
  const{o,n,fps,cw,ch,stick}=M,sgn=o.inv?-1:1,problems=[];
  const P=parseScene(em.xml);
  // --- structure
  if(typeof DOMParser!=='undefined'){try{const d=new DOMParser().parseFromString(em.xml,'application/xml');if(d.getElementsByTagName('parsererror').length)problems.push({text:'The XML is not well-formed.',frame:0,ids:[]})}catch{}}
  if(P.layers.length!==em.items.length)problems.push({text:`Layer count mismatch: built ${em.items.length}, found ${P.layers.length} in the file.`,frame:0,ids:[]});
  if(!(P.w>0&&P.h>0&&P.total>0))problems.push({text:'Scene size or length is invalid.',frame:0,ids:[]});
  const seen=new Set(),bad={ids:0,num:0,scale:0,time:0,off:0};
  const maxT=o.tmode==='frac'?1.0001:M.dur+.001;
  P.layers.forEach((L,li)=>{
    if(seen.has(L.id)||+L.id!==li+1)bad.ids++;seen.add(L.id);
    if(!L.ch.location||!L.ch.scale||!L.ch.size)bad.num++;
    for(const tag in L.ch){const c=L.ch[tag],vs=c.v?[c.v]:c.kfs.map(q=>q.v);
      for(const v of vs)if(!v.length||!v.every(Number.isFinite))bad.num++;
      if(tag==='scale')for(const v of vs)if(!(v[0]>0&&v[1]>0))bad.scale++;
      if(c.kfs){let last=-1;for(const q of c.kfs){if(!(q.t>=0&&q.t<=maxT&&q.t>last))bad.time++;last=q.t}}}
    if(L.ch.location){const vs=L.ch.location.v?[L.ch.location.v]:L.ch.location.kfs.map(q=>q.v);
      if(vs.some(v=>v[0]<-.5*P.w||v[0]>1.5*P.w||v[1]<-.5*P.h||v[1]>1.5*P.h))bad.off++}
  });
  if(bad.ids)problems.push({text:'Layer ids are duplicated or not in order.',frame:0,ids:[]});
  if(bad.num)problems.push({text:`${bad.num} channels are missing or hold invalid numbers.`,frame:0,ids:[]});
  if(bad.scale)problems.push({text:`${bad.scale} scale values are zero or negative.`,frame:0,ids:[]});
  if(bad.time)problems.push({text:`${bad.time} keyframe times are out of range or not increasing.`,frame:0,ids:[]});
  if(bad.off)problems.push({text:`${bad.off} layers drift far outside the canvas.`,frame:0,ids:[]});
  // --- geometry as Alight Motion will draw it
  const toF=t=>o.tmode==='frac'?t*M.dur*fps:t*fps;
  const lay=P.layers.map((L,li)=>({L,meta:em.items[li]||{}}));
  const geomAt=i=>lay.map(({L,meta})=>{
    const loc=evalCh(L.ch.location,i,o,toF),sc=evalCh(L.ch.scale,i,o,toF),rot=L.ch.rotation?evalCh(L.ch.rotation,i,o,toF)[0]:0;
    const sz=L.ch.size?evalCh(L.ch.size,i,o,toF):[50,50]; // 50 = this file's own convention if a shape somehow lacks the tag
    return{pid:meta.pid,role:meta.role,cx:loc[0],cy:loc[1],w:2*sz[0]*sc[0],h:2*sz[1]*sc[1],ang:sgn*rot,color:L.color};
  });
  const segsAt=i=>geomAt(i).filter(g=>g.role==='main'&&g.pid).map(g=>{
    if(stick&&g.pid==='head')return mkSeg(g.pid,g.cx,g.cy,0,1,0,g.w/2*1.1);
    const a=g.ang*Math.PI/180;return mkSeg(g.pid,g.cx,g.cy,g.w/2,Math.cos(a),Math.sin(a),g.h/2)});
  // --- does the file move like the analysis?
  let ep=0,ea=0,el=0,ef=0;const fr=frameList(n,300);
  for(const i of fr){const x=segsAt(i),m=M.segsAt(i,dyn);
    for(const A of m){const B=x.find(s=>s.id===A.id);if(!B)continue;
      const dp=Math.hypot(A.cx-B.cx,A.cy-B.cy),dl=Math.abs(A.hl-B.hl)*2;let da=0;
      if(A.hl>0){const d=((((Math.atan2(A.uy,A.ux)-Math.atan2(B.uy,B.ux))*180/Math.PI)%180)+180)%180;da=Math.min(d,180-d)}
      if(dp>ep){ep=dp;ef=i}if(da>ea)ea=da;if(dl>el)el=dl}}
  const tolP=3*M.k,tolL=3*M.k;
  if(ep>tolP)problems.push({text:`The file moves differently from the analysis: up to ${ep.toFixed(1)} px off at frame ${ef}. Check "Keyframe time unit" / "Easing stored on" with the calibration project.`,frame:ef,ids:[]});
  if(el>tolL)problems.push({text:`Part lengths in the file differ from the analysis by up to ${el.toFixed(1)} px.`,frame:ef,ids:[]});
  if(ea>1.5)problems.push({text:`Part angles in the file differ from the analysis by up to ${ea.toFixed(1)}°. Try "Invert rotation direction".`,frame:ef,ids:[]});
  // --- overlap scan on the drawn geometry
  const r=scan(M,segsAt,frameList(n,500));
  return{ok:!problems.length&&!r.pairs.length,problems,pairs:r.pairs,issues:[...problems,...r.issues],worstFrame:r.worstFrame,badFrames:r.badFrames,
    fid:{pos:ep,ang:ea,len:el,frame:ef},geomAt,hitsAt:i=>hitsAt(M,segsAt(i)),cw:P.w,ch:P.h,scanned:r.scanned};
}

/* ───────── pipeline ───────── */
export function evaluate(c,o,adjIn){
  const adj=Object.assign({auto:false,max:40,hide:new Set(),move:{}},adjIn||{});
  let M=prepare(c,o,adj,null),dyn=null,auto=null;
  if(adj.auto){
    const dup=findDups(M);
    if(dup.size)M=prepare(c,o,adj,dup);
    const t0=Date.now();dyn=solveDyn(M,Math.max(0,+adj.max||0)*M.k);
    let mx=0;if(dyn)for(const id in dyn)for(const v of dyn[id])mx=Math.max(mx,Math.abs(v));
    auto={left:[...dup].map(id=>M.D[id].n),moved:dyn?Object.keys(dyn).map(id=>M.D[id].n):[],maxPx:mx/M.k,ms:Date.now()-t0};
  }
  const em=emit(M,dyn),r1=check1(M,dyn),r2=check2(M,em,dyn);
  return{xml:em.xml,info:em.info,r1,r2,auto,M,dyn,ok:r1.ok&&r2.ok};
}
export function buildScene(c,o,adj){const r=evaluate(c,o,adj);return{xml:r.xml,info:r.info,r1:r.r1,r2:r.r2}}

/* Calibration project: tiny scene that shows which timing / easing / rotation conventions your Alight Motion build uses. */
export function buildCalibration(o){
  const totalMs=4000,tt=x=>(o.tmode==='frac'?x:x*4).toFixed(9),mk=(id,label,color,loc,extra='',shape='.rect')=>`  <shape id="${id}" label="${esc(label)}" startTime="0" endTime="${totalMs}" fillType="color" s="${esc(shape)}">
    <transform>
      ${loc}${extra}
    </transform>
    <fillColor value="${color}"/>
  </shape>`;
  const mv=(y,k)=>`<location>\n        ${k.map(([t,x,e])=>`<kf t="${t}" v="${x}.000000,${y}.000000,0.000000"${e?` e="${e}"`:''}/>`).join('\n        ')}\n      </location>`;
  const szVal=(Number.isFinite(+o.size)&&+o.size>0)?+o.size:50,szTag=`<size value="${f6(szVal)},${f6(szVal)}"/>`;
  const sc=`\n      ${szTag}\n      <scale value="1.200000,1.200000"/>`;
  const sizeScale=(880/(2*szVal)).toFixed(6); // bar width = 2 × szVal × scale, aimed at 880px of the 1080 canvas
  const layers=[
    mk(1,'TIME A - fraction mode: must ARRIVE at right edge exactly when clip ENDS','#FF3DE8FF',mv(200,[['0.000000000',160],['1.000000000',920]]),sc),
    mk(2,'TIME B - seconds mode: must ARRIVE at right edge exactly when clip ENDS','#FFFF8A4C',mv(380,[['0.000000000',160],['4.000000000',920]]),sc),
    mk(3,'EASE arrive-key: sits almost still in first half, then rushes (= easing is on ARRIVING key)','#FFFC01E6',mv(560,[[tt(0),160],[tt(.5),540,'cubicBezier 0.9 0 1 0.1'],[tt(1),920]]),sc),
    mk(4,'EASE depart-key: control, easing placed on DEPARTING key','#FF3DFFA6',mv(740,[[tt(0),160,'cubicBezier 0.9 0 1 0.1'],[tt(.5),540],[tt(1),920]]),sc),
    mk(5,'ROTATION - bar must turn CLOCKWISE and point DOWN at the end','#FFFFE45E',`<location value="540.000000,900.000000,0.000000"/>`,`\n      ${szTag}\n      <scale value="2.500000,0.300000"/>\n      <rotation>\n        <kf t="${tt(0)}" v="0.000000"/>\n        <kf t="${tt(1)}" v="90.000000"/>\n      </rotation>`),
    mk(6,'HEAD - must be a white SQUARE (same shape as the bars)','#FFFFFFFF',`<location value="160.000000,950.000000,0.000000"/>`,sc),
    mk(7,'ORDER first in file - RED square','#FFFF2D2D',`<location value="780.000000,950.000000,0.000000"/>`,sc),
    mk(8,'ORDER second in file - BLUE square: if BLUE covers RED then first layer = bottom, if RED covers BLUE set "First layer = top"','#FF2D6BFF',`<location value="830.000000,985.000000,0.000000"/>`,sc),
    mk(9,'SIZE - this bar must reach almost edge to edge (880 of 1080 px). If it stops well short (eg. around the middle), raise "Shape size" in Compatibility until it does','#FF8AFF3D',`<location value="540.000000,80.000000,0.000000"/>`,`\n      ${szTag}\n      <scale value="${sizeScale},0.300000"/>`)
  ].join('\n');
  // note: layers 1,3,4 are fraction-mode; layer 2 is seconds-mode. The scene stays valid either way, the wrong one simply overshoots.
  return `<?xml version="1.0" encoding="UTF-8"?>
<scene title="vid2anim calibration" width="1080" height="1080" exportWidth="1080" exportHeight="1080" bgcolor="#FF0D0F24" totalTime="${totalMs}" fps="30" modifiedTime="${Date.now()}" amver="1028425" ffver="106" am="com.alightcreative.motion/5.0.273.1028425" amplatform="android">
${layers}
</scene>
`;
}

/* ───────── UI wiring ───────── */
export function initAlight(c){
  const $=c.$,st=c.st,LS='vid2anim.alight';
  if(['amXml','amCal','amShare','amMsg'].some(i=>!$(i))){console.warn('Alight panel not found in index.html, export disabled');return}
  const F=['amLong','amBg','amThick','amTime','amEase','amDense','amFps','amInv','amStyle','amEdge','amHead','amOrder','amSkip','amSize'].filter(i=>$(i));
  const get=i=>!$(i)?undefined:$(i).type==='checkbox'?$(i).checked:$(i).value;
  try{const s=JSON.parse(localStorage.getItem(LS)||'{}');for(const i of F)if(i in s&&$(i)){if($(i).type==='checkbox')$(i).checked=s[i];else $(i).value=s[i]}}catch{}
  const save=()=>{try{localStorage.setItem(LS,JSON.stringify(Object.fromEntries(F.map(i=>[i,get(i)]))))}catch{}};
  for(const i of F)$(i).addEventListener('change',save);
  const opts=()=>({long:+get('amLong'),bg:get('amBg'),thick:+get('amThick')||100,tmode:get('amTime'),easeOn:get('amEase'),dense:get('amDense')==='dense',fps:get('amFps'),inv:get('amInv'),style:get('amStyle')||'stick',edge:$('amEdge')?+get('amEdge'):2,headPct:+get('amHead')||100,order:get('amOrder')||'bottom',size:$('amSize')?+get('amSize')||50:50,title:c.title()});
  const safe=s=>(s||'vid2anim').replace(/\.[^.]+$/,'').replace(/[^\w\- ]+/g,'_').slice(0,40)||'vid2anim';
  const msg=t=>{$('amMsg').textContent=t;st(t)};
  // adjustments made in the review (auto-fix switch, parts moved by hand, parts left out); reset when a different analysis is loaded
  let adj={auto:false,max:40,hide:new Set(),move:{}},adjKey='';
  const syncKey=()=>{const k=c.n()+'|'+c.act().join(',');if(k!==adjKey){adjKey=k;adj={auto:false,max:40,hide:new Set(),move:{}}}};
  const make=()=>{syncKey();const o=opts(),r=evaluate(c,o,adj);return{...r,name:safe(o.title)+'_alight.xml'}};
  const summary=r=>`${r.info.layers} layers · ${r.info.canvas} · ${(r.info.ms/1000).toFixed(2)}s @ ${r.info.fps} fps · ${(r.info.bytes/1024).toFixed(0)} KB · checks: ${r.r1.ok?'1 ✔':'1 ✖'} ${r.r2.ok?'2 ✔':'2 ✖'}`;
  const doDl=r=>{c.dl(r.name,r.xml,'application/xml');msg(`Saved ${r.name}: ${summary(r)}`)};
  const doShare=async r=>{try{const f=new File([r.xml],r.name,{type:'application/xml'});
      if(navigator.canShare&&navigator.canShare({files:[f]})){await navigator.share({files:[f],title:r.name});msg('Shared.')}else{c.dl(r.name,r.xml,'application/xml');msg('This browser cannot share files, so the file was downloaded instead.')}}catch(e){if(e.name!=='AbortError')msg(e.message)}};
  const hasReview=['amRev','amRCv','amRDl','amROk'].every(i=>$(i));
  // ---- review dialog (checks 1 + 2 results, preview, fixes, manual check 3)
  const R={r:null,frame:0,act:'dl',timer:0,bakedMove:{}};
  const part=()=>$('amRPart').value;
  const cvDraw=()=>{
    const r=R.r;if(!r)return;const g=r.r2,cv=$('amRCv'),W=Math.max(200,Math.min(560,($('amRCv').parentElement.clientWidth||400)-4)),sc=W/g.cw,H=Math.round(g.ch*sc);
    cv.width=W;cv.height=H;const x=cv.getContext('2d');
    x.fillStyle='#14172c';x.fillRect(0,0,W,H);const bg=opts().bg,al=parseInt(bg.slice(1,3),16);
    x.fillStyle='#'+bg.slice(3);x.globalAlpha=al/255||0.0;x.fillRect(0,0,W,H);x.globalAlpha=1;
    const hit=g.hitsAt(R.frame),lay=g.geomAt(R.frame);if(opts().order==='top')lay.reverse();
    const col=s=>{const a=parseInt(s.slice(1,3),16)/255,r=parseInt(s.slice(3,5),16),gg=parseInt(s.slice(5,7),16),b=parseInt(s.slice(7,9),16);return`rgba(${r},${gg},${b},${a})`};
    const sel=part();
    for(const L of lay){
      // the full recompute (evaluate) is debounced, so a move you're still dragging may not be baked into L.cx/cy yet —
      // draw the not-yet-applied part of the current nudge live, on top of the last computed position
      const cm=adj.move[L.pid]||[0,0],bm=R.bakedMove[L.pid]||[0,0],ex=cm[0]-bm[0],ey=cm[1]-bm[1];
      x.save();x.translate((L.cx+ex)*sc,(L.cy+ey)*sc);x.rotate(L.ang*Math.PI/180);
      x.fillStyle=col(L.color);x.fillRect(-L.w*sc/2,-L.h*sc/2,L.w*sc,L.h*sc);
      if(L.role==='main'&&hit.has(L.pid)){x.strokeStyle='#ff2d55';x.lineWidth=2;x.strokeRect(-L.w*sc/2-1,-L.h*sc/2-1,L.w*sc+2,L.h*sc+2)}
      if(L.role==='main'&&L.pid===sel){x.strokeStyle='#3de8ff';x.lineWidth=2;x.strokeRect(-L.w*sc/2-3,-L.h*sc/2-3,L.w*sc+6,L.h*sc+6)}
      x.restore()}
    $('amRTi').textContent=`frame ${R.frame} / ${c.n()-1} · ${hit.size?hit.size+' parts overlapping here':'no overlaps in this frame'}`;
  };
  const list=(el,issues)=>{
    el.innerHTML='';const top=issues.slice(0,10);
    for(const it of top){const b=document.createElement('button');b.textContent=it.text;b.onclick=()=>{R.frame=Math.max(0,Math.min(c.n()-1,it.frame||0));$('amRT').value=R.frame;cvDraw();
      if(it.ids&&it.ids[0]&&$('amRPart')&&[...$('amRPart').options].some(o=>o.value===it.ids[0])){$('amRPart').value=it.ids[0];loadPart()}};el.appendChild(b)}
    if(issues.length>top.length){const s=document.createElement('small');s.textContent=`+ ${issues.length-top.length} more`;el.appendChild(s)}
  };
  const loadPart=()=>{const id=part(),m=adj.move[id]||[0,0];$('amRDx').value=m[0];$('amRDy').value=m[1];$('amRHide').checked=adj.hide.has(id)};
  const buildParts=()=>{
    const cur=$('amRPart').value;const D=c.D();
    $('amRPart').innerHTML=c.act().map(id=>`<option value="${id}">${D[id].n}${adj.hide.has(id)?' (left out)':adj.move[id]?' (moved)':''}</option>`).join('');
    if([...$('amRPart').options].some(o=>o.value===cur))$('amRPart').value=cur;loadPart();
  };
  const paint=()=>{
    const r=R.r,row=(id,res,ok,bad)=>{const li=$(id);li.className=res.ok?'ok':'bad';li.querySelector('span').textContent=res.ok?ok:bad(res)};
    row('amRC1',r.r1,'passed – no part overlaps where it should not.',x=>`${x.pairs.length} problem${x.pairs.length===1?'':'s'} in the analysed layout (${x.badFrames} of ${x.scanned} checked frames).`);
    row('amRC2',r.r2,'passed – the .xml is valid, moves like the analysis and nothing overlaps in it.',x=>`${x.issues.length} problem${x.issues.length===1?'':'s'} found in the finished .xml.`);
    list($('amRI1'),r.r1.issues);list($('amRI2'),r.r2.issues);
    $('amRI1').parentElement.hidden=!r.r1.issues.length;$('amRI2').parentElement.hidden=!r.r2.issues.length;
    const a=r.auto;$('amRAutoMsg').textContent=adj.auto?(a?`Auto-fix is on. ${a.left.length?'Left out duplicate bars: '+a.left.join(', ')+'. ':''}${a.moved.length?`Moved ${a.moved.length} part${a.moved.length===1?'':'s'} apart by up to ${a.maxPx.toFixed(0)} px (limit ${adj.max}).`:'No part needed to be moved.'}`:''):'Auto-fix is off.';
    $('amRAuto').textContent=adj.auto?'Turn auto-fix off':'Auto-fix overlaps';
    $('amRSub').textContent=r.ok?'Both automatic checks passed.':'Automatic checks found problems – fix them below, or confirm anyway if the preview looks right.';
    $('amROk').checked=false;sync();
  };
  const sync=()=>{const ok=$('amROk').checked,r=R.r;$('amRDl').disabled=$('amRSh').disabled=!ok;
    $('amRDl').textContent=r&&r.ok?'Download .xml':'Download anyway';$('amRSh').textContent=r&&r.ok?'Share file…':'Share anyway…'};
  const refresh=(keepFrame)=>{
    try{const o=opts(),r=evaluate(c,o,adj);R.r={...r,name:safe(o.title)+'_alight.xml'};R.bakedMove={...adj.move};
      const n=c.n(),wf=r.r2.worstFrame>=0?r.r2.worstFrame:r.r1.worstFrame>=0?r.r1.worstFrame:n>>1;
      if(!keepFrame)R.frame=Math.max(0,Math.min(n-1,wf));$('amRT').max=n-1;$('amRT').value=R.frame;buildParts();paint();cvDraw()}
    catch(e){msg(e.message);closeRev()}
  };
  const later=()=>{clearTimeout(R.timer);$('amROk').checked=false;sync();R.timer=setTimeout(()=>refresh(true),180)};
  const openRev=act=>{R.act=act;$('amRev').hidden=false;document.body.style.overflow='hidden';refresh(false)};
  const closeRev=()=>{if($('amRev'))$('amRev').hidden=true;document.body.style.overflow=''};
  if(hasReview){
    $('amRX').onclick=closeRev;
    $('amRT').oninput=()=>{R.frame=+$('amRT').value;cvDraw()};
    $('amRWorst').onclick=()=>{const r=R.r;if(!r)return;const f=r.r2.worstFrame>=0?r.r2.worstFrame:r.r1.worstFrame;if(f>=0){R.frame=f;$('amRT').value=f;cvDraw()}};
    $('amRAuto').onclick=()=>{adj.auto=!adj.auto;refresh(true)};
    $('amRMax').onchange=()=>{adj.max=Math.max(0,+$('amRMax').value||0);if(adj.auto)refresh(true)};
    $('amRPart').onchange=()=>{loadPart();cvDraw()};
    const setMove=()=>{const id=part(),dx=+$('amRDx').value||0,dy=+$('amRDy').value||0;if(dx||dy)adj.move[id]=[dx,dy];else delete adj.move[id];cvDraw();later()};
    $('amRDx').oninput=$('amRDy').oninput=setMove;
    for(const[i,dx,dy]of[['amRL',-2,0],['amRR',2,0],['amRU',0,-2],['amRD',0,2]])$(i).onclick=()=>{$('amRDx').value=(+$('amRDx').value||0)+dx;$('amRDy').value=(+$('amRDy').value||0)+dy;setMove()};
    $('amRHide').onchange=()=>{const id=part();if($('amRHide').checked)adj.hide.add(id);else adj.hide.delete(id);refresh(true)};
    $('amRReset').onclick=()=>{adj={auto:false,max:adj.max,hide:new Set(),move:{}};refresh(true)};
    $('amROk').onchange=sync;
    $('amRDl').onclick=()=>{if(R.r&&$('amROk').checked){doDl(R.r);closeRev()}};
    $('amRSh').onclick=async()=>{if(R.r&&$('amROk').checked){const r=R.r;closeRev();await doShare(r)}};
    addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('amRev').hidden)closeRev()});
    $('amRMax').value=adj.max;
  }
  // ---- main buttons: build, run both checks, then either save straight away (review skipped) or open the review
  const go=act=>{
    try{
      if(!hasReview){const r=make();act==='dl'?doDl(r):doShare(r);return}
      syncKey();$('amRMax').value=adj.max;
      if(get('amSkip')){const r=make();if(r.ok){act==='dl'?doDl(r):doShare(r);return}}
      openRev(act);
    }catch(e){msg(e.message)}
  };
  $('amXml').onclick=()=>go('dl');
  $('amShare').onclick=()=>go('share');
  $('amCal').onclick=()=>{c.dl('vid2anim_calibration.xml',buildCalibration(opts()),'application/xml');msg('Calibration project saved. Import it in Alight Motion and follow the layer names.')};
}
