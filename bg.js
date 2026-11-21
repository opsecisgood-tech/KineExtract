// Rovio background rebuild.
//
//   1. For every Nth analysed frame (the "rebuild every" setting) the tracked person is cut out of the picture.
//      The cut-out uses an AI person segmenter (MediaPipe selfie segmenter, run on a crop around the tracked person)
//      together with the skeleton from the analysis, so it still works if the model cannot be downloaded.
//   2. The pixels that are left are the background. Each rebuild combines the nearby captures, so whatever the
//      person was standing in front of is filled in from moments where it was visible.
//        Static  = the camera does not move. Wide window, no alignment.
//        Moving  = the camera pans or shakes. Short window, and every capture is shifted by the camera movement
//                  that the analysis measured (and by the same stabilisation the exported character gets).
//   3. The result is a list of small colour "plates". The Alight export turns a plate into a grid of coloured
//      rectangle layers. Low / Mid / High = how many columns (and so how many parts) that grid has.
import {ImageSegmenter,FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const WASM="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";

/* ───────── quality → number of parts ───────── */
export const COLS={low:8,mid:16,high:32};
export const QNAME={low:'Low',mid:'Mid',high:'High'};
export function gridFor(q,w,h){const cols=COLS[q]||COLS.mid,rows=Math.max(1,Math.round(cols*h/Math.max(1,w)));return{cols,rows,parts:cols*rows}}

/* ───────── tiles: plates averaged down to cols × rows colours ───────── */
const tcache=new WeakMap();
function boxAvg(rgb,pw,ph,cols,rows){
  const out=new Uint8Array(cols*rows*3);
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
    const x0=c*pw/cols,x1=(c+1)*pw/cols,y0=r*ph/rows,y1=(r+1)*ph/rows;let R=0,G=0,B=0,W=0;
    for(let y=Math.floor(y0);y<Math.min(ph,Math.ceil(y1));y++){const wy=Math.min(y+1,y1)-Math.max(y,y0);
      for(let x=Math.floor(x0);x<Math.min(pw,Math.ceil(x1));x++){const w=wy*(Math.min(x+1,x1)-Math.max(x,x0)),k=3*(y*pw+x);R+=rgb[k]*w;G+=rgb[k+1]*w;B+=rgb[k+2]*w;W+=w}}
    const o=3*(r*cols+c);W=W||1;out[o]=Math.round(R/W);out[o+1]=Math.round(G/W);out[o+2]=Math.round(B/W)}
  return out;
}
// [{i, rgb:Uint8Array(cols*rows*3)}] for every capture, cached per grid
export function tileSeries(bg,cols,rows){
  let m=tcache.get(bg);if(!m){m=new Map();tcache.set(bg,m)}
  const k=cols+'x'+rows;if(m.has(k))return m.get(k);
  const s=bg.plates.map(p=>({i:p.i,rgb:boxAvg(p.rgb,bg.pw,bg.ph,cols,rows)}));m.set(k,s);return s;
}
// tile colours at analysis frame i, blended linearly between the two captures around it (how Alight Motion blends keyframes)
export function tilesAt(ser,i){
  let a=0;while(a<ser.length-1&&ser[a+1].i<=i)a++;
  const A=ser[a],B=ser[Math.min(a+1,ser.length-1)];if(A===B||B.i<=A.i||i<=A.i)return A.rgb;if(i>=B.i)return B.rgb;
  const f=(i-A.i)/(B.i-A.i),o=new Uint8Array(A.rgb.length);for(let k=0;k<o.length;k++)o[k]=Math.round(A.rgb[k]+(B.rgb[k]-A.rgb[k])*f);return o;
}

/* ───────── rebuilding the background from the captures (pure, no browser needed) ─────────
   S = [{i, t, rgb:Uint8(pw*ph*3), m:Uint8(pw*ph)}]   m = 1 where the person (or anything cut out) is
   o = {mode, every, fps, align(i,j)->[dx,dy] (fractions of the frame; where to read capture j when drawing frame i), onProg}   */
export async function buildPlates(S,pw,ph,o){
  const{mode,every,fps,align}=o,N=pw*ph,half=mode==='moving'?Math.max(2*every/fps,.6):Math.max(6*every/fps,2);
  const ra=new Uint8Array(S.length),ga=new Uint8Array(S.length),ba=new Uint8Array(S.length),plates=[],ALL=S.map((_,j)=>j);
  let holes=0;
  const med=(a,n)=>n===1?a[0]:n===2?(a[0]+a[1])>>1:(()=>{const s=a.subarray(0,n).sort();return s[n>>1]})();
  for(let ai=0;ai<S.length;ai++){
    const A=S[ai],J=[];for(let j=0;j<S.length;j++)if(Math.abs(S[j].t-A.t)<=half+1e-9)J.push(j);
    const sh=S.map(B=>{if(mode!=='moving'||!align)return[0,0];const a=align(A.i,B.i);return[Math.round(a[0]*pw),Math.round(a[1]*ph)]});
    const rgb=new Uint8ClampedArray(N*3),valid=new Uint8Array(N);
    const gather=(list,x,y)=>{let n=0;
      for(const j of list){const q=sh[j],qx=x+q[0],qy=y+q[1];if(qx<0||qy<0||qx>=pw||qy>=ph)continue;const k=qy*pw+qx;if(S[j].m[k])continue;
        ra[n]=S[j].rgb[3*k];ga[n]=S[j].rgb[3*k+1];ba[n]=S[j].rgb[3*k+2];n++}
      return n};
    let need=0;
    for(let y=0;y<ph;y++)for(let x=0;x<pw;x++){
      let n=gather(J,x,y);
      if(!n&&J.length<S.length)n=gather(ALL,x,y);        // never visible in the window: look through the whole clip
      const k=y*pw+x;if(!n){need++;continue}
      rgb[3*k]=med(ra,n);rgb[3*k+1]=med(ga,n);rgb[3*k+2]=med(ba,n);valid[k]=1}
    // spots the person covered in every capture: grow the surrounding colours inwards
    let left=need;
    for(let it=0;it<pw+ph&&left>0;it++){
      const add=[];
      for(let y=0;y<ph;y++)for(let x=0;x<pw;x++){const k=y*pw+x;if(valid[k])continue;
        let r=0,g=0,b=0,c=0;for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]){const X=x+dx,Y=y+dy;if(X<0||Y<0||X>=pw||Y>=ph)continue;const q=Y*pw+X;if(valid[q]){r+=rgb[3*q];g+=rgb[3*q+1];b+=rgb[3*q+2];c++}}
        if(c)add.push([k,r/c,g/c,b/c])}
      if(!add.length)break;
      for(const[k,r,g,b]of add){rgb[3*k]=r;rgb[3*k+1]=g;rgb[3*k+2]=b;valid[k]=1;left--}}
    if(left>0)for(let k=0;k<N;k++)if(!valid[k]){rgb[3*k]=rgb[3*k+1]=rgb[3*k+2]=18}  // nothing to go on at all (no capture had any background)
    holes+=need;plates.push({i:A.i,rgb:new Uint8ClampedArray(rgb)});
    if(o.onProg)o.onProg((ai+1)/S.length);
    if(ai%4===3)await new Promise(r=>setTimeout(r));
  }
  return{plates,holes};
}

/* ───────── person mask helpers ───────── */
const BONES=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32],[15,19],[16,20]];
function lmBox(lm){let x0=1,y0=1,x1=0,y1=0,n=0;for(const p of lm){if(!p||p.v<.3)continue;x0=Math.min(x0,p.x);x1=Math.max(x1,p.x);y0=Math.min(y0,p.y);y1=Math.max(y1,p.y);n++}return n>=4?{x0,y0,x1,y1}:null}
function drawSkel(g,lm,pw,ph,k){
  g.clearRect(0,0,pw,ph);g.strokeStyle=g.fillStyle='#fff';g.lineCap=g.lineJoin='round';
  const b=lmBox(lm);if(!b)return false;
  const ref=Math.max((b.y1-b.y0)*ph,(b.x1-b.x0)*pw*.9),P=i=>{const p=lm[i];return p&&p.v>.3?[p.x*pw,p.y*ph]:null};
  g.lineWidth=Math.max(2,ref*.085*k);
  for(const[a,c]of BONES){const A=P(a),B=P(c);if(A&&B){g.beginPath();g.moveTo(A[0],A[1]);g.lineTo(B[0],B[1]);g.stroke()}}
  const T=[11,12,24,23].map(P);if(T.every(Boolean)){g.beginPath();T.forEach((q,j)=>j?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1]));g.closePath();g.fill()}
  const e1=P(7),e2=P(8),no=P(0),hc=e1&&e2?[(e1[0]+e2[0])/2,(e1[1]+e2[1])/2]:no;
  if(hc){const r=Math.max(ref*.07,e1&&e2?Math.hypot(e1[0]-e2[0],e1[1]-e2[1])*.6:0)*k;g.beginPath();g.arc(hc[0],hc[1],r,0,7);g.fill();
    const s1=P(11),s2=P(12);if(s1&&s2){g.beginPath();g.moveTo(hc[0],hc[1]);g.lineTo((s1[0]+s2[0])/2,(s1[1]+s2[1])/2);g.stroke()}}
  return true;
}
const alphaMask=(g,pw,ph)=>{const d=g.getImageData(0,0,pw,ph).data,m=new Uint8Array(pw*ph);for(let i=0;i<m.length;i++)m[i]=d[4*i+3]>40?1:0;return m};
function dilate(m,pw,ph,n){for(let it=0;it<n;it++){const o=m.slice();for(let y=0;y<ph;y++)for(let x=0;x<pw;x++){const k=y*pw+x;if(m[k])continue;
  if((x>0&&m[k-1])||(x<pw-1&&m[k+1])||(y>0&&m[k-pw])||(y<ph-1&&m[k+pw])||(x>0&&y>0&&m[k-pw-1])||(x<pw-1&&y>0&&m[k-pw+1])||(x>0&&y<ph-1&&m[k+pw-1])||(x<pw-1&&y<ph-1&&m[k+pw+1]))o[k]=1}m.set(o)}}
// a model mask (any size, covers the square crop) → one value per plate pixel. Outside the crop there is never a person (whichever way the mask points).
function toPlane(mk,crop,pw,ph,W,H,inv){
  const out=new Float32Array(pw*ph),mw=mk.w,mh=mk.h;
  for(let y=0;y<ph;y++){const v=((y+.5)/ph*H-crop.y0)/crop.S*mh-.5;if(v<-.5||v>mh-.5)continue;const v0=Math.max(0,Math.min(mh-1,Math.floor(v))),v1=Math.min(mh-1,v0+1),fy=Math.max(0,Math.min(1,v-v0));
    for(let x=0;x<pw;x++){const u=((x+.5)/pw*W-crop.x0)/crop.S*mw-.5;if(u<-.5||u>mw-.5)continue;const u0=Math.max(0,Math.min(mw-1,Math.floor(u))),u1=Math.min(mw-1,u0+1),fx=Math.max(0,Math.min(1,u-u0));
      const val=(mk.d[v0*mw+u0]*(1-fx)+mk.d[v0*mw+u1]*fx)*(1-fy)+(mk.d[v1*mw+u0]*(1-fx)+mk.d[v1*mw+u1]*fx)*fy;out[y*pw+x]=inv?1-val:val}}
  return out;
}

/* ───────── wiring into the editor ───────── */
export function initBG(c){
  const $=c.$,st=c.st,V=c.V;
  let seg=null,segFail=false,loading=null,bg=null,busy=false;
  const fire=()=>window.dispatchEvent(new Event('rovio-bg'));
  const say=t=>{if($('bgMsg'))$('bgMsg').textContent=t;st(t)};
  const mk=(w,h)=>{const e=document.createElement('canvas');e.width=w;e.height=h;return e};
  const PW=128;

  async function ensureModel(){
    if(seg||segFail)return seg;
    if(loading)return loading;
    loading=(async()=>{
      try{st('Loading the background AI…');const fs=await FilesetResolver.forVisionTasks(WASM);
        seg=await ImageSegmenter.createFromOptions(fs,{baseOptions:{modelAssetPath:MODEL,delegate:'CPU'},runningMode:'IMAGE',outputCategoryMask:false,outputConfidenceMasks:true})}
      catch(e){console.warn('background AI',e);seg=null;segFail=true}
      loading=null;return seg})();
    return loading;
  }
  // run the segmenter on a canvas; returns [{w,h,d:Float32Array}] (works with both the callback and the return-value form of the API)
  function segRun(cv){
    let out=null;
    const grab=r=>{const ms=Array.isArray(r)?r:(r&&r.confidenceMasks)||[];out=ms.map(m=>({w:m.width,h:m.height,d:Float32Array.from(m.getAsFloat32Array())}))};
    const ret=seg.segment(cv,grab);
    if(!out&&ret){grab(ret);try{ret.close&&ret.close()}catch{}}
    return out||[];
  }

  async function capture(){
    if(busy)return;const fr=c.frames();if(!fr.length)return say('Analyze a video first, then the background can be rebuilt from it.');
    busy=true;c.resetStop();
    const bt=$('bgCap'),an=$('analyze'),cn=$('cancel');
    if(bt)bt.disabled=true;if(an)an.disabled=true;if(cn)cn.disabled=false;
    const t0=performance.now();
    try{
      await ensureModel();
      const n=fr.length,fps=c.fps(),start=c.start(),W=c.W(),H=c.H(),mode=($('bgMode')&&$('bgMode').value)||'static',
        every0=Math.max(1,Math.round(+($('bgEvery')&&$('bgEvery').value)||8)),every=Math.max(every0,Math.ceil(n/240));
      const idx=[];for(let i=0;i<n;i+=every)idx.push(i);if(idx[idx.length-1]!==n-1)idx.push(n-1);
      const pw=PW,ph=Math.max(16,Math.round(PW*H/W)),fc=mk(480,Math.max(8,Math.round(480*H/W))),fg=fc.getContext('2d',{willReadFrequently:true}),
        pc=mk(pw,ph),pg=pc.getContext('2d',{willReadFrequently:true}),cc=mk(256,256),cg=cc.getContext('2d'),kc=mk(pw,ph),kg=kc.getContext('2d',{willReadFrequently:true});
      const S=[];let aiFrames=0,skipped=0;
      // pass 1: cut the person out of every captured frame
      for(let q=0;q<idx.length&&!c.isStop();q++){
        const i=idx[q];await c.seek(start+i/fps);
        fg.drawImage(V,0,0,fc.width,fc.height);pg.clearRect(0,0,pw,ph);pg.drawImage(fc,0,0,pw,ph);
        const px=pg.getImageData(0,0,pw,ph).data,rgb=new Uint8ClampedArray(pw*ph*3);
        for(let k=0;k<pw*ph;k++){rgb[3*k]=px[4*k];rgb[3*k+1]=px[4*k+1];rgb[3*k+2]=px[4*k+2]}
        const m=new Uint8Array(pw*ph),lm=fr[i].lm,bb=lm&&lmBox(lm);
        if(bb){
          let ai=false;
          if(seg){
            drawSkel(kg,lm,pw,ph,.7);const sk=alphaMask(kg,pw,ph);let skN=0;for(const v of sk)skN+=v;
            const bw=(bb.x1-bb.x0)*W,bh=(bb.y1-bb.y0)*H,S_=Math.max(bw,bh)*1.4+8,cx=(bb.x0+bb.x1)/2*W,cy=(bb.y0+bb.y1)/2*H,crop={x0:cx-S_/2,y0:cy-S_/2,S:S_};
            cg.fillStyle='#808080';cg.fillRect(0,0,256,256);cg.drawImage(V,crop.x0,crop.y0,crop.S,crop.S,0,0,256,256);
            try{
              const ms=segRun(cc),cands=ms.map(x=>toPlane(x,crop,pw,ph,W,H,false));
              if(ms.length===1)cands.push(toPlane(ms[0],crop,pw,ph,W,H,true)); // one mask only: it may point at the person or at the background, the skeleton below decides
              let best=null,bm=0;
              for(const p of cands){let s=0;for(let k=0;k<p.length;k++)if(sk[k])s+=p[k];s=skN?s/skN:0;if(s>bm){bm=s;best=p}}
              if(best&&bm>.5){for(let k=0;k<best.length;k++)if(best[k]>.4)m[k]=1;ai=true}
            }catch(e){console.warn('segment',e)}
          }
          // the skeleton always counts, so a missed model mask never leaves the person in the background
          drawSkel(kg,lm,pw,ph,ai?1:1.45);const sk2=alphaMask(kg,pw,ph);for(let k=0;k<m.length;k++)if(sk2[k])m[k]=1;
          dilate(m,pw,ph,ai?1:2);if(ai)aiFrames++;
        }else skipped++;
        S.push({i,t:i/fps,rgb,m});
        $('prog').value=.5*(q+1)/idx.length;
        if(q%2===0){const el=(performance.now()-t0)/1000;say(`Background: cutting out the person, capture ${q+1}/${idx.length} · ~${Math.round(el/(q+1)*(idx.length-q-1))}s left`);await new Promise(r=>setTimeout(r))}
      }
      if(c.isStop()){say('Background capture cancelled.');return}
      // pass 2: rebuild the background around every capture
      const cs=mode==='moving'?c.shifts():null,CC=i=>fr[i].cc||[0,0];
      const align=(i,j)=>{const s=cs?cs[i]:[0,0],a=CC(i),b=CC(j);return[s[0]-a[0]+b[0],s[1]-a[1]+b[1]]};
      say('Background: rebuilding…');
      const r=await buildPlates(S,pw,ph,{mode,every,fps,align,onProg:p=>{$('prog').value=.5+.5*p}});
      bg={v:1,pw,ph,W,H,n,fps,mode,every,ai:aiFrames>0,aiFrames,plates:r.plates,holes:r.holes};
      const camNote=mode==='moving'&&!fr.some(f=>f.cc)?' Camera-shake compensation was off when you analyzed, so Moving mode could not follow the camera: turn it on in Scan and analyze again.':'';
      say(`Background rebuilt: ${mode==='moving'?'Moving':'Static'} · ${bg.plates.length} captures, one every ${every} frames · ${bg.ai?`AI person removal on ${aiFrames}/${S.length} captures`:segFail?'AI model could not load (needs internet), used the skeleton to remove the person':'skeleton-based person removal'}.${camNote}`);
      fire();c.redraw();
    }catch(e){console.error(e);say('Background capture failed: '+e.message)}
    finally{
      busy=false;if(bt)bt.disabled=false;if(an)an.disabled=false;if(cn)cn.disabled=true;
      try{c.V.currentTime=c.start()}catch{}
    }
  }

  // ---- panel controls
  if($('bgOn'))$('bgOn').onchange=async()=>{
    fire();
    if(!$('bgOn').checked){say('Background rebuild is off. The Alight export will not include a background.');c.redraw();return}
    segFail=false;const m=await ensureModel();
    say(m?'Background AI ready.':'The background AI could not load (it needs internet). The skeleton from the analysis will be used to remove the person instead.');
    if(c.frames().length&&!bg&&!busy&&$('analyze')&&!$('analyze').disabled)capture();
    else if(!c.frames().length)say((m?'Background AI ready. ':'')+'It runs on its own after you press Analyze.');
  };
  if($('bgCap'))$('bgCap').onclick=()=>capture();
  if($('bgClr'))$('bgClr').onclick=()=>{bg=null;say('Captured background cleared.');fire();c.redraw()};
  for(const i of['bgMode','bgEvery'])if($(i))$(i).onchange=()=>{if(bg)say('Settings changed. Press Capture background now to rebuild with them.')};

  return{
    capture,load:()=>ensureModel(),
    clear:()=>{if(bg){bg=null;fire()}},
    get:()=>bg,
    set:o=>{bg=o&&o.plates&&o.plates.length?o:null;fire();c.redraw()},
    // draw the rebuilt background (as the grid of parts the Alight export will use) onto a canvas
    paint:(g,w,h,i,q)=>{
      if(!bg)return false;const{cols,rows}=gridFor(q,bg.W,bg.H),t=tilesAt(tileSeries(bg,cols,rows),i);
      for(let r=0;r<rows;r++)for(let k=0;k<cols;k++){const o=3*(r*cols+k);g.fillStyle=`rgb(${t[o]},${t[o+1]},${t[o+2]})`;
        const x0=Math.floor(k*w/cols),y0=Math.floor(r*h/rows);g.fillRect(x0,y0,Math.floor((k+1)*w/cols)-x0+1,Math.floor((r+1)*h/rows)-y0+1)}
      return true;
    }
  };
}
