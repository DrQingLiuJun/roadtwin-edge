/* Complete-frame MJPEG playback: one decoder and one pending frame, never a backlog. */
(function(root){
'use strict';
class MultipartFrames {
 constructor(emit){this.emit=emit;this.buffer=new Uint8Array(0);this.header=null;this.length=0;}
 push(chunk){
  const merged=new Uint8Array(this.buffer.length+chunk.length);merged.set(this.buffer);merged.set(chunk,this.buffer.length);this.buffer=merged;
  while(this.buffer.length){
   if(!this.header){
    let end=-1;for(let i=0;i+3<this.buffer.length;i++)if(this.buffer[i]===13&&this.buffer[i+1]===10&&this.buffer[i+2]===13&&this.buffer[i+3]===10){end=i;break;}
    if(end<0){if(this.buffer.length>16384)throw Error('视频帧头异常');return;}
    const header=new TextDecoder().decode(this.buffer.subarray(0,end)),length=Number(header.match(/(?:^|\r\n)Content-Length:\s*(\d+)/i)?.[1]);
    if(!Number.isInteger(length)||length<4||length>8*1024*1024)throw Error('视频帧长度异常');
    this.header=header;this.length=length;this.buffer=this.buffer.subarray(end+4);
   }
   if(this.buffer.length<this.length)return;
   const jpeg=this.buffer.slice(0,this.length),header=this.header;this.buffer=this.buffer.subarray(this.length);this.header=null;
   if(jpeg[0]!==255||jpeg[1]!==216||jpeg[jpeg.length-2]!==255||jpeg[jpeg.length-1]!==217)throw Error('视频帧不完整');
   this.emit(jpeg,header);
  }
 }
}
class DetectionPlayer {
 constructor(canvas,url,onState=()=>{}){
  this.canvas=canvas;this.context=canvas.getContext('2d',{alpha:false});this.url=url;this.onState=onState;this.running=false;this.generation=0;this.lastFrame=0;this.frames=0;this.reconnects=0;this.decoding=false;this.pending=null;this.profile='wall';this.degraded=0;
 }
 start(profile='wall'){if(this.running&&this.profile===profile)return;this.stop();this.profile=profile;this.running=true;this.connect();}
 stop(){this.running=false;this.generation++;clearTimeout(this.retry);clearInterval(this.watchdog);this.controller?.abort();this.pending=null;}
 signal(state){this.onState(state,this);}
 async connect(){
  if(!this.running)return;const generation=++this.generation;this.controller=new AbortController();let lastBytes=performance.now(),windowStart=lastBytes,windowFrames=this.frames;
  this.signal(this.lastFrame?'reconnecting':'connecting');
  this.watchdog=setInterval(()=>{
   const now=performance.now();if(now-lastBytes>4500||this.lastFrame&&now-this.lastFrame>4500)this.controller.abort();
   if(this.lastFrame&&now-this.lastFrame>4500)this.signal('reconnecting');
   if(now-windowStart>6000){
    const fps=(this.frames-windowFrames)*1000/(now-windowStart);windowStart=now;windowFrames=this.frames;
    if(fps<2.5&&this.profile!=='low'){this.profile=this.profile==='native'?'wall':this.profile==='wall'?'compact':'low';this.degraded++;this.controller.abort();}
   }
  },500);
  try{
   const response=await fetch(this.url+`?profile=${this.profile}&t=${Date.now()}`,{cache:'no-store',signal:this.controller.signal});
   if(!response.ok||!response.body)throw Error(`视频连接 ${response.status}`);
   const parser=new MultipartFrames((jpeg,header)=>{this.pending={jpeg,header,generation};this.decode();});
   const reader=response.body.getReader();
   try{while(this.running&&generation===this.generation){const {value,done}=await reader.read();if(done)throw Error('视频连接结束');lastBytes=performance.now();parser.push(value);}}
   finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
  }catch(error){if(this.running&&generation===this.generation)this.signal('reconnecting');}
  finally{
   if(generation===this.generation){clearInterval(this.watchdog);this.controller.abort();if(this.running){this.reconnects++;this.retry=setTimeout(()=>this.connect(),500+Math.random()*300);}}
  }
 }
 async decode(){
  if(this.decoding)return;this.decoding=true;
  try{
   while(this.pending){
    const {jpeg,header,generation}=this.pending;this.pending=null;let bitmap;
    try{
     bitmap=await createImageBitmap(new Blob([jpeg],{type:'image/jpeg'}));
     if(!this.running||generation!==this.generation)continue;
     if(this.canvas.width!==bitmap.width||this.canvas.height!==bitmap.height){this.canvas.width=bitmap.width;this.canvas.height=bitmap.height;}
     this.context.drawImage(bitmap,0,0);this.lastFrame=performance.now();this.frames++;this.canvas.dataset.frames=String(this.frames);this.canvas.dataset.sequence=header.match(/X-Frame-Sequence:\s*(\d+)/i)?.[1]||'';this.signal('live');
    }catch(error){/* Keep the previous complete frame when decoding fails. */}
    finally{bitmap?.close();}
   }
  }finally{this.decoding=false;}
 }
}
root.MultipartFrames=MultipartFrames;root.DetectionPlayer=DetectionPlayer;
if(typeof module!=='undefined')module.exports={MultipartFrames,DetectionPlayer};
})(typeof window==='undefined'?globalThis:window);
