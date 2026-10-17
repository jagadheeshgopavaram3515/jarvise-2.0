'use client';

import { useEffect, useRef } from 'react';

const vertex = `
attribute vec4 particle;
uniform float energy;
uniform float time;
uniform float density;
uniform float rotation;
varying vec3 color;
varying float opacity;
void main(){
 vec3 p=particle.xyz;float seed=particle.w;
 float burst=pow(max(0.,sin(seed*97.+time*4.5)),12.)*energy;
 float wave=(sin(p.y*9.+time*4.2)*.055+sin(p.x*7.-time*3.1+p.z*4.)*.035)*energy*.25;
 p*=1.+wave+burst*.32;
 p.x+=sin(p.y*7.+time*3.8)*energy*.00875;
 p.y+=cos(p.x*6.-time*3.2)*energy*.00625;
 p+=vec3(sin(time*12.+seed*76.),cos(time*10.+seed*53.),0.)*energy*.012;
 float c=cos(rotation),s=sin(rotation);
 p=vec3(p.x*c+p.z*s,p.y,-p.x*s+p.z*c);
 float perspective=1.+p.z*.07;
 gl_Position=vec4(p.xy*perspective/1.42,0.,1.);
 gl_PointSize=density*(.72+seed*.62+burst*.35);
 float greenBlend=smoothstep(-.95,-.15,particle.x);
 color=mix(vec3(.12,.58,.78),vec3(.24,.93,.40),greenBlend);
 float yellowEdge=smoothstep(.45,.98,particle.x);
 color=mix(color,vec3(.98,.88,.20),yellowEdge);
 color*=.72+seed*.35;
 opacity=.65+smoothstep(-1.,1.,p.z)*.3;
}`;
const fragment = `
precision mediump float;
varying vec3 color;
varying float opacity;
void main(){float d=length(gl_PointCoord-vec2(.5));float alpha=1.-smoothstep(.18,.5,d);if(alpha<.01)discard;gl_FragColor=vec4(color,alpha*opacity);}
`;

export function VoiceSphere({ level, label, thinking = false }: { level: number; label: string; thinking?: boolean }) {
 const canvas=useRef<HTMLCanvasElement>(null),input=useRef(level);input.current=level;
 const thinkingState=useRef(thinking);thinkingState.current=thinking;
 useEffect(()=>{
  const element=canvas.current;
  const gl=element?.getContext('webgl',{alpha:true,antialias:true,premultipliedAlpha:false});
  if(!gl||!element){element?.classList.add('sphere-fallback');return;}
  const compile=(kind:number,source:string)=>{const shader=gl.createShader(kind)!;gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){gl.deleteShader(shader);return null;}return shader;};
  const vs=compile(gl.VERTEX_SHADER,vertex),fs=compile(gl.FRAGMENT_SHADER,fragment);
  if(!vs||!fs){element.classList.add('sphere-fallback');return;}
  const program=gl.createProgram()!;gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS)){element.classList.add('sphere-fallback');return;}
  gl.useProgram(program);
  let randomState=14731;
  const random=()=>{randomState=(Math.imul(randomState,1664525)+1013904223)>>>0;return randomState/4294967296;};
  const count=18000,points=new Float32Array(count*4);
  for(let i=0;i<count;i++){
   const z=random()*2-1,angle=random()*Math.PI*2,r=.93+random()*.075,ring=Math.sqrt(1-z*z);
   points.set([Math.cos(angle)*ring*r,Math.sin(angle)*ring*r,z*r,random()],i*4);
  }
  const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,points,gl.STATIC_DRAW);
  const attribute=gl.getAttribLocation(program,'particle');gl.enableVertexAttribArray(attribute);gl.vertexAttribPointer(attribute,4,gl.FLOAT,false,0,0);
  gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.clearColor(0,0,0,0);
  const energy=gl.getUniformLocation(program,'energy'),time=gl.getUniformLocation(program,'time'),density=gl.getUniformLocation(program,'density'),rotation=gl.getUniformLocation(program,'rotation');
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let frame=0,smoothed=0,activeTime=0,last=0,idleRotation=0;
  const render=(now:number)=>{
   const delta=Math.min(50,now-(last||now));last=now;smoothed+=(input.current-smoothed)*.2;
   if(input.current<.015&&smoothed<.02)smoothed=0;
   if(smoothed>0)activeTime+=delta/1000;
   if(!reduced&&smoothed===0&&!thinkingState.current)idleRotation=(idleRotation+delta*.00012)%(Math.PI*2);
   gl.viewport(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight);gl.clear(gl.COLOR_BUFFER_BIT);
   gl.uniform1f(energy,reduced?0:smoothed);gl.uniform1f(time,activeTime);
   gl.uniform1f(rotation,idleRotation);
   gl.uniform1f(density,Math.max(1,element.width/Math.max(1,element.clientWidth))*2.0);
   gl.drawArrays(gl.POINTS,0,count);frame=requestAnimationFrame(render);
  };
  const resize=new ResizeObserver(()=>{const size=Math.min(1600,Math.ceil(element.getBoundingClientRect().width*Math.max(2,window.devicePixelRatio)));if(size>0&&element.width!==size){element.width=size;element.height=size;}});
  resize.observe(element);frame=requestAnimationFrame(render);
  return()=>{resize.disconnect();cancelAnimationFrame(frame);gl.deleteBuffer(buffer);gl.deleteProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);};
 },[]);
 return <div className={`voice-sphere ${thinking?'is-thinking':''}`} aria-label={`${label} particle sphere`}><canvas ref={canvas} width={1000} height={1000} aria-hidden="true" /></div>;
}
