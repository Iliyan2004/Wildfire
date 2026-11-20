// PAKFIRE start page: a night conifer forest with a small fire that spreads from tree to tree when the button is pressed.
// Trees, flames, smoke, embers and lights are rendered with three.js; the spread itself comes from forestfire.js.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { buildForest, Fire, surgeControl, seedFire, mulberry32 } from "./forestfire.js";

const rnd = Math.random;
const smooth = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };

// ---------------------------------------------------------------- shaders
const BLACKBODY = `
// colour of a blackbody at T kelvin (Tanner Helland's fit of the Planckian locus), as linear RGB, and its radiance relative to 1500 K
vec3 blackbody(float T){
  float t = clamp(T, 900., 6000.)/100.;
  vec3 c;
  c.r = t <= 66. ? 1. : clamp(1.292936*pow(t-60., -0.1332047), 0., 1.);
  c.g = t <= 66. ? clamp(0.390081*log(t) - 0.631841, 0., 1.) : clamp(1.129891*pow(t-60., -0.0755148), 0., 1.);
  c.b = t >= 66. ? 1. : (t <= 19. ? 0. : clamp(0.543207*log(t-10.) - 1.196254, 0., 1.));
  return pow(c, vec3(2.2));
}
float radiance(float T){ float r = T/1500.; return r*r*r*r; }   // Stefan-Boltzmann: emitted power ~ T^4
`;
const NOISE = `
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.-2.*f);
  return mix(mix(hash(i), hash(i+vec2(1.,0.)), f.x), mix(hash(i+vec2(0.,1.)), hash(i+vec2(1.,1.)), f.x), f.y); }
`;
const FLAME_VS = `
attribute vec4 aData;   // age (s), max age (s), size (m), seed
attribute float aTemp;  // kelvin at birth
uniform float uPx; uniform float uTime;
varying float vAge; varying float vSeed; varying float vTemp; varying float vFade;
void main(){
  float age = aData.x/aData.y;
  vAge = age; vSeed = aData.w; vTemp = aTemp;
  vec4 mv = modelViewMatrix*vec4(position, 1.);
  float dist = max(-mv.z, 0.5);
  float grow = 0.55 + 0.75*sin(min(age, 1.)*3.14159*0.8);
  float px = aData.z*grow*uPx/dist;
  vFade = aData.x >= aData.y ? 0. : 1.;
  gl_PointSize = vFade > 0.5 ? clamp(px, 1., 640.) : 0.;
  gl_Position = projectionMatrix*mv;
}`;
const FLAME_FS = `
precision highp float;
uniform float uTime; uniform float uExposure;
varying float vAge; varying float vSeed; varying float vTemp; varying float vFade;
${NOISE}
${BLACKBODY}
void main(){
  if(vFade < 0.5) discard;
  vec2 p = gl_PointCoord*2. - 1.;
  p.y = -p.y;
  vec2 q = vec2(p.x*(1. + 0.9*(0.5 + 0.5*p.y)), p.y*0.9 + 0.12);         // teardrop: narrower towards the tip
  float d = length(q);
  float n = noise(p*3.2 + vec2(vSeed*17., -uTime*3.2 + vSeed*9.));
  float n2 = noise(p*7.5 + vec2(vSeed*5., -uTime*5.));
  float a = smoothstep(1.0, 0.10, d + (n - 0.5)*0.85*(0.5 + vAge) + (n2 - 0.5)*0.25);
  a *= smoothstep(0., 0.07, vAge)*(1. - smoothstep(0.55, 1.0, vAge));
  float T = vTemp*(1. - 0.50*vAge) * (0.80 + 0.35*(1. - d));               // cooler with age and towards the edge
  vec3 c = blackbody(T)*radiance(T);
  gl_FragColor = vec4(c*a*uExposure, a);
}`;
const TONGUE_VS = `
attribute vec3 iPos; attribute vec4 iData; attribute float iTemp;
uniform vec3 uWind;
varying vec2 vUv; varying float vAge; varying float vSeed; varying float vTemp; varying float vOn;
void main(){
  float age = iData.x/iData.y; vAge = age; vSeed = iData.w; vTemp = iTemp; vUv = uv; vOn = iData.x >= iData.y ? 1. : 0.; vOn = 1. - vOn;
  float size = iData.z*(0.65 + 0.55*sin(min(age, 1.)*3.14159*0.85));
  vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]); right = normalize(vec3(right.x, 0., right.z));
  float sh = position.y*position.y;
  vec3 wp = iPos + right*(position.x*size*(0.62 + 0.5*fract(iData.w*7.3))) + vec3(0., 1., 0.)*(position.y*size*(1.7 + 0.9*fract(iData.w*3.1))) + uWind*(sh*size*0.16);
  gl_Position = projectionMatrix*viewMatrix*vec4(wp, 1.);
  if(vOn < 0.5) gl_Position = vec4(2., 2., 2., 1.);
}`;
const TONGUE_FS = `
precision highp float;
uniform float uTime; uniform float uExposure;
varying vec2 vUv; varying float vAge; varying float vSeed; varying float vTemp; varying float vOn;
${NOISE}
${BLACKBODY}
void main(){
  if(vOn < 0.5) discard;
  vec2 p = vec2(vUv.x*2. - 1., vUv.y);
  float y = p.y;
  float wob = noise(vec2(vSeed*31. + y*3.2, uTime*3.1 + vSeed*13.))*2. - 1.;
  p.x += wob*0.38*y;
  float n = noise(vec2(p.x*3. + vSeed*17., y*4.2 - uTime*4.4 + vSeed*9.));
  float n2 = noise(vec2(p.x*8. + vSeed*5., y*9. - uTime*7.5));
  float w = pow(1. - y, 0.72)*(0.80 + 0.30*n) + 0.03;
  float d = abs(p.x)/w;
  float body = smoothstep(1.0, 0.30, d + (n2 - 0.5)*0.50)*smoothstep(0., 0.05, y);
  float tip = 1. - smoothstep(0.50, 1.0, y + (n - 0.5)*0.55);
  float a = body*tip*(1. - smoothstep(0.55, 1.0, vAge))*smoothstep(0., 0.06, vAge);
  float T = vTemp*(1. - 0.46*y)*(1. - 0.28*vAge)*(0.80 + 0.35*(1. - d));
  vec3 c = blackbody(T)*radiance(T);
  gl_FragColor = vec4(c*a*uExposure, a);
}`;
const SMOKE_VS = `
attribute vec4 aData; attribute float aTemp;   // aTemp: how much fire-light the puff carries at birth (0..1)
uniform float uPx;
varying float vAge; varying float vSeed; varying float vWarm; varying float vFade;
void main(){
  float age = aData.x/aData.y;
  vAge = age; vSeed = aData.w; vWarm = aTemp*pow(max(1. - age*1.6, 0.), 2.);
  vec4 mv = modelViewMatrix*vec4(position, 1.);
  float dist = max(-mv.z, 0.5);
  vFade = aData.x >= aData.y ? 0. : 1.;
  gl_PointSize = vFade > 0.5 ? clamp(aData.z*(0.6 + 1.4*age)*uPx/dist, 1., 900.) : 0.;
  gl_Position = projectionMatrix*mv;
}`;
const SMOKE_FS = `
precision highp float;
uniform float uTime; uniform vec3 uSky;
varying float vAge; varying float vSeed; varying float vWarm; varying float vFade;
${NOISE}
void main(){
  if(vFade < 0.5) discard;
  vec2 p = gl_PointCoord*2. - 1.;
  float d = length(p);
  float n = noise(p*2.2 + vec2(vSeed*13., uTime*0.15 + vSeed*7.)) * 0.6 + noise(p*5. + vSeed*3.) * 0.4;
  float a = smoothstep(1.0, 0.0, d + (n - 0.5)*0.8);
  a *= smoothstep(0., 0.12, vAge)*(1. - smoothstep(0.5, 1.0, vAge))*0.20;
  vec3 col = vec3(0.020, 0.020, 0.024) + uSky*0.4 + vec3(1.0, 0.36, 0.08)*vWarm*(0.5 + 0.8*n)*0.55;
  gl_FragColor = vec4(col, a);
}`;
const SPARK_VS = `
attribute vec4 aData; attribute float aTemp;
uniform float uPx;
varying float vA; varying float vTemp;
void main(){
  float age = aData.x/aData.y;
  vec4 mv = modelViewMatrix*vec4(position, 1.);
  float dist = max(-mv.z, 0.5);
  bool dead = aData.x >= aData.y;
  vA = dead ? 0. : (1. - smoothstep(0.6, 1., age))*(0.65 + 0.35*sin(aData.x*20. + aData.w*40.));
  vTemp = aTemp*(1. - 0.35*age);
  gl_PointSize = dead ? 0. : clamp(aData.z*uPx/dist, 1.5, 40.);
  gl_Position = projectionMatrix*mv;
}`;
const SPARK_FS = `
precision highp float;
uniform float uExposure; varying float vA; varying float vTemp;
${BLACKBODY}
void main(){
  vec2 p = gl_PointCoord*2. - 1.;
  float a = smoothstep(1., 0., length(p))*vA;
  if(a < 0.01) discard;
  gl_FragColor = vec4(blackbody(vTemp)*radiance(vTemp)*2.4*uExposure*a, a);
}`;
const HALO_VS = `
attribute float aSize; attribute float aPow;
uniform float uPx; varying float vPow;
void main(){
  vec4 mv = modelViewMatrix*vec4(position, 1.); float dist = max(-mv.z, 0.5);
  vPow = aPow; gl_PointSize = clamp(aSize*uPx/dist, 0., 1400.); gl_Position = projectionMatrix*mv;
}`;
const HALO_FS = `
precision highp float;
varying float vPow;
void main(){
  vec2 p = gl_PointCoord*2. - 1.; float d = length(p);
  float a = exp(-d*d*3.2)*smoothstep(1., 0.7, d)*vPow;
  gl_FragColor = vec4(vec3(1.0, 0.36, 0.08)*a, a);
}`;
const SKY_VS = `varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position, 1.); }`;
const SKY_FS = `
precision highp float;
varying vec3 vDir; uniform float uHeat; uniform float uTime;
${NOISE}
void main(){
  vec3 d = normalize(vDir);
  float y = clamp(d.y, 0., 1.);
  vec3 col = mix(vec3(0.020, 0.026, 0.044), vec3(0.002, 0.005, 0.014), pow(y, 0.55));
  // stars
  vec2 sp = vec2(atan(d.x, d.z)*60., d.y*120.);
  vec2 id = floor(sp); float r = hash(id);
  float star = step(0.9935, r)*smoothstep(0.5, 0., length(fract(sp) - 0.5))*(0.5 + 0.5*sin(uTime*(1.+r*3.) + r*40.));
  col += vec3(0.8, 0.88, 1.)*star*smoothstep(0.08, 0.5, d.y)*(1. - clamp(uHeat*0.4, 0., 1.));
  // faint moon
  vec3 mdir = normalize(vec3(-0.45, 0.55, -0.7));
  float md = dot(d, mdir);
  col += vec3(0.55, 0.65, 0.85)*(smoothstep(0.9990, 0.9996, md)*0.9 + pow(max(md, 0.), 220.)*0.25);
  // light of the fire on the horizon and smoke
  float glow = exp(-pow(d.y/0.35, 2.))*smoothstep(-0.7, 0.2, -d.z)*(0.15 + 0.85*clamp(uHeat, 0., 1.5));
  col += vec3(1.0, 0.30, 0.06)*glow*0.20*uHeat;
  gl_FragColor = vec4(col, 1.);
}`;
const FINAL = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uHeat: { value: 0 }, uFlash: { value: 0 }, uExposure: { value: 1 }, uRes: { value: new THREE.Vector2(1, 1) } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position, 1.); }`,
  fragmentShader: `
    precision highp float;
    uniform sampler2D tDiffuse; uniform float uTime; uniform float uHeat; uniform float uFlash; uniform float uExposure; uniform vec2 uRes;
    varying vec2 vUv;
    ${NOISE}
    vec3 aces(vec3 x){ return clamp((x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14), 0., 1.); }
    void main(){
      vec2 uv = vUv;
      // heat shimmer: a faint refractive wobble that grows with the fire
      float sh = 0.0012*clamp(uHeat, 0., 2.);
      uv += (vec2(noise(uv*vec2(40., 90.) + vec2(0., -uTime*2.4)), noise(uv*vec2(55., 70.) + vec2(uTime*0.7, -uTime*2.9))) - 0.5)*sh*(0.3 + smoothstep(0.0, 0.55, 1. - uv.y)*0.7);
      vec3 c = texture2D(tDiffuse, uv).rgb*uExposure;
      c = aces(c);
      c = pow(c, vec3(1./2.2));
      vec2 q = vUv - 0.5; c *= 1. - 0.9*dot(q, q)*0.9;
      c += (hash(gl_FragCoord.xy + uTime*60.) - 0.5)/110.;
      c = mix(c, vec3(1.0, 0.80, 0.45), uFlash);
      gl_FragColor = vec4(c, 1.);
    }`,
};

// ---------------------------------------------------------------- geometry
function branchTexture() {
  // a fir frond: a twig with side twigs and needles, drawn on a transparent card (alpha-tested)
  const W = 512, H = 256, c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d"), r = mulberry32(9);
  x.lineCap = "round";
  x.strokeStyle = "rgb(80,60,40)"; x.lineWidth = 5; x.beginPath(); x.moveTo(4, 128); x.quadraticCurveTo(W * 0.5, 124, W - 10, 130); x.stroke();
  for (let i = 0; i < 64; i++) {
    const t = i / 64, bx = 12 + t * (W - 40), by = 128 + (t < 0.5 ? -1 : 1) * 2 * Math.sin(t * 6);
    const len = (1 - t * 0.6) * (100 + 30 * r());
    for (const side of [-1, 1]) {
      const ang = side * (0.62 + 0.30 * r()), ex = bx + Math.cos(ang) * len * 0.30, ey = by + Math.sin(ang) * len;
      x.strokeStyle = "rgb(70,52,34)"; x.lineWidth = 2.2; x.beginPath(); x.moveTo(bx, by); x.lineTo(ex, ey); x.stroke();
      const nn = 12 + (len / 9) | 0;
      for (let k = 0; k < nn; k++) {
        const u = (k + 0.5) / nn, px = bx + (ex - bx) * u, py = by + (ey - by) * u;
        for (const s2 of [-1, 1]) {
          const na = Math.atan2(ey - by, ex - bx) + s2 * (0.85 + 0.45 * r()), nl = 8 + 8 * r() * (1 - 0.4 * u);
          x.strokeStyle = `rgb(${38 + 40 * r() | 0},${78 + 55 * r() | 0},${42 + 28 * r() | 0})`; x.lineWidth = 2.0;
          x.beginPath(); x.moveTo(px, py); x.lineTo(px + Math.cos(na) * nl, py + Math.sin(na) * nl); x.stroke();
        }
      }
    }
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}
function cardGeo(cards) {
  // cards: list of {o:[x,y,z] root, d:[x,y,z] unit direction, up:[x,y,z] width direction, L, hw, shade}
  const pos = [], uv = [], col = [], nor = [], idx = [];
  for (const c of cards) {
    const k = pos.length / 3, [ox, oy, oz] = c.o, [dx, dy, dz] = c.d, [sx, sy, sz] = c.w;
    const tipx = ox + dx * c.L, tipy = oy + dy * c.L, tipz = oz + dz * c.L;
    const hw = c.hw;
    const P = [[ox - sx * hw, oy - sy * hw, oz - sz * hw], [ox + sx * hw, oy + sy * hw, oz + sz * hw], [tipx + sx * hw, tipy + sy * hw, tipz + sz * hw], [tipx - sx * hw, tipy - sy * hw, tipz - sz * hw]];
    const U = [[0, 0], [0, 1], [1, 1], [1, 0]], ao = [0.35, 0.35, 1.0, 1.0];
    let nx = dx * 0.55 + c.out[0] * 0.5, ny = dy * 0.3 + 0.75, nz = dz * 0.55 + c.out[2] * 0.5; const nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;
    for (let i = 0; i < 4; i++) { pos.push(...P[i]); uv.push(...U[i]); const v = c.shade * ao[i]; col.push(0.55 * v, 0.95 * v, 0.62 * v); nor.push(nx, ny, nz); }
    idx.push(k, k + 1, k + 2, k, k + 2, k + 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3)); g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3)); g.setIndex(idx);
  return g;
}
function crownGeometry(seed, tiers = 20) {
  // a conifer crown of unit height, widest radius ~1: tiers of frond cards, each tier two crossed layers, drooping at the lower tiers
  const r = mulberry32(seed), cards = [];
  for (let k = 0; k < tiers; k++) {
    const f = k / (tiers - 1), y0 = 0.16 + 0.80 * f, R = Math.pow(1 - f, 0.85) * (0.95 + 0.14 * r()) + 0.10, M = 10 + Math.round(8 * (1 - f)), rot = r() * 6.28;
    for (let i = 0; i < M; i++) {
      const a = rot + (i + 0.5 * (k % 2)) / M * Math.PI * 2 + (r() - 0.5) * 0.3, ca = Math.cos(a), sa = Math.sin(a);
      const L = R * (0.46 + 0.30 * r()) + 0.06, droop = (0.32 - 0.24 * f) * (0.7 + 0.6 * r());
      const dlen = Math.hypot(1, droop), dx = ca / dlen, dy = -droop / dlen, dz = sa / dlen;
      for (let layer = 0; layer < 2; layer++) {
        const tilt = layer ? 1.05 : 0.0;
        const wx = -sa * Math.cos(tilt) + 0 * Math.sin(tilt), wz = ca * Math.cos(tilt), wy = Math.sin(tilt) * 0.9;
        const wl = Math.hypot(wx, wy, wz);
        cards.push({ o: [ca * 0.03, y0 + (r() - 0.5) * 0.02, sa * 0.03], d: [dx, dy, dz], w: [wx / wl, wy / wl, wz / wl], L, hw: L * 0.42, shade: (0.62 + 0.55 * r()) * (layer ? 0.85 : 1), out: [ca, 0, sa] });
      }
    }
  }
  // a short leader at the top
  for (let i = 0; i < 4; i++) { const a = i * 1.57 + r(); cards.push({ o: [0, 0.93, 0], d: [Math.cos(a) * 0.12, 0.99, Math.sin(a) * 0.12], w: [-Math.sin(a), 0, Math.cos(a)], L: 0.12, hw: 0.04, shade: 0.9, out: [0, 0, 0] }); }
  return cardGeo(cards);
}
function trunkGeometry() {
  const g = new THREE.CylinderGeometry(0.62, 1, 1, 8, 7, true); g.translate(0, 0.5, 0);
  const p = g.attributes.position, c = []; const r = mulberry32(3);
  for (let i = 0; i < p.count; i++) { const y = p.getY(i), a = Math.atan2(p.getZ(i), p.getX(i)); const k = 1 + (r() - 0.5) * 0.10 + 0.06 * Math.sin(a * 5 + y * 9); p.setXYZ(i, p.getX(i) * k, y, p.getZ(i) * k);
    const d = 0.65 + 0.5 * (0.5 + 0.5 * Math.sin(a * 7 + y * 5)); c.push(0.11 * d, 0.07 * d, 0.05 * d); }
  g.setAttribute("color", new THREE.Float32BufferAttribute(c, 3)); g.computeVertexNormals(); return g;
}
function fernGeometry(seed) {
  const r = mulberry32(seed), cards = [];
  const M = 9;
  for (let i = 0; i < M; i++) {
    const a = i / M * Math.PI * 2 + r() * 0.4, ca = Math.cos(a), sa = Math.sin(a), L = 0.7 + 0.5 * r(), up = 0.55 + 0.5 * r(), dl = Math.hypot(1, up);
    cards.push({ o: [0, 0.05, 0], d: [ca / dl, up / dl, sa / dl], w: [-sa, 0, ca], L, hw: L * 0.30, shade: 0.8 + 0.5 * r(), out: [ca, 0, sa] });
  }
  return cardGeo(cards);
}
function splitConifer(seed) { return { trunk: null, crown: crownGeometry(seed) }; }

function groundTexture() {
  const S = 512, c = document.createElement("canvas"); c.width = c.height = S;
  const x = c.getContext("2d"), img = x.createImageData(S, S), d = img.data;
  const h = (i, j) => { const s = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return s - Math.floor(s); };
  const n = (px, py, f) => { const X = px * f, Y = py * f, i = Math.floor(X), j = Math.floor(Y), u = X - i, v = Y - j, su = u * u * (3 - 2 * u), sv = v * v * (3 - 2 * v);
    return (h(i % f, j % f) * (1 - su) + h((i + 1) % f, j % f) * su) * (1 - sv) + (h(i % f, (j + 1) % f) * (1 - su) + h((i + 1) % f, (j + 1) % f) * su) * sv; };
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    const v = 0.55 * n(i, j, 8) + 0.30 * n(i, j, 32) + 0.15 * n(i, j, 128);
    const needle = n(i * 0.3, j * 4, 64) > 0.72 ? 0.5 : 0;                // pine needles and twigs
    const k = (j * S + i) * 4, L = 0.05 + 0.20 * v + 0.12 * needle;
    d[k] = 255 * L * 1.0; d[k + 1] = 255 * L * 0.72; d[k + 2] = 255 * L * 0.50; d[k + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(36, 36); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}
function ridgeTexture(seed, h0) {
  const W = 1024, H = 256, c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d"), r = mulberry32(seed); x.fillStyle = "#fff";
  x.beginPath(); x.moveTo(0, H);
  let y = h0, v = 0; for (let i = 0; i <= W; i += 4) { v += (r() - 0.5) * 7; v *= 0.92; y = Math.min(H * 0.92, Math.max(H * 0.2, y + v)); x.lineTo(i, H - y); }
  x.lineTo(W, H); x.closePath(); x.fill();
  const t = new THREE.CanvasTexture(c); return t;
}

// ---------------------------------------------------------------- the scene
export function init(opts = {}) {
  const canvas = opts.canvas;
  const q = new URLSearchParams(location.search);
  const quality = q.get("q") || "high";
  const Q = { low: { flame: 2600, smoke: 1300, spark: 600, pr: 0.55 }, mid: { flame: 4800, smoke: 2400, spark: 1000, pr: 0.8 }, high: { flame: 7600, smoke: 3600, spark: 1600, pr: 1.0 } }[quality] || { flame: 7600, smoke: 3600, spark: 1600, pr: 1 };
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance", alpha: false });
  let scale = q.has("scale") ? parseFloat(q.get("scale")) : Math.min(window.devicePixelRatio || 1, 1.5) * Q.pr;
  let qs = 1;                                                    // emission scale, lowered if the device is slow
  renderer.setClearColor(0x020408, 1);
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0c1626, 0.0085);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.3, 480);
  const camBase = new THREE.Vector3(0.8, 2.6, 46), look = new THREE.Vector3(-0.5, 7.2, -30);

  // sky and distant ridges
  const skyMat = new THREE.ShaderMaterial({ vertexShader: SKY_VS, fragmentShader: SKY_FS, side: THREE.BackSide, depthWrite: false, fog: false, uniforms: { uHeat: { value: 0 }, uTime: { value: 0 } } });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(380, 32, 20), skyMat); sky.renderOrder = -10; scene.add(sky);
  [[-185, 70, 0x0a1018, 41, 105], [-150, 50, 0x070c14, 17, 86]].forEach(([z, h, c, s, y0], i) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(720, h * 1.7), new THREE.MeshBasicMaterial({ color: c, alphaMap: ridgeTexture(s, y0), transparent: true, depthWrite: false, fog: true }));
    m.position.set(i * 40 - 20, h * 0.55 - 2, z); scene.add(m);
  });

  // lights: dim blue moonlight and sky, plus a pool of point lights that follow the fire
  scene.add(new THREE.HemisphereLight(0x2a3d66, 0x0a0705, 1.15));
  const moon = new THREE.DirectionalLight(0x7a98d0, 1.9); moon.position.set(-30, 50, -40); scene.add(moon);
  const NL = 10, lights = [];
  for (let i = 0; i < NL; i++) { const L = new THREE.PointLight(0xff7a2c, 0, 0, 2); L.position.set(0, 3, -30); scene.add(L); lights.push({ L, tx: 0, ty: 3, tz: -30, ti: 0, i: 0 }); }

  // ground
  const gMat = new THREE.MeshStandardMaterial({ map: groundTexture(), roughness: 1, metalness: 0, bumpScale: 1.5 });
  gMat.bumpMap = gMat.map;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(520, 520), gMat); ground.rotation.x = -Math.PI / 2; scene.add(ground);

  // forest
  const forest = buildForest(7);
  const fire = new Fire(forest, 11);
  const seeds = seedFire(fire);
  const nT = forest.nTrees, nAll = forest.n;
  const A = splitConifer(101), B = splitConifer(202), TRUNK = trunkGeometry(), BRANCH = branchTexture();
  const treeMat = (glow) => {
    const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0 });
    m.onBeforeCompile = (sh) => {
      sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nattribute float aGlow; varying float vGlow;").replace("#include <begin_vertex>", "#include <begin_vertex>\nvGlow = aGlow;");
      sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\nvarying float vGlow;")
        .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\ntotalEmissiveRadiance += vec3(1.0, 0.30, 0.05)*vGlow*(0.05 + 2.2*vColor.g);");
    };
    return m;
  };
  const trunkMat = treeMat(), crownMat = treeMat();
  crownMat.map = BRANCH; crownMat.alphaTest = 0.45; crownMat.side = THREE.DoubleSide; crownMat.roughness = 0.85;
  const idsA = [], idsB = []; for (let i = 0; i < nT; i++) (i % 2 ? idsB : idsA).push(i);
  const mk = (geo, mat, ids) => { const m = new THREE.InstancedMesh(geo, mat, ids.length); m.frustumCulled = false;
    m.geometry.setAttribute("aGlow", new THREE.InstancedBufferAttribute(new Float32Array(ids.length), 1)); m.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(ids.length * 3).fill(1), 3); scene.add(m); return m; };
  const trunkA = mk(TRUNK.clone(), trunkMat, idsA), crownA = mk(A.crown, crownMat, idsA), trunkB = mk(TRUNK.clone(), trunkMat, idsB), crownB = mk(B.crown, crownMat, idsB);
  const slot = new Int32Array(nT), grp = new Uint8Array(nT);
  idsA.forEach((id, k) => { slot[id] = k; grp[id] = 0; }); idsB.forEach((id, k) => { slot[id] = k; grp[id] = 1; });
  const rr = mulberry32(5), tint = new Float32Array(nT), yaw = new Float32Array(nT), wid = new Float32Array(nT), crownMax = new Float32Array(nT);
  for (let i = 0; i < nT; i++) { tint[i] = 0.75 + 0.5 * rr(); yaw[i] = rr() * 6.28; wid[i] = 0.20 + 0.06 * rr(); }
  const mtx = new THREE.Matrix4(), qt = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0), sc = new THREE.Vector3(), ps = new THREE.Vector3(), col = new THREE.Color();
  function setTree(i, glow) {
    const H = forest.H[i], f = fire.fuel[i], burnt = fire.st[i] > 0 ? Math.min(1, (1 - f) * 1.25) : 0;
    crownMax[i] = Math.max(crownMax[i], fire.crown[i]);
    const loss = crownMax[i] * Math.min(1, (1 - f) * 1.4);                       // needles are consumed where the crown burned
    qt.setFromAxisAngle(up, yaw[i]); ps.set(forest.X[i], 0, forest.Z[i]);
    const tr = (0.15 + 0.012 * H) * (1 - 0.15 * burnt);
    mtx.compose(ps, qt, sc.set(tr, H * 0.96, tr));
    const g = grp[i], s = slot[i];
    (g ? trunkB : trunkA).setMatrixAt(s, mtx);
    const cw = H * wid[i] * 1.25 * (1 - 0.6 * loss), ch = H * (1 - 0.25 * loss);
    mtx.compose(ps, qt, sc.set(cw, ch, cw)); (g ? crownB : crownA).setMatrixAt(s, mtx);
    const char = 1 - 0.8 * burnt; col.setScalar(tint[i] * char);
    (g ? crownB : crownA).setColorAt(s, col); col.setRGB(tint[i] * (1 - 0.82 * burnt), tint[i] * (1 - 0.86 * burnt), tint[i] * (1 - 0.86 * burnt)); (g ? trunkB : trunkA).setColorAt(s, col);
    (g ? crownB : crownA).geometry.attributes.aGlow.array[s] = glow * (0.4 + 0.6 * fire.crown[i]) * (1 - loss);
    (g ? trunkB : trunkA).geometry.attributes.aGlow.array[s] = glow * 0.5 + fire.glow[i] * 0.35;
  }
  for (let i = 0; i < nT; i++) setTree(i, 0);
  [trunkA, crownA, trunkB, crownB].forEach((m) => { m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true; m.geometry.attributes.aGlow.needsUpdate = true; });
  // shrubs
  const shrubGeo = new THREE.IcosahedronGeometry(1, 1);
  { const p = shrubGeo.attributes.position; for (let i = 0; i < p.count; i++) { const k = 0.8 + 0.4 * Math.abs(Math.sin(i * 12.9898)); p.setXYZ(i, p.getX(i) * k, p.getY(i) * k * 0.7, p.getZ(i) * k); } shrubGeo.computeVertexNormals(); }
  const shrubMat = new THREE.MeshStandardMaterial({ color: 0x0b1a10, roughness: 1 });
  const nS = nAll - nT, shrubs = new THREE.InstancedMesh(shrubGeo, shrubMat, nS); shrubs.frustumCulled = false;
  shrubs.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(nS * 3).fill(1), 3); scene.add(shrubs);
  function setShrub(i) {
    const k = i - nT, H = forest.H[i], burnt = fire.st[i] > 0 ? Math.min(1, (1 - fire.fuel[i]) * 1.2) : 0;
    ps.set(forest.X[i], H * 0.3, forest.Z[i]); qt.identity(); const s = H * 0.75 * (1 - 0.5 * burnt);
    mtx.compose(ps, qt, sc.set(s, s * 0.85, s)); shrubs.setMatrixAt(k, mtx); col.setScalar(1 - 0.85 * burnt); shrubs.setColorAt(k, col);
  }
  for (let i = nT; i < nAll; i++) setShrub(i);
  shrubs.instanceMatrix.needsUpdate = true; shrubs.instanceColor.needsUpdate = true;

  // ferns on the forest floor (decoration: lit by the fire, not fuel in the model)
  { const fg = fernGeometry(77), nF = 900, fm = new THREE.MeshStandardMaterial({ vertexColors: true, map: BRANCH, alphaTest: 0.45, side: THREE.DoubleSide, roughness: 0.9 });
    const ferns = new THREE.InstancedMesh(fg, fm, nF); ferns.frustumCulled = false; const fr = mulberry32(8), m4 = new THREE.Matrix4(), q4 = new THREE.Quaternion(), v3 = new THREE.Vector3(), s3 = new THREE.Vector3();
    for (let i = 0; i < nF; i++) { const sc2 = 0.7 + 0.9 * fr(); m4.compose(v3.set(-48 + 96 * fr(), 0, -75 + 100 * fr()), q4.setFromAxisAngle(up, fr() * 6.28), s3.set(sc2, sc2, sc2)); ferns.setMatrixAt(i, m4); }
    ferns.instanceMatrix.needsUpdate = true; scene.add(ferns); }

  // particles
  function makePool(n, vs, fs, blending, extra = {}, tongues = false) {
    const pos = new Float32Array(n * 3), data = new Float32Array(n * 4), temp = new Float32Array(n), vel = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { data[i * 4] = 1; data[i * 4 + 1] = 0.5; }       // all dead
    let g, pts, attrs;
    if (tongues) {                                                                 // one upright quad per flame tongue
      g = new THREE.InstancedBufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute([-0.5, 0, 0, 0.5, 0, 0, 0.5, 1, 0, -0.5, 1, 0], 3));
      g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2)); g.setIndex([0, 1, 2, 0, 2, 3]);
      attrs = [new THREE.InstancedBufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage), new THREE.InstancedBufferAttribute(data, 4).setUsage(THREE.DynamicDrawUsage), new THREE.InstancedBufferAttribute(temp, 1).setUsage(THREE.DynamicDrawUsage)];
      g.setAttribute("iPos", attrs[0]); g.setAttribute("iData", attrs[1]); g.setAttribute("iTemp", attrs[2]); g.instanceCount = n;
    } else {
      g = new THREE.BufferGeometry();
      attrs = [new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage), new THREE.BufferAttribute(data, 4).setUsage(THREE.DynamicDrawUsage), new THREE.BufferAttribute(temp, 1).setUsage(THREE.DynamicDrawUsage)];
      g.setAttribute("position", attrs[0]); g.setAttribute("aData", attrs[1]); g.setAttribute("aTemp", attrs[2]);
    }
    const mat = new THREE.ShaderMaterial({ vertexShader: vs, fragmentShader: fs, transparent: true, depthWrite: false, blending, uniforms: { uPx: { value: 1 }, uTime: { value: 0 }, uExposure: { value: 1 }, ...extra } });
    pts = tongues ? new THREE.Mesh(g, mat) : new THREE.Points(g, mat); pts.frustumCulled = false; scene.add(pts);
    return { n, pos, data, temp, vel, head: 0, g, mat, pts, attrs };
  }
  const flames = makePool(Q.flame, TONGUE_VS, TONGUE_FS, THREE.AdditiveBlending, { uWind: { value: new THREE.Vector3() } }, true);
  const smoke = makePool(Q.smoke, SMOKE_VS, SMOKE_FS, THREE.NormalBlending, { uSky: { value: new THREE.Color(0.02, 0.03, 0.05) } });
  const sparks = makePool(Q.spark, SPARK_VS, SPARK_FS, THREE.AdditiveBlending);
  const embersPool = makePool(1800, SPARK_VS, SPARK_FS, THREE.AdditiveBlending);
  smoke.pts.renderOrder = 1; flames.pts.renderOrder = 2; sparks.pts.renderOrder = 3; embersPool.pts.renderOrder = 3;
  function spawn(P, x, y, z, vx, vy, vz, life, size, seed, temp) {
    const i = P.head; P.head = (P.head + 1) % P.n;
    P.pos[i * 3] = x; P.pos[i * 3 + 1] = y; P.pos[i * 3 + 2] = z; P.vel[i * 3] = vx; P.vel[i * 3 + 1] = vy; P.vel[i * 3 + 2] = vz;
    P.data[i * 4] = 0; P.data[i * 4 + 1] = life; P.data[i * 4 + 2] = size; P.data[i * 4 + 3] = seed; P.temp[i] = temp;
  }
  // halos: soft glow round the hottest clusters, standing in for light scattered by smoke and air
  const haloG = new THREE.BufferGeometry(), hp = new Float32Array(NL * 3), hs = new Float32Array(NL), hw = new Float32Array(NL);
  haloG.setAttribute("position", new THREE.BufferAttribute(hp, 3)); haloG.setAttribute("aSize", new THREE.BufferAttribute(hs, 1)); haloG.setAttribute("aPow", new THREE.BufferAttribute(hw, 1));
  const haloMat = new THREE.ShaderMaterial({ vertexShader: HALO_VS, fragmentShader: HALO_FS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uPx: { value: 1 } } });
  const halo = new THREE.Points(haloG, haloMat); halo.frustumCulled = false; halo.renderOrder = 0; scene.add(halo);

  // post processing
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.85, 0.7, 0.9); composer.addPass(bloom);
  const fin = new ShaderPass(FINAL); composer.addPass(fin);

  function resize() {
    const w = Math.max(2, Math.floor(innerWidth * scale)), h = Math.max(2, Math.floor(innerHeight * scale));
    renderer.setPixelRatio(1); renderer.setSize(w, h, false); composer.setSize(w, h);
    canvas.style.width = "100%"; canvas.style.height = "100%";
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    const px = h / (2 * Math.tan(camera.fov * Math.PI / 360));
    [flames, smoke, sparks, embersPool].forEach((P) => (P.mat.uniforms.uPx.value = px)); haloMat.uniforms.uPx.value = px;
    fin.uniforms.uRes.value.set(w, h);
  }
  addEventListener("resize", resize); resize();

  // ------------------------------------------------------------ simulation state
  const ctl = { wx: 0.35 * 1.3, wz: 1.3, K: 1, surge: 0, emberMul: 1, spotHeat: 1 };
  let simAcc = 0, time = 0, k = 0, surging = false, surgeStart = 0, surgeDur = 4800, frozenK = null, flash = 0, heat = 0, mx = 0, tx = 0;
  const burning = [];

  function emit(dt) {
    burning.length = 0;
    for (let i = 0; i < nAll; i++) if (fire.st[i] === 1) burning.push(i);
    // demand (particles per second) over all burning nodes, then a global scale to fit the pool
    let demand = 0;
    for (const i of burning) { const I = fire.I[i], tree = forest.KIND[i] === 1; demand += tree ? (40 * I * (1 - fire.crown[i]) + 110 * I * fire.crown[i] * (forest.H[i] / 15)) : 55 * I; }
    const budget = flames.n / 1.15 * qs, scaleE = Math.min(1, budget / Math.max(demand, 1)), sizeMul = Math.min(2.4, 1 / Math.sqrt(scaleE));
    const W = Math.hypot(ctl.wx, ctl.wz);
    for (const i of burning) {
      const tree = forest.KIND[i] === 1, I = fire.I[i], H = forest.H[i], cr = fire.crown[i], x = forest.X[i], z = forest.Z[i];
      const ge = Math.max(0.05, Math.min(1, 1 - 0.2 * (1 - fire.fuel[i])));
      let n;
      if (!tree) {                                                  // shrub: bright, short flames
        n = 55 * I * dt * scaleE; for (; n > 0; n--) if (n >= 1 || rnd() < n) { const a = rnd() * 6.28, r = rnd() * H * 0.45;
          spawn(flames, x + Math.cos(a) * r, H * (0.1 + 0.5 * rnd()), z + Math.sin(a) * r, ctl.wx * 0.15, 1.6 + 2.2 * rnd() * (0.5 + I), ctl.wz * 0.15, 0.7 + 0.7 * rnd(), (0.55 + 0.55 * rnd()) * sizeMul, rnd(), 1150 + 450 * rnd() * I); }
        continue;
      }
      // surface fire at the foot of the tree
      n = 70 * I * (1 - cr) * dt * scaleE; for (; n > 0; n--) if (n >= 1 || rnd() < n) { const a = rnd() * 6.28, r = 0.2 + rnd() * 0.9;
        spawn(flames, x + Math.cos(a) * r, 0.1 + rnd() * 0.9, z + Math.sin(a) * r, ctl.wx * 0.15, 1.8 + 2.4 * rnd(), ctl.wz * 0.15, 0.7 + 0.8 * rnd(), (0.9 + 1.0 * rnd()) * sizeMul, rnd(), 1150 + 450 * rnd() * (0.4 + I)); }
      if (cr > 0.03) {                                              // fire climbing the trunk and through the crown
        n = 110 * I * cr * (H / 15) * dt * scaleE * ge; for (; n > 0; n--) if (n >= 1 || rnd() < n) {
          const f = 0.12 + 0.88 * Math.pow(rnd(), 0.8), y = H * f, R = H * 0.22 * Math.pow(1 - f, 0.8) * rnd(), a = rnd() * 6.28;
          spawn(flames, x + Math.cos(a) * R, y, z + Math.sin(a) * R, ctl.wx * 0.28, 2.5 + 5.5 * rnd() * (0.4 + cr), ctl.wz * 0.28, 0.9 + 0.9 * rnd(), (1.1 + 1.5 * rnd()) * (H / 15) * sizeMul, rnd(), 1250 + 550 * rnd() * (0.3 + I)); }
        // smoke from the burning crown
        const ns = 9 * I * cr * dt * qs; for (let m = ns; m > 0; m--) if (m >= 1 || rnd() < m) spawn(smoke, x + (rnd() - 0.5) * 3, H * (0.5 + 0.5 * rnd()), z + (rnd() - 0.5) * 3, ctl.wx * 0.5, 2 + 3 * rnd(), ctl.wz * 0.5, 4 + 4 * rnd(), 6 + 7 * rnd(), rnd(), I * cr);
      }
      // smoke and sparks from every burning tree
      const ns2 = (3 + 8 * (1 - cr)) * I * dt * qs; for (let m = ns2; m > 0; m--) if (m >= 1 || rnd() < m) spawn(smoke, x + (rnd() - 0.5), 0.8 + 3 * rnd() * (1 + cr * 3), z + (rnd() - 0.5), ctl.wx * 0.4, 1.4 + 2 * rnd(), ctl.wz * 0.4, 3.5 + 3.5 * rnd(), 3 + 3.5 * rnd(), rnd(), I * 0.7);
      const nsp = 5 * I * (0.4 + cr) * dt * qs; for (let m = nsp; m > 0; m--) if (m >= 1 || rnd() < m) spawn(sparks, x + (rnd() - 0.5) * 1.5, 0.6 + H * 0.5 * rnd() * cr + rnd(), z + (rnd() - 0.5) * 1.5, ctl.wx * 0.5 + (rnd() - 0.5) * 2, 2.5 + 5 * rnd(), ctl.wz * 0.5 + (rnd() - 0.5) * 2, 1.4 + 2.2 * rnd(), 0.10 + 0.10 * rnd(), rnd(), 1100 + 350 * rnd());
    }
  }
  function updatePools(dt) {
    const Wx = ctl.wx, Wz = ctl.wz;
    // flames: buoyant gas, pulled sideways by the wind, with a little turbulence
    { const P = flames; for (let i = 0; i < P.n; i++) { const o = i * 4; if (P.data[o] >= P.data[o + 1]) continue; P.data[o] += dt;
      const a = P.data[o] / P.data[o + 1], seed = P.data[o + 3], j = i * 3;
      P.vel[j] += (Wx * 0.35 - P.vel[j]) * Math.min(1, 1.2 * dt) + Math.sin(time * 5 + seed * 40) * 0.9 * dt; P.vel[j + 2] += (Wz * 0.35 - P.vel[j + 2]) * Math.min(1, 1.2 * dt) + Math.cos(time * 4.3 + seed * 31) * 0.9 * dt; P.vel[j + 1] += (0.9 - 0.8 * a) * dt;
      P.pos[j] += P.vel[j] * dt; P.pos[j + 1] += P.vel[j + 1] * dt; P.pos[j + 2] += P.vel[j + 2] * dt; } }
    { const P = smoke; for (let i = 0; i < P.n; i++) { const o = i * 4; if (P.data[o] >= P.data[o + 1]) continue; P.data[o] += dt; const j = i * 3;
      P.vel[j] += (Wx * 0.6 - P.vel[j]) * Math.min(1, 0.5 * dt); P.vel[j + 2] += (Wz * 0.6 - P.vel[j + 2]) * Math.min(1, 0.5 * dt); P.vel[j + 1] += (0.25 - 0.18 * (P.data[o] / P.data[o + 1])) * dt;
      P.pos[j] += P.vel[j] * dt; P.pos[j + 1] += P.vel[j + 1] * dt; P.pos[j + 2] += P.vel[j + 2] * dt; } }
    { const P = sparks; for (let i = 0; i < P.n; i++) { const o = i * 4; if (P.data[o] >= P.data[o + 1]) continue; P.data[o] += dt; const j = i * 3;
      P.vel[j] += (Wx * 0.6 - P.vel[j]) * Math.min(1, 0.7 * dt); P.vel[j + 2] += (Wz * 0.6 - P.vel[j + 2]) * Math.min(1, 0.7 * dt); P.vel[j + 1] += (-1.5) * dt * 0.5 + Math.sin(time * 9 + P.data[o + 3] * 50) * 0.6 * dt;
      P.pos[j] += P.vel[j] * dt; P.pos[j + 1] += P.vel[j + 1] * dt; P.pos[j + 2] += P.vel[j + 2] * dt; } }
    // embers come straight from the fire model, so what lands is what ignites
    { const P = embersPool, E = fire.embers; for (let i = 0; i < P.n; i++) {
      if (i < E.length) { const e = E[i], o = i * 4; P.pos[i * 3] = e.x; P.pos[i * 3 + 1] = e.y; P.pos[i * 3 + 2] = e.z; P.data[o] = e.age; P.data[o + 1] = e.life; P.data[o + 2] = 0.22; P.data[o + 3] = i * 0.37; P.temp[i] = 1150 + 300 * e.heat; }
      else P.data[i * 4] = P.data[i * 4 + 1] = 1; } }
    for (const P of [flames, smoke, sparks, embersPool]) for (const at of P.attrs) at.needsUpdate = true;
  }
  // the lights: group burning nodes into 9 m cells, give the strongest cells a light
  const cells = new Map();
  function updateLights(dt) {
    cells.clear();
    for (const i of burning) {
      const tree = forest.KIND[i] === 1, w = fire.I[i] * (tree ? (forest.H[i] / 12) * (0.35 + 0.65 * fire.crown[i]) : forest.H[i] / 3);
      if (w < 0.02) continue;
      const key = Math.floor(forest.X[i] / 9) * 1000 + Math.floor(forest.Z[i] / 9);
      let c = cells.get(key); if (!c) { c = { w: 0, x: 0, y: 0, z: 0 }; cells.set(key, c); }
      c.w += w; c.x += forest.X[i] * w; c.z += forest.Z[i] * w; c.y += (tree ? 0.5 + forest.H[i] * 0.45 * fire.crown[i] : forest.H[i] * 0.5) * w;
    }
    const arr = [...cells.values()].map((c) => ({ w: c.w, x: c.x / c.w, y: c.y / c.w, z: c.z / c.w, s: c.w / (1 + 0.0035 * Math.hypot(c.x / c.w - camera.position.x, c.z / c.w - camera.position.z) ** 2) }));
    arr.sort((a, b) => b.s - a.s);
    let total = 0;
    for (let l = 0; l < NL; l++) {
      const o = lights[l], c = arr[l];
      if (c) { o.tx = c.x; o.ty = c.y; o.tz = c.z; o.ti = Math.min(900, 120 * Math.pow(c.w, 0.82)); total += c.w; } else o.ti = 0;
      const a = Math.min(1, 3 * dt);
      o.L.position.x += (o.tx - o.L.position.x) * a; o.L.position.y += (o.ty - o.L.position.y) * a; o.L.position.z += (o.tz - o.L.position.z) * a;
      const fl = 0.82 + 0.18 * Math.sin(time * 11 + l * 2.3) * Math.sin(time * 7.3 + l) + 0.1 * Math.sin(time * 23 + l * 5);
      o.i += (o.ti - o.i) * a; o.L.intensity = o.i * fl;
      hp[l * 3] = o.L.position.x; hp[l * 3 + 1] = o.L.position.y; hp[l * 3 + 2] = o.L.position.z;
      hs[l] = o.i > 1 ? Math.min(60, 8 + 5 * Math.pow(o.i / 120, 0.5)) : 0; hw[l] = Math.min(0.34, 0.08 + 0.04 * Math.pow(o.i / 120, 0.6)) * (o.i > 1 ? 1 : 0);
    }
    haloG.attributes.position.needsUpdate = true; haloG.attributes.aSize.needsUpdate = true; haloG.attributes.aPow.needsUpdate = true;
    heat += (Math.min(2.2, total / 3.5) - heat) * Math.min(1, 2 * dt);
  }

  function physics(dt) {
    fire.step(dt, ctl);
    for (let i = 0; i < nT; i++) if (fire.st[i] !== 0 || fire.glow[i] > 0.01) { setTree(i, fire.st[i] === 1 ? fire.I[i] : 0); dirtyTrees = true; }
    for (let i = nT; i < nAll; i++) if (fire.st[i] !== 0) { setShrub(i); dirtyShrubs = true; }
  }
  let dirtyTrees = false, dirtyShrubs = false;
  function advance(seconds, withSurge) {
    const dt = 1 / 30;
    for (let t = 0; t < seconds; t += dt) {
      if (withSurge) Object.assign(ctl, surgeControl(Math.min(1, (t / seconds)), { wx: 0.35, wz: 1 }));
      physics(dt); time += dt; emit(dt); updatePools(dt);
    }
  }
  function applyFlush() {
    if (dirtyTrees) { [trunkA, crownA, trunkB, crownB].forEach((m) => { m.instanceMatrix.needsUpdate = true; m.instanceColor.needsUpdate = true; m.geometry.attributes.aGlow.needsUpdate = true; }); dirtyTrees = false; }
    if (dirtyShrubs) { shrubs.instanceMatrix.needsUpdate = true; shrubs.instanceColor.needsUpdate = true; dirtyShrubs = false; }
  }

  // debug: pre-run the idle fire for `t` seconds, and optionally the surge up to progress `rise`
  const warm = q.has("t") ? parseFloat(q.get("t")) : 7;
  advance(warm, false);
  const rise = q.has("rise") ? parseFloat(q.get("rise")) : null;
  if (rise !== null && rise > 0) { const dur = surgeDur / 1000; const dt = 1 / 30; for (let t = 0; t < rise * dur; t += dt) { Object.assign(ctl, surgeControl(t / dur, { wx: 0.35, wz: 1 })); physics(dt); time += dt; emit(dt); updatePools(dt); } frozenK = rise; k = rise; }
  if (q.has("flash")) flash = parseFloat(q.get("flash"));

  // ------------------------------------------------------------ frame loop
  let last = performance.now(), acc = 0, frames = 0, tuned = false, nFrames = 0;
  const maxFrames = q.has("frames") ? parseInt(q.get("frames"), 10) : Infinity;      // debug: draw N frames and stop
  const cam = camera;
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (!tuned && !q.has("notune") && q.get("t") === null && dt < 0.5) { acc += dt; if (++frames === 50) { tuned = true; const avg = acc / frames * 1000; if (avg > 36) { qs = Math.max(0.35, qs * 30 / avg); scale = Math.max(0.4, scale * Math.sqrt(30 / avg)); resize(); } } }
    mx += (tx - mx) * 0.05;
    if (surging) { k = Math.min(1, (now - surgeStart) / surgeDur); Object.assign(ctl, surgeControl(k, { wx: 0.35, wz: 1 })); flash = smooth((k - 0.88) / 0.12); }
    else if (frozenK === null) { Object.assign(ctl, { wx: 0.35 * 1.3, wz: 1.3, K: 1, surge: 0, emberMul: 1 }); }
    simAcc += dt; while (simAcc >= 1 / 30) { physics(1 / 30); simAcc -= 1 / 30; }
    time += dt; emit(dt); updatePools(dt); updateLights(dt); applyFlush();
    // camera: a slow drift at rest; during the surge it walks forward into the forest and shakes with the heat
    const dolly = smooth((k - 0.30) / 0.70), z = 46 - 32 * dolly * dolly, shake = 0.045 * Math.sqrt(Math.max(heat, 0)) * (surging || frozenK !== null ? 1 : 0.3);
    cam.position.set(camBase.x + mx * 1.4 + Math.sin(time * 0.15) * 0.25 + (Math.random() - 0.5) * shake, camBase.y + dolly * 1.2 + Math.sin(time * 0.11) * 0.08 + (Math.random() - 0.5) * shake, z);
    cam.lookAt(look.x + mx * 2.5, look.y + dolly * 2.0, look.z + dolly * 14);
    flames.mat.uniforms.uTime.value = smoke.mat.uniforms.uTime.value = skyMat.uniforms.uTime.value = fin.uniforms.uTime.value = time;
    flames.mat.uniforms.uWind.value.set(ctl.wx, 0, ctl.wz);
    skyMat.uniforms.uHeat.value = heat; fin.uniforms.uHeat.value = heat; fin.uniforms.uFlash.value = flash;
    const ex = 1.15 / (1 + 0.10 * Math.max(heat, 0)); fin.uniforms.uExposure.value = ex; flames.mat.uniforms.uExposure.value = sparks.mat.uniforms.uExposure.value = embersPool.mat.uniforms.uExposure.value = 1.0;
    bloom.strength = 0.8 + 0.25 * Math.min(1, heat); bloom.threshold = 0.9;
    sky.position.copy(cam.position);
    composer.render();
    if (++nFrames < maxFrames) requestAnimationFrame(frame); else window.__stillDone = true;
  }
  addEventListener("pointermove", (e) => { tx = e.clientX / innerWidth - 0.5; }, { passive: true });
  requestAnimationFrame((n) => { last = n; frame(n); });

  return {
    surge(ms = 4800) { surgeDur = ms; surging = true; surgeStart = performance.now(); frozenK = null; return new Promise((res) => setTimeout(res, ms)); },
    stats: () => ({ ...fire.stats(), flames: flames.n, lights: lights.filter((l) => l.L.intensity > 1).length, heat }),
  };
}
