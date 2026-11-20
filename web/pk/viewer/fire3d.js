/* PAKFIRE 3D viewer: the animated scene shown in the lower pane of the main page (and full page).
 * One arrival-time raster per run is downloaded; any time T is painted from it in the browser, so scrubbing,
 * replay, speed and the four display modes need no further requests. The camera follows the fire front. */
'use strict';
mapboxgl.accessToken = window.MAPBOX_TOKEN;
const $ = (id) => document.getElementById(id);
const PKT_MS = 5 * 3600e3, TS = 8, AGE_FINAL = 240;
const STYLES = { dark: 'mapbox://styles/mapbox/dark-v11', light: 'mapbox://styles/mapbox/light-v11', satellite: 'mapbox://styles/mapbox/satellite-streets-v12',
  streets: 'mapbox://styles/mapbox/streets-v12', terrain: 'mapbox://styles/mapbox/outdoors-v12', night: 'mapbox://styles/mapbox/navigation-night-v1' };
const S = { mode: 'fire', style: 'dark', labels: false, terrain: true, exag: 1.4, opacity: 1, path: true, spots: true, wind: false, playing: false,
  follow: true, T: 0, lastT: 0, speed: 1, lastPaint: 0, particles: 500, meta: null, R: null,
  glow: true, cols: true, smoke: false, fh: 1, flameList: [], tick: 0, lastFront: 0, preset: null };
const BASE_SPEED = 30;                                    // 1x = 30 simulated minutes per real second
const pad2 = (x) => String(x).padStart(2, '0');
const fmt = (x, d = 0) => (x == null || isNaN(x) ? '–' : Number(x).toLocaleString(undefined, { maximumFractionDigits: d }));
const EMPTY = { type: 'FeatureCollection', features: [] };
let map, probeMarker = null;

// ---------------------------------------------------------------- data
async function init() {
  const id = window.RUN_ID;
  const meta = await (await fetch(`/api/runs/${id}/meta.json`)).json();
  const get = (n, T) => fetch(`/api/runs/${id}/${n}.bin`).then((r) => (r.ok ? r.arrayBuffer() : null)).then((b) => (b ? new T(b) : null));
  const [A, I, PR] = await Promise.all([get('arrival', Uint16Array), get('intensity', Uint8Array), meta.has_prob ? get('prob', Uint8Array) : null]);
  const [w, h] = meta.size, n = w * h, never = meta.never;
  const cnt = new Uint32Array(never + 1);
  let nb = 0;
  for (let i = 0; i < n; i++) if (A[i] < never) { cnt[A[i]]++; nb++; }
  let acc = 0;
  for (let k = 0; k <= never; k++) { const c = cnt[k]; cnt[k] = acc; acc += c; }
  const ord = new Uint32Array(nb);
  for (let i = 0; i < n; i++) if (A[i] < never) ord[cnt[A[i]]++] = i;
  const canvas = document.createElement('canvas'); canvas.id = 'fireCanvas'; canvas.width = w; canvas.height = h; canvas.style.display = 'none';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(w, h);
  S.meta = meta;
  S.R = { A, I, PR, ord, nb, w, h, ctx, img, px: new Uint32Array(img.data.buffer), pB: 0, pF: 0, lut: makeLut(), arrLut: rampLut(ARR), intLut: rampLut(INT), probLut: rampLut(PROB), probDone: false,
    ha: meta.stats.burned_ha / nb, canvas };
  $('title').textContent = `Fire spread · ${meta.ignitions[0][1].toFixed(3)}°N ${meta.ignitions[0][0].toFixed(3)}°E`;
  $('panelSub').textContent = `${meta.hours} h from ${clockAt(0)} · ${fmt(meta.stats.burned_ha)} ha burned · ${String(meta.stats.backend).toUpperCase()} solver`;
  $('aboutText').innerHTML = `Colour in <b>Fire over time</b> is the time since the front passed: white-yellow flaming front, orange burning, red smouldering, charcoal burnt. <b>Time of arrival</b> shades each burned cell by when the fire reached it; <b>Fire intensity</b> by Byram fireline intensity. ${meta.has_prob ? '<b>Burn probability</b> is the share of the ' + meta.ensemble.members + ' ensemble members that burned the cell.' : ''} The dashed line is the spread axis (head of the fire) with timed waypoints. Surface fire only: crown fire is not modelled and rates are not calibrated.`;
  if (!meta.has_prob) $('cardProb').classList.add('dis');
  $('timeSlider').max = Math.ceil(meta.t_end_min);
  if (window.innerHeight < 640) { document.body.classList.add('compact'); $('panel').classList.add('collapsed'); document.body.classList.add('panel-collapsed'); }
  $('timelineBar').style.display = '';
  makeMap(meta);
  legend();
}

// ---------------------------------------------------------------- colours
const ARR = [[0, [255, 244, 160]], [0.2, [255, 190, 60]], [0.5, [240, 100, 30]], [0.8, [170, 30, 20]], [1, [60, 15, 20]]];
const INT = [[0, [255, 235, 140]], [0.25, [255, 160, 40]], [0.5, [235, 80, 25]], [0.75, [170, 25, 30]], [1, [90, 10, 60]]];
const PROB = [[0, [255, 240, 150]], [0.3, [255, 190, 70]], [0.6, [240, 100, 40]], [1, [150, 15, 25]]];
function rampLut(stops) {
  const lut = new Uint32Array(256);
  for (let k = 0; k < 256; k++) {
    const t = k / 255; let j = 0; while (j < stops.length - 2 && t > stops[j + 1][0]) j++;
    const [t0, c0] = stops[j], [t1, c1] = stops[j + 1], f = Math.min(Math.max((t - t0) / (t1 - t0), 0), 1);
    const r = c0[0] + (c1[0] - c0[0]) * f, g = c0[1] + (c1[1] - c0[1]) * f, b = c0[2] + (c1[2] - c0[2]) * f;
    lut[k] = (230 << 24) | ((b | 0) << 16) | ((g | 0) << 8) | (r | 0);
  }
  return lut;
}
function makeLut() {
  const buckets = AGE_FINAL * 2 + 1, lut = new Uint32Array(buckets * 8);
  const lerp = (a, b, t) => a + (b - a) * Math.min(Math.max(t, 0), 1);
  for (let q = 0; q < 8; q++) {
    const bright = 0.72 + 0.28 * (q / 7);
    for (let k = 0; k < buckets; k++) {
      const a = k / 2; let r, g, b, al;
      if (a < 4) { const t = a / 4; r = 255; g = lerp(246, 160, t); b = lerp(190, 40, t); al = 1; }
      else if (a < 15) { const t = (a - 4) / 11; r = lerp(255, 232, t); g = lerp(160, 76, t); b = lerp(40, 18, t); al = 0.97; }
      else if (a < 60) { const t = (a - 15) / 45; r = lerp(232, 160, t); g = lerp(76, 24, t); b = lerp(18, 14, t); al = lerp(0.95, 0.82, t); }
      else { const t = (a - 60) / (AGE_FINAL - 60); r = lerp(160, 44, t); g = lerp(24, 28, t); b = lerp(14, 22, t); al = lerp(0.82, 0.70, t); }
      if (a < 15) { r *= bright; g *= bright; b *= bright; }
      lut[k * 8 + q] = ((al * 255) << 24) | ((b | 0) << 16) | ((g | 0) << 8) | (r | 0);
    }
  }
  return lut;
}
function legend() {
  const m = S.meta, bar = $('legendBar'), lbl = $('legendLbl'), note = $('legendNote');
  const grad = (st) => `linear-gradient(90deg,${st.map((s) => `rgb(${s[1].join(',')}) ${s[0] * 100}%`).join(',')})`;
  const L = {
    fire: ['linear-gradient(90deg,#fff2a8,#ff8a1f 30%,#a3170f 62%,#2a1d1a)', ['flaming front', 'burnt'], 'Colour = time since the fire front passed (4 h to charcoal).'],
    arrival: [grad(ARR), ['0 h', `${(m.t_end_min / 60).toFixed(1)} h`], 'Time the front reached each cell, from ignition.'],
    intensity: [grad(INT), ['0', '6,000 kW/m'], 'Byram fireline intensity when the front passed.'],
    prob: [grad(PROB), ['0%', '100%'], m.has_prob ? `Share of ${m.ensemble.members} perturbed runs that burned the cell.` : ''],
  }[S.mode];
  bar.style.background = L[0]; lbl.innerHTML = `<span>${L[1][0]}</span><span>${L[1][1]}</span>`; note.textContent = L[2];
}

// ---------------------------------------------------------------- map
function makeMap(meta) {
  const ig = meta.ignitions[0];
  map = new mapboxgl.Map({ container: 'map', style: STYLES[S.style], center: ig, zoom: 12.4, pitch: 58, bearing: 0, antialias: true, preserveDrawingBuffer: true, projection: 'mercator' });
  map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), 'top-right');
  map.on('style.load', () => { addAll(); paint(S.T, true); });
  map.on('click', probe);
  ['mousedown', 'wheel', 'touchstart'].forEach((ev) => map.on(ev, () => { if (S.playing && S.follow) setFollow(false); }));
  map.once('load', () => { setTimeout(() => startPlay(), 600); });
}
function bearingAt(ax, i) { const a = ax[Math.max(i - 1, 0)], b = ax[Math.min(i + 1, ax.length - 1)]; return (Math.atan2((b[1] - a[1]) * Math.cos((a[2] * Math.PI) / 180), b[2] - a[2]) * 180 / Math.PI + 360) % 360; }
function addAll() {
  const m = S.meta, R = S.R;
  if (!map.getSource('mapbox-dem')) map.addSource('mapbox-dem', { type: 'raster-dem', url: 'mapbox://mapbox.mapbox-terrain-dem-v1', tileSize: 512, maxzoom: 14 });
  map.setTerrain(S.terrain ? { source: 'mapbox-dem', exaggeration: S.exag } : null);
  const night = S.style === 'night' || S.style === 'dark';
  map.setFog({ range: [0.6, 11], color: night ? 'rgb(8,12,18)' : 'rgb(186,210,229)', 'high-color': night ? 'rgb(14,22,38)' : 'rgb(36,92,223)', 'horizon-blend': 0.08, 'space-color': 'rgb(6,9,14)', 'star-intensity': night ? 0.5 : 0.1 });
  if (S.style !== 'satellite' && !map.getLayer('hills')) {                  // flat basemaps: add relief shading so the terrain reads
    const dark = S.style === 'dark' || S.style === 'night';
    const first = map.getStyle().layers.find((l) => l.type === 'symbol');
    map.addLayer({ id: 'hills', type: 'hillshade', source: 'mapbox-dem', paint: { 'hillshade-exaggeration': dark ? 0.75 : 0.45, 'hillshade-shadow-color': dark ? '#000000' : '#3b4a56',
      'hillshade-highlight-color': dark ? '#6f7f8c' : '#ffffff', 'hillshade-accent-color': dark ? '#2b3640' : '#9aa7b1', 'hillshade-illumination-direction': 315 } }, first && first.id);
  }
  if (!map.getLayer('sky')) map.addLayer({ id: 'sky', type: 'sky', paint: { 'sky-type': 'atmosphere', 'sky-atmosphere-sun': [0, 80], 'sky-atmosphere-sun-intensity': 12 } });
  if (!S.labels) map.getStyle().layers.forEach((l) => { if (l.type === 'symbol') map.setLayoutProperty(l.id, 'visibility', 'none'); });
  const [W, Sb, E, N] = m.bounds;
  map.addSource('fire', { type: 'canvas', canvas: 'fireCanvas', animate: false, coordinates: [[W, N], [E, N], [E, Sb], [W, Sb]] });
  map.addLayer({ id: 'fire', type: 'raster', source: 'fire', paint: { 'raster-fade-duration': 0, 'raster-opacity': S.opacity, 'raster-resampling': 'linear' } });
  map.addSource('smoke', { type: 'geojson', data: EMPTY });
  map.addLayer({ id: 'smoke', type: 'heatmap', source: 'smoke', layout: { visibility: S.smoke ? 'visible' : 'none' }, paint: {
    'heatmap-weight': ['get', 'w'], 'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 9, 1.0, 15, 2.2], 'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 9, 9, 12, 26, 15, 80],
    'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(140,135,130,0)', 0.2, 'rgba(150,142,135,0.2)', 0.55, 'rgba(120,112,106,0.45)', 1, 'rgba(92,86,84,0.72)'], 'heatmap-opacity': 0.75 } });
  map.addSource('sparks', { type: 'geojson', data: EMPTY });
  map.addLayer({ id: 'sparks', type: 'circle', source: 'sparks', layout: { visibility: S.glow ? 'visible' : 'none' }, paint: {
    'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 1.4, 13, 2.4, 15, 4], 'circle-color': ['interpolate', ['linear'], ['get', 'f'], 0, '#ff6a1a', 1, '#fff0b0'], 'circle-blur': 0.3, 'circle-opacity': ['get', 'o'] } });
  map.addSource('flames', { type: 'geojson', data: EMPTY });
  map.addLayer({ id: 'flames', type: 'heatmap', source: 'flames', layout: { visibility: S.glow ? 'visible' : 'none' }, paint: {
    'heatmap-weight': ['get', 'w'], 'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 9, 1.0, 15, 2.6], 'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 9, 12, 12, 30, 15, 85],
    'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(255,60,10,0)', 0.2, 'rgba(200,40,10,0.45)', 0.5, 'rgba(255,120,20,0.75)', 0.8, 'rgba(255,200,70,0.9)', 1, 'rgba(255,250,220,1)'], 'heatmap-opacity': 0.9 } });
  map.addSource('spots', { type: 'geojson', data: { type: 'FeatureCollection', features: m.spots.map((s) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: [s[0], s[1]] }, properties: { t: s[2] } })) } });
  map.addLayer({ id: 'spots-glow', type: 'circle', source: 'spots', filter: ['<=', ['get', 't'], S.T], layout: { visibility: S.spots ? 'visible' : 'none' }, paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 8, 14, 22], 'circle-color': 'rgba(76,201,255,.35)', 'circle-blur': 1 } });
  map.addLayer({ id: 'spots', type: 'circle', source: 'spots', filter: ['<=', ['get', 't'], S.T], layout: { visibility: S.spots ? 'visible' : 'none' },
    paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 3, 14, 9], 'circle-color': 'rgba(76,201,255,.25)', 'circle-stroke-color': '#4cc9ff', 'circle-stroke-width': 2 } });
  const ax = m.axis;
  map.addSource('axis', { type: 'geojson', data: { type: 'Feature', geometry: { type: 'LineString', coordinates: ax.map((p) => [p[1], p[2]]) } } });
  const wp = [];
  const NW = ax[ax.length - 1][4] < 3 ? 3 : 5;
  for (let k = 1; k <= NW; k++) { const p = ax[Math.min(Math.round((k / NW) * (ax.length - 1)), ax.length - 1)]; wp.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [p[1], p[2]] }, properties: { label: `+${Math.floor(p[0] / 60)}h${pad2(Math.round(p[0] % 60))} · ${p[4].toFixed(1)} km`, bearing: bearingAt(ax, Math.min(Math.round((k / NW) * (ax.length - 1)), ax.length - 1)) } }); }
  map.addSource('wpts', { type: 'geojson', data: { type: 'FeatureCollection', features: wp } });
  map.addSource('head', { type: 'geojson', data: EMPTY });
  const vis = S.path ? 'visible' : 'none';
  map.addSource('axis-done', { type: 'geojson', data: EMPTY });
  map.addLayer({ id: 'axis-todo', type: 'line', source: 'axis', layout: { visibility: vis }, paint: { 'line-color': '#ffd24a', 'line-width': 2.2, 'line-opacity': 0.75, 'line-dasharray': [2, 1.6] } });
  map.addLayer({ id: 'axis-done', type: 'line', source: 'axis-done', layout: { visibility: vis, 'line-cap': 'round' }, paint: { 'line-color': '#ffe27a', 'line-width': 3.4 } });
  map.addLayer({ id: 'axis-pts', type: 'circle', source: 'wpts', layout: { visibility: vis }, paint: { 'circle-radius': 5, 'circle-color': '#ffd24a', 'circle-stroke-color': '#1b1205', 'circle-stroke-width': 1.5 } });
  map.addLayer({ id: 'axis-lbl', type: 'symbol', source: 'wpts', layout: { visibility: vis, 'text-field': ['get', 'label'], 'text-size': 12, 'text-offset': [0, -1.3], 'text-allow-overlap': true },
    paint: { 'text-color': '#ffe9a8', 'text-halo-color': '#1b1205', 'text-halo-width': 1.6 } });
  map.addLayer({ id: 'axis-arrow', type: 'symbol', source: 'wpts', layout: { visibility: vis, 'text-field': '\u25B2', 'text-size': 20, 'text-rotate': ['get', 'bearing'], 'text-rotation-alignment': 'map', 'text-allow-overlap': true, 'text-ignore-placement': true, 'text-offset': [0, 0] },
    paint: { 'text-color': '#ff5a1a', 'text-halo-color': 'rgba(20,8,4,.9)', 'text-halo-width': 1.6 } });
  map.addLayer({ id: 'head-pt', type: 'circle', source: 'head', layout: { visibility: vis }, paint: { 'circle-radius': 7, 'circle-color': '#fff', 'circle-stroke-color': '#ff5a1a', 'circle-stroke-width': 3 } });
  map.addSource('ign', { type: 'geojson', data: { type: 'FeatureCollection', features: m.ignitions.map((c) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: c }, properties: { label: 'Ignition\n' + clockAt(0) } })) } });
  map.addLayer({ id: 'ign-glow2', type: 'circle', source: 'ign', paint: { 'circle-radius': 46, 'circle-color': 'rgba(255,200,110,.28)', 'circle-blur': 1 } });
  map.addLayer({ id: 'ign-glow', type: 'circle', source: 'ign', paint: { 'circle-radius': 24, 'circle-color': 'rgba(255,245,210,.85)', 'circle-blur': 0.9 } });
  map.addLayer({ id: 'ign-pt', type: 'circle', source: 'ign', paint: { 'circle-radius': 5, 'circle-color': '#ff3b1a', 'circle-stroke-color': '#fff', 'circle-stroke-width': 1.6 } });
  map.addLayer({ id: 'ign-lbl', type: 'symbol', source: 'ign', layout: { 'text-field': ['get', 'label'], 'text-size': 11.5, 'text-anchor': 'left', 'text-offset': [1.3, 0.6], 'text-allow-overlap': true, 'text-justify': 'left' },
    paint: { 'text-color': '#fff', 'text-halo-color': 'rgba(18,14,12,.92)', 'text-halo-width': 2.2 } });
  R.source = map.getSource('fire');
}

// ---------------------------------------------------------------- painting
function bsearch(R, Tq) { let lo = 0, hi = R.nb; while (lo < hi) { const m = (lo + hi) >> 1; if (R.A[R.ord[m]] <= Tq) lo = m + 1; else hi = m; } return lo; }
function paint(T, full) {
  const R = S.R; if (!R || !map || !R.source) return;
  const { A, I, ord, px } = R, Tq = T * TS;
  if (full || T < S.lastT - 1e-6) { px.fill(0); R.pF = 0; R.pB = 0; R.probDone = false; }
  S.lastT = T;
  if (S.mode === 'prob') {
    if (!R.probDone && R.PR) { for (let i = 0; i < R.PR.length; i++) px[i] = R.PR[i] ? R.probLut[Math.min(R.PR[i] * 2.55 | 0, 255)] : 0; R.probDone = true; }
  } else {
    let pB = R.pB; while (pB < R.nb && A[ord[pB]] <= Tq) pB++;
    if (S.mode === 'fire') {
      const finalQ = (T - AGE_FINAL) * TS, FIN = R.lut[(AGE_FINAL * 2) * 8];
      R.pB = pB;
      let pF = R.pF; while (pF < pB && A[ord[pF]] <= finalQ) { px[ord[pF]] = FIN; pF++; }
      R.pF = pF;
      for (let k = pF; k < pB; k++) {
        const i = ord[k], age = (Tq - A[i]) / TS; let q = Math.min(I[i] >> 4, 7), ab = age;
        if (age > 6 && age < 45 && ((Math.imul(i, 2654435761) >>> 24) < 26 + 20 * q)) ab = 5 + ((i >> 3) & 7);   // glowing embers scattered behind the front
        px[i] = R.lut[(Math.min(ab * 2, AGE_FINAL * 2) | 0) * 8 + q];
      }
    } else {
      const tend = Math.max(S.meta.t_end_min * TS, 1);
      for (let k = R.pB; k < pB; k++) { const i = ord[k]; px[i] = S.mode === 'arrival' ? R.arrLut[Math.min((A[i] / tend) * 255 | 0, 255)] : R.intLut[Math.min((I[i] * 40 / 6000) * 255 | 0, 255)]; }
      R.pB = pB; R.pF = pB;
    }
  }
  R.ctx.putImageData(R.img, 0, 0);
  kick();
  front(T, full);
  if (map.getLayer('spots')) { map.setFilter('spots', ['<=', ['get', 't'], T]); map.setFilter('spots-glow', ['<=', ['get', 't'], T]); }
  hud(T);
}
// ---- flame glow, columns and smoke are rebuilt from the cells that ignited in the last 8 minutes
const FRONT_MIN = 8, MAXP = 1100;
function front(T, force) {
  const R = S.R, m = S.meta; if (!R || !map.getSource('flames')) return;
  const now = performance.now(); if (!force && now - S.lastFront < 110) return; S.lastFront = now;
  if (S.mode !== 'fire' || (!S.glow && !S.cols && !S.smoke)) { ['flames', 'smoke', 'sparks'].forEach((s) => map.getSource(s).setData(EMPTY)); S.flameList = []; return; }
  const Tq = T * TS, lo = bsearch(R, Tq - FRONT_MIN * TS), hi = bsearch(R, Tq), stride = Math.max(1, Math.ceil((hi - lo) / MAXP));
  const [W, Sb, E, N] = m.bounds, dx = (E - W) / R.w, dy = (N - Sb) / R.h;
  const hh = Math.min(Math.floor(T / 60), m.weather.wind.length - 1), to = ((m.weather.wdir[hh] + 180) * Math.PI) / 180, ws = m.weather.wind[hh];
  const flist = [], fstep = Math.max(1, Math.ceil((hi - lo) / 300)), flames = [], smoke = [], sparks = [], lo2 = S.smoke ? bsearch(R, Tq - 30 * TS) : hi, st2 = Math.max(1, Math.ceil((hi - lo2) / 900));
  S.tick = now;
  for (let k = lo; k < hi; k += stride) {
    const i = R.ord[k], c = i % R.w, r = (i / R.w) | 0, I = R.I[i] * 40, lon = W + (c + 0.5) * dx, lat = N - (r + 0.5) * dy;
    const age = (Tq - R.A[i]) / TS, fade = 1 - age / FRONT_MIN, w = Math.min(0.25 + I / 4000, 1) * fade;
    if (S.glow) for (let q = 0; q < 3; q++) { const j2 = ((Math.imul(i + q * 7919, 2246822519) ^ (now / 90 | 0)) >>> 0) % 1000 / 1000, d = (4 + j2 * 70) / 111320, a2 = j2 * 53 + q * 2.1; sparks.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [lon + Math.sin(a2) * d, lat + Math.cos(a2) * d] }, properties: { f: j2, o: Math.max(0.35, fade) * (1 - j2 * 0.4) } }); }
    if (S.glow) flames.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [lon, lat] }, properties: { w } });
    if (S.cols && I >= 60 && (k % fstep) === 0) flist.push([lon, lat, I, Math.max(fade, 0.2), i % 251]);
  }
  for (let k = lo2; k < hi; k += st2) {
    const i = R.ord[k], c = i % R.w, r = (i / R.w) | 0, I = R.I[i] * 40, lon = W + (c + 0.5) * dx, lat = N - (r + 0.5) * dy, age = (Tq - R.A[i]) / TS;
    const d = (age * 25 * (0.6 + ws / 6) + 100 + ((i * 2654435761) >>> 0) % 400) / 111320, a = to + ((((i * 40503) >>> 0) % 1000) / 1000 - 0.5) * 0.6;
    smoke.push({ type: 'Feature', geometry: { type: 'Point', coordinates: [lon + (Math.sin(a) * d) / Math.cos((lat * Math.PI) / 180), lat + Math.cos(a) * d] }, properties: { w: 0.15 + 0.35 * Math.min(I / 3000, 1) } });
  }
  map.getSource('flames').setData({ type: 'FeatureCollection', features: flames });
  S.flameList = flist;
  map.getSource('smoke').setData({ type: 'FeatureCollection', features: smoke });
  map.getSource('sparks').setData({ type: 'FeatureCollection', features: sparks });
}
let _pt = null;
function kick() { const s = S.R && S.R.source; if (!s) return; s.play(); clearTimeout(_pt); _pt = setTimeout(() => s.pause(), S.playing ? 400 : 180); }
function clockAt(T, short) {
  const d = new Date(Date.parse(S.meta.start + 'Z') + PKT_MS + T * 60000);
  const hm = `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
  return short ? hm : `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${hm} PKT`;
}
function axisAt(T) {
  const ax = S.meta.axis; if (T <= ax[0][0]) return ax[0]; const last = ax[ax.length - 1]; if (T >= last[0]) return last;
  let lo = 0, hi = ax.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ax[m][0] <= T) lo = m; else hi = m; }
  const a = ax[lo], b = ax[hi], f = (T - a[0]) / (b[0] - a[0] || 1);
  return [T, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
const seriesAt = (s, T) => { const k = Math.min(Math.max(Math.ceil(T / 30) - 1, 0), s.length - 1); return s[k] ? s[k][1] : 0; };
function pktHour(T) { const d = new Date(Date.parse(S.meta.start + 'Z') + PKT_MS + T * 60000); return d.getUTCHours() + d.getUTCMinutes() / 60; }
function presetFor(h) { return h < 5 ? 'night' : h < 7 ? 'dawn' : h < 17 ? 'day' : h < 19.5 ? 'dusk' : 'night'; }
function applyPreset(force) {
  const h = pktHour(S.T), p = presetFor(h);
  $('chipIcon').textContent = p === 'night' ? '\u263E' : p === 'dawn' ? '\u25D0' : p === 'dusk' ? '\u25D1' : '\u2600';
  S.preset = p;
}
function hud(T) {
  const m = S.meta, R = S.R;
  $('chipTime').textContent = clockAt(T);
  if (!S._pp || Math.abs(S._pp - T) > 20) { S._pp = T; applyPreset(false); }
  $('timeBig').textContent = `+${Math.floor(T / 60)}:${pad2(Math.floor(T % 60))}`;
  $('timeSub').textContent = clockAt(T);
  $('timeLabel').textContent = clockAt(T, true);
  $('timeSlider').value = T;
  $('stArea').textContent = `${fmt(bsearch(R, T * TS) * R.ha)} ha`;
  $('stRos').textContent = `${fmt(seriesAt(m.stats.ros_series, T), 1)} m/min`;
  { const L0 = seriesAt(m.stats.flame_series, T), w0 = m.weather.wind[Math.min(Math.floor(T / 60), m.weather.wind.length - 1)], I0 = Math.pow(L0 / 0.0775, 1 / 0.46), p0 = flamePhys(I0, w0); $('stFl').textContent = `${fmt(L0, 1)} m · tilt ${Math.round((p0.th * 180) / Math.PI)}°`; }
  const hh = Math.min(Math.floor(T / 60), m.weather.wind.length - 1);
  $('stWind').textContent = `${m.weather.wind[hh]} m/s ${['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round(m.weather.wdir[hh] / 45) % 8]}`;
  if (map.getSource('head')) {
    const a = axisAt(T), ax = m.axis, done = ax.filter((p) => p[0] <= T).map((p) => [p[1], p[2]]); done.push([a[1], a[2]]);
    map.getSource('head').setData({ type: 'Feature', geometry: { type: 'Point', coordinates: [a[1], a[2]] } });
    if (map.getSource('axis-done')) map.getSource('axis-done').setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: done.length > 1 ? done : [done[0], done[0]] } });
    if (map.getLayer('ign-glow')) { const k = 0.5 + 0.5 * Math.sin(performance.now() / 260); map.setPaintProperty('ign-glow', 'circle-radius', 20 + 8 * k); map.setPaintProperty('ign-glow2', 'circle-radius', 40 + 14 * k); map.setPaintProperty('ign-glow', 'circle-opacity', T < 90 ? 1 : Math.max(0.15, 1 - (T - 90) / 400)); }
  }
  drawChart(T);
}

// ---------------------------------------------------------------- chart
let chartGeom = null;
function drawChart(T, hover) {
  const c = $('chart'), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight; if (!W || !H) return;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const st = S.meta.stats, ar = [[0, 0], ...st.area_series], te = S.meta.t_end_min;
  const pl = 36, pr = 34, pt = 16, pb = 14, amax = Math.max(...ar.map((p) => p[1]), 1) * 1.08, rmax = Math.max(...st.ros_series.map((p) => p[1]), 1) * 1.15;
  const X = (t) => pl + (t / te) * (W - pl - pr), YA = (v) => pt + (1 - v / amax) * (H - pt - pb), YR = (v) => pt + (1 - v / rmax) * (H - pt - pb);
  chartGeom = { pl, pr, W, te, X };
  g.font = '9.5px Public Sans, system-ui'; g.strokeStyle = 'rgba(255,255,255,.07)'; g.fillStyle = 'rgba(255,255,255,.45)';
  for (let k = 0; k <= 2; k++) { const y = pt + (H - pt - pb) * k / 2; g.beginPath(); g.moveTo(pl, y); g.lineTo(W - pr, y); g.stroke(); g.fillText(fmt(amax * (1 - k / 2)), 2, y + 3); g.fillText(fmt(rmax * (1 - k / 2), 1), W - pr + 3, y + 3); }
  g.fillStyle = 'rgba(224,73,44,.55)'; const bw = Math.max((W - pl - pr) / st.ros_series.length - 1, 1);
  st.ros_series.forEach(([t, v]) => g.fillRect(X(t) - bw, YR(v), bw, YR(0) - YR(v)));
  g.beginPath(); ar.forEach(([t, v], i) => (i ? g.lineTo(X(t), YA(v)) : g.moveTo(X(t), YA(v)))); g.strokeStyle = '#ff9a3c'; g.lineWidth = 2; g.stroke();
  g.lineTo(X(te), YA(0)); g.lineTo(X(0), YA(0)); g.fillStyle = 'rgba(255,154,60,.16)'; g.fill();
  S.meta.events.forEach((e) => { g.fillStyle = '#fff'; g.fillRect(X(e.t) - 1, H - pb, 2, 4); });
  const cx = X(hover != null ? hover : T); g.strokeStyle = '#fff'; g.lineWidth = 1; g.beginPath(); g.moveTo(cx, pt); g.lineTo(cx, H - pb); g.stroke();
  g.fillStyle = 'rgba(255,255,255,.5)'; g.fillText('0', pl, H - 2); g.fillText(`${(te / 60).toFixed(0)} h`, W - pr - 22, H - 2);
  g.fillStyle = '#ff9a3c'; g.fillText('burned area (ha)', pl + 6, 10); g.fillStyle = 'rgba(224,73,44,.9)'; g.fillText('peak spread (m/min)', W - pr - 118, 10);
}
$('chart').addEventListener('mousemove', (e) => {
  if (!chartGeom) return; const r = $('chart').getBoundingClientRect(), x = e.clientX - r.left;
  const t = Math.min(Math.max(((x - chartGeom.pl) / (chartGeom.W - chartGeom.pl - chartGeom.pr)) * chartGeom.te, 0), chartGeom.te);
  const st = S.meta.stats, ar = [[0, 0], ...st.area_series]; let area = 0; for (const p of ar) if (p[0] <= t) area = p[1];
  const tip = $('chartTip'); tip.style.display = 'block'; tip.style.left = Math.min(x + 12, r.width - 150) + 'px'; tip.style.top = '-8px';
  tip.innerHTML = `<b>${clockAt(t, true)}</b> · +${Math.floor(t / 60)}h${pad2(Math.floor(t % 60))}<br>${fmt(area)} ha burned<br>peak ${fmt(seriesAt(st.ros_series, t), 1)} m/min`;
  drawChart(S.T, t);
});
$('chart').addEventListener('mouseleave', () => { $('chartTip').style.display = 'none'; drawChart(S.T); });
$('chart').addEventListener('click', (e) => {
  const r = $('chart').getBoundingClientRect(), x = e.clientX - r.left;
  seek(Math.min(Math.max(((x - chartGeom.pl) / (chartGeom.W - chartGeom.pl - chartGeom.pr)) * chartGeom.te, 0), chartGeom.te));
});

// ---------------------------------------------------------------- playback and camera
function camAt(T) {
  const tr = S.meta.camera; if (!tr.length) return null; if (T <= tr[0][0]) return tr[0]; const last = tr[tr.length - 1]; if (T >= last[0]) return last;
  let lo = 0, hi = tr.length - 1; while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (tr[mid][0] <= T) lo = mid; else hi = mid; }
  const a = tr[lo], b = tr[hi], f = (T - a[0]) / (b[0] - a[0]), db = ((b[4] - a[4] + 540) % 360) - 180;
  return [T, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f, (a[4] + db * f + 360) % 360];
}
function camPad() { const cb = document.body.classList.contains('compact'), pc = document.body.classList.contains('panel-collapsed'); return { top: 50, bottom: cb ? 110 : 270, left: pc ? 20 : 330, right: 20 }; }
function follow(T) { const c = camAt(T); if (c && map) map.jumpTo({ center: [c[1], c[2]], zoom: c[3], bearing: c[4], pitch: 58, padding: camPad() }); }
let _prev = 0;
function frame(ts) {
  requestAnimationFrame(frame);
  if (!S.playing || !S.R) { _prev = ts; return; }
  const dtS = Math.min((ts - _prev) / 1000, 0.1); _prev = ts; const end = S.meta.t_end_min;
  S.T = Math.min(S.T + dtS * BASE_SPEED * S.speed, end);
  if (ts - S.lastPaint > 50) { S.lastPaint = ts; paint(S.T); }
  if (S.follow) follow(S.T);
  if (S.T >= end) { paint(S.T); pause(); }
}
requestAnimationFrame(frame);
function startPlay() { if (!S.R) return; if (S.T >= S.meta.t_end_min - 0.01) { S.T = 0; paint(0, true); } S.playing = true; $('playPauseBtn').innerHTML = '&#10074;&#10074;'; if (S.follow) follow(S.T); }
function pause() { S.playing = false; $('playPauseBtn').innerHTML = '&#9654;'; if (S.R) kick(); }
function seek(T) { S.T = T; paint(T, true); if (S.follow) follow(T); }
function setFollow(v) { S.follow = v; $('followBtn').classList.toggle('on', v); }
$('playPauseBtn').onclick = () => (S.playing ? pause() : startPlay());
$('timeSlider').oninput = (e) => seek(+e.target.value);
$('speedSelect').onchange = (e) => (S.speed = +e.target.value);
$('followBtn').onclick = () => { setFollow(!S.follow); if (S.follow) { const c = camAt(S.T); if (c) map.easeTo({ center: [c[1], c[2]], zoom: c[3], bearing: c[4], pitch: 58, duration: 900, padding: camPad() }); } };
$('resetBtn').onclick = () => { setFollow(false); map.flyTo({ center: S.meta.ignitions[0], zoom: 12.4, pitch: 58, bearing: 0, duration: 1200 }); };

// ---------------------------------------------------------------- panel controls
document.querySelectorAll('.card').forEach((b) => (b.onclick = () => {
  if (b.classList.contains('dis')) return;
  document.querySelectorAll('.card').forEach((x) => x.classList.toggle('active', x === b)); S.mode = b.dataset.mode; legend(); paint(S.T, true);
}));
document.querySelectorAll('.tile').forEach((b) => (b.onclick = () => {
  document.querySelectorAll('.tile').forEach((x) => x.classList.toggle('active', x === b)); S.style = b.dataset.style; map.setStyle(STYLES[S.style]);
}));
const pill = (id, key, apply) => { $(id).onclick = () => { S[key] = !S[key]; $(id).classList.toggle('on', S[key]); apply(); }; };
pill('pillTerrain', 'terrain', () => { map.setTerrain(S.terrain ? { source: 'mapbox-dem', exaggeration: S.exag } : null); });
pill('pillLabels', 'labels', () => { map.getStyle().layers.forEach((l) => { if (l.type === 'symbol' && l.id !== 'axis-lbl') map.setLayoutProperty(l.id, 'visibility', S.labels ? 'visible' : 'none'); }); });
pill('pillPath', 'path', () => ['axis-todo', 'axis-done', 'axis-pts', 'axis-lbl', 'axis-arrow', 'head-pt'].forEach((l) => map.setLayoutProperty(l, 'visibility', S.path ? 'visible' : 'none')));
pill('pillSpots', 'spots', () => ['spots', 'spots-glow'].forEach((l) => map.setLayoutProperty(l, 'visibility', S.spots ? 'visible' : 'none')));
pill('pillGlow', 'glow', () => { ['flames', 'sparks'].forEach((l) => map.setLayoutProperty(l, 'visibility', S.glow ? 'visible' : 'none')); front(S.T, true); });
pill('pillCols', 'cols', () => { front(S.T, true); });
pill('pillSmoke', 'smoke', () => { map.setLayoutProperty('smoke', 'visibility', S.smoke ? 'visible' : 'none'); front(S.T, true); });
pill('pillWind', 'wind', () => { if (!S.wind) wctx.clearRect(0, 0, wc.width, wc.height); });
const slider = (id, out, fmtf, apply) => { const f = () => { $(out).textContent = fmtf($(id).value); apply(+$(id).value); }; $(id).oninput = f; f(); };
slider('exagSlider', 'exagOut', (v) => `${(+v).toFixed(1)}×`, (v) => { S.exag = v; $('exagVal').textContent = v.toFixed(1); if (map && map.getSource('mapbox-dem') && S.terrain) map.setTerrain({ source: 'mapbox-dem', exaggeration: v }); });
slider('opSlider', 'opOut', (v) => `${v}%`, (v) => { S.opacity = v / 100; if (map && map.getLayer('fire')) map.setPaintProperty('fire', 'raster-opacity', S.opacity); });
slider('fhSlider', 'fhOut', (v) => `${(+v).toFixed(1)}×`, (v) => { S.fh = v; });
slider('partSlider', 'partOut', (v) => v, (v) => { S.particles = v; });
$('chartBtn').onclick = () => { document.body.classList.toggle('compact'); $('chartBtn').classList.toggle('on', !document.body.classList.contains('compact')); drawChart(S.T); };
$('collapseBtn').onclick = () => { $('panel').classList.toggle('collapsed'); document.body.classList.toggle('panel-collapsed'); };
$('fsBtn').onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); };
window.addEventListener('resize', () => S.R && drawChart(S.T));

// ---------------------------------------------------------------- probe
function probe(e) {
  const R = S.R, m = S.meta; if (!R) return; const [W, Sb, E, N] = m.bounds;
  const c = Math.floor(((e.lngLat.lng - W) / (E - W)) * R.w), r = Math.floor(((N - e.lngLat.lat) / (N - Sb)) * R.h);
  if (c < 0 || c >= R.w || r < 0 || r >= R.h) return;
  const i = r * R.w + c, a = R.A[i], el = map.queryTerrainElevation(e.lngLat);
  $('probe').style.display = 'block';
  if (a >= m.never) { $('probeDepth').textContent = 'unburned'; $('probeWhen').textContent = ''; $('probeArr').textContent = '–'; $('probeInt').textContent = '–'; $('probeNote').textContent = 'The fire did not reach this cell in the simulated period.'; }
  else { const t = a / TS; $('probeDepth').textContent = `+${Math.floor(t / 60)}h ${pad2(Math.floor(t % 60))}m`; $('probeWhen').textContent = clockAt(t); $('probeArr').textContent = clockAt(t, true); $('probeInt').textContent = `${fmt(R.I[i] * 40)} kW/m`; $('probeNote').textContent = R.PR ? `Burn probability here: ${R.PR[i]}%` : ''; }
  $('probeGround').textContent = el == null ? '–' : `${fmt(el)} m`;
  if (probeMarker) probeMarker.remove();
  const el2 = document.createElement('div'); el2.className = 'pin'; el2.textContent = '1';
  probeMarker = new mapboxgl.Marker(el2).setLngLat(e.lngLat).addTo(map);
}
$('probeClose').onclick = () => { $('probe').style.display = 'none'; if (probeMarker) probeMarker.remove(); };

// ---------------------------------------------------------------- animated flame sprites (screen-space, additive)
const fc = $('flameCanvas'), fctx = fc.getContext('2d');
const SPR = (() => {                                   // one pre-rendered flame bitmap, stamped per sprite (cheap on software GL too)
  const c = document.createElement('canvas'); c.width = 64; c.height = 128; const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 128, 0, 0);
  grad.addColorStop(0, 'rgba(255,248,205,0.95)'); grad.addColorStop(0.3, 'rgba(255,170,45,0.8)'); grad.addColorStop(0.7, 'rgba(225,70,22,0.45)'); grad.addColorStop(1, 'rgba(120,20,8,0)');
  g.fillStyle = grad; g.beginPath(); g.moveTo(2, 128); g.bezierCurveTo(0, 82, 20, 52, 32, 0); g.bezierCurveTo(44, 52, 64, 82, 62, 128); g.closePath(); g.fill();
  return c;
})();
// Flame geometry follows the physics. Length: Byram (1959) L = 0.0775 I^0.46 (I in kW/m). Tilt from vertical: tan(theta) = U / w, with U the mid-flame
// wind (0.4 of the 10 m wind) and w = (g I / (rho cp T))^(1/3) the buoyant velocity scale. Flames are intermittent (visible length 0.5-1.3 of the mean) and
// pulse at the Heskestad frequency 1.5 / sqrt(D) Hz. The base is one model cell wide. At 1x nothing is exaggerated, so flames only become visible at close zoom.
function flamePhys(I, ws) {
  const L = 0.0775 * Math.pow(I, 0.46), wc = Math.cbrt((9.81 * I) / (1.1 * 1.0 * 300)), th = Math.atan((0.4 * ws) / Math.max(wc, 0.3));
  return { L, th, H: L * Math.cos(th), tan: Math.tan(th) };
}
function flameFrame(ts) {
  requestAnimationFrame(flameFrame);
  if (!map || !S.meta) return;
  const w = fc.clientWidth, h = fc.clientHeight; if (fc.width !== w) { fc.width = w; fc.height = h; }
  fctx.setTransform(1, 0, 0, 1, 0, 0); fctx.clearRect(0, 0, w, h);
  if (!S.cols || S.mode !== 'fire' || !S.flameList.length || map.getZoom() < 9.5) return;
  const hh = Math.min(Math.floor(S.T / 60), S.meta.weather.wind.length - 1), ws = S.meta.weather.wind[hh];
  const to = ((S.meta.weather.wdir[hh] + 180 - map.getBearing()) * Math.PI) / 180;
  const mpp = 156543.03 * Math.cos((map.getCenter().lat * Math.PI) / 180) / Math.pow(2, map.getZoom()), cellPx = Math.max((S.meta.cell_m || 50) / mpp, 2), pitchK = Math.max(Math.cos((map.getPitch() * Math.PI) / 180), 0.25);
  fctx.globalCompositeOperation = 'lighter';
  for (const f of S.flameList) {
    let P; try { P = map.project([f[0], f[1]]); } catch (e) { return; }
    if (P.x < -60 || P.x > w + 60 || P.y < -60 || P.y > h + 60) continue;
    const ph = flamePhys(f[2], ws), freq = 1.5 / Math.sqrt(Math.max(ph.L, 0.5)), t = ts / 1000, ph0 = f[4] * 1.7;
    const fl = 0.9 + 0.4 * Math.sin(2 * Math.PI * freq * t + ph0) * Math.sin(2 * Math.PI * freq * 0.37 * t + ph0 * 0.6) - 0.35 * (0.5 + 0.5 * Math.sin(ts * 0.0031 + ph0));
    let ht = (ph.H * S.fh * Math.max(fl, 0.35) * pitchK) / mpp * f[3];
    if (S.fh > 1) ht = Math.min(Math.max(ht, 4), 90 * f[3]);
    if (ht < 0.6) continue;
    const lean = Math.sin(to) * ph.tan * (1 + 0.06 * Math.sin(ts * 0.009 + ph0));
    fctx.globalAlpha = Math.min(1, 0.4 + f[3] * 0.6);
    // Close up, a model cell is a line of separate flame tongues, each about L/3 wide (flame width scales with its length), spread along the fire line
    // (perpendicular to the wind) and a little along the wind; far away it is drawn as one sheet one cell wide.
    const tw = (ph.L / 3) * S.fh / mpp, nT = tw > 2 ? Math.min(Math.max(Math.round(cellPx / (tw * 1.4)), 1), 14) : 1;
    if (nT === 1) {
      const sy = ht / 128, sx = Math.max(cellPx * 0.9, ht * 0.18) / 64;
      fctx.setTransform(sx, 0, -lean * sy, sy, P.x, P.y); fctx.drawImage(SPR, -32, -128); continue;
    }
    const along = Math.sin(to + Math.PI / 2), alongY = -Math.cos(to + Math.PI / 2) * pitchK;
    for (let q = 0; q < nT; q++) {
      const hq = ((Math.imul(f[4] * 131 + q * 7919, 2654435761) >>> 0) % 1000) / 1000, hq2 = ((Math.imul(f[4] * 17 + q * 104729, 2246822519) >>> 0) % 1000) / 1000;
      const off = ((q + 0.5) / nT - 0.5) * cellPx, back = (hq - 0.5) * (ph.L * 0.5 * S.fh) / mpp;
      const fq = 0.75 + 0.45 * hq2 + 0.3 * Math.sin(2 * Math.PI * freq * t * (0.8 + 0.4 * hq) + hq * 40) * Math.sin(ts * 0.004 + hq2 * 9) ;
      const htq = Math.max(ht * fq / Math.max(fl, 0.35), 0), sy = htq / 128, sx = Math.max(tw, 1.5) / 64;
      fctx.setTransform(sx, 0, -lean * sy, sy, P.x + along * off + Math.sin(to) * back, P.y + alongY * off - Math.cos(to) * back * pitchK);
      fctx.drawImage(SPR, -32, -128);
    }
  }
  fctx.setTransform(1, 0, 0, 1, 0, 0); fctx.globalAlpha = 1; fctx.globalCompositeOperation = 'source-over';
}
requestAnimationFrame(flameFrame);

// ---------------------------------------------------------------- wind streaks
const wc = $('windCanvas'), wctx = wc.getContext('2d'); let parts = [];
function windFrame() {
  requestAnimationFrame(windFrame);
  if (!S.wind || !S.meta || !map) return;
  const w = wc.clientWidth, h = wc.clientHeight; if (wc.width !== w) { wc.width = w; wc.height = h; }
  while (parts.length < S.particles) parts.push([Math.random() * w, Math.random() * h, Math.random() * 60]);
  parts.length = S.particles;
  const hh = Math.min(Math.floor(S.T / 60), S.meta.weather.wind.length - 1), ws = S.meta.weather.wind[hh], to = S.meta.weather.wdir[hh] + 180;
  const ang = ((to - map.getBearing()) * Math.PI) / 180, vx = Math.sin(ang), vy = -Math.cos(ang), sp = 1.2 + ws * 0.9;
  wctx.globalCompositeOperation = 'destination-out'; wctx.fillStyle = 'rgba(0,0,0,0.12)'; wctx.fillRect(0, 0, w, h); wctx.globalCompositeOperation = 'source-over';
  wctx.strokeStyle = 'rgba(190,225,255,.55)'; wctx.lineWidth = 1.1; wctx.beginPath();
  for (const p of parts) { const x0 = p[0], y0 = p[1]; p[0] += vx * sp; p[1] += vy * sp; p[2]--; wctx.moveTo(x0, y0); wctx.lineTo(p[0], p[1]); if (p[2] <= 0 || p[0] < 0 || p[0] > w || p[1] < 0 || p[1] > h) { p[0] = Math.random() * w; p[1] = Math.random() * h; p[2] = 40 + Math.random() * 50; } }
  wctx.stroke();
}
requestAnimationFrame(windFrame);

// ---------------------------------------------------------------- export
$('exportToggle').onclick = () => $('exportMenu').classList.toggle('open');
$('exportBtn').onclick = () => { $('exportMenu').classList.remove('open'); map.once('idle', () => { const a = document.createElement('a'); a.href = map.getCanvas().toDataURL('image/png'); a.download = `pakfire_${window.RUN_ID}_${Math.round(S.T)}min.png`; a.click(); }); map.triggerRepaint(); };
$('exportVideoBtn3d').onclick = () => {
  $('exportMenu').classList.remove('open');
  const stream = map.getCanvas().captureStream(30), rec = new MediaRecorder(stream, { mimeType: 'video/webm' }), chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' })); a.download = `pakfire_${window.RUN_ID}.webm`; a.click(); };
  setFollow(true); S.T = 0; paint(0, true); rec.start(); startPlay();
  const t = setInterval(() => { if (!S.playing) { clearInterval(t); rec.stop(); } }, 300);
};
init().catch((e) => { document.body.insertAdjacentHTML('beforeend', `<div class="glass" style="position:absolute;left:50%;top:40%;transform:translateX(-50%);padding:16px 22px">Could not load this run: ${e.message}</div>`); });
