// Node test of the 3D scene script (web/pk/viewer/fire3d.js) with a stubbed map and DOM, against a run on the server.
// Usage: node tools/viewer_test.js <run_id> [base_url]
const vm = require('vm'), fs = require('fs'), path = require('path');
const RUN = process.argv[2], BASE = process.argv[3] || 'http://localhost:8020';
const noop = () => {};
const ctx2d = new Proxy({}, { get: (t, k) => (k in t ? t[k] : (k === 'createLinearGradient' ? () => ({ addColorStop: noop }) : k === 'createImageData' ? (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }) : (k === 'measureText' ? () => ({ width: 10 }) : noop))), set: (t, k, v) => { t[k] = v; return true; } });
const els = {};
const mkEl = (id) => ({ id, style: {}, classList: { toggle: noop, add: noop, remove: noop }, addEventListener: noop, appendChild: noop, querySelectorAll: () => [], clientWidth: 0, clientHeight: 0, value: '0', textContent: '', innerHTML: '', width: 0, height: 0, getContext: () => ctx2d, getBoundingClientRect: () => ({ left: 0, top: 0, width: 300, height: 100 }), remove: noop });
const el = (id) => (els[id] ||= mkEl(id));
const reg = { sources: {}, layers: {}, handlers: [] };
const mkSrc = (spec) => ({ data: spec.data, setData(d) { this.data = d; }, play: noop, pause: noop });
const map = { on(ev, a, b) { reg.handlers.push([ev, a, b]); }, once: noop, addControl: noop, addSource(id, s) { reg.sources[id] = mkSrc(s); }, getSource: (id) => reg.sources[id],
  addLayer(l) { reg.layers[l.id] = l; }, getLayer: (id) => reg.layers[id], setLayoutProperty: noop, setPaintProperty: noop, setFilter: noop, setTerrain: noop, setFog: noop,
  getStyle: () => ({ layers: [] }), jumpTo: noop, easeTo: noop, flyTo: noop, getBearing: () => 0, setStyle: noop, queryTerrainElevation: () => 1234, setConfigProperty: noop,
  getCanvas: () => ({ captureStream: noop, toDataURL: () => '' }), triggerRepaint: noop, resize: noop, isMoving: () => false };
const sandbox = { console, fetch: (u, o) => fetch(BASE + u, o), setTimeout, clearTimeout, setInterval, clearInterval, requestAnimationFrame: noop, performance, Uint16Array, Uint32Array, Uint8Array, Uint8ClampedArray, Date, Math, Object, Promise, Array, Number, String, JSON, URL,
  document: { getElementById: el, createElement: (t) => mkEl('new-' + t), body: { appendChild: noop, insertAdjacentHTML: noop, classList: { toggle: noop } }, documentElement: {}, fullscreenElement: null, querySelectorAll: () => [] },
  window: { MAPBOX_TOKEN: 'x', RUN_ID: RUN, addEventListener: noop, devicePixelRatio: 1 }, mapboxgl: { Map: function () { return map; }, NavigationControl: function () {}, Marker: function () { return { setLngLat() { return this; }, addTo() { return this; }, remove: noop }; } }, MediaRecorder: function () {} };
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
const run = (c) => vm.runInContext(c, sandbox);
const results = {}; const check = (name, ok, info) => { results[name] = ok ? 'ok' : 'FAIL ' + (info || ''); };
(async () => {
  run(fs.readFileSync(path.join(__dirname, '../web/pk/viewer/fire3d.js'), 'utf8'));
  for (let i = 0; i < 100 && !run('S.R'); i++) await new Promise((r) => setTimeout(r, 100));
  check('data loaded', !!run('S.R'));
  const sl = reg.handlers.find((h) => h[0] === 'style.load'); sl[1]();           // the style finished loading: add sources and layers
  const need = ['fire', 'flames', 'sparks', 'smoke', 'spots', 'axis', 'axis-done', 'wpts', 'head', 'ign'];
  check('sources added', need.every((s) => reg.sources[s]), need.filter((s) => !reg.sources[s]).join(','));
  const alive = () => run('(()=>{let c=0;const p=S.R.px;for(let i=0;i<p.length;i++)if(p[i]>>>24)c++;return c})()');
  const te = run('S.meta.t_end_min'), nb = run('S.R.nb');
  const hasProb = run('S.meta.has_prob'), maxI = run('S.meta.stats.max_intensity_kw_m');
  for (const mode of ['fire', 'arrival', 'intensity'].concat(hasProb ? ['prob'] : [])) {
    run(`S.mode='${mode}'`); run('paint(0,true)'); const a0 = alive(); run(`paint(${te / 2},true)`); const am = alive(); run(`paint(${te},true)`); const ae = alive();
    check(`mode ${mode} paints`, mode === 'prob' ? (a0 === am && am === ae && ae > 0) : (am > a0 && ae >= am && ae > 0), `${a0}/${am}/${ae}`);
    if (mode === 'arrival' || mode === 'intensity') check(`mode ${mode} covers every burnt cell at the end`, ae === nb, `${ae} vs ${nb}`);
  }
  run("S.mode='fire'"); run('paint(0,true)');
  let same = true; const snap = () => Uint32Array.from(run('S.R.px'));
  for (let t = 0; t <= te && same; t += te / 33) { run(`paint(${t})`); const a = snap(); run(`paint(${t},true)`); const b = run('S.R.px'); for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { same = false; break; } }
  run(`paint(${te * 0.7})`); run(`paint(${te * 0.2})`); { const a = snap(); run(`paint(${te * 0.2},true)`); const b = run('S.R.px'); for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { same = false; break; } }
  check('incremental paint equals full repaint (fire mode, incl. backward scrub)', same);
  run('S.glow=true;S.cols=true;S.smoke=true;S.mode="fire"'); run(`paint(${te * 0.5},true)`);
  const n = (s) => reg.sources[s].data.features.length;
  check('flame glow points', n('flames') > 0, n('flames')); check('sparks (3 per glow point)', n('sparks') === 3 * n('flames'), `${n('sparks')} vs ${n('flames')}`);
  let tongues = 0; for (let f = 0.05; f < 1; f += 0.05) { run(`paint(${te * f},true)`); tongues = Math.max(tongues, run('S.flameList.length')); } run(`paint(${te * 0.5},true)`);
  if (maxI >= 60) check('flame sprites appear at some time', tongues > 0, tongues); else results['flame sprites'] = 'skipped (fire weaker than 60 kW/m)'; check('smoke points', n('smoke') > 0, n('smoke'));
  run('S.mode="arrival"'); run(`paint(${te * 0.5},true)`); check('flame layers empty outside fire mode', n('flames') === 0 && run('S.flameList.length') === 0);
  run('S.mode="fire"'); run(`paint(${te * 0.5},true)`);
  const done = reg.sources['axis-done'].data.geometry.coordinates; check('travelled spread axis grows with time', done.length >= 2);
  check('hud text', /^\+\d+:\d\d$/.test(els.timeBig.textContent) && /PKT/.test(els.timeSub.textContent), els.timeBig.textContent + '|' + els.timeSub.textContent);
  const cam = run(`camAt(${te / 2})`); check('camera interpolation', cam && cam.length === 5 && cam.every(Number.isFinite));
  const [W, Sb, E, N] = run('S.meta.bounds'); const pr = reg.handlers.find((h) => h[0] === 'click' && typeof h[1] === 'function');
  pr[1]({ lngLat: { lng: (W + E) / 2 + 0.01, lat: (Sb + N) / 2 + 0.01 } }); check('probe reads a cell', /^\+|unburned/.test(els.probeDepth.textContent), els.probeDepth.textContent);
  check('preset by clock', run('presetFor(3)') === 'night' && run('presetFor(6)') === 'dawn' && run('presetFor(12)') === 'day' && run('presetFor(18)') === 'dusk' && run('presetFor(22)') === 'night');
  console.log(JSON.stringify(results, null, 1));
  process.exit(Object.values(results).some((v) => v !== 'ok') ? 1 : 0);
})().catch((e) => { console.error('FAIL', e); process.exit(1); });
