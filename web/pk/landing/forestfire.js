// Tree-to-tree wildfire model for the landing scene. Pure JavaScript, no rendering, so it can be tested in Node (tools/forestfire_test.mjs).
// It is a small physical model, not PAKFIRE's spread engine:
//   * every tree and shrub is a fuel node with a position, height, fuel load, heat and a state (unburnt, burning, burnt);
//   * a burning node heats its neighbours by radiation and convection: flux ~ flame size / (1 + (d/d0)^2), stronger downwind;
//   * an unburnt node ignites when the heat it has accumulated reaches 1 (heat is lost to the air at rate LAMBDA);
//   * a surface fire climbs into the crown when its line intensity reaches a threshold (Van Wagner 1977), and wind raises that intensity;
//   * crowning trees throw embers that loft, drift with the wind and start spot fires where they land (spotting, after Albini 1983 in spirit).
// SPREAD (K) multiplies the heat transfer and the crowning rate: the landing page uses it to compress minutes of spread into seconds.

export function mulberry32(a) {
  return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

export const P = {
  C0: 3.0,            // heat transfer coefficient
  D0: 3.0,            // m, distance at which the flux has fallen to a half
  LAMBDA: 0.40,       // 1/s, heat lost by an unburnt node
  SURF_TREE: 0.22,    // normalised surface-fire intensity under a tree
  SURF_SHRUB: 0.55,
  I_CROWN: 0.45,      // Van Wagner initiation threshold on the normalised line intensity
  EMBER_RATE: 0.9,    // embers per second from a fully crowning 15 m tree at 1 m/s wind
  R_NB: 16,           // m, neighbour search radius
};

export function buildForest(seed = 7, o = {}) {
  const rnd = mulberry32(seed);
  const X0 = o.x0 ?? -72, X1 = o.x1 ?? 72, Z0 = o.z0 ?? -104, Z1 = o.z1 ?? 9, cell = o.cell ?? 4.4;
  const x = [], z = [], h = [], kind = [];
  for (let gz = Z0; gz < Z1; gz += cell) for (let gx = X0; gx < X1; gx += cell) {
    if (rnd() < 0.10) continue;
    const tx = gx + (rnd() - 0.5) * cell * 0.85, tz = gz + (rnd() - 0.5) * cell * 0.85;
    if (o.lane !== false && Math.abs(tx - 0.8) < 10.5 && tz > -5) continue;    // a clearing in front of the camera, with trees left at the edges of the view
    if (o.lane !== false && Math.abs(tx + 2.0) < 5.5 && tz > -9) continue;     // a sightline to where the fire starts
    x.push(tx); z.push(tz);
    h.push(10 + 12 * Math.pow(rnd(), 1.35)); kind.push(1);
  }
  const nTrees = x.length, nShrubs = o.shrubs ?? 360;
  for (let i = 0; i < nShrubs; i++) { const sx = X0 + rnd() * (X1 - X0), sz = Z0 + rnd() * (Z1 - Z0), hh = 0.7 + 1.1 * rnd(); if (o.lane !== false && Math.abs(sx + 2.0) < 5.5 && sz > -9 && !(Math.abs(sx + 1.2) < 4.5 && sz < -9.5 && sz > -14)) continue; x.push(sx); z.push(sz); h.push(hh); kind.push(0); }
  const n = x.length;
  const X = Float32Array.from(x), Z = Float32Array.from(z), H = Float32Array.from(h), KIND = Uint8Array.from(kind);
  // neighbour lists (compressed rows): index, distance, unit vector from the burning node towards the neighbour
  const g = new Map(), gs = P.R_NB;
  const key = (a, b) => a * 4096 + b;
  for (let i = 0; i < n; i++) { const k = key(Math.floor((X[i] + 200) / gs), Math.floor((Z[i] + 200) / gs)); (g.get(k) || g.set(k, []).get(k)).push(i); }
  const off = new Int32Array(n + 1), idx = [], dd = [], ux = [], uz = [];
  for (let i = 0; i < n; i++) {
    const cx = Math.floor((X[i] + 200) / gs), cz = Math.floor((Z[i] + 200) / gs);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (const j of g.get(key(cx + a, cz + b)) || []) {
      if (j === i) continue;
      const dx = X[j] - X[i], dz = Z[j] - Z[i], d = Math.hypot(dx, dz);
      if (d < P.R_NB && d > 0.05) { idx.push(j); dd.push(d); ux.push(dx / d); uz.push(dz / d); }
    }
    off[i + 1] = idx.length;
  }
  return { n, nTrees, X, Z, H, KIND, off, idx: Int32Array.from(idx), dd: Float32Array.from(dd), ux: Float32Array.from(ux), uz: Float32Array.from(uz), grid: g, gs, key };
}

export class Fire {
  constructor(forest, seed = 11) {
    this.f = forest; const n = forest.n;
    this.rnd = mulberry32(seed);
    this.heat = new Float32Array(n); this.I = new Float32Array(n); this.crown = new Float32Array(n);
    this.fuel = new Float32Array(n).fill(1); this.st = new Uint8Array(n); this.age = new Float32Array(n); this.glow = new Float32Array(n);
    this.keep = new Uint8Array(n);                       // persistent seeds: the fire under them is fed until the surge
    this.qin = new Float32Array(n);                      // heat flux received this step
    this.embers = []; this.maxEmbers = 1800; this.t = 0; this.spots = 0;
  }
  ignite(i, keep = false) { if (this.st[i] === 0) { this.st[i] = 1; this.I[i] = 0.08; this.keep[i] = keep ? 1 : 0; } }
  nearest(x, z, maxD = 3) {
    const f = this.f; let best = -1, bd = maxD;
    const cx = Math.floor((x + 200) / f.gs), cz = Math.floor((z + 200) / f.gs);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (const j of f.grid.get(f.key(cx + a, cz + b)) || []) {
      const d = Math.hypot(f.X[j] - x, f.Z[j] - z) - (f.KIND[j] ? 0.3 : 0.2);
      if (this.st[j] === 0 && d < bd) { bd = d; best = j; }
    }
    return best;
  }
  // ctl: { wx, wz (m/s, direction the wind blows towards), K (spread multiplier), surge (0..1) }
  step(dt, ctl) {
    const f = this.f, n = f.n, W = Math.hypot(ctl.wx, ctl.wz), K = ctl.K, ex = Math.exp(-P.LAMBDA * dt);
    const wux = W > 1e-3 ? ctl.wx / W : 0, wuz = W > 1e-3 ? ctl.wz / W : 0;
    this.qin.fill(0);
    for (let j = 0; j < n; j++) {
      if (this.st[j] !== 1) continue;
      const tree = f.KIND[j] === 1;
      const size = tree ? f.H[j] / 12 : f.H[j] / 3;
      const Q = this.I[j] * size * (tree ? 0.35 + 0.65 * this.crown[j] : 1.0);
      if (Q < 0.002) continue;
      for (let k = f.off[j]; k < f.off[j + 1]; k++) {
        const i = f.idx[k]; if (this.st[i] !== 0) continue;
        const d = f.dd[k], r = d / P.D0;
        const cos = wux * f.ux[k] + wuz * f.uz[k];
        const wf = 0.30 + (0.35 + 0.16 * W) * Math.max(0, cos + 0.15);
        const q = P.C0 * K * Q / (1 + r * r) * wf * (f.KIND[i] ? 0.7 : 1.5);
        this.heat[i] += q * dt; this.qin[i] += q;
      }
    }
    for (let i = 0; i < n; i++) {
      const s = this.st[i];
      if (s === 0) { this.heat[i] *= ex; if (this.heat[i] >= 1) { this.st[i] = 1; this.I[i] = 0.06; this.heat[i] = 1; } continue; }
      if (s === 1) {
        this.age[i] += dt;
        const tree = f.KIND[i] === 1;
        const surf = tree ? P.SURF_TREE : P.SURF_SHRUB;
        // crown fire: line intensity grows with wind and with heat from burning neighbours; above the threshold the crown takes
        let drive = 0;
        if (tree) {
          const iLine = surf * (1 + 0.25 * W) + 1.2 * this.qin[i];
          const target = Math.min(1, Math.max(0, (iLine - P.I_CROWN) / 0.30));
          const tc = 2.2 / (1 + 0.12 * (K - 1));
          this.crown[i] += (target - this.crown[i]) * Math.min(1, dt / tc);
          drive = this.crown[i];
        }
        const imax = surf + (1 - surf) * drive;
        this.I[i] += (imax - this.I[i]) * Math.min(1, dt / 4);
        const burn = (tree ? 0.004 : 0.02) + (tree ? 0.05 : 0.04) * this.I[i];
        this.fuel[i] -= burn * dt;
        if (this.keep[i] && ctl.surge <= 0 && this.fuel[i] < 0.4) this.fuel[i] = 0.4;
        if (this.fuel[i] <= 0) { this.st[i] = 2; this.glow[i] = 1; }
        // embers from crowning trees
        if (tree && this.crown[i] > 0.4 && this.embers.length < this.maxEmbers) {
          const rate = P.EMBER_RATE * this.crown[i] * this.I[i] * (f.H[i] / 15) * (0.5 + 0.25 * W) * (ctl.emberMul ?? 1);
          if (this.rnd() < rate * dt) this.spawnEmber(i, W, ctl);
        }
      } else {
        this.I[i] *= Math.exp(-dt / 3); this.glow[i] *= Math.exp(-dt / 14); this.crown[i] *= Math.exp(-dt / 5);
      }
    }
    this.stepEmbers(dt, ctl, W);
    this.t += dt;
  }
  spawnEmber(i, W, ctl) {
    const f = this.f, r = this.rnd;
    const top = f.H[i] * (0.55 + 0.45 * r());
    this.embers.push({ x: f.X[i] + (r() - 0.5) * 2, y: top, z: f.Z[i] + (r() - 0.5) * 2,
      vx: ctl.wx * (0.5 + 0.5 * r()) + (r() - 0.5) * 2, vy: 3 + 7 * r() * (0.5 + this.I[i]), vz: ctl.wz * (0.5 + 0.5 * r()) + (r() - 0.5) * 2,
      life: 7 + 14 * r(), age: 0, heat: 0.7 + 0.5 * r() });
  }
  stepEmbers(dt, ctl, W) {
    const E = this.embers;
    for (let k = E.length - 1; k >= 0; k--) {
      const e = E[k];
      e.age += dt;
      e.vy += (-2.8 - e.vy) * Math.min(1, 0.55 * dt);                 // loft decays towards a fall at the ember's terminal speed (a few m/s for a bark flake)
      e.vx += (ctl.wx * 0.45 - e.vx) * Math.min(1, 0.8 * dt); e.vz += (ctl.wz * 0.45 - e.vz) * Math.min(1, 0.8 * dt);
      e.x += e.vx * dt; e.y += e.vy * dt; e.z += e.vz * dt;
      if (e.y <= 0.3 || e.age > e.life) {
        if (e.y <= 0.6 && e.age < e.life) {                            // landed alive: may start a spot fire
          const j = this.nearest(e.x, e.z, 3);
          if (j >= 0) { this.heat[j] += e.heat * (ctl.spotHeat ?? 0.9); if (this.heat[j] >= 1) { this.ignite(j); this.spots++; } }
        }
        E[k] = E[E.length - 1]; E.pop();
      }
    }
  }
  stats(box) {
    const f = this.f; let burning = 0, burnt = 0, total = 0, bT = 0, tT = 0;
    for (let i = 0; i < f.n; i++) {
      if (box && !(f.X[i] > box.x0 && f.X[i] < box.x1 && f.Z[i] > box.z0 && f.Z[i] < box.z1)) continue;
      total++; if (this.st[i] === 1) burning++; else if (this.st[i] === 2) burnt++;
      if (f.KIND[i] === 1) { tT++; if (this.st[i] !== 0) bT++; }
    }
    return { total, burning, burnt, fracAlight: (burning + burnt) / Math.max(total, 1), treeFrac: bT / Math.max(tT, 1), embers: this.embers.length, spots: this.spots };
  }
}

// the surge schedule used by the landing page: progress k in 0..1 -> wind, spread multiplier
export function surgeControl(k, base = { wx: 0.35, wz: 1.0 }) {
  const s = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
  const W0 = 1.3, W1 = 11.0, w = W0 + (W1 - W0) * s(k / 0.45);
  const len = Math.hypot(base.wx, base.wz);
  return { wx: base.wx / len * w, wz: base.wz / len * w, K: 1 + 30 * s(k / 0.50), surge: k, emberMul: 1 + 3.0 * s(k / 0.45), spotHeat: 1.7 };
}

export function seedFire(fire, at = [-3.5, -12]) {
  const f = fire.f; const picks = [];
  const near = (x, z, kind) => { let b = -1, bd = 1e9; for (let i = 0; i < f.n; i++) { if (kind !== undefined && f.KIND[i] !== kind) continue; if (picks.includes(i)) continue; const d = Math.hypot(f.X[i] - x, f.Z[i] - z); if (d < bd) { bd = d; b = i; } } return b; };
  picks.push(near(at[0], at[1], 1)); picks.push(near(at[0] + 4.6, at[1] - 1.5, 1));
  picks.push(near(at[0] + 2.4, at[1] + 1.0, 0)); picks.push(near(at[0] - 2.2, at[1] - 1.2, 0)); picks.push(near(at[0] + 6.5, at[1] - 2.5, 0));
  picks.forEach((i) => fire.ignite(i, true));
  return picks;
}
