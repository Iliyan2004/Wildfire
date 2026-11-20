// Node test of the landing page's fire model: node tools/forestfire_test.mjs
import { buildForest, Fire, surgeControl, seedFire } from "../web/pk/landing/forestfire.js";
const forest = buildForest(7);
console.log("nodes", forest.n, "trees", forest.nTrees);
let bad = 0; const ok = (c, m) => { console.log(c ? "ok  " : "FAIL", m); if (!c) bad++; };

// idle: a small fire that stays small for two minutes
let fire = new Fire(forest); const seeds = seedFire(fire);
const idle = { wx: 0.35 * 1.3, wz: 1.3, K: 1, surge: 0, emberMul: 1 };
let maxB = 0;
for (let t = 0; t < 120; t += 1 / 30) { fire.step(1 / 30, idle); if ((t * 30 | 0) % 30 === 0) maxB = Math.max(maxB, fire.stats().burning); }
const s0 = fire.stats();
ok(maxB <= 6, `idle fire stays small (at most ${maxB} nodes burning over 120 s, ${s0.burning} now)`);
ok(seeds.every((i) => fire.st[i] === 1), "the seeds are still burning after 120 s (fed)");

// surge: progress over DURATION seconds, with the camera region as the box
const DUR = +(process.argv[2] || 4.8);
fire = new Fire(forest); seedFire(fire);
for (let t = 0; t < 8; t += 1 / 30) fire.step(1 / 30, idle);               // settle as the page would
const box = { x0: -50, x1: 50, z0: -70, z1: 8 };
const marks = [0.25, 0.5, 0.65, 0.8, 0.9, 1.0]; let mi = 0;
for (let t = 0; t <= DUR + 1e-6; t += 1 / 30) {
  const k = Math.min(1, t / DUR); fire.step(1 / 30, surgeControl(k));
  if (mi < marks.length && k >= marks[mi]) { const s = fire.stats(box); console.log(`k=${marks[mi]}  alight ${(s.fracAlight * 100).toFixed(0)}%  burning ${s.burning}  embers ${s.embers}  spot fires ${s.spots}`); mi++; }
}
const end = fire.stats(box);
ok(end.fracAlight > 0.80, `the surge covers most of the view by the end (${(end.fracAlight * 100).toFixed(0)}% alight)`);
fire = new Fire(forest); seedFire(fire); for (let t = 0; t < 8; t += 1 / 30) fire.step(1 / 30, idle);
for (let t = 0; t <= DUR * 0.4; t += 1 / 30) fire.step(1 / 30, surgeControl(t / DUR));
const early = fire.stats(box);
ok(early.fracAlight < 0.35, `it takes time to spread (${(early.fracAlight * 100).toFixed(0)}% alight at 40% of the surge)`);
// spotting alone: with heat transfer between trees switched off, only embers can spread the fire
import { P } from "../web/pk/landing/forestfire.js";
const c0 = P.C0; P.C0 = 0;
fire = new Fire(forest); seedFire(fire);
const gale = { wx: 11, wz: 3, K: 1, surge: 1, emberMul: 4, spotHeat: 1.7 };
for (let t = 0; t < 90; t += 1 / 30) fire.step(1 / 30, gale);
P.C0 = c0;
ok(fire.spots > 0, `embers alone start spot fires in a windy burn (${fire.spots} in 90 s, ${fire.stats().burning + fire.stats().burnt} nodes alight)`);
process.exit(bad ? 1 : 0);
