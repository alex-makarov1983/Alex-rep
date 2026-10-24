// Проверка разбора на синтетических «скелетах»: правильный и неправильный форхенд, подача.
// Запуск: node --test coach/test/*.test.js
const test = require("node:test");
const assert = require("node:assert");
const A = require("../analysis.js");

const { FPS, pose, forehandClip, smooth, lerp } = require("./synthetic.js");

test("находит три удара и отличает хороший форхенд от плохого", () => {
  const good = A.analyze(forehandClip(true), { mode: "ground", hand: "right" });
  const bad = A.analyze(forehandClip(false), { mode: "ground", hand: "right" });
  assert.ok(!good.error && !bad.error, JSON.stringify({ g: good.error, b: bad.error }));
  assert.equal(good.strokes.length, 3);
  assert.equal(bad.strokes.length, 3);
  assert.ok(good.types.forehand, "удары распознаны как форхенд");
  const g = good.types.forehand.metrics, b = bad.types.forehand.metrics;
  for (const k of ["contactFront", "knee", "base", "turn", "followThrough"]) {
    assert.ok(g[k].s > b[k].s, `${k}: ${g[k].s} > ${b[k].s}`);
  }
  assert.ok(good.overall >= 75, "хороший форхенд: " + good.overall);
  assert.ok(bad.overall < 55, "плохой форхенд: " + bad.overall);
});

test("без людей в кадре возвращает ошибку nopose", () => {
  const frames = Array.from({ length: 30 }, (_, i) => ({ t: i / FPS, p: null, w: null }));
  assert.equal(A.analyze(frames, { mode: "ground" }).error, "nopose");
});

test("подача: высокая точка удара и толчок ногами", () => {
  const frames = [];
  for (let i = 0; i < 4 * FPS; i++) {
    const t = i / FPS, d = t - 2;
    let wx = 430, wy = 450, drop = 0, knee = 0, off = 380;
    if (d > -1.3 && d < -0.2) { const k = smooth((d + 1.3) / 1.1); drop = 40 * k; knee = 70 * k; off = lerp(380, 120, k); wx = 300; wy = 330; }
    else if (d >= -0.2 && d <= 0) { const k = smooth((d + 0.2) / 0.2); drop = lerp(40, -40, k); wx = lerp(300, 410, k); wy = lerp(330, 60, k); off = 250; }
    else if (d > 0 && d < 0.8) { const k = smooth(d / 0.8); drop = lerp(-40, 0, k); wx = lerp(410, 380, k); wy = lerp(60, 640, k); }
    const f = pose({ shift: 0, drop, knee, stance: 160, wx, wy, elbowBend: 0, yaw: 0, headDx: 0 });
    f.p[15] = [370, off, 0.95];
    frames.push({ t, ...f });
  }
  const r = A.analyze(frames, { mode: "serve", hand: "right" });
  assert.ok(!r.error, r.error);
  assert.equal(r.strokes.length, 1);
  const m = r.types.serve.metrics;
  assert.ok(m.contactHeight.s >= 80, "высота " + JSON.stringify(m.contactHeight));
  assert.ok(m.legDrive.s >= 80, "толчок " + JSON.stringify(m.legDrive));
  assert.ok(m.tossArm.s >= 80, "рука " + JSON.stringify(m.tossArm));
});

test("scoreValue: 100 в диапазоне и 0 за допуском", () => {
  assert.equal(A.scoreValue(5, [0, 10], 4), 100);
  assert.equal(A.scoreValue(12, [0, 10], 4), 50);
  assert.equal(A.scoreValue(20, [0, 10], 4), 0);
});
