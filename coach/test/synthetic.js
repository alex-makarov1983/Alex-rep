// Синтетические «скелеты» для проверок: форхенд снятый сбоку, правильный или с ошибками.
const FPS = 15;
const lerp = (a, b, k) => a + (b - a) * k;
const smooth = (k) => k * k * (3 - 2 * k);

// Правша снят сбоку, бьёт вправо (сетка справа). Торс 200 px.
function pose(o) {
  const p = Array.from({ length: 33 }, () => [0, 0, 0]);
  const set = (i, x, y) => { p[i] = [x, y, 0.95]; };
  const hx = 400 + o.shift, hy = 500 + o.drop;
  const kneeOut = o.knee; // сдвиг колена вперёд даёт сгибание
  set(23, hx - 10, hy); set(24, hx + 10, hy);
  set(11, hx - 15, hy - 200); set(12, hx + 15, hy - 200);
  set(0, hx + 5 + o.headDx, hy - 260);
  set(25, hx - 10 - o.stance / 4 + kneeOut, hy + 120 - o.drop / 2); set(26, hx + 10 + o.stance / 4 + kneeOut, hy + 120 - o.drop / 2);
  set(27, hx - o.stance / 2, hy + 240 - o.drop); set(28, hx + o.stance / 2, hy + 240 - o.drop);
  set(16, o.wx, o.wy); set(14, (hx + 15 + o.wx) / 2 - o.elbowBend, (hy - 200 + o.wy) / 2 + o.elbowBend);
  set(15, hx - 40, hy - 120); set(13, hx - 30, hy - 160);
  const yaw = o.yaw * Math.PI / 180;
  const w = Array.from({ length: 33 }, () => [0, 0, 0]);
  w[11] = [-0.2 * Math.cos(yaw), -0.5, -0.2 * Math.sin(yaw)];
  w[12] = [0.2 * Math.cos(yaw), -0.5, 0.2 * Math.sin(yaw)];
  // На замахе форхенда кисть позади, со стороны правого плеча.
  w[16] = o.wx < hx ? [w[12][0] * 2.5, -0.4, w[12][2] * 2.5 + 0.3] : [0.5, -0.4, -0.3];
  return { p, w };
}

function forehandClip(good) {
  const frames = [];
  const hits = [1.5, 3.5, 5.5];
  for (let i = 0; i < 7.5 * FPS; i++) {
    const t = i / FPS;
    const h = hits.reduce((a, b) => (Math.abs(b - t) < Math.abs(a - t) ? b : a));
    const d = t - h;
    const hx = 400;
    let wx = hx + 60, wy = 380, yaw = 0, knee = 0, drop = 0;
    const contactX = hx + (good ? 170 : 10);
    if (d >= -1.0 && d < 0) {
      const k = smooth((d + 1.0) / 1.0);
      wx = d < -0.3 ? lerp(hx + 60, hx - 200, smooth(Math.min(1, (d + 1) / 0.5))) : lerp(hx - 200, contactX, smooth((d + 0.3) / 0.3));
      wy = 400;
      yaw = good ? lerp(0, 90, Math.min(1, k * 2)) * (d < -0.3 ? 1 : 1 - smooth((d + 0.3) / 0.3)) : 15 * k;
      knee = good ? 60 * Math.min(1, k * 1.5) : 5;
      drop = good ? 30 * Math.min(1, k * 1.5) : 0;
    } else if (d >= 0 && d < 0.8) {
      const k = smooth(d / 0.8);
      wx = lerp(contactX, hx - (good ? 40 : -100), k);
      wy = lerp(400, good ? 220 : 380, k);
      knee = good ? 60 * (1 - k) : 5;
      drop = good ? 30 * (1 - k) : 0;
    }
    frames.push({ t, ...pose({ shift: 0, drop, knee, stance: good ? 240 : 110, wx, wy, elbowBend: good ? 0 : 60, yaw, headDx: good ? 0 : (d > 0 && d < 0.2 ? 60 : 0) }) });
  }
  return frames;
}

module.exports = { FPS, pose, forehandClip, smooth, lerp };
