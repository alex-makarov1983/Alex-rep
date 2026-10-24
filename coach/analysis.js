// Разбор ударов по последовательности поз (33 точки MediaPipe Pose).
// Кадр: { t: секунды, p: [[x, y, visibility] × 33] в пикселях, w: [[x, y, z] × 33] в метрах или null }.
// Камеру лучше ставить сбоку от игрока на уровне пояса: так точнее всего видна точка удара.
(function (root) {
  "use strict";

  const K = typeof module !== "undefined" && module.exports ? require("./knowledge.js") : root.TennisKnowledge;

  const L = { nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lHip: 23, rHip: 24, lKnee: 25, rKnee: 26, lAnk: 27, rAnk: 28 };
  const MIN_VIS = 0.4;

  function side(hand) {
    const r = hand !== "left";
    return {
      sh: r ? L.rSh : L.lSh, el: r ? L.rEl : L.lEl, wr: r ? L.rWr : L.lWr,
      offSh: r ? L.lSh : L.rSh, offWr: r ? L.lWr : L.rWr,
    };
  }

  function pt(f, i) {
    if (!f || !f.p) return null;
    const q = f.p[i];
    return q && q[2] >= MIN_VIS ? q : null;
  }
  function mid(a, b) { return a && b ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.min(a[2], b[2])] : null; }
  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
  function angle(a, b, c) {
    if (!a || !b || !c) return null;
    const v1 = [a[0] - b[0], a[1] - b[1]], v2 = [c[0] - b[0], c[1] - b[1]];
    const n = Math.hypot(...v1) * Math.hypot(...v2);
    if (!n) return null;
    return Math.acos(Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / n))) * 180 / Math.PI;
  }
  function torso(f) {
    const s = mid(pt(f, L.lSh), pt(f, L.rSh)), h = mid(pt(f, L.lHip), pt(f, L.rHip));
    return s && h ? dist(s, h) : null;
  }
  function hipC(f) { return mid(pt(f, L.lHip), pt(f, L.rHip)); }
  function median(xs) {
    const a = xs.filter((x) => x != null && isFinite(x)).sort((x, y) => x - y);
    if (!a.length) return null;
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }
  function inWindow(frames, t0, t1) { return frames.filter((f) => f.t >= t0 && f.t <= t1); }
  function nearest(frames, t) {
    let best = null;
    for (const f of frames) if (!best || Math.abs(f.t - t) < Math.abs(best.t - t)) best = f;
    return best;
  }
  function maxOf(xs) { const a = xs.filter((x) => x != null && isFinite(x)); return a.length ? Math.max(...a) : null; }
  function minOf(xs) { const a = xs.filter((x) => x != null && isFinite(x)); return a.length ? Math.min(...a) : null; }

  // Оценка 0–100: внутри идеального диапазона 100, дальше линейно падает до 0 на расстоянии tol.
  function scoreValue(v, range, tol) {
    if (v == null) return null;
    const [lo, hi] = range;
    const d = v < lo ? lo - v : v > hi ? v - hi : 0;
    return Math.round(Math.max(0, 100 * (1 - d / tol)));
  }

  // Поиск ударов: пики скорости ударной кисти, измеренной в длинах корпуса в секунду.
  function detectStrokes(frames, hand, mode) {
    const s = side(hand);
    const scale = median(frames.map(torso)) || 1;
    const speed = frames.map((f, i) => {
      if (!i) return 0;
      const a = pt(frames[i - 1], s.wr), b = pt(f, s.wr), dt = f.t - frames[i - 1].t;
      return a && b && dt > 0 ? dist(a, b) / dt / scale : 0;
    });
    const sm = speed.map((_, i) => (speed[Math.max(0, i - 1)] + speed[i] + speed[Math.min(speed.length - 1, i + 1)]) / 3);
    const top = Math.max(0, ...sm);
    const thr = Math.max(2.5, top * 0.45);
    const gap = mode === "serve" ? 2.0 : 0.9;
    const peaks = [];
    for (let i = 1; i < sm.length - 1; i++) {
      if (sm[i] < thr || sm[i] < sm[i - 1] || sm[i] < sm[i + 1]) continue;
      const last = peaks[peaks.length - 1];
      if (last && frames[i].t - last.t < gap) {
        if (sm[i] > last.v) { last.t = frames[i].t; last.v = sm[i]; }
        continue;
      }
      peaks.push({ t: frames[i].t, v: sm[i] });
    }
    return { peaks, scale };
  }

  // Направление удара в кадре: куда движется кисть в момент контакта (+1 вправо, −1 влево).
  function hitDirection(frames, t, wr) {
    const a = pt(nearest(frames, t - 0.07), wr), b = pt(nearest(frames, t + 0.07), wr);
    return a && b && b[0] !== a[0] ? Math.sign(b[0] - a[0]) : 1;
  }

  // Форхенд или бэкхенд: на замахе форхенда кисть со стороны ударной руки.
  function classify(frames, t, hand) {
    const s = side(hand);
    const votes = inWindow(frames, t - 0.45, t - 0.15).map((f) => {
      if (f.w) {
        const sh = f.w[s.sh], off = f.w[s.offSh], wr = f.w[s.wr];
        const ax = [sh[0] - off[0], sh[2] - off[2]];
        const c = [(sh[0] + off[0]) / 2, (sh[2] + off[2]) / 2];
        return Math.sign((wr[0] - c[0]) * ax[0] + (wr[2] - c[1]) * ax[1]);
      }
      const sh = pt(f, s.sh), off = pt(f, s.offSh), wr = pt(f, s.wr);
      if (!sh || !off || !wr) return 0;
      return Math.sign((wr[0] - (sh[0] + off[0]) / 2) * (sh[0] - off[0]));
    });
    return votes.reduce((a, b) => a + b, 0) >= 0 ? "forehand" : "backhand";
  }

  // Поворот линии плеч (в градусах) за окно: по 3D-точкам, если есть, иначе по сужению плеч в кадре.
  function shoulderTurn(frames, t) {
    const win = inWindow(frames, t - 1.0, t + 0.3);
    const yaws = [];
    for (const f of win) {
      if (!f.w) continue;
      const a = f.w[L.lSh], b = f.w[L.rSh];
      yaws.push(Math.atan2(b[2] - a[2], b[0] - a[0]) * 180 / Math.PI);
    }
    if (yaws.length >= 3) {
      // Разворачиваем углы, чтобы переход через ±180° не давал скачков.
      const u = [yaws[0]];
      for (let i = 1; i < yaws.length; i++) {
        let d = yaws[i] - yaws[i - 1];
        while (d > 180) d -= 360;
        while (d < -180) d += 360;
        u.push(u[i - 1] + d);
      }
      return Math.max(...u) - Math.min(...u);
    }
    const ratios = win.map((f) => {
      const a = pt(f, L.lSh), b = pt(f, L.rSh), tr = torso(f);
      return a && b && tr ? dist(a, b) / tr : null;
    }).filter((x) => x != null);
    if (ratios.length < 3) return null;
    const mx = Math.max(...ratios), mn = Math.min(...ratios);
    return Math.acos(Math.max(0, Math.min(1, mn / mx))) * 180 / Math.PI;
  }

  function groundstrokeMetrics(frames, t, hand, scale) {
    const s = side(hand);
    const c = nearest(frames, t);
    const tr = torso(c) || scale;
    const dir = hitDirection(frames, t, s.wr);
    const wr = pt(c, s.wr), hip = hipC(c);

    const knees = inWindow(frames, t - 0.7, t + 0.05).map((f) => minOf([
      angle(pt(f, L.lHip), pt(f, L.lKnee), pt(f, L.lAnk)),
      angle(pt(f, L.rHip), pt(f, L.rKnee), pt(f, L.rAnk)),
    ]));

    const follow = inWindow(frames, t, t + 0.7).map((f) => {
      const sh = mid(pt(f, L.lSh), pt(f, L.rSh)), w = pt(f, s.wr), tt = torso(f) || scale;
      return sh && w ? (sh[1] - w[1]) / tt : null;
    });

    const n0 = pt(c, L.nose);
    const head = n0 ? maxOf(inWindow(frames, t - 0.15, t + 0.15).map((f) => {
      const n = pt(f, L.nose);
      return n ? dist(n, n0) / tr : null;
    })) : null;

    const la = pt(c, L.lAnk), ra = pt(c, L.rAnk);

    return {
      turn: shoulderTurn(frames, t),
      knee: minOf(knees),
      contactFront: wr && hip ? ((wr[0] - hip[0]) * dir) / tr : null,
      extension: angle(pt(c, s.sh), pt(c, s.el), wr),
      followThrough: maxOf(follow),
      head,
      base: la && ra ? dist(la, ra) / tr : null,
    };
  }

  function serveMetrics(frames, peakT, hand, scale) {
    const s = side(hand);
    // Контакт на подаче — самая высокая точка кисти рядом с пиком скорости.
    let c = nearest(frames, peakT), best = Infinity;
    for (const f of inWindow(frames, peakT - 0.3, peakT + 0.1)) {
      const w = pt(f, s.wr);
      if (w && w[1] < best) { best = w[1]; c = f; }
    }
    const t = c.t;
    const tr = torso(c) || scale;
    const nose = pt(c, L.nose), wr = pt(c, s.wr);

    let kneeMin = null, kneeT = null;
    for (const f of inWindow(frames, t - 1.3, t - 0.1)) {
      const a = minOf([
        angle(pt(f, L.lHip), pt(f, L.lKnee), pt(f, L.lAnk)),
        angle(pt(f, L.rHip), pt(f, L.rKnee), pt(f, L.rAnk)),
      ]);
      if (a != null && (kneeMin == null || a < kneeMin)) { kneeMin = a; kneeT = f.t; }
    }

    let legDrive = null;
    if (kneeT != null) {
      const h0 = hipC(nearest(frames, kneeT));
      const top = minOf(inWindow(frames, kneeT, t + 0.15).map((f) => { const h = hipC(f); return h ? h[1] : null; }));
      if (h0 && top != null) legDrive = (h0[1] - top) / tr;
    }

    const toss = maxOf(inWindow(frames, t - 1.5, t - 0.2).map((f) => {
      const n = pt(f, L.nose), w = pt(f, s.offWr), tt = torso(f) || scale;
      return n && w ? (n[1] - w[1]) / tt : null;
    }));

    const finish = maxOf(inWindow(frames, t + 0.1, t + 1.0).map((f) => {
      const h = hipC(f), w = pt(f, s.wr), tt = torso(f) || scale;
      return h && w ? (w[1] - h[1]) / tt : null;
    }));

    return {
      t,
      m: {
        trophy: kneeMin,
        contactHeight: nose && wr ? (nose[1] - wr[1]) / tr : null,
        serveExtension: angle(pt(c, s.sh), pt(c, s.el), wr),
        tossArm: toss,
        legDrive,
        serveFinish: finish,
      },
    };
  }

  function scoreMetrics(values) {
    const out = {};
    let sum = 0, wsum = 0;
    for (const [k, v] of Object.entries(values)) {
      const def = K.METRICS[k];
      if (!def || v == null) continue;
      const sc = scoreValue(v, def.range, def.tol);
      out[k] = { v: Math.round(v * 100) / 100, s: sc };
      sum += sc * def.weight; wsum += def.weight;
    }
    return { metrics: out, score: wsum ? Math.round(sum / wsum) : null };
  }

  function levelFor(score) {
    let lv = K.LEVELS[0];
    for (const l of K.LEVELS) if (score >= l.min) lv = l;
    return lv;
  }

  // Главная функция: кадры → удары → метрики по каждому типу удара → общая оценка и уровень.
  function analyze(frames, opts) {
    const hand = opts.hand || "right";
    const mode = opts.mode === "serve" ? "serve" : "ground";
    const usable = frames.filter((f) => pt(f, side(hand).wr) && torso(f));
    if (usable.length < 8) return { error: "nopose" };

    const { peaks, scale } = detectStrokes(frames, hand, mode);
    const strokes = [];
    for (const p of peaks.slice(0, 20)) {
      if (mode === "serve") {
        const r = serveMetrics(frames, p.t, hand, scale);
        strokes.push({ type: "serve", t: r.t, ...scoreMetrics(r.m) });
      } else {
        const type = classify(frames, p.t, hand);
        strokes.push({ type, t: p.t, ...scoreMetrics(groundstrokeMetrics(frames, p.t, hand, scale)) });
      }
    }
    if (!strokes.length) return { error: "nostrokes" };

    // По каждому типу удара берём медиану значений: один неудачный удар не портит картину.
    const byType = {};
    for (const st of strokes) (byType[st.type] = byType[st.type] || []).push(st);
    const types = {};
    let total = 0, n = 0;
    for (const [type, list] of Object.entries(byType)) {
      const values = {};
      for (const k of Object.keys(K.METRICS)) {
        if (!K.METRICS[k].strokes.includes(type)) continue;
        values[k] = median(list.map((x) => (x.metrics[k] ? x.metrics[k].v : null)));
      }
      const r = scoreMetrics(values);
      const scores = list.map((x) => x.score).filter((x) => x != null);
      const mean = scores.reduce((a, b) => a + b, 0) / (scores.length || 1);
      const spread = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / (scores.length || 1));
      types[type] = { count: list.length, score: r.score, metrics: r.metrics, consistency: Math.max(0, Math.round(100 - spread * 2)) };
      if (r.score != null) { total += r.score * list.length; n += list.length; }
    }
    const overall = n ? Math.round(total / n) : null;
    return { strokes, types, overall, level: levelFor(overall || 0) };
  }

  const api = { analyze, scoreValue, levelFor, detectStrokes, angle, LANDMARKS: L };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TennisAnalysis = api;
})(typeof window !== "undefined" ? window : globalThis);
