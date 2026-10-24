// Интерфейс тренера: распознавание позы на видео (MediaPipe), разбор, история и план тренировок.
(function () {
  "use strict";

  const K = window.TennisKnowledge;
  const A = window.TennisAnalysis;

  const params = new URLSearchParams(location.search);
  const CDN = params.get("cdn") || "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
  const MODEL = params.get("model") ||
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
  const FPS = 15;          // кадров в секунду для разбора: хватает, чтобы поймать удар
  const MAX_SECONDS = 45;  // длиннее разбирать долго, а пользы мало

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------- Хранилище ----------
  const store = {
    get(key, def) {
      try { const v = localStorage.getItem("coach." + key); return v ? JSON.parse(v) : def; } catch (e) { return def; }
    },
    set(key, v) {
      try { localStorage.setItem("coach." + key, JSON.stringify(v)); return true; } catch (e) { return false; }
    },
  };
  let sessions = store.get("sessions", []);
  const prefs = store.get("prefs", { mode: "ground", hand: "right" });

  // ---------- Переключатели ----------
  function bindSeg(id, key) {
    const seg = $(id);
    const sync = () => seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === prefs[key])));
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      prefs[key] = b.dataset.v;
      store.set("prefs", prefs);
      sync();
    });
    sync();
  }
  bindSeg("modeSeg", "mode");
  bindSeg("handSeg", "hand");

  function showTab(name) {
    document.querySelectorAll("nav.tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll("section.tab").forEach((s) => { s.hidden = s.id !== "tab-" + name; });
    if (name === "progress") renderProgress();
    if (name === "plan") renderPlan();
    window.scrollTo(0, 0);
  }
  document.querySelector("nav.tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (b) showTab(b.dataset.tab);
  });

  // ---------- Распознавание позы ----------
  let landmarkerPromise = null;
  let lastTs = 0; // метки времени для модели должны расти и между разными видео
  function getLandmarker() {
    if (!landmarkerPromise) {
      landmarkerPromise = (async () => {
        const vision = await import(CDN + "/vision_bundle.mjs");
        const files = await vision.FilesetResolver.forVisionTasks(CDN + "/wasm");
        const make = (delegate) => vision.PoseLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: MODEL, delegate },
          runningMode: "VIDEO",
          numPoses: 2,
          minPoseDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        try { return await make("GPU"); } catch (e) { return await make("CPU"); }
      })();
      landmarkerPromise.catch(() => { landmarkerPromise = null; });
    }
    return landmarkerPromise;
  }

  function seek(video, t) {
    return new Promise((resolve) => {
      const done = () => { video.removeEventListener("seeked", done); resolve(); };
      video.addEventListener("seeked", done);
      video.currentTime = t;
    });
  }

  // Если в кадре двое (например, партнёр на другой стороне), берём того, кто крупнее, то есть ближе к камере.
  function pickPlayer(res) {
    let best = -1, bestSize = -1;
    (res.landmarks || []).forEach((lm, i) => {
      const xs = lm.map((p) => p.x), ys = lm.map((p) => p.y);
      const size = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
      if (size > bestSize) { bestSize = size; best = i; }
    });
    return best;
  }

  async function extractFrames(video, onProgress) {
    const lm = await getLandmarker();
    const W = video.videoWidth, H = video.videoHeight;
    const end = Math.min(video.duration, MAX_SECONDS);
    const frames = [];
    for (let t = 0; t < end; t += 1 / FPS) {
      await seek(video, t);
      lastTs += Math.max(1, Math.round(1000 / FPS));
      const res = lm.detectForVideo(video, lastTs);
      const i = pickPlayer(res);
      if (i < 0) frames.push({ t, p: null, w: null });
      else {
        frames.push({
          t,
          p: res.landmarks[i].map((q) => [q.x * W, q.y * H, q.visibility == null ? 1 : q.visibility]),
          w: res.worldLandmarks && res.worldLandmarks[i] ? res.worldLandmarks[i].map((q) => [q.x, q.y, q.z]) : null,
        });
      }
      onProgress(t / end);
    }
    return frames;
  }

  // ---------- Разбор ----------
  let current = null; // { frames, result, session }

  $("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) await processFile(file);
  });

  function setWork(title, text, frac) {
    $("workCard").hidden = false;
    $("workTitle").textContent = title;
    $("workText").textContent = text;
    $("workBar").style.width = Math.round((frac || 0) * 100) + "%";
  }

  async function processFile(file) {
    const video = $("video");
    $("result").hidden = true;
    setWork("Разбираю видео…", "Загружаю модель распознавания позы", 0.02);
    try {
      if (video.src) URL.revokeObjectURL(video.src);
      video.src = URL.createObjectURL(file);
      await new Promise((ok, fail) => {
        video.onloadeddata = ok;
        video.onerror = () => fail(new Error("video"));
      });
      try { await getLandmarker(); } catch (e) { throw new Error("model"); }
      const frames = await extractFrames(video, (f) => setWork("Разбираю видео…", "Распознаю движения: " + Math.round(f * 100) + "%", 0.05 + f * 0.9));
      await showAnalysis(frames);
    } catch (err) {
      console.error(err);
      const msg = err && err.message === "video" ? "Не получилось открыть это видео. Попробуйте MP4 или MOV."
        : err && err.message === "model" ? "Не удалось загрузить модель распознавания. Нужен интернет при первом запуске."
        : "Что-то пошло не так при разборе. Попробуйте ещё раз или другое видео.";
      setWork("Ошибка", msg, 0);
      $("workTitle").classList.add("error");
    }
  }

  // Отдельная точка входа: разбор готовых кадров (используется и в тестах).
  async function showAnalysis(frames) {
    $("workTitle").classList.remove("error");
    const result = A.analyze(frames, { mode: prefs.mode, hand: prefs.hand });
    if (result.error) {
      const text = result.error === "nopose"
        ? "Не вижу игрока. Снимите так, чтобы игрок был виден целиком, с головы до ног, и ближе к камере."
        : "Не нашёл ни одного удара. Проверьте, что выбран правильный тип (удары с отскока или подача) и на видео есть полные замахи.";
      setWork("Не получилось разобрать", text, 0);
      return null;
    }
    $("workCard").hidden = true;
    const session = {
      id: Date.now().toString(36),
      date: new Date().toISOString(),
      mode: prefs.mode,
      hand: prefs.hand,
      overall: result.overall,
      level: result.level.name,
      types: result.types,
      thumb: await thumbnail(result.strokes[0].t),
    };
    sessions.push(session);
    if (!store.set("sessions", sessions)) {
      // Место закончилось: сначала выбрасываем картинки у старых разборов.
      sessions.forEach((s, i) => { if (i < sessions.length - 1) delete s.thumb; });
      store.set("sessions", sessions);
    }
    current = { frames, result, session };
    renderResult();
    return result;
  }

  function thumbnail(t) {
    const video = $("video");
    if (!video.videoWidth) return Promise.resolve(null);
    return seek(video, t).then(() => {
      const c = document.createElement("canvas");
      c.width = 128; c.height = Math.round(128 * video.videoHeight / video.videoWidth);
      c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
      try { return c.toDataURL("image/jpeg", 0.6); } catch (e) { return null; }
    });
  }

  function colorFor(s) { return s >= 80 ? "var(--good)" : s >= 60 ? "var(--mid)" : "var(--bad)"; }
  function labelFor(s) { return s >= 80 ? "Хорошо" : s >= 60 ? "Можно лучше" : "Исправить"; }

  function ring(score) {
    const r = 44, c = 2 * Math.PI * r, f = Math.max(0, Math.min(100, score)) / 100;
    return `<svg class="ring" viewBox="0 0 104 104" role="img" aria-label="Оценка ${score} из 100">
      <circle cx="52" cy="52" r="${r}" fill="none" stroke="var(--panel-2)" stroke-width="10"/>
      <circle cx="52" cy="52" r="${r}" fill="none" stroke="${colorFor(score)}" stroke-width="10" stroke-linecap="round"
        stroke-dasharray="${c * f} ${c}" transform="rotate(-90 52 52)"/>
      <text x="52" y="63" text-anchor="middle">${score}</text></svg>`;
  }

  function renderResult() {
    const { result } = current;
    $("result").hidden = false;

    $("chips").innerHTML = result.strokes.map((s, i) =>
      `<button class="chip" data-t="${s.t}"><i class="dot" style="background:${colorFor(s.score)}"></i>${i + 1}. ${K.STROKES[s.type].short} · ${s.score}</button>`
    ).join("");

    const lv = result.level;
    const counts = Object.entries(result.types).map(([t, v]) => `${K.STROKES[t].short}: ${v.count}`).join(", ");
    $("summaryCard").innerHTML = `
      <div class="summary">
        ${ring(result.overall)}
        <div>
          <p class="level">${esc(lv.name)}</p>
          <span class="pill">≈ NTRP ${esc(lv.ntrp)}</span><span class="pill">Ударов: ${result.strokes.length}</span>
          <p class="dim" style="margin-top:8px;font-size:14px">${esc(lv.text)}</p>
        </div>
      </div>
      <p class="dim" style="font-size:13px;margin:10px 0 0">Найдено: ${esc(counts)}. Уровень — ориентир по технике на этом видео, а не официальный рейтинг.</p>`;

    $("typeCards").innerHTML = Object.entries(result.types).map(([type, data]) => typeCard(type, data)).join("");
    $("drillsCard").innerHTML = "<h2>Упражнения с партнёром</h2>" + drillsFor(weakest(result.types), true);
  }

  function typeCard(type, data) {
    const entries = Object.entries(data.metrics).sort((a, b) => a[1].s - b[1].s);
    const good = entries.filter(([, m]) => m.s >= 80).reverse();
    const fix = entries.filter(([, m]) => m.s < 65);
    return `<div class="card">
      <h2>${esc(K.STROKES[type].name)} — ${data.score}</h2>
      <p class="dim" style="font-size:14px">Ударов: ${data.count} · стабильность ${data.consistency}/100</p>
      ${good.length ? `<h3>Что получается</h3><ul class="list-good">${good.map(([k]) => `<li>${esc(K.METRICS[k].good)}</li>`).join("")}</ul>` : ""}
      ${fix.length ? `<h3>Что исправить в первую очередь</h3><ul class="list-fix">${fix.slice(0, 3).map(([k]) => `<li>${esc(K.METRICS[k].fix)}</li>`).join("")}</ul>` : ""}
      <h3>Разбор по элементам</h3>
      ${entries.map(([k, m]) => metricRow(k, m)).join("")}
    </div>`;
  }

  function metricRow(k, m) {
    const def = K.METRICS[k];
    const ok = m.s >= 80;
    return `<div class="metric">
      <div class="metric-head"><b>${esc(def.title)}</b><span style="color:${colorFor(m.s)}">${labelFor(m.s)} · ${m.s}</span></div>
      <div class="bar"><i style="width:${m.s}%;background:${colorFor(m.s)}"></i></div>
      <p>${esc(ok ? def.good : def.fix)}</p>
      ${ok ? "" : `<div class="cue">${esc(def.cue)}</div>`}
      <details><summary>Почему это важно</summary><p style="margin-top:6px">${esc(def.why)}</p></details>
    </div>`;
  }

  // Слабые места: метрики с самыми низкими оценками.
  function weakest(types) {
    const all = [];
    for (const d of Object.values(types)) for (const [k, m] of Object.entries(d.metrics)) all.push([k, m.s]);
    return all.sort((a, b) => a[1] - b[1]).filter(([, s]) => s < 80).map(([k]) => k)
      .filter((k, i, a) => a.indexOf(k) === i).slice(0, 3);
  }

  function drillsFor(keys, withIntro) {
    if (!keys.length) {
      return `<p class="dim">Серьёзных ошибок не видно. Поддерживайте форму игровыми упражнениями:</p>` +
        ["crossRally", "deepRally", "splitStep"].map((d) => drillHtml(d)).join("");
    }
    const ids = [];
    keys.forEach((k) => K.METRICS[k].drills.slice(0, 2).forEach((d) => { if (!ids.includes(d)) ids.push(d); }));
    return (withIntro ? `<p class="dim">Подобраны под ваши слабые места: ${keys.map((k) => esc(K.METRICS[k].title.toLowerCase())).join(", ")}.</p>` : "") +
      ids.slice(0, 5).map((d) => drillHtml(d)).join("");
  }

  function drillHtml(id, check) {
    const d = K.DRILLS[id];
    return `<div class="drill"><b>${esc(d.name)}</b><p>${esc(d.partner)}</p>
      <p class="meta">${esc(d.dose)} · Цель: ${esc(d.goal)}</p>
      ${check ? `<label><input type="checkbox" data-drill="${id}" ${check.done ? "checked" : ""}> Сделано</label>` : ""}</div>`;
  }

  // ---------- Плеер со скелетом ----------
  const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
  function drawOverlay() {
    const video = $("video"), cv = $("overlay");
    if (!current || !video.videoWidth) return;
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    cv.width = rect.width * dpr; cv.height = rect.height * dpr;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    // Видео вписано в элемент с сохранением пропорций: считаем масштаб и отступы.
    const s = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight) * dpr;
    const ox = (cv.width - video.videoWidth * s) / 2, oy = (cv.height - video.videoHeight * s) / 2;
    let f = null;
    for (const fr of current.frames) if (!f || Math.abs(fr.t - video.currentTime) < Math.abs(f.t - video.currentTime)) f = fr;
    if (!f || !f.p || Math.abs(f.t - video.currentTime) > 0.2) return;
    const P = (i) => [ox + f.p[i][0] * s, oy + f.p[i][1] * s];
    ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(255, 211, 77, 0.9)"; ctx.lineCap = "round";
    for (const [a, b] of BONES) {
      if (f.p[a][2] < 0.4 || f.p[b][2] < 0.4) continue;
      ctx.beginPath(); ctx.moveTo(...P(a)); ctx.lineTo(...P(b)); ctx.stroke();
    }
    ctx.fillStyle = "#fff";
    for (const i of [0, 15, 16, 25, 26, 27, 28]) {
      if (f.p[i][2] < 0.4) continue;
      ctx.beginPath(); ctx.arc(...P(i), 4 * dpr, 0, Math.PI * 2); ctx.fill();
    }
  }
  (function loop() { if (!$("result").hidden) drawOverlay(); requestAnimationFrame(loop); })();

  $("chips").addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    const v = $("video");
    v.pause();
    v.currentTime = Math.max(0, parseFloat(b.dataset.t) - 0.05);
    v.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // ---------- Прогресс ----------
  function fmtDate(iso) { return new Date(iso).toLocaleDateString("ru-RU", { day: "numeric", month: "short" }); }

  function lineChart(points) {
    const W = 640, H = 200, pl = 30, pr = 12, pt = 12, pb = 26;
    const n = points.length;
    const x = (i) => pl + (n === 1 ? (W - pl - pr) / 2 : (i * (W - pl - pr)) / (n - 1));
    const y = (v) => pt + (1 - v / 100) * (H - pt - pb);
    const grid = [0, 50, 100].map((v) => `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${pl - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("");
    const path = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.v)}`).join(" ");
    const step = Math.max(1, Math.ceil(n / 6));
    const labels = points.map((p, i) => (i % step === 0 || i === n - 1) ? `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${esc(p.label)}</text>` : "").join("");
    const dots = points.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.v)}" r="4" fill="var(--ball)"><title>${esc(p.label)}: ${p.v}</title></circle>`).join("");
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="График общей оценки по разборам">${grid}<path d="${path}" fill="none" stroke="var(--ball)" stroke-width="2.5"/>${dots}${labels}</svg>`;
  }

  // История каждой метрики по всем разборам: [{date, s}], от старых к новым.
  function metricHistory() {
    const h = {};
    for (const s of sessions) for (const d of Object.values(s.types)) for (const [k, m] of Object.entries(d.metrics)) {
      (h[k] = h[k] || []).push({ date: s.date, s: m.s });
    }
    return h;
  }

  // Над чем работать: средняя оценка за последние три разбора по каждому элементу.
  function focusAreas() {
    const h = metricHistory();
    return Object.entries(h).map(([k, list]) => {
      const last = list.slice(-3);
      return [k, Math.round(last.reduce((a, b) => a + b.s, 0) / last.length)];
    }).sort((a, b) => a[1] - b[1]).filter(([, s]) => s < 80).slice(0, 3);
  }

  function renderProgress() {
    const el = $("tab-progress");
    if (!sessions.length) {
      el.innerHTML = `<div class="card empty"><h2>Пока пусто</h2><p class="dim">Загрузите первое видео во вкладке «Разбор», и здесь появятся уровень, график и слабые места.</p></div>`;
      return;
    }
    const last = sessions[sessions.length - 1];
    const lv = A.levelFor(last.overall);
    const first = sessions[0];
    const delta = last.overall - first.overall;
    const hist = metricHistory();
    const focus = focusAreas();

    const trends = Object.entries(hist).map(([k, list]) => {
      const now = list[list.length - 1].s;
      const prev = list.length > 1 ? list[list.length - 2].s : null;
      const d = prev == null ? "" : now - prev;
      return `<span>${esc(K.METRICS[k].title)}</span>
        <span class="${d > 0 ? "up" : d < 0 ? "down" : "dim"}">${d === "" ? "—" : (d > 0 ? "▲ +" : d < 0 ? "▼ " : "= ") + (d || "")}</span>
        <span class="v" style="color:${colorFor(now)}">${now}</span>`;
    }).join("");

    el.innerHTML = `
      <div class="card">
        <div class="summary">${ring(last.overall)}
          <div><p class="level">${esc(lv.name)}</p><span class="pill">≈ NTRP ${esc(lv.ntrp)}</span><span class="pill">Разборов: ${sessions.length}</span>
          ${sessions.length > 1 ? `<p style="margin-top:8px" class="${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "+" : ""}${delta} с первого разбора</p>` : ""}</div>
        </div>
      </div>
      <div class="card"><h2>Общая оценка</h2>${lineChart(sessions.map((s) => ({ v: s.overall, label: fmtDate(s.date) })))}</div>
      <div class="card"><h2>Над чем работать</h2>
        ${focus.length ? `<ul class="list-fix">${focus.map(([k, s]) => `<li><b>${esc(K.METRICS[k].title)}</b> — ${s}/100. <span class="dim">${esc(K.METRICS[k].cue)}</span></li>`).join("")}</ul>
        <button class="btn ghost" style="margin-top:12px" data-goto="plan">План тренировки с партнёром →</button>`
          : `<p class="dim">Все элементы на хорошем уровне. Продолжайте загружать видео, чтобы не терять форму.</p>`}
      </div>
      <div class="card"><h2>Элементы техники</h2><p class="dim" style="font-size:13px">Последняя оценка и изменение к предыдущему разбору</p><div class="trend">${trends}</div></div>
      <div class="card"><h2>История разборов</h2><div class="history">
        ${sessions.slice().reverse().map((s) => `<div class="hist-item">
          ${s.thumb ? `<img src="${s.thumb}" alt="">` : `<div class="ph"></div>`}
          <div><b>${s.mode === "serve" ? "Подача" : Object.keys(s.types).map((t) => K.STROKES[t].short).join(" + ")} · ${s.overall}</b>
          <small>${new Date(s.date).toLocaleString("ru-RU", { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })} · ${esc(s.level)}</small></div>
          <button class="icon-btn" data-del="${s.id}" aria-label="Удалить разбор">✕</button></div>`).join("")}
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px">
        <button class="btn ghost" id="exportBtn">Сохранить в файл</button>
        <label class="btn ghost" style="position:relative">Загрузить из файла<input type="file" id="importInput" accept="application/json" style="position:absolute;inset:0;opacity:0;width:100%"></label>
      </div></div>`;
  }

  $("tab-progress").addEventListener("click", (e) => {
    const del = e.target.closest("[data-del]");
    if (del && confirm("Удалить этот разбор из истории?")) {
      sessions = sessions.filter((s) => s.id !== del.dataset.del);
      store.set("sessions", sessions);
      renderProgress();
    }
    if (e.target.closest("[data-goto]")) showTab("plan");
    if (e.target.id === "exportBtn") {
      const blob = new Blob([JSON.stringify({ sessions, plan: store.get("plan", {}) }, null, 1)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "tennis-coach-" + new Date().toISOString().slice(0, 10) + ".json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }
  });
  $("tab-progress").addEventListener("change", async (e) => {
    if (e.target.id !== "importInput" || !e.target.files[0]) return;
    try {
      const data = JSON.parse(await e.target.files[0].text());
      const known = new Set(sessions.map((s) => s.id));
      const incoming = (data.sessions || []).filter((s) => s && s.id && s.types && !known.has(s.id));
      sessions = sessions.concat(incoming).sort((a, b) => a.date.localeCompare(b.date));
      store.set("sessions", sessions);
      renderProgress();
    } catch (err) {
      alert("Не получилось прочитать файл.");
    }
  });

  // ---------- План тренировки ----------
  function renderPlan() {
    const el = $("tab-plan");
    const focus = focusAreas().map(([k]) => k);
    const done = store.get("plan", {});
    const ids = [];
    focus.forEach((k) => K.METRICS[k].drills.slice(0, 2).forEach((d) => { if (!ids.includes(d)) ids.push(d); }));
    if (!ids.length) ids.push("dropFeed", "contactCone");
    const plan = ids.slice(0, 4);
    const game = "crossRally";

    el.innerHTML = `
      <div class="card"><h2>Тренировка с партнёром</h2>
        <p class="dim">${focus.length ? "Фокус: " + focus.map((k) => esc(K.METRICS[k].title.toLowerCase())).join(", ") + "." : "Загрузите видео, и план подстроится под ваши слабые места. Пока — базовая техника."}</p>
        <p class="dim" style="font-size:14px">≈ 60–75 минут. Меняйтесь ролями: сначала вы бьёте, а партнёр набрасывает и смотрит, потом наоборот.</p>
      </div>
      <div class="card"><h3 style="margin-top:0">1. Разминка · 10 минут</h3>
        <p class="dim" style="font-size:14px">Лёгкий бег, приставные шаги, махи руками и ногами. Затем мини-теннис от линий подачи: мягко, с полной техникой, 3–4 минуты.</p>
        <h3>2. Техника · 35–45 минут</h3>
        ${plan.map((d) => drillHtml(d, { done: !!done[d] })).join("")}
        <h3>3. Игровое упражнение · 15 минут</h3>
        ${drillHtml(game, { done: !!done[game] })}
        <h3>4. После тренировки</h3>
        <p class="dim" style="font-size:14px">Попросите партнёра снять 10–20 секунд ударов сбоку и загрузите их во вкладке «Разбор». Так вы увидите, перешли ли упражнения в игру.</p>
        <button class="btn ghost" id="resetPlan" style="margin-top:8px">Сбросить отметки</button>
      </div>
      <div class="card"><h2>Все упражнения</h2>
        ${Object.keys(K.DRILLS).map((d) => drillHtml(d)).join("")}
      </div>`;
  }
  $("tab-plan").addEventListener("change", (e) => {
    const id = e.target.dataset.drill;
    if (!id) return;
    const done = store.get("plan", {});
    done[id] = e.target.checked;
    store.set("plan", done);
  });
  $("tab-plan").addEventListener("click", (e) => {
    if (e.target.id === "resetPlan") { store.set("plan", {}); renderPlan(); }
  });

  // Для автоматических проверок.
  window.CoachApp = { showAnalysis, showTab, prefs };
})();
