// Глубокий разбор через Claude: ключевые кадры каждого удара (замах, контакт, завершение)
// плюс измерения по позе уходят в модель, а обратно приходит разбор в виде JSON по схеме.
// Ключ API хранится только в браузере пользователя и уходит только в api.anthropic.com.
(function (root) {
  "use strict";

  const params = new URLSearchParams(location.search);
  const SDK_URL = params.get("sdk") || "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.131.0/+esm";
  const MODEL = "claude-opus-5-5";
  const MAX_STROKES = 12;            // больше кадров — дороже и не точнее
  const PHASES = [-0.45, 0, 0.4];    // замах, контакт, завершение (секунды от контакта)
  const FRAME_SIDE = 512;

  const STROKE_TYPES = ["forehand", "backhand_two_handed", "backhand_one_handed", "slice", "volley", "smash", "serve", "return", "other"];
  const TYPE_NAMES = {
    forehand: "Форхенд", backhand_two_handed: "Бэкхенд двумя руками", backhand_one_handed: "Бэкхенд одной рукой",
    slice: "Резаный удар", volley: "Удар с лёта", smash: "Смэш", serve: "Подача", return: "Приём подачи", other: "Другое",
  };

  const str = { type: "string" };
  const obj = (props) => ({ type: "object", properties: props, required: Object.keys(props), additionalProperties: false });
  const arr = (items) => ({ type: "array", items });
  const SCHEMA = obj({
    summary: str,
    level: obj({ ntrp: str, comment: str }),
    strokes: arr(obj({
      index: { type: "integer" },
      is_stroke: { type: "boolean" },
      type: { type: "string", enum: STROKE_TYPES },
      grip: str,
      verdict: str,
      good: arr(str),
      fix: arr(str),
    })),
    strengths: arr(obj({ title: str, detail: str })),
    priorities: arr(obj({ title: str, why: str, cue: str, pro_reference: str })),
    pro_comparison: str,
    drills: arr(obj({ name: str, partner: str, dose: str, goal: str })),
    next_video_tip: str,
  });

  const SYSTEM = `Ты — опытный теннисный тренер. Ты работаешь по методикам ITF, USTA и PTR и знаешь, как ставят технику ведущие тренеры (Мамулашвили, Макки, Муратоглу, Кахилл и другие).
Тебе присылают кадры ударов игрока-любителя. По каждому удару идут три кадра: замах, момент контакта и завершение. Ещё приходят измерения по позе, которые посчитал алгоритм.

Как работать:
- Сначала смотри на кадры, а не на цифры. Измерения по позе бывают неточными, особенно при съёмке сзади или издалека. Если кадр противоречит цифре, верь кадру.
- Кандидаты в удары алгоритм нашёл по скорости рук, и среди них бывают ложные (игрок просто идёт или поправляет одежду). Если на кадрах нет удара по мячу, поставь is_stroke = false и не разбирай его.
- Определи тип каждого удара и, если видно, хват (континентальный, восточный, полузападный, западный).
- Пиши по-русски, тепло и конкретно, обращайся к игроку на «ты» и по имени.
- Учитывай уровень игрока. Для начинающих (NTRP 1.5–2.5) главное — база: подготовка, разворот, точка удара, баланс, завершение. Не перегружай: выбери 2–3 приоритета, которые дадут наибольший рост.
- В priorities для каждого пункта дай короткую установку («cue»), которую можно повторять на корте, и сравнение с профессионалом. Для pro_reference выбирай игрока с похожим стилем: например, форхенд Свёнтек или Синнера, бэкхенд двумя руками Джокович или Соболенко. Опиши, что именно у профессионала делается иначе.
- Упражнения давай для пары: что делает игрок, что делает партнёр, объём.
- Если на кадре что-то не видно (игрок вне кадра, смазано), так и скажи. Не выдумывай.`;

  let sdkPromise = null;
  function loadSdk() {
    if (!sdkPromise) {
      sdkPromise = import(SDK_URL).then((m) => m.default || m.Anthropic);
      sdkPromise.catch(() => { sdkPromise = null; });
    }
    return sdkPromise;
  }

  function seek(video, t) {
    return new Promise((resolve) => {
      const done = () => { video.removeEventListener("seeked", done); resolve(); };
      video.addEventListener("seeked", done);
      video.currentTime = Math.max(0, Math.min(video.duration - 0.05, t));
    });
  }

  // Рамка вокруг игрока по точкам позы (объединение по всем фазам удара) с запасом под ракетку.
  function playerBox(frames, t0, t1, W, H) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const f of frames) {
      if (f.t < t0 || f.t > t1 || !f.p) continue;
      for (const q of f.p) {
        if (q[2] < 0.3) continue;
        x0 = Math.min(x0, q[0]); y0 = Math.min(y0, q[1]); x1 = Math.max(x1, q[0]); y1 = Math.max(y1, q[1]);
      }
    }
    if (!isFinite(x0)) return { x: 0, y: 0, w: W, h: H };
    // Сверху и по бокам оставляем место под ракетку (она длиннее руки), снизу немного корта.
    const h = (y1 - y0) * 1.35, w = Math.max(x1 - x0 + h * 0.5, h * 0.75);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2 - (y1 - y0) * 0.08;
    const side = Math.min(Math.max(w, h), Math.max(W, H));
    const bw = Math.min(W, side), bh = Math.min(H, side);
    return { x: Math.max(0, Math.min(W - bw, cx - bw / 2)), y: Math.max(0, Math.min(H - bh, cy - bh / 2)), w: bw, h: bh };
  }

  // Удары для разбора: равномерно по видео, не больше MAX_STROKES.
  function pickStrokes(strokes) {
    if (strokes.length <= MAX_STROKES) return strokes.map((s, i) => ({ ...s, n: i + 1 }));
    const out = [];
    for (let k = 0; k < MAX_STROKES; k++) {
      const i = Math.round((k * (strokes.length - 1)) / (MAX_STROKES - 1));
      out.push({ ...strokes[i], n: i + 1 });
    }
    return out;
  }

  async function captureKeyFrames(video, frames, strokes, onProgress) {
    const W = video.videoWidth, H = video.videoHeight;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const picked = pickStrokes(strokes);
    const shots = [];
    let done = 0;
    for (const s of picked) {
      const box = playerBox(frames, s.t + PHASES[0], s.t + PHASES[PHASES.length - 1], W, H);
      const scale = Math.min(2.5, FRAME_SIDE / Math.max(box.w, box.h));
      canvas.width = Math.round(box.w * scale); canvas.height = Math.round(box.h * scale);
      const images = [];
      for (const dt of PHASES) {
        await seek(video, s.t + dt);
        ctx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, canvas.width, canvas.height);
        images.push(canvas.toDataURL("image/jpeg", 0.82));
        onProgress && onProgress(++done / (picked.length * PHASES.length));
      }
      shots.push({ n: s.n, t: s.t, type: s.type, metrics: s.metrics, images });
    }
    return shots;
  }

  function poseNotes(shot, K) {
    const lines = Object.entries(shot.metrics || {}).map(([k, m]) => `${K.METRICS[k].title}: ${m.s}/100`);
    return lines.length ? lines.join("; ") : "нет надёжных измерений";
  }

  async function analyze({ apiKey, shots, session, profile, K }) {
    const Anthropic = await loadSdk();
    const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });

    const content = [{
      type: "text",
      text: `Игрок: ${profile.name || "игрок"}, ${profile.hand === "left" ? "левша" : "правша"}, уровень по самооценке NTRP ${profile.level || "не указан"}.` +
        (profile.goal ? ` Цель: ${profile.goal}.` : "") +
        ` Видео: ${session.mode === "serve" ? "подачи" : "удары с отскока в игре"}, снято ${session.view === "back" ? "сзади" : "сбоку"}.` +
        (session.view === "back" ? " При съёмке сзади точку удара и проводку по позе не измерить, оценивай их по кадрам." : ` Оценка по позе: ${session.overall}/100.`) +
        ` Ниже ${shots.length} удар(ов), у каждого 3 кадра: замах, контакт, завершение.`,
    }];
    for (const shot of shots) {
      content.push({ type: "text", text: `Удар №${shot.n} (${shot.t.toFixed(1)} с), алгоритм считает: ${K.STROKES[shot.type] ? K.STROKES[shot.type].short : shot.type}. Измерения по позе: ${poseNotes(shot, K)}.` });
      shot.images.forEach((url) => content.push({
        type: "image",
        source: { type: "base64", media_type: "image/jpeg", data: url.slice(url.indexOf(",") + 1) },
      }));
    }
    content.push({ type: "text", text: "Сделай разбор строго по схеме. В strokes укажи index — номер удара из подписи." });

    const stream = client.beta.messages.stream({
      model: MODEL,
      max_tokens: 32000,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      output_config: { effort: "high", format: { type: "json_schema", schema: SCHEMA } },
      system: SYSTEM,
      messages: [{ role: "user", content }],
    });
    const msg = await stream.finalMessage();
    if (msg.stop_reason === "refusal") throw new Error("refusal");
    if (msg.stop_reason === "max_tokens") throw new Error("truncated");
    const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
    return JSON.parse(text);
  }

  // Понятное сообщение об ошибке по типу исключения SDK.
  async function describeError(err) {
    let A = null;
    try { A = await loadSdk(); } catch (e) { return "Не удалось загрузить модуль Claude. Проверьте интернет."; }
    if (err && err.message === "refusal") return "Claude отказался разбирать это видео. Попробуйте другое.";
    if (err && err.message === "truncated") return "Разбор получился слишком длинным и оборвался. Попробуйте ролик покороче.";
    if (A && err instanceof A.AuthenticationError) return "Ключ API не подходит. Проверьте его в настройках.";
    if (A && err instanceof A.PermissionDeniedError) return "У ключа нет доступа к модели. Проверьте настройки аккаунта Anthropic.";
    if (A && err instanceof A.RateLimitError) return "Слишком много запросов. Подождите минуту и попробуйте снова.";
    if (A && err instanceof A.BadRequestError) return "Запрос отклонён: " + err.message;
    if (A && err instanceof A.APIConnectionError) return "Нет связи с сервером Anthropic. Проверьте интернет.";
    if (A && err instanceof A.APIError) return "Ошибка сервера (" + err.status + "). Попробуйте позже.";
    return "Не получилось сделать глубокий разбор.";
  }

  root.TennisAI = { captureKeyFrames, analyze, describeError, TYPE_NAMES, MODEL };
})(window);
